from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
pytestmark = pytest.mark.usefixtures("private_provider_adapter")

from lib.providers.worker import StageRecorder
from tools.avatar.sync_lipsync import SyncLipsync


class _Response:
    def __init__(self, payload: dict | None = None, *, content: bytes = b"") -> None:
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_cost_scales_with_billed_seconds() -> None:
    tool = SyncLipsync()
    assert tool.estimate_cost({"duration_seconds": 8}) == pytest.approx(0.4)
    assert tool.estimate_cost({"duration_seconds": 60, "model": "lipsync-2-pro"}) == pytest.approx(5.0)


def test_uploads_both_inputs_then_submits_once(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "line.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    posts: list[tuple[str, dict]] = []
    puts: list[str] = []

    def fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        posts.append((url, json))
        if url.endswith("/storage/upload/initiate"):
            name = json["file_name"]
            return _Response({"upload_url": f"https://upload.test/{name}", "file_url": f"https://files.test/{name}"})
        return _Response({"status_url": "https://queue.test/status", "response_url": "https://queue.test/result"})

    def fake_put(url, *, headers, data, timeout):  # noqa: ANN001
        puts.append(url)
        return _Response()

    def fake_get(url, *, headers=None, timeout):  # noqa: ANN001
        if url == "https://queue.test/status":
            return _Response({"status": "COMPLETED"})
        if url == "https://queue.test/result":
            return _Response({"video": {"url": "https://media.test/out.mp4"}})
        return _Response(content=b"synced")

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.put", fake_put)
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("tools.avatar.sync_lipsync.time.sleep", lambda _: None)
    monkeypatch.setattr("tools.video._shared.probe_output", lambda _: {"duration_seconds": 8.0})

    output = tmp_path / "out.mp4"
    result = SyncLipsync().execute(
        {"video_path": str(video), "audio_path": str(audio), "output_path": str(output)}
    )

    assert result.success, result.error
    assert output.read_bytes() == b"synced"
    assert puts == ["https://upload.test/clip.mp4", "https://upload.test/line.wav"]
    submit_url, payload = posts[-1]
    assert submit_url == "https://queue.fal.run/fal-ai/sync-lipsync/v2"
    assert payload == {
        "model": "lipsync-2",
        "video_url": "https://files.test/clip.mp4",
        "audio_url": "https://files.test/line.wav",
        "sync_mode": "cut_off",
    }
    assert result.cost_usd == pytest.approx(0.4)


def test_rejects_unsupported_input_before_any_paid_submit(monkeypatch, tmp_path: Path) -> None:
    posts = []
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda url, **kw: posts.append(url))
    result = SyncLipsync().execute(
        {"video_path": str(tmp_path / "missing.mp4"), "audio_path": str(tmp_path / "a.txt")}
    )
    assert not result.success
    assert posts == []


def test_fal_storage_upload_is_free_and_queue_submit_is_billable(monkeypatch) -> None:
    from lib.providers import worker

    events = []
    monkeypatch.setattr(worker, "public_target", lambda url: urlsplit(url).hostname)
    monkeypatch.setattr(worker, "emit", lambda event, **kwargs: events.append(event))
    recorder = StageRecorder("fal")
    recorder.before("POST", "https://rest.alpha.fal.ai/storage/upload/initiate", {}, b'{"file_name":"clip.mp4"}')
    recorder.before("PUT", "https://upload.test/clip.mp4", {"Authorization": "Key x"}, b"video")
    recorder.before(
        "POST", "https://queue.fal.run/fal-ai/sync-lipsync/v2", {}, json.dumps({"model": "lipsync-2"}).encode()
    )
    assert [event["billable"] for event in events if event["event"] == "sending"] == [False, False, True]
