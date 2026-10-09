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
    posts: list[tuple[str, dict, object]] = []

    def fake_post(url, *, headers, timeout, json=None, data=None):  # noqa: ANN001
        posts.append((url, headers, json))
        if "/storage/auth/token" in url:
            assert headers["Authorization"] == "Key test-key"
            return _Response({"base_url": "https://v3.fal.media", "token": "cdn", "token_type": "Bearer"})
        if url == "https://v3.fal.media/files/upload":
            assert headers["Authorization"] == "Bearer cdn"
            return _Response({"access_url": "https://files.test/" + headers["X-Fal-File-Name"]})
        return _Response({"status_url": "https://queue.test/status", "response_url": "https://queue.test/result"})

    def fake_get(url, *, headers=None, timeout):  # noqa: ANN001
        if url == "https://queue.test/status":
            return _Response({"status": "COMPLETED"})
        if url == "https://queue.test/result":
            return _Response({"video": {"url": "https://media.test/out.mp4"}})
        return _Response(content=b"synced")

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("tools.avatar.sync_lipsync.time.sleep", lambda _: None)
    monkeypatch.setattr("tools.video._shared.probe_output", lambda _: {"duration_seconds": 8.0})

    output = tmp_path / "out.mp4"
    result = SyncLipsync().execute(
        {"video_path": str(video), "audio_path": str(audio), "output_path": str(output)}
    )

    assert result.success, result.error
    assert output.read_bytes() == b"synced"
    assert [url for url, _, _ in posts[:3]] == [
        "https://rest.alpha.fal.ai/storage/auth/token?storage_type=fal-cdn-v3",
        "https://v3.fal.media/files/upload",
        "https://v3.fal.media/files/upload",
    ]
    submit_url, _, payload = posts[-1]
    assert submit_url == "https://queue.fal.run/fal-ai/sync-lipsync/v2"
    assert payload == {
        "model": "lipsync-2",
        "video_url": "https://files.test/clip.mp4",
        "audio_url": "https://files.test/line.wav",
        "sync_mode": "cut_off",
    }
    assert result.cost_usd == pytest.approx(0.4)


def test_rejects_foreign_cdn_host(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "c.mp4").write_bytes(b"v")
    (tmp_path / "a.wav").write_bytes(b"a")
    posts = []

    def fake_post(url, **kw):  # noqa: ANN001
        posts.append(url)
        return _Response({"base_url": "https://evil.test", "token": "t", "token_type": "Bearer"})

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("requests.post", fake_post)
    result = SyncLipsync().execute({"video_path": str(tmp_path / "c.mp4"), "audio_path": str(tmp_path / "a.wav")})
    assert not result.success
    assert len(posts) == 1


def test_rejects_unsupported_input_before_any_paid_submit(monkeypatch, tmp_path: Path) -> None:
    posts = []
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda url, **kw: posts.append(url))
    result = SyncLipsync().execute(
        {"video_path": str(tmp_path / "missing.mp4"), "audio_path": str(tmp_path / "a.txt")}
    )
    assert not result.success
    assert posts == []


def test_fal_cdn_upload_is_free_keeps_only_cdn_token_and_submit_is_billable(monkeypatch) -> None:
    from lib.providers import worker

    events = []
    monkeypatch.setattr(worker, "public_target", lambda url: urlsplit(url).hostname)
    monkeypatch.setattr(worker, "emit", lambda event, **kwargs: events.append(event))
    recorder = StageRecorder("fal")
    token = recorder.before(
        "POST", "https://rest.alpha.fal.ai/storage/auth/token?storage_type=fal-cdn-v3", {}, b"{}"
    )
    recorder.after(token, 200, json.dumps({"base_url": "https://v3.fal.media", "token": "t", "token_type": "Bearer"}).encode())
    cdn_headers = {"Authorization": "Bearer t"}
    recorder.before("POST", "https://v3.fal.media/files/upload", cdn_headers, b"video")
    assert cdn_headers == {"Authorization": "Bearer t"}
    leaked = {"Authorization": "Key account-secret"}
    recorder.before("POST", "https://v3.fal.media/files/upload", leaked, b"audio")
    assert "Authorization" not in leaked
    with pytest.raises(RuntimeError, match="Unregistered"):
        recorder.before("POST", "https://other.fal.media/files/upload", {}, b"x")
    recorder.before(
        "POST", "https://queue.fal.run/fal-ai/sync-lipsync/v2", {}, json.dumps({"model": "lipsync-2"}).encode()
    )
    assert [event["billable"] for event in events if event["event"] == "sending"] == [False, False, False, True]
