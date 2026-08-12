from __future__ import annotations

from pathlib import Path

import pytest

from tools.video.kling_video import KlingVideo


class _Response:
    def __init__(self, payload: dict | None = None, *, content: bytes = b"") -> None:
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_v3_standard_cost_uses_current_per_second_rate() -> None:
    tool = KlingVideo()

    assert tool.estimate_cost({"model_variant": "v3/standard", "duration": "3"}) == pytest.approx(0.252)
    assert tool.estimate_cost(
        {"model_variant": "v3/standard", "duration": "3", "generate_audio": True}
    ) == pytest.approx(0.378)


def test_v3_standard_disables_native_audio_by_default(monkeypatch, tmp_path: Path) -> None:
    submitted: dict = {}

    def fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        submitted.update(json)
        return _Response(
            {
                "status_url": "https://queue.test/status",
                "response_url": "https://queue.test/result",
            }
        )

    def fake_get(url, *, headers=None, timeout):  # noqa: ANN001
        if url == "https://queue.test/status":
            return _Response({"status": "COMPLETED"})
        if url == "https://queue.test/result":
            return _Response({"video": {"url": "https://media.test/output.mp4"}})
        return _Response(content=b"video-bytes")

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("tools.video.kling_video.time.sleep", lambda _: None)
    monkeypatch.setattr("tools.video._shared.probe_output", lambda _: {"duration_seconds": 3.0})

    output = tmp_path / "clip.mp4"
    result = KlingVideo().execute(
        {
            "prompt": "A complete pot remains fully visible.",
            "model_variant": "v3/standard",
            "duration": "3",
            "aspect_ratio": "9:16",
            "output_path": str(output),
        }
    )

    assert result.success is True
    assert submitted["duration"] == "3"
    assert submitted["generate_audio"] is False
    assert output.read_bytes() == b"video-bytes"
