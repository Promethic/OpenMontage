from __future__ import annotations

from pathlib import Path

import pytest

import tools.video._shared as video_shared
from tools.base_tool import ToolStatus
from tools.video.higgsfield_video import HiggsFieldVideo
from lib.providers.context import service_worker


_CREDENTIAL_NAMES = (
    "HF_KEY",
    "HF_CREDENTIALS",
    "HF_API_KEY_ID",
    "HF_API_KEY_SECRET",
    "HIGGSFIELD_KEY",
    "HIGGSFIELD_API_KEY",
    "HIGGSFIELD_API_SECRET",
)


class FakeResponse:
    def __init__(self, payload=None, *, content=b""):
        self._payload = payload or {}
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture
def trusted_worker():
    with service_worker():
        yield


@pytest.fixture
def clean_credentials(monkeypatch):
    for name in _CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_current_and_legacy_credentials_are_supported(monkeypatch, clean_credentials):
    tool = HiggsFieldVideo()
    assert tool.get_status() == ToolStatus.UNAVAILABLE

    monkeypatch.setenv("HF_API_KEY_ID", "current-id")
    monkeypatch.setenv("HF_API_KEY_SECRET", "current-secret")
    assert tool._get_credentials() == ("current-id", "current-secret")

    monkeypatch.delenv("HF_API_KEY_ID")
    monkeypatch.delenv("HF_API_KEY_SECRET")
    monkeypatch.setenv("HIGGSFIELD_KEY", "legacy-id:legacy-secret")
    assert tool._get_credentials() == ("legacy-id", "legacy-secret")


def test_model_endpoints_and_costs_match_current_catalog():
    tool = HiggsFieldVideo()

    assert tool._endpoint("seedance_2.5", "text_to_video") == (
        "bytedance/seedance-2.5/text-to-video"
    )
    assert tool._endpoint("kling_3.0_standard", "text_to_video") == (
        "kling-video/v3.0/std/text-to-video"
    )
    assert tool.estimate_cost(
        {"model": "seedance_2.5", "operation": "text_to_video", "duration": 5}
    ) == pytest.approx(0.72)
    assert tool.estimate_cost(
        {"model": "kling_3.0_standard", "operation": "text_to_video", "duration": 5}
    ) == pytest.approx(0.63)
    assert tool.estimate_cost(
        {"model": "kling_3.0_standard", "duration": 8, "generate_audio": False}
    ) == pytest.approx(0.672)


def test_rejects_unsupported_model_operation():
    tool = HiggsFieldVideo()
    with pytest.raises(ValueError, match="does not support"):
        tool._endpoint("kling_3.0_standard", "reference_to_video")


def test_kling_standard_image_payload_uses_native_audio_and_reference_dimensions():
    endpoint, payload = HiggsFieldVideo()._build_payload(
        {
            "prompt": "A Polish mother speaks naturally",
            "model": "kling_3.0_standard",
            "operation": "image_to_video",
            "duration": 8,
            "image_url": "https://cdn.example.test/mother.png",
            "end_image_url": "https://cdn.example.test/end.png",
            "aspect_ratio": "9:16",
            "generate_audio": True,
        },
        {},
        None,
    )
    assert endpoint == "kling-video/v3.0/std/image-to-video"
    assert payload == {
        "prompt": "A Polish mother speaks naturally",
        "duration": 8,
        "sound": "on",
        "cfg_scale": 0.5,
        "multi_shots": False,
        "image_url": "https://cdn.example.test/mother.png",
        "last_image_url": "https://cdn.example.test/end.png",
    }


def test_kling_duration_is_checked_before_upload_or_submit():
    with pytest.raises(ValueError, match="accepts duration 3-15s"):
        HiggsFieldVideo()._build_payload(
            {"prompt": "test", "model": "kling_3.0_standard", "duration": 16},
            {},
            None,
        )


def test_higgsfield_cannot_submit_outside_studio(monkeypatch, clean_credentials):
    monkeypatch.setenv("HIGGSFIELD_KEY", "test-id:test-secret")
    monkeypatch.delenv("STUDIO_API_URL", raising=False)
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    result = HiggsFieldVideo().execute({"prompt": "test"})
    assert not result.success
    assert result.billing_status == "notSubmitted"


def test_execute_uses_current_auth_lifecycle_and_downloads_output(
    monkeypatch, clean_credentials, tmp_path, trusted_worker
):
    import requests

    monkeypatch.setenv("HF_KEY", "key-id:key-secret")
    calls = []

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        return FakeResponse(
            {
                "status": "queued",
                "request_id": "request-1",
                "status_url": "https://api.higgsfield.ai/requests/request-1/status",
            }
        )

    def fake_get(url, **kwargs):
        calls.append(("get", url, kwargs))
        if url.endswith("/status"):
            return FakeResponse(
                {
                    "status": "completed",
                    "request_id": "request-1",
                    "video": {"url": "https://cdn.higgsfield.test/result.mp4"},
                }
            )
        return FakeResponse(content=b"video-bytes")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(video_shared, "probe_output", lambda path: {"duration": 5.0})

    output = tmp_path / "result.mp4"
    result = HiggsFieldVideo().execute(
        {
            "prompt": "A product rotates in soft studio light",
            "model": "seedance_2.5",
            "duration": 5,
            "output_path": str(output),
        }
    )

    assert result.success is True
    assert output.read_bytes() == b"video-bytes"
    assert result.data["gateway"] == "higgsfield"
    assert result.data["request_id"] == "request-1"
    assert result.data["model_id"] == "bytedance/seedance-2.5/text-to-video"
    assert calls[0][2]["headers"]["Authorization"] == "Key key-id:key-secret"


def test_local_image_uses_higgsfield_presigned_upload_without_api_credentials(
    monkeypatch, clean_credentials, tmp_path, trusted_worker
):
    import requests

    monkeypatch.setenv("HF_KEY", "key-id:key-secret")
    image = tmp_path / "input.png"
    image.write_bytes(b"png")
    puts = []

    def fake_post(url, **kwargs):
        if url.endswith("/files/generate-upload-url"):
            return FakeResponse(
                {
                    "upload_url": "https://storage.example.test/upload",
                    "public_url": "https://cdn.example.test/input.png",
                    "upload_headers": {"Content-Type": "image/png", "x-amz-tagging": "temporary"},
                }
            )
        return FakeResponse(
            {
                "request_id": "request-2",
                "status_url": "https://api.higgsfield.ai/requests/request-2/status",
            }
        )

    def fake_put(url, **kwargs):
        puts.append((url, kwargs))
        return FakeResponse()

    def fake_get(url, **kwargs):
        if url.endswith("/status"):
            return FakeResponse(
                {
                    "status": "completed",
                    "request_id": "request-2",
                    "video": {"url": "https://cdn.example.test/result.mp4"},
                }
            )
        return FakeResponse(content=b"video")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "put", fake_put)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(video_shared, "probe_output", lambda path: {})

    result = HiggsFieldVideo().execute(
        {
            "prompt": "Slow camera push",
            "operation": "image_to_video",
            "image_path": str(image),
            "output_path": str(tmp_path / "result.mp4"),
        }
    )

    assert result.success is True
    assert puts[0][0] == "https://storage.example.test/upload"
    assert "Authorization" not in puts[0][1]["headers"]


def test_untrusted_status_url_is_rejected(monkeypatch, clean_credentials, trusted_worker):
    import requests

    monkeypatch.setenv("HF_KEY", "key-id:key-secret")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {
                "request_id": "request-3",
                "status_url": "https://attacker.example/status",
            }
        ),
    )

    result = HiggsFieldVideo().execute({"prompt": "test"})
    assert result.success is False
    assert "untrusted status URL" in result.error


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.example.test/result.mp4",
        "https://127.0.0.1/result.mp4",
        "https://user:password@cdn.example.test/result.mp4",
    ],
)
def test_untrusted_external_media_url_is_rejected(url):
    with pytest.raises(ValueError, match="invalid|untrusted"):
        HiggsFieldVideo._safe_external_url(url, "output media")
