from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

from lib.providers.catalog import FREE_API_TOOLS, PAID_TOOLS, catalog, route_inputs
from lib.providers.context import service_worker
from lib.providers.worker import StageRecorder
from tools.base_tool import BaseTool, ToolResult, ToolRuntime
from tools.tool_registry import ToolRegistry


def test_every_paid_api_tool_is_classified_and_guarded():
    registry = ToolRegistry()
    registry.discover()
    paid = {
        name
        for name, tool in registry._tools.items()
        if tool.runtime == ToolRuntime.API and name not in FREE_API_TOOLS
    }
    assert paid <= PAID_TOOLS.keys()
    for name in PAID_TOOLS:
        tool = registry.get(name)
        assert tool is not None
        assert getattr(tool.execute, "_studio_brokered", False)
    with service_worker():
        items = catalog()
    assert {item["toolId"] for item in items} == PAID_TOOLS.keys()
    assert all(item["gateway"] in PAID_TOOLS[item["toolId"]] for item in items)


def test_direct_import_cannot_buy_with_ambient_provider_keys(monkeypatch):
    from tools.graphics.flux_image import FluxImage

    monkeypatch.setenv("FAL_KEY", "test-only-key")
    monkeypatch.delenv("STUDIO_API_URL", raising=False)
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    called = []
    monkeypatch.setattr(FluxImage, "_execute_provider", lambda *_: called.append(True))
    result = FluxImage().execute({"prompt": "test", "studio_project_id": "film"})
    assert not result.success
    assert result.cost_usd is None
    assert called == []


def test_local_execution_remains_independent():
    class Local(BaseTool):
        name = "local_boundary_test"

        def execute(self, inputs):
            return ToolResult(success=True, data=inputs)

    assert Local().execute({"value": 1}).success


def test_new_unreviewed_paid_tool_fails_closed():
    class Unreviewed(BaseTool):
        name = "unreviewed_paid"
        runtime = ToolRuntime.API

        def execute(self, inputs):
            raise AssertionError("Paid implementation must not execute")

    result = Unreviewed().execute({})
    assert not result.success
    assert result.billing_status == "notSubmitted"


def test_route_never_silently_switches_gateway():
    assert route_inputs("veo_video", "fal", {"backend": "auto"})["backend"] == "fal"
    with pytest.raises(ValueError):
        route_inputs("veo_video", "fal", {"backend": "google"})


def test_stages_are_persisted_before_returning_and_retry_is_blocked(monkeypatch):
    from lib.providers import worker

    events = []
    monkeypatch.setattr(worker, "public_target", lambda url: urlsplit(url).hostname)
    monkeypatch.setattr(
        worker, "emit", lambda event, **kwargs: events.append((event, kwargs))
    )
    recorder = StageRecorder("fal")
    headers = {}
    stage = recorder.before(
        "POST", "https://queue.fal.run/fal-ai/flux/dev", headers, b'{"prompt":"test"}'
    )
    assert headers["X-Fal-No-Retry"] == "1"
    assert events[0][1] == {"acknowledge": True}
    recorder.after(
        stage,
        200,
        json.dumps(
            {
                "request_id": "original-order",
                "status_url": "https://queue.fal.run/fal-ai/flux/requests/original-order/status",
            }
        ).encode(),
    )
    assert events[1][0]["providerOrderId"] == "original-order"
    with pytest.raises(RuntimeError, match="retry blocked"):
        recorder.before(
            "POST",
            "https://queue.fal.run/fal-ai/flux/dev",
            headers,
            b'{"prompt":"test"}',
        )


def test_sdk_download_does_not_receive_provider_auth(monkeypatch):
    from lib.providers import worker

    monkeypatch.setattr(worker, "public_target", lambda url: urlsplit(url).hostname)
    recorder = StageRecorder("fal")
    headers = {"Authorization": "Key test-only"}
    assert (
        recorder.before("GET", "https://cdn.example.org/result.mp4", headers, b"")
        is None
    )
    assert "Authorization" not in headers


def test_higgsfield_upload_is_free_and_generation_is_one_durable_paid_stage(monkeypatch):
    from lib.providers import worker

    events = []
    monkeypatch.setattr(worker, "public_target", lambda url: urlsplit(url).hostname)
    monkeypatch.setattr(worker, "emit", lambda event, **kwargs: events.append((event, kwargs)))
    recorder = StageRecorder("higgsfield")
    allocation = recorder.before(
        "POST", "https://api.higgsfield.ai/files/generate-upload-url",
        {"Authorization": "Key test-id:test-secret"}, b'{"content_type":"image/png"}',
    )
    upload_headers = {"Authorization": "Key must-not-leave-api", "x-amz-tagging": "temporary"}
    recorder.before("PUT", "https://storage.example.org/upload", upload_headers, b"png")
    generation = recorder.before(
        "POST", "https://api.higgsfield.ai/kling-video/v3.0/std/image-to-video",
        {"Authorization": "Key test-id:test-secret"}, b'{"duration":8}',
    )
    sending = [event for event, _ in events if event["event"] == "sending"]
    assert [event["billable"] for event in sending] == [False, False, True]
    assert all(options == {"acknowledge": True} for _, options in events)
    assert "Authorization" not in upload_headers
    assert upload_headers["x-amz-tagging"] == "temporary"
    recorder.after(generation, 200, json.dumps({
        "request_id": "higgsfield-original-order",
        "status_url": "https://api.higgsfield.ai/requests/higgsfield-original-order/status",
    }).encode())
    assert events[-1][0]["providerOrderId"] == "higgsfield-original-order"
    assert events[-1][0]["status_url"].endswith("/status")
    assert allocation != generation
    with pytest.raises(RuntimeError, match="retry blocked"):
        recorder.before(
            "POST", "https://api.higgsfield.ai/kling-video/v3.0/std/image-to-video",
            {}, b'{"duration":8}',
        )


def test_worker_requires_owned_invocation(monkeypatch):
    from io import StringIO
    from lib.providers import worker

    monkeypatch.delenv("STUDIO_PROVIDER_WORKER_NONCE", raising=False)
    monkeypatch.setattr(
        worker.sys, "__stdin__", StringIO('{"contractVersion":1,"action":"execute"}\n')
    )
    with pytest.raises(RuntimeError, match="service-owned"):
        worker.main()


def test_response_stream_is_bounded(monkeypatch):
    import httpx
    from lib.providers import worker

    monkeypatch.setattr(worker, "MAX_RESPONSE_BYTES", 8)
    response = httpx.Response(200, stream=httpx.ByteStream(b"large response"))
    with pytest.raises(RuntimeError, match="bounded result"):
        worker.bounded_httpx_response(response)
    assert response.is_closed


def test_lost_submit_response_remains_unresolved(monkeypatch):
    from lib.providers import client

    monkeypatch.setenv("STUDIO_PROJECT_ID", "registered")

    def call(method, path, payload=None):
        if path.endswith("/quote"):
            return {"quoteId": "test"}
        raise RuntimeError("Response lost")

    monkeypatch.setattr(client, "_call", call)
    result = client.execute("flux_image", {"prompt": "test"})
    assert result.billing_status == "unresolved"
    assert result.cost_usd is None


def test_kling_paid_post_has_no_transport_retry(monkeypatch):
    import requests
    from tools._kling.client import KlingClient
    from tools._kling.errors import KlingAPIError

    monkeypatch.setenv("KLING_API_KEY", "test-only")

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise requests.Timeout("Response lost")

    session = Session()
    with pytest.raises(KlingAPIError):
        KlingClient(session=session, max_retries=2).post("/v1/videos/text2video", {})
    assert session.calls == 1


def test_selector_keeps_explicit_platform_and_strips_only_selector_metadata(
    monkeypatch,
):
    from lib.providers.selectors import delegate, explicit_tool
    from tools.graphics.flux_image import FluxImage
    from lib.providers import client

    received = []
    monkeypatch.setattr(
        client, "execute", lambda name, inputs: received.append((name, inputs))
    )
    tool = FluxImage()
    inputs = {
        "prompt": "test",
        "preferred_provider": "auto",
        "studio_route_id": "flux_image:fal",
        "scene_id": "scene-1",
    }
    assert explicit_tool(inputs, [tool]) == (True, tool)
    delegate(tool, inputs)
    assert received == [
        (
            "flux_image",
            {
                "prompt": "test",
                "studio_route_id": "flux_image:fal",
                "scene_id": "scene-1",
            },
        )
    ]
    assert explicit_tool({"studio_route_id": "recraft_image:fal"}, [tool]) == (
        True,
        None,
    )
