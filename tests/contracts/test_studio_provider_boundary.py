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
    paid = {name for name, tool in registry._tools.items() if tool.runtime == ToolRuntime.API and name not in FREE_API_TOOLS}
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
    monkeypatch.setattr(worker, "emit", lambda event, **kwargs: events.append((event, kwargs)))
    recorder = StageRecorder("fal")
    headers = {}
    stage = recorder.before("POST", "https://queue.fal.run/fal-ai/flux/dev", headers, b'{"prompt":"test"}')
    assert headers["X-Fal-No-Retry"] == "1"
    assert events[0][1] == {"acknowledge": True}
    recorder.after(stage, 200, json.dumps({"request_id": "original-order", "status_url": "https://queue.fal.run/fal-ai/flux/requests/original-order/status"}).encode())
    assert events[1][0]["providerOrderId"] == "original-order"
    with pytest.raises(RuntimeError, match="retry blocked"):
        recorder.before("POST", "https://queue.fal.run/fal-ai/flux/dev", headers, b'{"prompt":"test"}')


def test_sdk_download_does_not_receive_provider_auth(monkeypatch):
    from lib.providers import worker
    monkeypatch.setattr(worker, "public_target", lambda url: urlsplit(url).hostname)
    recorder = StageRecorder("fal")
    headers = {"Authorization": "Key test-only"}
    assert recorder.before("GET", "https://cdn.example.org/result.mp4", headers, b"") is None
    assert "Authorization" not in headers


def test_worker_requires_owned_invocation(monkeypatch):
    from io import StringIO
    from lib.providers import worker
    monkeypatch.delenv("STUDIO_PROVIDER_WORKER_NONCE", raising=False)
    monkeypatch.setattr(worker.sys, "__stdin__", StringIO('{"contractVersion":1,"action":"execute"}\n'))
    with pytest.raises(RuntimeError, match="service-owned"):
        worker.main()
