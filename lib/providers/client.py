"""Scoped Studio client used by standalone and supervised OpenMontage tools.

No provider credentials, paid retries, local reservations or alternate gateways.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from urllib.parse import quote, urlsplit

_CATALOG: tuple[float, tuple[str, str], dict] | None = None


def _connection() -> tuple[str, str]:
    url = os.environ.get("STUDIO_API_URL", "").rstrip("/")
    token = os.environ.get("STUDIO_API_TOKEN", "")
    address = urlsplit(url)
    if address.scheme not in {"http", "https"} or address.hostname not in {"localhost", "127.0.0.1", "::1"} or address.path or address.query or address.fragment or address.username or address.password or not token:
        raise RuntimeError("Paid tools require a local Studio connection and scoped Studio grant")
    return url, token


def _call(method: str, path: str, body: dict | None = None) -> dict:
    import requests

    url, token = _connection()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.request(method, url + path, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=30, allow_redirects=False)
        if response.status_code >= 300:
            # Never include raw server/provider bodies or scoped tokens in events.
            raise RuntimeError(f"Studio rejected the operation (HTTP {response.status_code}); inspect Studio")
        return response.json()
    finally:
        session.close()


def available(tool_id: str) -> bool:
    global _CATALOG
    try:
        identity = _connection()
        if _CATALOG is None or _CATALOG[1] != identity or time.monotonic() - _CATALOG[0] > 30:
            _CATALOG = (time.monotonic(), identity, _call("GET", "/api/provider-tools"))
        return any(item["toolId"] == tool_id and item.get("available") for item in _CATALOG[2]["items"])
    except Exception:
        return False


def execute(tool_id: str, inputs: dict):
    from tools.base_tool import ToolResult

    try:
        from lib.providers.catalog import PAID_TOOLS

        project = inputs.get("studio_project_id") or os.environ.get("STUDIO_PROJECT_ID")
        if not project:
            raise RuntimeError("Register the project in Studio and set STUDIO_PROJECT_ID")
        parameters = {key: value for key, value in inputs.items() if key not in {"studio_project_id", "studio_route_id", "studio_idempotency_key", "studio_purpose", "scene_id", "project_dir", "output_path", "output_dir"}}
        gateways = PAID_TOOLS[tool_id]
        route = inputs.get("studio_route_id")
        if not route:
            if len(gateways) != 1:
                raise RuntimeError("Select studio_route_id explicitly; provider fallback is forbidden")
            route = f"{tool_id}:{gateways[0]}"
        base = f"/api/projects/{quote(str(project), safe='')}/workflow/provider-operations"
        payload = {"toolId": tool_id, "routeId": route, "parameters": parameters, "purpose": inputs.get("studio_purpose", "production"), "sceneId": inputs.get("scene_id")}
        pricing = _call("POST", base + "/quote", payload)
        # Stable by intent across interrupted invocations; an intentional new take
        # requires an explicit new key after the preceding order is resolved.
        key = inputs.get("studio_idempotency_key") or hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        operation = _call("POST", base, {"quoteId": pricing["quoteId"], "idempotencyKey": key})
        deadline = time.monotonic() + 30
        while operation["state"] in {"reserved", "submitted"} and time.monotonic() < deadline:
            time.sleep(1)
            operation = _call("POST", base + f"/{operation['id']}/refresh")
        cost = operation.get("actualCostUsd")
        data = {**operation.get("data", {}), "studioOperation": operation}
        paths = operation.get("outputPaths", [])
        if paths:
            data["output"] = paths[0]
        return ToolResult(success=operation["state"] == "succeeded", data=data, artifacts=paths, error=None if operation["state"] == "succeeded" else f"Studio operation {operation['id']}: {operation['state']}; refresh this order, do not resubmit", cost_usd=cost, operation_id=operation["id"], billing_status="settled" if cost is not None else "unresolved")
    except Exception as exc:
        return ToolResult(success=False, error=str(exc), cost_usd=None, billing_status="notSubmitted")
