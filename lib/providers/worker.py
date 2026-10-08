"""Private JSON-lines worker launched by Studio, never a public paid CLI.

The parent acknowledges each write intent and provider ID after committing it.
SDK retries cannot silently turn one reservation into repeated paid submissions.
Tool adapters reuse the pinned provider implementations inside this boundary.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import ipaddress
import json
import os
import socket
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from lib.providers.catalog import GATEWAYS, catalog, route_inputs
from lib.providers.context import service_worker

MAX_RESPONSE_BYTES = 128 * 1024**2
_RESOLVE = socket.getaddrinfo
_DNS = {}


def emit(payload: dict, *, acknowledge: bool = False):
    sys.__stdout__.write(
        json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
    )
    sys.__stdout__.flush()
    if acknowledge:
        answer = json.loads(sys.__stdin__.readline())
        if answer != {"ack": payload["stageId"]}:
            raise RuntimeError("Studio did not persist the provider stage")


def atomic_json(path: Path, payload: dict):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        os.chmod(temporary, 0o600)
        json.dump(payload, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def public_target(url: str) -> str:
    address = urlsplit(url)
    if (
        address.scheme != "https"
        or not address.hostname
        or address.username
        or address.password
        or address.port not in {None, 443}
    ):
        raise RuntimeError("Provider URLs must be public HTTPS endpoints")
    # Reject private DNS results as well as literal addresses; no redirects.
    addresses = _DNS.get(address.hostname) or _RESOLVE(
        address.hostname, 443, type=socket.SOCK_STREAM
    )
    if not addresses or any(
        not ipaddress.ip_address(item[4][0]).is_global for item in addresses
    ):
        raise RuntimeError("Provider URL resolves to a non-public address")
    _DNS[address.hostname] = addresses
    return address.hostname


def pinned_dns(host, port, *args, **kwargs):
    hostname = host.decode() if isinstance(host, bytes) else str(host)
    if hostname in _DNS:
        return _DNS[hostname]
    try:
        if ipaddress.ip_address(hostname).is_global:
            return _RESOLVE(host, port, *args, **kwargs)
    except ValueError:
        pass
    raise RuntimeError("Connection target was not validated by the provider adapter")


def gateway_hosts(gateway: str) -> set[str]:
    hosts = set(GATEWAYS[gateway].hosts)
    if gateway == "azure":
        endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")
        if endpoint:
            hosts.add(public_target(endpoint))
        region = os.environ.get("AZURE_SPEECH_REGION", "")
        if region and region.replace("-", "").isalnum():
            hosts.add(f"{region}.api.cognitive.microsoft.com")
    if gateway == "modal":
        endpoint = os.environ.get("MODAL_LTX2_ENDPOINT_URL")
        if endpoint:
            hosts.add(public_target(endpoint))
    if gateway == "kling" and os.environ.get("KLING_API_BASE_URL"):
        hosts.add(public_target(os.environ["KLING_API_BASE_URL"]))
    return hosts


def identifier(payload):
    if not isinstance(payload, dict):
        return None
    for name in ("request_id", "task_id", "id", "name", "video_id"):
        value = payload.get(name)
        if (
            isinstance(value, str)
            and 0 < len(value) <= 300
            and all(char.isalnum() or char in "_./:-" for char in value)
        ):
            return value
    return identifier(payload.get("data")) or identifier(payload.get("output"))


class StageRecorder:
    def __init__(self, gateway: str):
        self.gateway = gateway
        self.hosts = gateway_hosts(gateway)
        self.writes = set()
        self.sequence = 0
        self.upload_hosts = set()

    def before(self, method, url, headers, content) -> str | None:
        host = public_target(str(url))
        method = method.upper()
        authenticated = host in self.hosts or (
            self.gateway == "google" and host.endswith("-aiplatform.googleapis.com")
        )
        if not authenticated:
            if method not in {"GET", "HEAD", "PUT"} and host not in self.upload_hosts:
                raise RuntimeError("Unregistered provider write endpoint")
            for header in list(headers):
                if header.lower() in {
                    "authorization",
                    "x-api-key",
                    "xi-api-key",
                    "x-goog-api-key",
                    "ocp-apim-subscription-key",
                }:
                    del headers[header]
        if self.gateway == "fal" and authenticated:
            headers["X-Fal-No-Retry"] = "1"
        read_only_post = method == "POST" and (
            self.gateway == "volcengine"
            and (
                urlsplit(str(url)).path.endswith("/query")
                or parse_qs(urlsplit(str(url)).query).get("Action")
                == ["CVSync2AsyncGetResult"]
            )
        )
        if method in {"GET", "HEAD", "OPTIONS"} or read_only_post:
            return None
        billable = not (
            host in self.upload_hosts
            or method == "PUT"
            or host == "oauth2.googleapis.com"
            or urlsplit(str(url)).path.endswith("/files")
            or "/upload/" in urlsplit(str(url)).path
            or "/storage/" in urlsplit(str(url)).path
        )
        raw = content if isinstance(content, bytes) else str(content).encode()
        signature = hashlib.sha256(
            method.encode() + str(url).encode() + raw
        ).hexdigest()
        if signature in self.writes:
            raise RuntimeError(
                "Automatic provider write retry blocked; inspect the original stage"
            )
        self.writes.add(signature)
        self.sequence += 1
        stage = f"stage-{self.sequence}"
        emit(
            {
                "event": "sending",
                "stageId": stage,
                "gateway": self.gateway,
                "method": method,
                "host": host,
                "path": urlsplit(str(url)).path,
                "requestDigest": signature,
                "billable": billable,
            },
            acknowledge=True,
        )
        return stage

    def after(self, stage, status, content):
        payload = None
        if len(content) <= MAX_RESPONSE_BYTES:
            try:
                payload = json.loads(content)
            except (ValueError, UnicodeError):
                pass
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            upload_host = payload["data"].get("upload_host")
            if (
                self.gateway == "dashscope"
                and isinstance(upload_host, str)
                and public_target(upload_host).endswith(".aliyuncs.com")
            ):
                self.upload_hosts.add(urlsplit(upload_host).hostname)
        if stage is None:
            return
        event = {
            "event": "response",
            "stageId": stage,
            "statusCode": status,
            "providerOrderId": identifier(payload),
        }
        if isinstance(payload, dict):
            for field in ("status_url", "response_url", "cancel_url"):
                value = payload.get(field)
                if (
                    isinstance(value, str)
                    and len(value) <= 1000
                    and urlsplit(value).hostname in self.hosts
                    and not urlsplit(value).query
                ):
                    event[field] = value
        emit(event, acknowledge=True)


@contextlib.contextmanager
def instrument_network(gateway: str):
    """Observe the two transports used by all paid tools in this pinned tree."""
    import httpx
    import requests

    recorder = StageRecorder(gateway)
    request_send = requests.sessions.Session.send
    httpx_send = httpx.Client.send
    async_send = httpx.AsyncClient.send
    resolver = socket.getaddrinfo

    def requests_send(session, request, **kwargs):
        session.trust_env = False
        kwargs["allow_redirects"] = False
        kwargs["proxies"] = {}
        kwargs["stream"] = True
        stage = recorder.before(
            request.method, request.url, request.headers, request.body
        )
        response = request_send(session, request, **kwargs)
        chunks, size = [], 0
        try:
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise RuntimeError(
                        "Provider response exceeds the bounded result size"
                    )
                chunks.append(chunk)
            response._content = b"".join(chunks)
            response._content_consumed = True
        finally:
            response.close()
        recorder.after(stage, response.status_code, response.content)
        return response

    def sync_send(client, request, **kwargs):
        kwargs["follow_redirects"] = False
        kwargs["stream"] = True
        stage = recorder.before(
            request.method, request.url, request.headers, request.read()
        )
        response = httpx_send(client, request, **kwargs)
        bounded_httpx_response(response)
        recorder.after(stage, response.status_code, response.content)
        return response

    async def asynchronous_send(client, request, **kwargs):
        kwargs["follow_redirects"] = False
        kwargs["stream"] = True
        stage = recorder.before(
            request.method, request.url, request.headers, await request.aread()
        )
        response = await async_send(client, request, **kwargs)
        chunks, size = [], 0
        try:
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise RuntimeError(
                        "Provider response exceeds the bounded result size"
                    )
                chunks.append(chunk)
            response._content = b"".join(chunks)
        finally:
            await response.aclose()
        recorder.after(stage, response.status_code, response.content)
        return response

    requests.sessions.Session.send = requests_send
    httpx.Client.send = sync_send
    httpx.AsyncClient.send = asynchronous_send
    socket.getaddrinfo = pinned_dns
    try:
        yield
    finally:
        requests.sessions.Session.send = request_send
        httpx.Client.send = httpx_send
        httpx.AsyncClient.send = async_send
        socket.getaddrinfo = resolver


def bounded_httpx_response(response):
    chunks, size = [], 0
    try:
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RuntimeError("Provider response exceeds the bounded result size")
            chunks.append(chunk)
        response._content = b"".join(chunks)
    finally:
        response.close()


def http_request(request: dict) -> dict:
    """Shared low-level adapter for Studio's compatibility APIs."""
    import httpx

    gateway = request["gateway"]
    host = public_target(request["url"])
    if host not in gateway_hosts(gateway):
        raise RuntimeError("HTTP request does not belong to its provider adapter")
    headers = {
        key: value
        for key, value in request.get("headers", {}).items()
        if key.lower()
        not in {
            "authorization",
            "x-api-key",
            "xi-api-key",
            "cookie",
            "host",
            "content-length",
        }
    }
    token = next(
        (
            os.environ.get(name)
            for name in GATEWAYS[gateway].credentials
            if os.environ.get(name)
        ),
        None,
    )
    if not token:
        raise RuntimeError("Provider credential missing")
    if gateway == "anthropic":
        headers.update({"x-api-key": token, "anthropic-version": "2023-06-01"})
    elif gateway == "elevenlabs":
        headers["xi-api-key"] = token
    elif gateway == "fal":
        headers["Authorization"] = f"Key {token}"
        headers["X-Fal-No-Retry"] = "1"
    else:
        headers["Authorization"] = f"Bearer {token}"
    body = base64.b64decode(request.get("body", ""), validate=True)
    with httpx.Client(
        timeout=request.get("timeout", 240), follow_redirects=False, trust_env=False
    ) as client:
        response = client.send(
            client.build_request(
                request["method"], request["url"], headers=headers, content=body
            ),
            stream=True,
        )
        bounded_httpx_response(response)
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Provider response exceeds the bounded result size")
    path = Path(request["resultRoot"]) / "http-body"
    with path.open("wb") as stream:
        stream.write(response.content)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    metadata = {
        "status": response.status_code,
        "headers": {
            name: value
            for name, value in response.headers.items()
            if name
            not in {
                "content-encoding",
                "content-length",
                "transfer-encoding",
                "set-cookie",
            }
        },
        "bodyPath": str(path),
    }
    atomic_json(path.parent / "http-response.json", metadata)
    return metadata


def observe_tool(request: dict) -> dict:
    """Read/cancel an already recorded async order; never create a new one."""
    import requests

    gateway = request["gateway"]
    if gateway not in {"fal", "replicate"}:
        raise RuntimeError("This adapter has no read-only order recovery")
    stage = request["stage"]
    token = next(
        (
            os.environ.get(name)
            for name in GATEWAYS[gateway].credentials
            if os.environ.get(name)
        ),
        None,
    )
    if not token:
        raise RuntimeError("Provider credential missing")
    headers = {
        "Authorization": f"{'Key' if gateway == 'fal' else 'Bearer'} {token}",
        "X-Fal-No-Retry": "1",
    }
    order_id = stage.get("providerOrderId")
    if not order_id:
        raise RuntimeError("The original order has no durable provider ID")
    if gateway == "replicate":
        base = f"https://api.replicate.com/v1/predictions/{quote(order_id, safe='')}"
        url = base + "/cancel" if request["action"] == "cancel" else base
    else:
        field = "cancel_url" if request["action"] == "cancel" else "status_url"
        url = stage.get(field)
        if not url:
            raise RuntimeError(
                "The original queue URLs are missing; inspect the provider console"
            )
    if public_target(url) not in gateway_hosts(gateway):
        raise RuntimeError("Original order URL does not belong to its adapter")
    session = requests.Session()
    session.trust_env = False
    if request["action"] == "cancel":
        response = session.request(
            "PUT" if gateway == "fal" else "POST",
            url,
            headers=headers,
            timeout=30,
            allow_redirects=False,
        )
        response.raise_for_status()
        return {"cancelRequested": True}
    response = session.get(url, headers=headers, timeout=30, allow_redirects=False)
    response.raise_for_status()
    payload = response.json()
    state = payload.get("status")
    if state in {"starting", "processing", "IN_QUEUE", "IN_PROGRESS"}:
        return {"pending": True}
    if (gateway == "replicate" and state != "succeeded") or (
        gateway == "fal" and state != "COMPLETED"
    ):
        return {"success": False, "pending": False}
    if gateway == "fal":
        url = stage.get("response_url")
        if not url or public_target(url) not in gateway_hosts(gateway):
            raise RuntimeError("Original result URL is unavailable")
        response = session.get(url, headers=headers, timeout=30, allow_redirects=False)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            return {"success": False, "pending": False}
    urls = []

    def collect(value):
        if isinstance(value, str) and value.startswith("https://"):
            urls.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key in ("output", "images", "video", "audio", "url"):
                if key in value:
                    collect(value[key])

    collect(payload.get("output") if gateway == "replicate" else payload)
    if not urls or len(urls) > 20:
        raise RuntimeError("Expected bounded media outputs from the original order")
    directory = Path(request["resultRoot"]) / "output"
    directory.mkdir(mode=0o700, exist_ok=True)
    artifacts = []
    for index, url in enumerate(urls):
        public_target(url)
        download = session.get(url, timeout=120, stream=True, allow_redirects=False)
        download.raise_for_status()
        mime = download.headers.get("content-type", "").split(";", 1)[0]
        suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
        }.get(mime, Path(urlsplit(url).path).suffix)
        if suffix not in {".png", ".jpg", ".webp", ".mp4", ".mp3", ".wav"}:
            raise RuntimeError("Unsupported recovered output type")
        path = directory / f"recovered-{index}{suffix}"
        total = 0
        with path.open("wb") as stream:
            for chunk in download.iter_content(1024 * 1024):
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise RuntimeError("Recovered output exceeds its size bound")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        artifacts.append(str(path))
    result = {"success": True, "artifacts": artifacts, "data": {}, "cost_usd": None}
    atomic_json(Path(request["resultRoot"]) / "result.json", result)
    return result


def execute_tool(request: dict) -> dict:
    from tools.tool_registry import registry

    registry.ensure_discovered()
    tool = registry.get(request["toolId"])
    if tool is None:
        raise RuntimeError("Unknown pinned provider tool")
    inputs = route_inputs(request["toolId"], request["gateway"], request["parameters"])
    with instrument_network(request["gateway"]), contextlib.redirect_stdout(sys.stderr):
        result = tool._execute_provider(inputs)
    payload = asdict(result)
    # Legacy tool estimates are not evidence of an actual provider charge.
    payload["cost_usd"] = None
    payload["error"] = (
        None
        if result.success
        else "Provider tool failed; inspect the retained operation and result"
    )
    atomic_json(Path(request["resultRoot"]) / "result.json", payload)
    return payload


def main():
    request = json.loads(sys.__stdin__.readline())
    if (
        request.get("contractVersion") != 1
        or not os.environ.get("STUDIO_PROVIDER_WORKER_NONCE")
        or request.get("nonce") != os.environ["STUDIO_PROVIDER_WORKER_NONCE"]
    ):
        raise RuntimeError("Provider worker requires a service-owned invocation")
    with service_worker():
        if request["action"] == "catalog":
            result = {"items": catalog()}
        elif request["action"] == "http":
            result = http_request(request)
        elif request["action"] == "execute":
            result = execute_tool(request)
        elif request["action"] in {"observe", "cancel"}:
            result = observe_tool(request)
        else:
            raise RuntimeError("Unsupported provider worker action")
    emit({"event": "result", "result": result})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # SDK exception strings frequently include keys or signed URLs.
        emit(
            {
                "event": "error",
                "message": "Provider worker failed; no paid request will be retried",
            }
        )
        raise SystemExit(1)
