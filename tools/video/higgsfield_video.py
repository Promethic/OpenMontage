"""Video generation through the current Higgsfield pay-as-you-go API.

Higgsfield is an optional multi-model gateway. OpenMontage keeps model and
gateway identity separate so fal.ai can remain the default while Higgsfield is
available for comparison, fallback, and Higgsfield-specific creative paths.
"""

from __future__ import annotations

import ipaddress
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


_API_ORIGIN = "https://api.higgsfield.ai"
_DEFAULT_MODEL = "seedance_2.5"
_PRICE_CHECKED_AT = "2026-10-09"
_TERMINAL_STATUSES = {"completed", "failed", "nsfw", "canceled"}

# Public self-serve rates visible on the model pages on _PRICE_CHECKED_AT.
# Account-specific promotions can differ, so these values are estimates.
_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "seedance_2.5": {
        "label": "Seedance 2.5",
        "endpoints": {
            "text_to_video": "bytedance/seedance-2.5/text-to-video",
            "image_to_video": "bytedance/seedance-2.5/image-to-video",
            "reference_to_video": "bytedance/seedance-2.5/reference-to-video",
        },
        "rates": {
            "text_to_video": 0.144,
            "image_to_video": 0.144,
            "reference_to_video": 0.1728,
        },
        "duration": (4, 30),
        "runtime": 120.0,
    },
    "kling_3.0_standard": {
        "label": "Kling 3.0 Standard",
        "endpoints": {
            "text_to_video": "kling-video/v3.0/std/text-to-video",
            "image_to_video": "kling-video/v3.0/std/image-to-video",
        },
        "rates": {"text_to_video": 0.126, "image_to_video": 0.126},
        "duration": (3, 15),
        "runtime": 90.0,
    },
    "kling_3.0_pro": {
        "label": "Kling 3.0 Pro",
        "endpoints": {
            "text_to_video": "kling-video/v3.0/pro/text-to-video",
            "image_to_video": "kling-video/v3.0/pro/image-to-video",
        },
        "rates": {"text_to_video": 0.0616, "image_to_video": 0.0616},
        "duration": (3, 15),
        "runtime": 100.0,
    },
    "minimax_h3": {
        "label": "MiniMax H3",
        "endpoints": {
            "text_to_video": "minimax/h3/text-to-video",
            "image_to_video": "minimax/h3/image-to-video",
            "reference_to_video": "minimax/h3/reference-to-video",
        },
        "rates": {
            "text_to_video": 0.091,
            "image_to_video": 0.13,
            "reference_to_video": 0.091,
        },
        "duration": (5, 15),
        "runtime": 120.0,
    },
}

_UPLOAD_CONTENT_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".webp": "image/webp",
}


class HiggsFieldVideo(BaseTool):
    name = "higgsfield_video"
    version = "0.2.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "higgsfield"
    gateway = "higgsfield"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Create server-side credentials at https://open.higgsfield.ai/api-keys, then set "
        "HIGGSFIELD_KEY=key_id:key_secret in Studio's private provider configuration "
        "(or HIGGSFIELD_API_KEY and HIGGSFIELD_API_SECRET)."
    )
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video", "reference_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": True,
        "reference_image": True,
        "multiple_reference_images": True,
        "multi_model_routing": True,
        "native_audio": True,
        "cinematic_quality": True,
        "camera_direction": True,
        "lip_sync": True,
        "multi_shot": True,
        "aspect_ratio": True,
    }
    best_for = [
        "optional multi-model gateway for comparing Seedance, Kling, and MiniMax",
        "Higgsfield-specific creator workflows and character-consistent creative production",
        "pay-as-you-go video generation with one server-side credential pair",
        "cinematic clips, native audio, multi-shot video, and reference-conditioned generation",
    ]
    not_good_for = [
        "offline generation",
        "custom model hosting or LoRA training",
        "using as the only gateway when provider independence is required",
    ]
    fallback_tools = [
        "seedance_video",
        "seedance_replicate",
        "kling_video",
        "veo_video",
        "minimax_video",
    ]
    quality_score = 0.9

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video"],
                "default": "text_to_video",
            },
            "model": {
                "type": "string",
                "enum": list(_MODEL_CONFIGS),
                "default": _DEFAULT_MODEL,
                "description": "Underlying model served through Higgsfield.",
            },
            "duration": {"type": "integer", "minimum": 3, "maximum": 30, "default": 5},
            "aspect_ratio": {
                "type": "string",
                "enum": ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                "default": "16:9",
            },
            "resolution": {
                "type": "string",
                "enum": ["480p", "720p", "2K"],
                "default": "720p",
            },
            "generate_audio": {"type": "boolean", "default": True},
            "image_url": {"type": "string"},
            "image_path": {
                "type": "string",
                "description": "Local image uploaded through Higgsfield's presigned upload flow.",
            },
            "end_image_url": {"type": "string"},
            "reference_image_urls": {"type": "array", "items": {"type": "string"}},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}},
            "reference_video_urls": {"type": "array", "items": {"type": "string"}},
            "reference_video_paths": {"type": "array", "items": {"type": "string"}},
            "reference_audio_urls": {"type": "array", "items": {"type": "string"}},
            "reference_audio_paths": {"type": "array", "items": {"type": "string"}},
            "cfg_scale": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "multi_shots": {"type": "boolean", "default": False},
            "aigc_watermark": {"type": "boolean", "default": False},
            "output_path": {"type": "string"},
            "poll_interval_seconds": {"type": "number", "minimum": 1, "default": 2},
            "timeout_seconds": {"type": "integer", "minimum": 30, "default": 900},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0, retryable_errors=[])
    idempotency_key_fields = ["prompt", "model", "operation", "duration", "aspect_ratio"]
    side_effects = ["writes video file to output_path", "calls Higgsfield API"]
    user_visible_verification = ["Watch generated clip for motion coherence and visual quality"]

    @staticmethod
    def _split_credentials(value: str | None) -> tuple[str, str] | None:
        if value and ":" in value:
            key_id, key_secret = value.split(":", 1)
            if key_id and key_secret:
                return key_id, key_secret
        return None

    def _get_credentials(self) -> tuple[str, str] | None:
        for name in ("HF_KEY", "HF_CREDENTIALS", "HIGGSFIELD_KEY"):
            credentials = self._split_credentials(os.environ.get(name))
            if credentials:
                return credentials

        for key_name, secret_name in (
            ("HF_API_KEY_ID", "HF_API_KEY_SECRET"),
            ("HIGGSFIELD_API_KEY", "HIGGSFIELD_API_SECRET"),
        ):
            key_id = os.environ.get(key_name)
            key_secret = os.environ.get(secret_name)
            if key_id and key_secret:
                return key_id, key_secret
        return None

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._get_credentials() else ToolStatus.UNAVAILABLE

    @staticmethod
    def _model_config(model: str) -> dict[str, Any]:
        try:
            return _MODEL_CONFIGS[model]
        except KeyError as error:
            supported = ", ".join(_MODEL_CONFIGS)
            raise ValueError(f"Unsupported Higgsfield model {model!r}; choose one of: {supported}") from error

    def _endpoint(self, model: str, operation: str) -> str:
        config = self._model_config(model)
        endpoint = config["endpoints"].get(operation)
        if not endpoint:
            raise ValueError(f"Higgsfield model {model!r} does not support {operation!r}")
        return endpoint

    def _validate_duration(self, model: str, duration: int) -> None:
        minimum, maximum = self._model_config(model)["duration"]
        if not minimum <= duration <= maximum:
            raise ValueError(
                f"Higgsfield model {model!r} accepts duration {minimum}-{maximum}s; got {duration}s"
            )

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = str(inputs.get("model", _DEFAULT_MODEL))
        operation = str(inputs.get("operation", "text_to_video"))
        duration = int(inputs.get("duration", 5))
        config = self._model_config(model)
        rate = config["rates"].get(operation)
        if model == "kling_3.0_standard" and not inputs.get("generate_audio", True):
            rate = 0.084
        if rate is None:
            self._endpoint(model, operation)
            return 0.0
        return round(float(rate) * duration, 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        model = str(inputs.get("model", _DEFAULT_MODEL))
        return float(self._model_config(model)["runtime"])

    @staticmethod
    def _headers(credentials: tuple[str, str]) -> dict[str, str]:
        key_id, key_secret = credentials
        return {
            "Authorization": f"Key {key_id}:{key_secret}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _safe_status_url(value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.higgsfield.ai"
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Higgsfield returned an untrusted status URL")
        return value

    @staticmethod
    def _safe_external_url(value: str, label: str) -> str:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(f"Higgsfield returned an invalid {label} URL")
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            raise ValueError(f"Higgsfield returned an untrusted {label} URL")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return value
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ValueError(f"Higgsfield returned an untrusted {label} URL")
        return value

    @staticmethod
    def _content_type(path: Path) -> str:
        detected = _UPLOAD_CONTENT_TYPES.get(path.suffix.lower())
        if not detected:
            detected, _ = mimetypes.guess_type(path.name)
        if detected not in set(_UPLOAD_CONTENT_TYPES.values()):
            raise ValueError(f"Unsupported Higgsfield upload type for {path.name}")
        return detected

    def _upload_file(self, path_value: str, headers: dict[str, str], requests_module: Any) -> str:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        content_type = self._content_type(path)
        allocation = requests_module.post(
            f"{_API_ORIGIN}/files/generate-upload-url",
            headers=headers,
            json={"content_type": content_type},
            timeout=30,
        )
        allocation.raise_for_status()
        upload = allocation.json()
        upload_url = self._safe_external_url(str(upload["upload_url"]), "upload")
        public_url = self._safe_external_url(str(upload["public_url"]), "public media")

        upload_headers = dict(upload.get("upload_headers") or {"Content-Type": content_type})
        with path.open("rb") as source:
            response = requests_module.put(
                upload_url,
                headers=upload_headers,
                data=source,
                timeout=120,
            )
        response.raise_for_status()
        return public_url

    def _collect_urls(
        self,
        url_key: str,
        path_key: str,
        inputs: dict[str, Any],
        headers: dict[str, str],
        requests_module: Any,
    ) -> list[str]:
        urls = list(inputs.get(url_key) or [])
        urls.extend(
            self._upload_file(str(path), headers, requests_module)
            for path in (inputs.get(path_key) or [])
        )
        return urls

    def _build_payload(
        self,
        inputs: dict[str, Any],
        headers: dict[str, str],
        requests_module: Any,
    ) -> tuple[str, dict[str, Any]]:
        model = str(inputs.get("model", _DEFAULT_MODEL))
        operation = str(inputs.get("operation", "text_to_video"))
        duration = int(inputs.get("duration", 5))
        self._validate_duration(model, duration)
        endpoint = self._endpoint(model, operation)

        payload: dict[str, Any] = {"prompt": str(inputs["prompt"]), "duration": duration}
        aspect_ratio = str(inputs.get("aspect_ratio", "16:9"))
        generate_audio = bool(inputs.get("generate_audio", True))

        if model == "seedance_2.5":
            payload.update(
                resolution=str(inputs.get("resolution", "720p")),
                output_format="mp4",
                generate_audio=generate_audio,
            )
            if operation != "image_to_video":
                payload["aspect_ratio"] = aspect_ratio
        elif model.startswith("kling_3.0"):
            if aspect_ratio not in {"16:9", "9:16", "1:1"}:
                raise ValueError("Kling 3.0 supports aspect ratios 16:9, 9:16, and 1:1")
            payload.update(
                sound="on" if generate_audio else "off",
                cfg_scale=float(inputs.get("cfg_scale", 0.5)),
                multi_shots=bool(inputs.get("multi_shots", False)),
            )
            if operation == "text_to_video":
                payload["aspect_ratio"] = aspect_ratio
        elif model == "minimax_h3":
            payload.update(
                resolution="2K",
                aspect_ratio=aspect_ratio,
                aigc_watermark=bool(inputs.get("aigc_watermark", False)),
            )

        if operation == "image_to_video":
            image_url = inputs.get("image_url")
            if not image_url and inputs.get("image_path"):
                image_url = self._upload_file(str(inputs["image_path"]), headers, requests_module)
            if not image_url:
                raise ValueError("image_to_video requires image_url or image_path")
            payload["image_url"] = str(image_url)
            if model == "seedance_2.5" and inputs.get("end_image_url"):
                payload["end_image_url"] = str(inputs["end_image_url"])
            elif model.startswith("kling_3.0") and inputs.get("end_image_url"):
                payload["last_image_url"] = str(inputs["end_image_url"])

        if operation == "reference_to_video":
            image_urls = self._collect_urls(
                "reference_image_urls", "reference_image_paths", inputs, headers, requests_module
            )
            video_urls = self._collect_urls(
                "reference_video_urls", "reference_video_paths", inputs, headers, requests_module
            )
            audio_urls = self._collect_urls(
                "reference_audio_urls", "reference_audio_paths", inputs, headers, requests_module
            )
            if not image_urls and not video_urls and not audio_urls:
                raise ValueError("reference_to_video requires at least one reference input")
            if image_urls:
                payload["image_urls"] = image_urls
            if video_urls:
                payload["video_urls"] = video_urls
            if audio_urls:
                payload["audio_urls"] = audio_urls

        return endpoint, payload

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        credentials = self._get_credentials()
        if not credentials:
            return ToolResult(
                success=False,
                error="Higgsfield credentials not set. " + self.install_instructions,
            )

        import requests

        start = time.time()
        headers = self._headers(credentials)
        model = str(inputs.get("model", _DEFAULT_MODEL))
        operation = str(inputs.get("operation", "text_to_video"))

        try:
            endpoint, payload = self._build_payload(inputs, headers, requests)
            submit_response = requests.post(
                f"{_API_ORIGIN}/{endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            submit_response.raise_for_status()
            request_data = submit_response.json()
            request_id = str(request_data["request_id"])
            status_url = self._safe_status_url(str(request_data["status_url"]))

            delay = float(inputs.get("poll_interval_seconds", 2))
            deadline = time.time() + int(inputs.get("timeout_seconds", 900))
            result_data: dict[str, Any] | None = None

            while time.time() < deadline:
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_response.raise_for_status()
                result_data = status_response.json()
                status = str(result_data.get("status", "")).lower()
                if status in _TERMINAL_STATUSES:
                    break
                time.sleep(delay + random.uniform(0, min(0.5, delay / 4)))
                delay = min(delay * 1.5, 10.0)

            if not result_data or str(result_data.get("status", "")).lower() not in _TERMINAL_STATUSES:
                return ToolResult(
                    success=False,
                    error=f"Higgsfield request {request_id} timed out",
                    model=model,
                )

            status = str(result_data.get("status", "")).lower()
            if status != "completed":
                detail = result_data.get("error") or status
                return ToolResult(
                    success=False,
                    error=f"Higgsfield request {request_id} ended with {status}: {detail}",
                    model=model,
                )

            video = result_data.get("video") or {}
            video_url = video.get("url") if isinstance(video, dict) else None
            if not video_url:
                raise ValueError("Higgsfield completed the request without a video URL")

            video_response = requests.get(
                self._safe_external_url(str(video_url), "output media"),
                timeout=120,
            )
            video_response.raise_for_status()
            output_path = Path(inputs.get("output_path", "higgsfield_output.mp4"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(video_response.content)
        except Exception as error:
            return ToolResult(
                success=False,
                error=f"Higgsfield video generation failed: {error}",
                model=model,
            )

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "gateway": self.gateway,
                "model": model,
                "model_id": endpoint,
                "request_id": request_id,
                "prompt": inputs["prompt"],
                "operation": operation,
                "aspect_ratio": inputs.get("aspect_ratio", "16:9"),
                "resolution": payload.get("resolution"),
                "generate_audio": inputs.get("generate_audio", True),
                "pricing_checked_at": _PRICE_CHECKED_AT,
                "pricing_note": "Estimated public self-serve rate; Higgsfield billing is authoritative.",
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
