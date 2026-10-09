"""Sync Labs lipsync-2 through fal.ai.

Replaces the mouth movement of an existing clip so it follows a separately
generated speech track. Studio supplies both inputs as hash-pinned local files;
they are uploaded to fal storage (free) before one billable queue submission.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

ENDPOINT = "fal-ai/sync-lipsync/v2"
# Public list price for lipsync-2; lipsync-2-pro costs roughly 1.67x.
USD_PER_MINUTE = {"lipsync-2": 3.0, "lipsync-2-pro": 5.0}
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}


class SyncLipsync(BaseTool):
    name = "sync_lipsync"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "avatar"
    provider = "fal"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:FAL_KEY"]
    install_instructions = "Set FAL_KEY in .env (https://fal.ai/dashboard/keys)."
    agent_skills = ["avatar-video"]

    capabilities = ["lip_sync", "audio_video_alignment", "dubbing_support"]
    supports = {"lip_sync": True, "offline": False, "cloud_render": True}
    best_for = [
        "replacing mouth movement of a generated clip with a separate speech track",
        "languages the video model cannot speak natively (e.g. Polish dialogue)",
    ]
    not_good_for = ["multi-speaker clips without a single dominant face", "offline work"]
    fallback_tools = ["kling_lip_sync", "lip_sync"]

    input_schema = {
        "type": "object",
        "required": ["video_path", "audio_path"],
        "properties": {
            "video_path": {"type": "string", "description": "Source clip with one visible face"},
            "audio_path": {"type": "string", "description": "Speech track the lips must follow"},
            "model": {"type": "string", "enum": list(USD_PER_MINUTE), "default": "lipsync-2"},
            "sync_mode": {
                "type": "string",
                "enum": ["cut_off", "loop", "bounce", "silence", "remap"],
                "default": "cut_off",
            },
            "duration_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 300,
                "description": "Billed clip length, used for cost estimation",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True)
    side_effects = ["uploads inputs to fal storage", "writes lip-synced video to output_path"]
    user_visible_verification = ["Watch the mouth against the speech track before approval"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        minutes = float(inputs.get("duration_seconds") or 60) / 60
        return round(USD_PER_MINUTE[inputs.get("model", "lipsync-2")] * minutes, 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 120.0

    @staticmethod
    def _upload(path_value: str, headers: dict[str, str], requests_module: Any) -> str:
        path = Path(path_value)
        content_type = CONTENT_TYPES.get(path.suffix.lower())
        if content_type is None or not path.is_file():
            raise ValueError(f"Unsupported or missing media input: {path.name}")
        initiated = requests_module.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            headers=headers,
            json={"content_type": content_type, "file_name": path.name},
            timeout=30,
        )
        initiated.raise_for_status()
        data = initiated.json()
        uploaded = requests_module.put(
            data["upload_url"],
            headers={"Content-Type": content_type},
            data=path.read_bytes(),
            timeout=120,
        )
        uploaded.raise_for_status()
        return data["file_url"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(success=False, error="FAL_KEY not set. " + self.install_instructions)

        import requests

        start = time.time()
        model = inputs.get("model", "lipsync-2")
        headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
        try:
            payload = {
                "model": model,
                "video_url": self._upload(inputs["video_path"], headers, requests),
                "audio_url": self._upload(inputs["audio_path"], headers, requests),
                "sync_mode": inputs.get("sync_mode", "cut_off"),
            }
            submitted = requests.post(
                f"https://queue.fal.run/{ENDPOINT}", headers=headers, json=payload, timeout=30
            )
            submitted.raise_for_status()
            queue = submitted.json()
            deadline = time.time() + 900
            while True:
                time.sleep(5)
                status = requests.get(queue["status_url"], headers=headers, timeout=15)
                status.raise_for_status()
                state = status.json().get("status", "UNKNOWN")
                if state == "COMPLETED":
                    break
                if state in ("FAILED", "CANCELLED") or time.time() > deadline:
                    return ToolResult(success=False, error=f"Lipsync {state.lower()}")
            result = requests.get(queue["response_url"], headers=headers, timeout=30)
            result.raise_for_status()
            video = requests.get(result.json()["video"]["url"], timeout=180)
            video.raise_for_status()
            output_path = Path(inputs.get("output_path", "sync_lipsync_output.mp4"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(video.content)
        except Exception as e:
            return ToolResult(success=False, error=f"Lipsync failed: {e}")

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "fal",
                "model": f"{ENDPOINT}:{model}",
                "sync_mode": payload["sync_mode"],
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost({**inputs, **({"duration_seconds": probed["duration_seconds"]} if probed.get("duration_seconds") else {})}),
            duration_seconds=round(time.time() - start, 2),
            model=f"{ENDPOINT}:{model}",
        )
