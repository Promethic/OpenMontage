"""ElevenLabs text-to-speech provider tool."""

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
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class ElevenLabsTTS(BaseTool):
    name = "elevenlabs_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "elevenlabs"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set the ELEVENLABS_API_KEY environment variable:\n"
        "  export ELEVENLABS_API_KEY=your_key_here\n"
        "Get a key at https://elevenlabs.io"
    )
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "piper_tts"]
    agent_skills = ["elevenlabs", "text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "ssml_support",
        "pronunciation_control",
    ]
    supports = {
        "voice_cloning": True,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "high-quality narration",
        "voice-sensitive spokesperson videos",
        "multilingual spoken delivery",
    ]
    not_good_for = [
        "fully offline production",
        "privacy-constrained local-only workflows",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to convert to speech"},
            "voice_id": {
                "type": "string",
                "description": "ElevenLabs voice ID (default: Rachel)",
            },
            "model_id": {
                "type": "string",
                "default": "eleven_multilingual_v2",
                "description": "TTS model to use",
            },
            "previous_text": {
                "type": "string",
                "maxLength": 1000,
                "description": "Text spoken just before this clip; keeps prosody continuous",
            },
            "next_text": {
                "type": "string",
                "maxLength": 1000,
                "description": "Text spoken just after this clip; keeps prosody continuous",
            },
            "previous_request_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": (
                    "request_id values of the clips generated before this one "
                    "(request stitching; ignored for eleven_v3)"
                ),
            },
            "next_request_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": (
                    "request_id values of the clips that follow this one, used when "
                    "regenerating a clip between existing ones (ignored for eleven_v3)"
                ),
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4294967295,
                "description": "Best-effort deterministic sampling; keep one seed per narration",
            },
            "stability": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
            },
            "similarity_boost": {
                "type": "number",
                "default": 0.75,
                "minimum": 0,
                "maximum": 1,
            },
            "style": {
                "type": "number",
                "default": 0.0,
                "minimum": 0,
                "maximum": 1,
            },
            "speed": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.7,
                "maximum": 1.2,
            },
            "use_speaker_boost": {
                "type": "boolean",
                "default": True,
            },
            "output_path": {"type": "string"},
            "output_format": {
                "type": "string",
                "default": "mp3_44100_128",
                "enum": ["mp3_44100_128", "mp3_44100_192", "pcm_16000", "pcm_24000"],
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "text",
        "voice_id",
        "model_id",
        "stability",
        "similarity_boost",
        "style",
        "speed",
        "use_speaker_boost",
        "previous_text",
        "next_text",
        "previous_request_ids",
        "next_request_ids",
        "seed",
    ]
    side_effects = ["writes audio file to output_path", "calls ElevenLabs API"]
    user_visible_verification = ["Listen to generated audio for natural speech quality"]

    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
    DEFAULT_MODEL_ID = "eleven_multilingual_v2"
    # USD per character from https://elevenlabs.io/pricing/api, checked 2026-09-29
    # (list prices; temporary promotions are ignored so estimates stay conservative).
    PRICE_PER_CHARACTER = {
        "eleven_v4": 0.00008,
        "eleven_v4_turbo": 0.00004,
        "eleven_v3": 0.00008,
    }
    FALLBACK_PRICE_PER_CHARACTER = 0.0003
    # Request stitching (previous/next_request_ids) is unavailable on these models.
    NO_STITCHING_MODELS = {"eleven_v3"}

    def get_status(self) -> ToolStatus:
        if os.environ.get("ELEVENLABS_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model_id = inputs.get("model_id", self.DEFAULT_MODEL_ID)
        rate = self.PRICE_PER_CHARACTER.get(model_id, self.FALLBACK_PRICE_PER_CHARACTER)
        return round(len(inputs.get("text", "")) * rate, 4)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            return ToolResult(success=False, error="No ElevenLabs API key. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"TTS generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _build_request(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model_id = inputs.get("model_id", self.DEFAULT_MODEL_ID)
        voice_settings = {
            "stability": inputs.get("stability", 0.5),
            "similarity_boost": inputs.get("similarity_boost", 0.75),
            "style": inputs.get("style", 0.0),
            "speed": inputs.get("speed", 1.0),
            "use_speaker_boost": inputs.get("use_speaker_boost", True),
        }
        payload: dict[str, Any] = {
            "text": inputs["text"],
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        warnings = []
        for field in ("previous_text", "next_text"):
            if inputs.get(field):
                payload[field] = inputs[field]
        for field in ("previous_request_ids", "next_request_ids"):
            ids = list(inputs.get(field) or [])
            if not ids:
                continue
            if len(ids) > 3:
                raise ValueError(f"{field} accepts at most 3 request ids")
            if model_id in self.NO_STITCHING_MODELS:
                warnings.append(f"{field} ignored: request stitching is unavailable on {model_id}")
                continue
            payload[field] = ids
        if inputs.get("seed") is not None:
            seed = int(inputs["seed"])
            if not 0 <= seed <= 4294967295:
                raise ValueError("seed must be between 0 and 4294967295")
            payload["seed"] = seed
        return {
            "voice_id": inputs.get("voice_id", self.DEFAULT_VOICE_ID),
            "output_format": inputs.get("output_format", "mp3_44100_128"),
            "payload": payload,
            "warnings": warnings,
        }

    def _generate(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        request = self._build_request(inputs)
        voice_id = request["voice_id"]
        output_format = request["output_format"]
        payload = request["payload"]

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json=payload,
            params={"output_format": output_format},
            timeout=120,
        )
        response.raise_for_status()

        ext = "mp3" if "mp3" in output_format else "wav"
        output_path = Path(inputs.get("output_path", f"tts_output.{ext}"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": payload["model_id"],
                "voice_id": voice_id,
                "voice_settings": payload["voice_settings"],
                "text_length": len(payload["text"]),
                "output": str(output_path),
                "format": output_format,
                # Pass this as previous/next_request_ids to neighbouring clips.
                "request_id": response.headers.get("request-id"),
                "continuity": {
                    key: payload[key]
                    for key in (
                        "previous_text",
                        "next_text",
                        "previous_request_ids",
                        "next_request_ids",
                        "seed",
                    )
                    if key in payload
                },
                "warnings": request["warnings"],
            },
            artifacts=[str(output_path)],
            model=payload["model_id"],
        )
