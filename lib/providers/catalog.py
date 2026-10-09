"""Billing boundaries for the pinned catalog, independent of vendor branding.

Free stock APIs are intentionally absent. A paid route is never inferred from an
estimated cost (some older cloud tools incorrectly estimate zero dollars).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class Gateway:
    credentials: tuple[str, ...]
    hosts: tuple[str, ...]


GATEWAYS = {
    "replicate": Gateway(("REPLICATE_API_TOKEN",), ("api.replicate.com",)),
    "fal": Gateway(("FAL_KEY", "FAL_AI_API_KEY"), ("fal.run", "queue.fal.run", "rest.alpha.fal.ai", "api.fal.ai")),
    "elevenlabs": Gateway(("ELEVENLABS_API_KEY",), ("api.elevenlabs.io",)),
    "anthropic": Gateway(("ANTHROPIC_API_KEY",), ("api.anthropic.com",)),
    "openai": Gateway(("OPENAI_API_KEY",), ("api.openai.com",)),
    "google": Gateway(("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"), ("generativelanguage.googleapis.com", "aiplatform.googleapis.com", "oauth2.googleapis.com")),
    "google-cloud": Gateway(("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_API_KEY", "GEMINI_API_KEY"), ("texttospeech.googleapis.com", "oauth2.googleapis.com")),
    "xai": Gateway(("XAI_API_KEY",), ("api.x.ai",)),
    "heygen": Gateway(("HEYGEN_API_KEY",), ("api.heygen.com",)),
    "runway": Gateway(("RUNWAY_API_KEY", "RUNWAYML_API_SECRET"), ("api.dev.runwayml.com",)),
    "higgsfield": Gateway(("HIGGSFIELD_API_KEY", "HIGGSFIELD_API_SECRET", "HIGGSFIELD_KEY"), ("api.higgsfield.ai",)),
    "dashscope": Gateway(("DASHSCOPE_API_KEY", "DASHSCOPE_REGION"), ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com")),
    "volcengine": Gateway(("VOLC_ACCESSKEY", "VOLC_SECRETKEY", "VOLCENGINE_ACCESS_KEY", "VOLCENGINE_SECRET_KEY", "JIMENG_ACCESS_KEY", "JIMENG_SECRET_KEY", "DOUBAO_SPEECH_API_KEY", "DOUBAO_SPEECH_VOICE_TYPE"), ("visual.volcengineapi.com", "openspeech.bytedance.com")),
    "kling": Gateway(("KLING_API_KEY", "KLING_API_BASE_URL", "KLING_API_REGION"), ("api.klingai.com", "api-singapore.klingai.com", "api-beijing.klingai.com", "api.klingai.com.cn")),
    "azure": Gateway(("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT"), ()),
    "sunoapi": Gateway(("SUNO_API_KEY",), ("api.sunoapi.org",)),
    "modal": Gateway(("MODAL_LTX2_ENDPOINT_URL",), ()),
}

# These are implementation metadata, not an agent-facing hardcoded tool menu.
PAID_TOOLS = {
    "azure_stt": ("azure",), "dashscope_asr": ("dashscope",),
    "dashscope_tts": ("dashscope",), "doubao_tts": ("volcengine",),
    "elevenlabs_tts": ("elevenlabs",), "google_tts": ("google-cloud",),
    "kling_tts": ("kling",), "openai_tts": ("openai",),
    "google_music": ("google",), "music_gen": ("elevenlabs",),
    "suno_music": ("sunoapi",), "kling_avatar": ("kling",),
    "kling_lip_sync": ("kling",), "dashscope_image": ("dashscope",),
    "flux_image": ("fal",), "google_imagen": ("google",),
    "grok_image": ("xai",), "kling_official_image": ("kling",),
    "openai_image": ("openai",), "recraft_image": ("fal",),
    "gemini_omni_video": ("google",), "grok_video": ("xai",),
    "heygen_video": ("heygen",), "higgsfield_video": ("higgsfield",),
    "jimeng_video": ("volcengine",), "kling_official_video": ("kling",),
    "kling_video": ("fal",), "ltx_video_modal": ("modal",),
    "minimax_video": ("fal",), "runway_video": ("runway",),
    "seedance_replicate": ("replicate",), "seedance_video": ("fal",),
    "sora_video": ("openai",), "veo_video": ("google", "fal"),
    "image_gen": ("openai", "fal"),
}
FREE_API_TOOLS = frozenset({"freesound_music", "pixabay_music", "pexels_image", "pixabay_image", "pexels_video", "pixabay_video"})
FREE_ENVIRONMENT = frozenset({"PEXELS_API_KEY", "PIXABAY_API_KEY", "FREESOUND_API_KEY", "HF_TOKEN"})


def credential_groups(tool_id, gateway):
    if gateway == "google" or gateway == "google-cloud":
        return [["GEMINI_API_KEY"], ["GOOGLE_API_KEY"], ["GOOGLE_APPLICATION_CREDENTIALS"]]
    if gateway == "volcengine":
        return [["DOUBAO_SPEECH_API_KEY"]] if tool_id == "doubao_tts" else [["VOLC_ACCESSKEY", "VOLC_SECRETKEY"]]
    if gateway == "higgsfield":
        return [["HIGGSFIELD_API_KEY", "HIGGSFIELD_API_SECRET"], ["HIGGSFIELD_KEY"]]
    if gateway == "kling":
        return [["KLING_API_KEY"]]
    if gateway == "azure":
        return [["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"], ["AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT"]]
    return [[name] for name in GATEWAYS[gateway].credentials if not name.endswith("REGION")]


def paid_path(name: str, inputs: dict) -> bool:
    # The deprecated monolith still supports an explicitly chosen local branch.
    return name in PAID_TOOLS and not (name == "image_gen" and inputs.get("provider") == "local")


def route_inputs(tool_id: str, gateway: str, parameters: dict) -> dict:
    if gateway not in PAID_TOOLS.get(tool_id, ()):
        raise ValueError("Unknown provider route")
    result = copy.deepcopy(parameters)
    if tool_id == "veo_video":
        if result.get("backend") not in {None, "auto", gateway}:
            raise ValueError("Backend conflicts with the quoted route")
        result["backend"] = gateway
    elif tool_id == "image_gen":
        selected = "flux" if gateway == "fal" else "openai"
        if result.get("provider") not in {None, selected}:
            raise ValueError("Provider conflicts with the quoted route")
        result["provider"] = selected
    return result


def catalog() -> list[dict]:
    from tools.tool_registry import registry

    registry.ensure_discovered()
    items = []
    for name, gateways in PAID_TOOLS.items():
        tool = registry.get(name)
        if tool is None:
            raise RuntimeError(f"Paid tool missing from pinned registry: {name}")
        for gateway in gateways:
            schema = copy.deepcopy(tool.input_schema)
            # Paths are supplied by Studio, never by the agent or SDK defaults.
            properties = schema.setdefault("properties", {})
            for field in ("output_path", "output_dir", "project_dir"):
                properties.pop(field, None)
            schema["required"] = [key for key in schema.get("required", []) if key in properties]
            schema["additionalProperties"] = False
            items.append({
                "toolId": name, "routeId": f"{name}:{gateway}", "gateway": gateway,
                "capability": tool.capability, "inputSchema": schema,
                "credentials": list(GATEWAYS[gateway].credentials),
                "credentialGroups": credential_groups(name, gateway),
                "supportsCancel": gateway in {"fal", "replicate"},
                "supportsRecover": gateway in {"fal", "replicate"},
                "contractVersion": 1,
            })
    return items
