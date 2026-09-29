"""Contract tests for ElevenLabs narration continuity (context text, stitching, seed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.audio.elevenlabs_tts import ElevenLabsTTS


def test_continuity_fields_are_part_of_schema_and_idempotency():
    tool = ElevenLabsTTS()
    props = tool.input_schema["properties"]
    for field in ("previous_text", "next_text", "previous_request_ids", "next_request_ids", "seed"):
        assert field in props
        assert field in tool.idempotency_key_fields
    assert props["previous_request_ids"]["maxItems"] == 3


def test_v4_payload_carries_stitching_context_and_seed():
    request = ElevenLabsTTS()._build_request(
        {
            "text": "Druga scena.",
            "voice_id": "voice-a",
            "model_id": "eleven_v4",
            "previous_text": "Pierwsza scena.",
            "next_text": "Trzecia scena.",
            "previous_request_ids": ["req-1"],
            "next_request_ids": ["req-3"],
            "seed": 42,
        }
    )
    payload = request["payload"]
    assert request["voice_id"] == "voice-a"
    assert payload["model_id"] == "eleven_v4"
    assert payload["previous_text"] == "Pierwsza scena."
    assert payload["next_text"] == "Trzecia scena."
    assert payload["previous_request_ids"] == ["req-1"]
    assert payload["next_request_ids"] == ["req-3"]
    assert payload["seed"] == 42
    assert request["warnings"] == []


def test_v3_drops_request_ids_but_keeps_text_context():
    request = ElevenLabsTTS()._build_request(
        {
            "text": "Druga scena.",
            "model_id": "eleven_v3",
            "previous_text": "Pierwsza scena.",
            "previous_request_ids": ["req-1"],
        }
    )
    payload = request["payload"]
    assert payload["previous_text"] == "Pierwsza scena."
    assert "previous_request_ids" not in payload
    assert any("eleven_v3" in warning for warning in request["warnings"])


def test_plain_request_is_unchanged_without_continuity_inputs():
    payload = ElevenLabsTTS()._build_request({"text": "hello"})["payload"]
    assert set(payload) == {"text", "model_id", "voice_settings"}


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"text": "x", "model_id": "eleven_v4", "previous_request_ids": ["a", "b", "c", "d"]},
        {"text": "x", "seed": -1},
        {"text": "x", "seed": 2**32},
    ],
)
def test_invalid_continuity_inputs_are_rejected(bad_inputs):
    with pytest.raises(ValueError):
        ElevenLabsTTS()._build_request(bad_inputs)


def test_cost_estimate_uses_model_price():
    tool = ElevenLabsTTS()
    text = "x" * 1000
    assert tool.estimate_cost({"text": text, "model_id": "eleven_v4"}) == 0.08
    assert tool.estimate_cost({"text": text, "model_id": "eleven_v3"}) == 0.08
    assert tool.estimate_cost({"text": text}) == 0.3
