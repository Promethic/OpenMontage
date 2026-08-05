"""Contract tests for the strict four-value `scene_plan.scenes[].type` taxonomy.

The type answers one question: how is the scene's visible material made? Everything
else a scene carries — narrative role, transitions, shot language, asset provenance —
stays orthogonal, so those fields must keep validating exactly as before.
"""

import sys
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.artifacts import load_schema, validate_artifact  # noqa: E402

CANONICAL_TYPES = ["genesio_capture", "generated_video", "image_scene", "graphic_card"]

REMOVED_TYPES = [
    "talking_head",
    "broll",
    "animation",
    "character_scene",
    "diagram",
    "text_card",
    "transition",
    "generated",
    "screen_recording",
]


def scene_plan(scene_type: str = "generated_video", version: str = "2.0") -> dict:
    return {
        "version": version,
        "scenes": [
            {
                "id": "scene-1",
                "type": scene_type,
                "description": "Opening beat",
                "start_seconds": 0,
                "end_seconds": 4,
            }
        ],
    }


class TestCanonicalTypes:
    @pytest.mark.parametrize("scene_type", CANONICAL_TYPES)
    def test_accepts_each_canonical_type(self, scene_type):
        validate_artifact("scene_plan", scene_plan(scene_type))

    def test_enum_is_exactly_the_four_values(self):
        schema = load_schema("scene_plan")
        enum = schema["properties"]["scenes"]["items"]["properties"]["type"]["enum"]
        assert enum == CANONICAL_TYPES

    def test_accepts_a_plan_mixing_all_four(self):
        document = {
            "version": "2.0",
            "scenes": [
                {
                    "id": f"scene-{index + 1}",
                    "type": scene_type,
                    "description": f"Beat {index + 1}",
                    "start_seconds": index * 2,
                    "end_seconds": index * 2 + 2,
                }
                for index, scene_type in enumerate(CANONICAL_TYPES)
            ],
        }
        validate_artifact("scene_plan", document)


class TestRejectedTypes:
    def test_rejects_version_one(self):
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", scene_plan(version="1.0"))

    @pytest.mark.parametrize("scene_type", REMOVED_TYPES)
    def test_rejects_every_removed_legacy_value(self, scene_type):
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", scene_plan(scene_type))

    def test_rejects_custom_value(self):
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", scene_plan("agent_invented_scene"))

    def test_rejects_empty_type(self):
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", scene_plan(""))

    def test_rejects_missing_type(self):
        document = scene_plan()
        del document["scenes"][0]["type"]
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)


class TestOrthogonalFieldsStillValidate:
    """A scene type change must not weaken any other part of the contract."""

    def rich_scene_plan(self) -> dict:
        return {
            "version": "2.0",
            "style_playbook": "clean-professional",
            "scenes": [
                {
                    "id": "scene-1",
                    "type": "genesio_capture",
                    "description": "Operator walks through the Genesio brief screen",
                    "start_seconds": 0,
                    "end_seconds": 6,
                    "script_section_id": "hook",
                    "transition_in": "fade",
                    "transition_out": "dissolve",
                    "narrative_role": "establish_context",
                    "hero_moment": True,
                    "shot_language": {
                        "shot_size": "medium",
                        "camera_movement": "static",
                        "lens_mm": 35,
                        "lighting_key": "natural",
                        "depth_of_field": "medium",
                        "color_temperature": "neutral",
                    },
                    "required_assets": [
                        {
                            "type": "video",
                            "description": "Genesio brief screen capture",
                            "source": "record",
                        }
                    ],
                }
            ],
            "metadata": {"remotion_components": {}},
        }

    def test_accepts_fully_populated_scene(self):
        validate_artifact("scene_plan", self.rich_scene_plan())

    def test_still_rejects_negative_timing(self):
        document = self.rich_scene_plan()
        document["scenes"][0]["start_seconds"] = -1
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)

    def test_still_rejects_unknown_narrative_role(self):
        document = self.rich_scene_plan()
        document["scenes"][0]["narrative_role"] = "vibes"
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)

    def test_still_rejects_unknown_shot_language_value(self):
        document = self.rich_scene_plan()
        document["scenes"][0]["shot_language"]["shot_size"] = "cosmic"
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)

    def test_still_rejects_unknown_required_asset_source(self):
        document = self.rich_scene_plan()
        document["scenes"][0]["required_assets"][0]["source"] = "borrow"
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)

    def test_still_rejects_unknown_scene_field(self):
        document = self.rich_scene_plan()
        document["scenes"][0]["render_target"] = "mobile"
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)

    def test_transitions_are_not_scene_types(self):
        """A transition is a property of adjacent scenes, never a scene of its own."""
        document = deepcopy(self.rich_scene_plan())
        assert document["scenes"][0]["transition_in"] == "fade"
        validate_artifact("scene_plan", document)
        document["scenes"][0]["type"] = "transition"
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", document)

    def test_still_requires_at_least_one_scene(self):
        with pytest.raises(ValidationError):
            validate_artifact("scene_plan", {"version": "2.0", "scenes": []})
