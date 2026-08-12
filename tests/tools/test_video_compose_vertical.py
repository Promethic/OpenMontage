"""Vertical / arbitrary-resolution support for video_compose's FFmpeg compose.

Regression test for a silent-dimension bug: the compose target resolution was
resolved from `profile` (and the documented `metadata.compose_target` hook) but
the per-segment scale/pad filter hardcoded 1920x1080, so vertical profiles like
`tiktok` silently produced landscape output. These tests run the real FFmpeg
path on a tiny lavfi fixture and assert the output dimensions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.base_tool import ToolResult
from tools.video.video_compose import VideoCompose

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_clip(path: Path, w: int = 1280, h: int = 720, d: int = 2) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=teal:s={w}x{h}:d={d}:r=30",
         "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
         "-g", "30", "-keyint_min", "30", str(path)],
        capture_output=True, check=True,
    )


def _dims(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)]
    ).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def _fps(path: Path) -> str:
    return subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(path)]
    ).decode().strip()


def _video_duration_and_frames(path: Path) -> tuple[float, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration,nb_frames", "-of", "json", str(path)]
    ).decode()
    stream = __import__("json").loads(out)["streams"][0]
    return float(stream["duration"]), int(stream["nb_frames"])


def _edit_decisions(src: Path, metadata: dict | None = None) -> dict:
    ed = {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "cuts": [{"id": "c1", "source": str(src), "in_seconds": 0, "out_seconds": 2}],
    }
    if metadata:
        ed["metadata"] = metadata
    return ed


def test_compose_default_is_landscape_hd(tmp_path):
    """No profile / no target → unchanged 1920x1080 default (backward compatible)."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": _edit_decisions(src), "output_path": str(out)}
    )
    assert r.success, r.error
    assert _dims(out) == (1920, 1080)


def test_compose_vertical_profile(tmp_path):
    """profile='tiktok' → 1080x1920 (the bug: previously stayed 1920x1080)."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": _edit_decisions(src),
         "profile": "tiktok", "output_path": str(out)}
    )
    assert r.success, r.error
    assert _dims(out) == (1080, 1920)


def test_compose_target_override_cover(tmp_path):
    """metadata.compose_target with fit='cover' → exact requested dims, cropped to fill."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    ed = _edit_decisions(src, metadata={"compose_target": {"width": 720, "height": 1280, "fit": "cover"}})
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": ed, "output_path": str(out)}
    )
    assert r.success, r.error
    assert _dims(out) == (720, 1280)


def test_compose_target_override_fps(tmp_path):
    """metadata.compose_target.fps controls the rendered frame rate."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    ed = _edit_decisions(src, metadata={"compose_target": {"fps": 24}})
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": ed, "output_path": str(out)}
    )
    assert r.success, r.error
    assert _fps(out) == "24/1"


def test_disabled_subtitles_accept_schema_string_style(tmp_path):
    """The edit schema defines subtitles.style as a display-mode string."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    ed = _edit_decisions(src)
    ed["subtitles"] = {"enabled": False, "style": "none"}
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": ed, "output_path": str(out)}
    )
    assert r.success, r.error


def test_ffmpeg_final_review_receives_resolved_cut_paths(tmp_path, monkeypatch):
    """Promise review must inspect real media extensions, not asset IDs."""
    tool = VideoCompose()
    out = tmp_path / "out.mp4"
    out.touch()
    resolved = [{"id": "c1", "source": str(tmp_path / "source.mp4"),
                 "in_seconds": 0, "out_seconds": 1}]
    edit = {"version": "1.0", "render_runtime": "ffmpeg",
            "cuts": [{"id": "c1", "source": "asset-id",
                      "in_seconds": 0, "out_seconds": 1}]}
    seen = {}

    monkeypatch.setattr(
        tool,
        "_compose",
        lambda _inputs: ToolResult(success=True, data={}, artifacts=[str(out)]),
    )

    def review(_output, reviewed_edit, *_args, **_kwargs):
        seen["source"] = reviewed_edit["cuts"][0]["source"]
        return {"status": "pass"}

    monkeypatch.setattr(tool, "_run_final_review", review)
    result = tool._render_via_ffmpeg(
        inputs={"options": {"subtitle_burn": False}},
        edit_decisions=edit,
        resolved_cuts=resolved,
        output_path=out,
        profile=None,
    )
    assert result.success
    assert seen["source"].endswith("source.mp4")


def test_target_duration_pads_video_stream_to_exact_frame_count(tmp_path):
    """Concat/AAC timestamps must not leave video shorter than target audio."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    cuts = [
        {"id": f"c{i}", "source": str(src), "in_seconds": 0, "out_seconds": 0.4}
        for i in range(5)
    ]
    ed = {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "cuts": cuts,
        "metadata": {
            "compose_target": {"width": 720, "height": 1280, "fit": "pad", "fps": 24},
            "target_duration_seconds": 2.0,
        },
    }
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": ed, "output_path": str(out)}
    )
    assert r.success, r.error
    duration, frames = _video_duration_and_frames(out)
    assert duration == pytest.approx(2.0, abs=0.01)
    assert frames == 48


def test_precompose_accepts_canonical_scene_plan_object(monkeypatch):
    """The risk gate must score scene_plan['scenes'], not the wrapper object."""
    tool = VideoCompose()
    seen = {}

    def fake_score(scenes, *_args):
        seen["scenes"] = scenes
        return {"average": 0.0, "verdict": "strong"}

    monkeypatch.setattr("lib.slideshow_risk.score_slideshow_risk", fake_score)
    edit = {
        "version": "1.0",
        "renderer_family": "documentary-montage",
        "render_runtime": "ffmpeg",
        "cuts": [{"id": "c1", "source": "clip.mp4", "in_seconds": 0, "out_seconds": 1}],
    }
    resolved = [{"id": "c1", "source": "clip.mp4", "in_seconds": 0, "out_seconds": 1}]
    canonical_scene_plan = {"version": "2.0", "scenes": [{"id": "s1", "type": "generated_video"}]}

    result = tool._pre_compose_validation(edit, resolved, canonical_scene_plan)

    assert result is None
    assert seen["scenes"] == canonical_scene_plan["scenes"]
