"""Atelier renders must not leave their staged copy inside remotion-composer/projects/."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.video.video_compose as video_compose
from tools.video.video_compose import VideoCompose


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A fake checkout: remotion-composer with node_modules and a project entry."""
    composer = tmp_path / "remotion-composer"
    (composer / "node_modules").mkdir(parents=True)
    project = tmp_path / "projects" / "demo-film"
    project.mkdir(parents=True)
    (project / "index.tsx").write_text("export {};\n")
    (project / "Composition.tsx").write_text("export {};\n")
    # composer_dir is derived from the module file: tools/video/video_compose.py -> checkout root.
    monkeypatch.setattr(video_compose, "__file__", str(tmp_path / "tools" / "video" / "video_compose.py"))
    return composer, project


def render(project: Path, output: Path):
    return VideoCompose()._render_via_atelier(
        {"output_path": str(output)},
        {"bespoke": {"entry": str(project / "index.tsx"), "composition_id": "Demo"}},
    )


def test_successful_render_removes_staged_copy(checkout, monkeypatch, tmp_path):
    composer, project = checkout
    output = tmp_path / "out.mp4"
    seen = {}

    def fake_run(cmd, timeout, cwd):
        staged = composer / "projects" / "demo-film"
        seen["staged_during_render"] = (staged / "index.tsx").is_file()
        output.write_bytes(b"mp4")

    monkeypatch.setattr(VideoCompose, "run_command", lambda self, cmd, timeout, cwd: fake_run(cmd, timeout, cwd))
    monkeypatch.setattr(VideoCompose, "_probe_output", lambda self, path: {}, raising=False)
    render(project, output)
    assert seen["staged_during_render"]
    assert not (composer / "projects").exists()
    assert (project / "index.tsx").is_file()


def test_failed_render_also_removes_staged_copy(checkout, monkeypatch, tmp_path):
    composer, project = checkout

    def boom(self, cmd, timeout, cwd):
        raise RuntimeError("remotion crashed")

    monkeypatch.setattr(VideoCompose, "run_command", boom)
    result = render(project, tmp_path / "out.mp4")
    assert not result.success
    assert not (composer / "projects").exists()


def test_other_staged_projects_are_left_alone(checkout, monkeypatch, tmp_path):
    composer, project = checkout
    other = composer / "projects" / "someone-else"
    other.mkdir(parents=True)
    monkeypatch.setattr(VideoCompose, "run_command", lambda self, cmd, timeout, cwd: None)
    render(project, tmp_path / "out.mp4")
    assert other.is_dir()
    assert not (composer / "projects" / "demo-film").exists()
