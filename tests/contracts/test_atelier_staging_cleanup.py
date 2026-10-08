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
        staged = Path(cmd[3]).parent
        seen["staged_during_render"] = (staged / "index.tsx").is_file()
        seen["staged"] = staged
        assert not staged.is_relative_to(composer.parent)
        assert (staged.parent.parent / "node_modules").resolve() == (composer / "node_modules").resolve()
        output.write_bytes(b"mp4")

    monkeypatch.setattr(VideoCompose, "run_command", lambda self, cmd, timeout, cwd: fake_run(cmd, timeout, cwd))
    monkeypatch.setattr(VideoCompose, "_probe_output", lambda self, path: {}, raising=False)
    render(project, output)
    assert seen["staged_during_render"]
    assert not seen["staged"].exists()
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
    other = VideoCompose._atelier_runtime_composer(composer) / "projects" / "someone-else"
    other.mkdir(parents=True)
    monkeypatch.setattr(VideoCompose, "run_command", lambda self, cmd, timeout, cwd: None)
    render(project, tmp_path / "out.mp4")
    assert other.is_dir()
    assert not (composer / "projects" / "demo-film").exists()


def test_concurrent_render_stages_are_distinct(checkout):
    composer, project = checkout
    first, second = VideoCompose(), VideoCompose()
    one = first._stage_atelier_project(project / "index.tsx", composer)
    two = second._stage_atelier_project(project / "index.tsx", composer)
    assert one != two and one.is_file() and two.is_file()
    first._unstage_atelier_project()
    assert not one.exists() and two.is_file()
    second._unstage_atelier_project()
    assert (project / "index.tsx").is_file()


def test_staging_inside_checkout_is_refused(checkout, monkeypatch):
    composer, project = checkout
    monkeypatch.setenv("OPENMONTAGE_RUNTIME_ROOT", str(composer.parent / "runtime"))
    with pytest.raises(ValueError, match="outside the code checkout"):
        VideoCompose()._stage_atelier_project(project / "index.tsx", composer)
