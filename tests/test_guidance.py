from __future__ import annotations

from pathlib import Path

from swaag.context_builder import build_context
from swaag.tokens import ConservativeEstimator
from swaag.types import ExpandedTask, SessionState


def _state(root: Path, cwd: Path) -> SessionState:
    state = SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
    )
    state.environment.workspace.root = str(root)
    state.environment.workspace.cwd = str(cwd)
    return state


def test_guidance_inheritance_and_directory_override(make_config, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    subdir = root / "pkg"
    subdir.mkdir(parents=True)
    global_file = tmp_path / "global.md"
    global_file.write_text("Global: always verify.\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("Repo: keep patches minimal.\n", encoding="utf-8")
    (subdir / "AGENTS.md").write_text("Directory: prefer pkg-specific naming.\n", encoding="utf-8")
    state = _state(root, subdir)
    config = make_config(guidance__global_paths=[str(global_file)])

    bundle = build_context(config, state, ConservativeEstimator(), goal="edit pkg module")

    assert "Global: always verify." in bundle.guidance_text
    assert "Repo: keep patches minimal." in bundle.guidance_text
    assert "Directory: prefer pkg-specific naming." in bundle.guidance_text
    assert any(item.item_type == "guidance" and item.selected for item in bundle.selection_trace)


def test_task_local_and_role_guidance_are_included(make_config, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    state = _state(root, root)
    state.active_role = "verifier"
    state.expanded_task = ExpandedTask(
        original_goal="fix it",
        expanded_goal="fix it with care",
        constraints=["Do not widen the patch"],
        expected_outputs=["exact fix"],
    )

    bundle = build_context(make_config(), state, ConservativeEstimator(), goal="fix it")

    assert "Do not widen the patch" in bundle.guidance_text
    assert "Reject unsupported or partial results." in bundle.guidance_text
    assert bundle.guidance_sources


def test_irrelevant_directory_guidance_is_excluded_when_not_relevant(make_config, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    subdir = root / "pkg"
    subdir.mkdir(parents=True)
    (root / "AGENTS.md").write_text("Repo: verify changes.\n", encoding="utf-8")
    (subdir / "AGENTS.md").write_text("Directory: focus only on CSS theme palettes.\n", encoding="utf-8")
    state = _state(root, subdir)

    bundle = build_context(
        make_config(),
        state,
        ConservativeEstimator(),
        goal="Fix the parser configuration loading bug in pkg/parser.py",
    )

    assert "Repo: verify changes." in bundle.guidance_text
    assert "CSS theme palettes" not in bundle.guidance_text
    assert any(item.item_type == "guidance" and item.selected for item in bundle.selection_trace)
