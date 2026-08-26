from __future__ import annotations

from pathlib import Path


def test_runtime_installer_refreshes_noneditable_project_package() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "install-runtime-env.sh"
    ).read_text(encoding="utf-8")

    assert "--no-editable" in script
    assert "--reinstall-package swaag" in script
    assert 'UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/venv"' in script
