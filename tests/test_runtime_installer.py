from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest


def test_runtime_installer_refreshes_noneditable_project_package() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "install-runtime-env.sh"
    ).read_text(encoding="utf-8")

    assert "--no-editable" in script
    assert "--reinstall-package swaag" in script
    assert 'UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/venv"' in script


# The real installer is tested with an isolated fake uv and fake runtime
# interpreter. These tests verify ordering and failure propagation without
# fetching dependencies, mutating deployed environments, or contacting a model.
@pytest.mark.parametrize("sqlite_version,modify_installed,expected_success", [
    ("3.50.4", False, False),
    ("3.53.1", False, True),
    ("3.53.1", True, False),
])
def test_install_runtime_fails_closed_on_actual_preflight_exit(
    tmp_path: Path, sqlite_version: str, modify_installed: bool,
    expected_success: bool,
) -> None:
    root = Path(__file__).resolve().parents[1]
    installed = tmp_path / "installed-swaag-package"
    shutil.copytree(root / "src/swaag", installed)
    if modify_installed:
        init_file = installed / "__init__.py"
        init_file.write_bytes(init_file.read_bytes() + b"\n# drift test\n")

    # This helper mimics an interpreter's isolated *reported* capabilities.
    # The real preflight and exact source-to-install byte comparison still run.
    runtime_python = tmp_path / "stub-runtime-python"
    runtime_python.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        "if sys.argv[1:3] == ['-I', '-c']:\n"
        "    print(json.dumps({'root': " + repr(str(installed))
        + ", 'sqlite': " + repr(sqlite_version) + ", 'fts5': True}))\n"
        "elif sys.argv[1] == '-c':\n"
        "    print('swaag runtime environment ok')\n"
        "else: sys.exit(9)\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o700)
    cli = tmp_path / "stub-swaag-cli"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)
    uv = tmp_path / "stub-uv"
    uv.write_text(
        "#!/bin/sh\nset -eu\n"
        'case "$1" in\n'
        '  python) test "$2" = install ;;\n'
        '  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; '
        'cp "$MOCK_RUNTIME_PYTHON" "$UV_PROJECT_ENVIRONMENT/bin/python"; '
        'cp "$MOCK_RUNTIME_CLI" "$UV_PROJECT_ENVIRONMENT/bin/swaag" ;;\n'
        "  *) exit 12 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    uv.chmod(0o700)
    target = tmp_path / "owned-test-runtime"
    env = os.environ.copy()
    env.update({
        "UV_BIN": str(uv),
        "SWAAG_RUNTIME_ROOT": str(target),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "MOCK_RUNTIME_PYTHON": str(runtime_python),
        "MOCK_RUNTIME_CLI": str(cli),
    })
    result = subprocess.run(
        ["bash", str(root / "scripts/install-runtime-env.sh")],
        cwd=tmp_path, env=env, text=True, capture_output=True,
        check=False, timeout=30,
    )
    assert (result.returncode == 0) is expected_success, result.stderr
    report, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
    assert report["runtime"]["sqlite_compatible"] is (sqlite_version != "3.50.4")
    assert report["comparison"]["passed"] is (not modify_installed)
    assert report["passed"] is expected_success
    assert ("swaag runtime environment ok" in result.stdout) is expected_success
    assert (str(target / "venv") in result.stdout) is expected_success
