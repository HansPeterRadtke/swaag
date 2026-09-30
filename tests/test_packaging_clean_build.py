from __future__ import annotations

import subprocess
import shutil
import zipfile
from pathlib import Path


def test_wheel_build_does_not_ship_stale_build_lib_modules(tmp_path):
    source_repo = Path(__file__).resolve().parents[1]
    # Build a disposable source snapshot: the regression must neither mutate the
    # developer's build tree nor depend on ownership left by a prior deployment.
    repo = tmp_path / "source"
    repo.mkdir()
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(source_repo / name, repo / name)
    shutil.copytree(
        source_repo / "src",
        repo / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.egg-info"),
    )
    stale = repo / "build/lib/swaag/__stale_packaging_probe__.py"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("SHOULD_NOT_SHIP = True\n", encoding="utf-8")
    out = tmp_path / "dist"
    try:
        subprocess.run(
            [
                "/data/var/cache/uv/bin/uv",
                "build",
                "--wheel",
                "--no-cache",
                "--out-dir",
                str(out),
                str(repo),
            ],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        wheel = next(out.glob("swaag-*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        assert "swaag/__stale_packaging_probe__.py" not in names
        source_names = {
            path.relative_to(repo / "src").as_posix()
            for path in (repo / "src/swaag").rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        }
        wheel_package_names = {
            name for name in names if name.startswith("swaag/") and not name.endswith("/")
        }
        assert wheel_package_names == source_names
    finally:
        stale.unlink(missing_ok=True)
