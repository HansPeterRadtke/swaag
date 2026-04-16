from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


EXCLUDE_PARTS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
SUBSET_TESTS = [
    "tests/test_imports.py",
    "tests/test_scaled_catalog.py",
    "tests/test_runtime_verification_flow.py",
    "tests/test_end_to_end.py",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name.startswith(".coverage"):
        return False
    return True


def _build_archive(project_root: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project_root.rglob("*")):
            if path.is_dir() or not _should_include(path.relative_to(project_root)):
                continue
            archive.write(path, arcname=str(path.relative_to(project_root)))


def _tracked_project_files(project_root: Path) -> set[str]:
    parent = project_root.parent
    result = subprocess.run(
        ["git", "-C", str(parent), "ls-files", project_root.name],
        check=True,
        text=True,
        capture_output=True,
    )
    tracked: set[str] = set()
    prefix = f"{project_root.name}/"
    for line in result.stdout.splitlines():
        if not line.startswith(prefix):
            continue
        tracked.add(line[len(prefix) :])
    return tracked


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an archive from the current working tree, extract it, and run proof tests.")
    parser.add_argument("--archive", default="agent_worktree.zip", help="Archive file name to create inside the temporary proof directory.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary extraction directory.")
    args = parser.parse_args(argv)

    project_root = _project_root()
    temp_dir = Path(tempfile.mkdtemp(prefix="swaag-archive-proof-"))
    archive_path = temp_dir / args.archive
    extract_root = temp_dir / "extracted"
    try:
        _build_archive(project_root, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            tracked_files = _tracked_project_files(project_root)
            archived_files = set(archive.namelist())
            missing_tracked = sorted(path for path in tracked_files if path not in archived_files)
            if missing_tracked:
                raise SystemExit(f"Archive is missing tracked project files: {', '.join(missing_tracked[:20])}")
            archive.extractall(extract_root)

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", "src")

        for test_path in SUBSET_TESTS:
            _run([sys.executable, "-m", "pytest", "-q", test_path], cwd=extract_root, env=env)
        _run([sys.executable, "-m", "pytest", "-q"], cwd=extract_root, env=env)
        print(f"archive={archive_path}")
        print(f"extracted={extract_root}")
        return 0
    finally:
        if args.keep_temp:
            print(f"kept_temp={temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
