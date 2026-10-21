#!/usr/bin/env python3
"""Read-only SWAAG source-to-installed-package and SQLite release preflight.

This checks disk contents, NOT modules already loaded by a running daemon.
It must never be used as sole evidence that a service is current or functional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

MIN_SQLITE = (3, 51, 3)
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _manifest(root: Path) -> tuple[dict[str, str], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"package directory not found: {root}")
    files: dict[str, str] = {}
    symlinks: list[str] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if "__pycache__" in relative.parts or candidate.suffix in IGNORED_SUFFIXES:
            continue
        if candidate.is_symlink():
            symlinks.append(relative.as_posix())
        elif candidate.is_file():
            files[relative.as_posix()] = _digest(candidate)
    return files, symlinks


def compare_packages(source: Path, installed: Path) -> dict[str, Any]:
    source_files, source_symlinks = _manifest(source)
    installed_files, installed_symlinks = _manifest(installed)
    only_source = sorted(source_files.keys() - installed_files.keys())
    only_installed = sorted(installed_files.keys() - source_files.keys())
    changed = sorted(
        key for key in source_files.keys() & installed_files.keys()
        if source_files[key] != installed_files[key]
    )
    return {
        "passed": not (only_source or only_installed or changed or source_symlinks or installed_symlinks),
        "source_count": len(source_files),
        "installed_count": len(installed_files),
        "source_only": only_source,
        "installed_only": only_installed,
        "different": changed,
        "source_symlinks": source_symlinks,
        "installed_symlinks": installed_symlinks,
    }


def inspect_venv(venv: Path) -> dict[str, Any]:
    """Query only the designated interpreter, ignoring inherited PYTHONPATH.

    The SQLite FTS probe writes exclusively to an in-memory database.
    """
    python = venv / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"missing runtime interpreter: {python}")
    script = """
import importlib.util,json,pathlib,sqlite3
spec=importlib.util.find_spec('swaag')
if spec is None or spec.origin is None:
    raise RuntimeError('swaag package is not installed')
db=sqlite3.connect(':memory:')
db.execute('CREATE VIRTUAL TABLE preflight_fts USING fts5(text)')
print(json.dumps({'root':str(pathlib.Path(spec.origin).resolve().parent),
                  'sqlite':sqlite3.sqlite_version,'fts5':True}))
"""
    run = subprocess.run(
        [str(python), "-I", "-c", script],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if run.returncode:
        raise RuntimeError(
            f"runtime interpreter probe failed (exit {run.returncode}); "
            "verify package installation and SQLite FTS5 support"
        )
    data = json.loads(run.stdout)
    version = tuple(int(p) for p in data["sqlite"].split(".")[:3])
    return {
        "root": str(Path(data["root"]).resolve()),
        "sqlite": data["sqlite"],
        "fts5": data["fts5"],
        "sqlite_compatible": version >= MIN_SQLITE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/data/src/github/swaag"))
    parser.add_argument("--venv", type=Path, default=Path("/data/var/swaag/venv"))
    parser.add_argument(
        "--installed", type=Path,
        help="Installed package directory override for isolated/offline comparison",
    )
    parser.add_argument("--output", type=Path, help="Optional local JSON report")
    args = parser.parse_args(argv)
    try:
        source = args.repo.resolve() / "src/swaag"
        runtime = None if args.installed is not None else inspect_venv(args.venv.resolve())
        installed = args.installed.resolve() if args.installed is not None else Path(runtime["root"])
        comparison = compare_packages(source, installed)
        passed = comparison["passed"] and (runtime is None or runtime["sqlite_compatible"])
        report: dict[str, Any] = {
            "passed": passed,
            "source": str(source),
            "installed": str(installed),
            "comparison": comparison,
            "runtime": runtime,
            "scope": "Disk package bytes and interpreter SQLite only; live process acceptance not tested",
        }
    except (FileNotFoundError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        report = {
            "passed": False,
            "error": type(error).__name__,
            "reason": str(error),
            "scope": "Preflight could not complete; do not infer installed source equality",
        }
    serialized = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
