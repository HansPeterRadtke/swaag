from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _src_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "src")


def test_benchmark_modules_import_without_optional_datasets_dependency() -> None:
    script = """
import builtins
real_import = builtins.__import__
def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "datasets" or name.startswith("datasets."):
        raise ModuleNotFoundError("No module named 'datasets'")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = blocked_import
import swaag.benchmark
import swaag.benchmark.external
import swaag.benchmark.benchmark_runner
print("imports-ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = _src_path() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "imports-ok"
