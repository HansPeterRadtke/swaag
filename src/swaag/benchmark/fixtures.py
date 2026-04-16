from __future__ import annotations

from pathlib import Path


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "swebench"


def bounded_swebench_fixture_paths() -> dict[str, Path]:
    root = _fixture_root()
    return {
        "swebench_lite": root / "swebench_lite_sympy__sympy-15345.json",
        "swebench_verified": root / "swebench_verified_pallets__flask-5014.json",
        "swebench_full": root / "swebench_full_pydata__xarray-4098.json",
        "swebench_multilingual": root / "swebench_multilingual_astral-sh__ruff-15309.json",
    }


def terminal_bench_task_root() -> Path:
    return Path(__file__).resolve().parent / "terminal_tasks"
