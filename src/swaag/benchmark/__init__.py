from __future__ import annotations

from typing import Any, Sequence

__all__ = ["main", "run_benchmarks"]


def main(argv: Sequence[str] | None = None) -> int:
    from swaag.benchmark.benchmark_runner import main as _main

    return _main(argv)


def run_benchmarks(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from swaag.benchmark.benchmark_runner import run_benchmarks as _run_benchmarks

    return _run_benchmarks(*args, **kwargs)
