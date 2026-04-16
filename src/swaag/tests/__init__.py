from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(verbose: bool = False) -> int:
    test_dir = Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "pytest", str(test_dir)]
    if verbose:
        cmd.append("-v")
    print(f"[SWAAG] Running package smoke tests: {' '.join(cmd)}")
    return subprocess.call(cmd)
