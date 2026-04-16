from __future__ import annotations

import subprocess
import sys


def main() -> int:
    print("\n================= RUNNING SWAAG TEST SUITE =================\n")
    cmd = ["pytest", "-q"]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
