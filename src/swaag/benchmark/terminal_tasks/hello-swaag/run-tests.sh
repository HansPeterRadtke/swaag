#!/usr/bin/env bash
set -euo pipefail

cd /workspace
/usr/bin/python3 -m pytest "$TEST_DIR/test_outputs.py" -q -rA
