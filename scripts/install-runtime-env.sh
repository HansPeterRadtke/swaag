#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-/data/var/cache/uv/bin/uv}"
RUNTIME_ROOT="${SWAAG_RUNTIME_ROOT:-/data/var/swaag}"
PYTHON_VERSION="${SWAAG_PYTHON_VERSION:-3.13}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/var/cache/uv/cache}"
export UV_PYTHON_INSTALL_DIR="$RUNTIME_ROOT/python"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/venv"
mkdir -p "$RUNTIME_ROOT" "$UV_CACHE_DIR"
"$UV_BIN" python install --install-dir "$UV_PYTHON_INSTALL_DIR" "$PYTHON_VERSION"
cd "$ROOT"
"$UV_BIN" sync --frozen --no-dev --no-editable --reinstall-package swaag --python "$PYTHON_VERSION"
# A successful dependency sync is not a usable durable runtime. Before
# announcing success, reject an unsupported SQLite/FTS5 interpreter or a
# package whose installed bytes differ from the source being deployed.
# This checks files on disk only; it does not restart or validate live services.
python3 "$ROOT/scripts/verify-installed-package.py" --repo "$ROOT" --venv "$RUNTIME_ROOT/venv"
"$RUNTIME_ROOT/venv/bin/python" -c 'import swaag, requests; print("swaag runtime environment ok")'
"$RUNTIME_ROOT/venv/bin/swaag" --help >/dev/null
printf '%s\n' "$RUNTIME_ROOT/venv"
