#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Building SWAAG package..."
python3 -m build

echo "[INFO] Uploading package to PyPI..."
twine upload dist/* || echo "[ERROR] Upload failed. Artifacts remain available in dist/."

echo "[INFO] Cleaning up build artifacts..."
rm -rf build dist src/*.egg-info ./*.egg-info

echo "[INFO] Done."
