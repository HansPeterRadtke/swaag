#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
path = Path('/workspace/app.py')
path.write_text('def greet(name: str) -> str:\\n    return f\"Hello, {name}!\"\\n', encoding='utf-8')
PY
