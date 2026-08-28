#!/usr/bin/env bash
set -euo pipefail

VERSION="1.1.0"
RUNTIME_ROOT="${SWAAG_RUNTIME_ROOT:-/data/var/swaag}"
INSTALL_ROOT="${A2A_CONFORMANCE_ROOT:-${RUNTIME_ROOT}/protocol-conformance/a2a-js-${VERSION}}"

verify() {
  node - "$INSTALL_ROOT" "$VERSION" <<'NODE'
const fs = require('node:fs');
const path = require('node:path');
const [root, expected] = process.argv.slice(2);
const file = path.join(root, 'node_modules', '@a2a-js', 'sdk', 'package.json');
const payload = JSON.parse(fs.readFileSync(file, 'utf8'));
if (payload.version !== expected) {
  throw new Error(`${payload.name} version ${payload.version} does not match ${expected}`);
}
NODE
}

if verify 2>/dev/null; then
  printf '%s\n' "$INSTALL_ROOT"
  exit 0
fi

mkdir -p "$INSTALL_ROOT"
npm install --prefix "$INSTALL_ROOT" --no-save --package-lock=false \
  "@a2a-js/sdk@${VERSION}"
verify
printf '%s\n' "$INSTALL_ROOT"
