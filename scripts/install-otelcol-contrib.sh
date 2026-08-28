#!/usr/bin/env bash
set -euo pipefail

VERSION="0.159.0"
ARCHIVE_NAME="otelcol-contrib_${VERSION}_linux_arm64.tar.gz"
ARCHIVE_SHA256="abb8665cc963e886c2d1286c50b38bcb2e53d968b192c3d8fe4d1ed6b91c3901"
DOWNLOAD_URL="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${VERSION}/${ARCHIVE_NAME}"
RUNTIME_ROOT="${SWAAG_RUNTIME_ROOT:-/data/var/swaag}"
INSTALL_ROOT="${OTELCOL_INSTALL_ROOT:-${RUNTIME_ROOT}/otelcol-contrib}"
DOWNLOAD_ROOT="${OTELCOL_DOWNLOAD_ROOT:-${INSTALL_ROOT}/downloads}"
TARGET_ROOT="${INSTALL_ROOT}/${VERSION}"
TARGET="${TARGET_ROOT}/otelcol-contrib"
ARCHIVE="${DOWNLOAD_ROOT}/${ARCHIVE_NAME}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  printf 'otelcol-contrib %s is pinned for Linux ARM64, found %s/%s\n' \
    "$VERSION" "$(uname -s)" "$(uname -m)" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_ROOT" "$INSTALL_ROOT"

if [[ ! -f "$ARCHIVE" ]]; then
  partial="${ARCHIVE}.part.$$"
  trap 'rm -f "$partial"' EXIT
  curl -fL --retry 3 --connect-timeout 20 --max-time 1800 \
    -o "$partial" "$DOWNLOAD_URL"
  mv "$partial" "$ARCHIVE"
  trap - EXIT
fi

printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256sum -c -

if [[ -x "$TARGET" ]]; then
  "$TARGET" --version | grep -Fx "otelcol-contrib version ${VERSION}"
  printf '%s\n' "$TARGET"
  exit 0
fi

if [[ -e "$TARGET_ROOT" ]]; then
  printf 'refusing to replace incomplete collector install: %s\n' "$TARGET_ROOT" >&2
  exit 1
fi

staging="${INSTALL_ROOT}/.install-${VERSION}.$$"
mkdir -p "$staging"
trap 'rm -rf "$staging"' EXIT
tar -xzf "$ARCHIVE" -C "$staging" otelcol-contrib
chmod 0755 "$staging/otelcol-contrib"
"$staging/otelcol-contrib" --version | grep -Fx "otelcol-contrib version ${VERSION}"
mv "$staging" "$TARGET_ROOT"
trap - EXIT
printf '%s\n' "$TARGET"
