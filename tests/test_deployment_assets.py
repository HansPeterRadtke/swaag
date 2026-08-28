from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_collector_config_is_loopback_only_and_bounded() -> None:
    config = (ROOT / "deploy" / "otelcol-contrib.yaml").read_text(encoding="utf-8")

    assert "endpoint: 127.0.0.1:13501" in config
    assert "endpoint: 127.0.0.1:13502" in config
    assert "0.0.0.0" not in config
    assert "path: /data/var/swaag/telemetry/otlp.json" in config
    assert "max_megabytes:" in config
    assert "max_days:" in config
    assert "max_backups:" in config
    assert "level: none" in config


def test_collector_installer_is_valid_and_checksum_pinned() -> None:
    installer = ROOT / "scripts" / "install-otelcol-contrib.sh"
    source = installer.read_text(encoding="utf-8")

    completed = subprocess.run(
        ["bash", "-n", str(installer)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'VERSION="0.159.0"' in source
    assert 'ARCHIVE_SHA256="abb8665cc963e886c2d1286c50b38bcb2e53d968b192c3d8fe4d1ed6b91c3901"' in source
    assert "sha256sum -c -" in source
    assert "/releases/download/v${VERSION}/" in source
