from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_collector_config_is_loopback_only_and_bounded() -> None:
    config = (ROOT / "deploy" / "otelcol-contrib.yaml").read_text(encoding="utf-8")

    assert "endpoint: 127.0.0.1:13501" in config
    assert "endpoint: 127.0.0.1:13502" in config
    assert "0.0.0.0" not in config
    assert "file/traces:" in config
    assert "path: /data/var/swaag/telemetry/traces.json" in config
    assert "file/metrics:" in config
    assert "path: /data/var/swaag/telemetry/metrics.json" in config
    assert "exporters: [file/traces]" in config
    assert "exporters: [file/metrics]" in config
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


def test_mcp_sdk_conformance_assets_are_pinned_and_parse() -> None:
    installer = ROOT / "scripts" / "install-mcp-conformance-env.sh"
    probe = ROOT / "scripts" / "mcp-sdk-conformance.mjs"
    installer_source = installer.read_text(encoding="utf-8")
    probe_source = probe.read_text(encoding="utf-8")

    shell_check = subprocess.run(
        ["bash", "-n", str(installer)],
        text=True,
        capture_output=True,
        check=False,
    )
    node_check = subprocess.run(
        ["node", "--check", str(probe)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert shell_check.returncode == 0, shell_check.stderr
    assert node_check.returncode == 0, node_check.stderr
    assert 'VERSION="2.0.0"' in installer_source
    assert "@modelcontextprotocol/client@${VERSION}" in installer_source
    assert "@modelcontextprotocol/core@${VERSION}" in installer_source
    assert "mode: { pin: \"2026-07-28\" }" in probe_source
    assert "client.listTools" in probe_source
    assert "client.callTool" in probe_source


def test_a2a_sdk_conformance_assets_are_pinned_and_parse() -> None:
    installer = ROOT / "scripts" / "install-a2a-conformance-env.sh"
    probe = ROOT / "scripts" / "a2a-sdk-conformance.mjs"
    installer_source = installer.read_text(encoding="utf-8")
    probe_source = probe.read_text(encoding="utf-8")

    shell_check = subprocess.run(
        ["bash", "-n", str(installer)],
        text=True,
        capture_output=True,
        check=False,
    )
    node_check = subprocess.run(
        ["node", "--check", str(probe)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert shell_check.returncode == 0, shell_check.stderr
    assert node_check.returncode == 0, node_check.stderr
    assert 'VERSION="1.1.0"' in installer_source
    assert "@a2a-js/sdk@${VERSION}" in installer_source
    assert "new JsonRpcTransportFactory()" in probe_source
    assert "TaskState.TASK_STATE_UNSPECIFIED" in probe_source
    assert "factory.createFromUrl" in probe_source
    assert "client.listTasks" in probe_source
    assert "client.getTask" in probe_source
    assert ".resubscribeTask" in probe_source
    assert "client.cancelTask" in probe_source
