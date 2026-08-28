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
    http_probe = ROOT / "scripts" / "mcp-http-sdk-conformance.mjs"
    http_runner = ROOT / "scripts" / "run-mcp-http-sdk-conformance.py"
    installer_source = installer.read_text(encoding="utf-8")
    probe_source = probe.read_text(encoding="utf-8")
    http_probe_source = http_probe.read_text(encoding="utf-8")
    http_runner_source = http_runner.read_text(encoding="utf-8")

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
    http_node_check = subprocess.run(
        ["node", "--check", str(http_probe)],
        text=True,
        capture_output=True,
        check=False,
    )
    http_python_check = subprocess.run(
        ["python", "-m", "py_compile", str(http_runner)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert shell_check.returncode == 0, shell_check.stderr
    assert node_check.returncode == 0, node_check.stderr
    assert http_node_check.returncode == 0, http_node_check.stderr
    assert http_python_check.returncode == 0, http_python_check.stderr
    assert 'VERSION="2.0.0"' in installer_source
    assert "@modelcontextprotocol/client@${VERSION}" in installer_source
    assert "@modelcontextprotocol/core@${VERSION}" in installer_source
    assert 'mode: { pin: "2026-07-28" }' in probe_source
    assert "client.listTools" in probe_source
    assert "client.callTool" in probe_source
    assert "new StreamableHTTPClientTransport" in http_probe_source
    assert '"x-mcp-header"' in http_probe_source
    assert "_NoInferenceClient" in http_runner_source
    assert "model_client=no_inference" in http_runner_source


def test_a2a_sdk_conformance_assets_are_pinned_and_parse() -> None:
    installer = ROOT / "scripts" / "install-a2a-conformance-env.sh"
    probe = ROOT / "scripts" / "a2a-sdk-conformance.mjs"
    runner = ROOT / "scripts" / "run-a2a-sdk-conformance.py"
    installer_source = installer.read_text(encoding="utf-8")
    probe_source = probe.read_text(encoding="utf-8")
    runner_source = runner.read_text(encoding="utf-8")

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
    python_check = subprocess.run(
        ["python", "-m", "py_compile", str(runner)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert shell_check.returncode == 0, shell_check.stderr
    assert node_check.returncode == 0, node_check.stderr
    assert python_check.returncode == 0, python_check.stderr
    assert 'VERSION="1.1.0"' in installer_source
    assert "@a2a-js/sdk@${VERSION}" in installer_source
    assert "new JsonRpcTransportFactory()" in probe_source
    assert "new RestTransportFactory()" in probe_source
    assert 'transport === "http-json"' in probe_source
    assert "TaskState.TASK_STATE_UNSPECIFIED" in probe_source
    assert "factory.createFromUrl" in probe_source
    assert "client.listTasks" in probe_source
    assert "client.getTask" in probe_source
    assert ".resubscribeTask" in probe_source
    assert "client.cancelTask" in probe_source
    assert "client.sendMessage(" in probe_source
    assert ".sendMessageStream(" in probe_source
    assert "model_client=no_inference" in runner_source
    assert "queue_without_executor" in runner_source
    assert 'choices=("jsonrpc", "http-json")' in runner_source
    assert '"--exercise-existing-task"' in runner_source


def test_ag_ui_sdk_conformance_assets_are_pinned_and_parse() -> None:
    installer = ROOT / "scripts" / "install-ag-ui-conformance-env.sh"
    probe = ROOT / "scripts" / "ag-ui-sdk-conformance.mjs"
    preparer = ROOT / "scripts" / "prepare-ag-ui-conformance.py"
    runner = ROOT / "scripts" / "run-ag-ui-sdk-conformance.py"
    client_tool_probe = ROOT / "scripts" / "ag-ui-client-tool-conformance.mjs"
    client_tool_runner = ROOT / "scripts" / "run-ag-ui-client-tool-conformance.py"
    installer_source = installer.read_text(encoding="utf-8")
    probe_source = probe.read_text(encoding="utf-8")
    runner_source = runner.read_text(encoding="utf-8")
    client_tool_probe_source = client_tool_probe.read_text(encoding="utf-8")
    client_tool_runner_source = client_tool_runner.read_text(encoding="utf-8")

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
    client_tool_node_check = subprocess.run(
        ["node", "--check", str(client_tool_probe)],
        text=True,
        capture_output=True,
        check=False,
    )
    python_check = subprocess.run(
        [
            "python",
            "-m",
            "py_compile",
            str(preparer),
            str(runner),
            str(client_tool_runner),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert shell_check.returncode == 0, shell_check.stderr
    assert node_check.returncode == 0, node_check.stderr
    assert client_tool_node_check.returncode == 0, client_tool_node_check.stderr
    assert python_check.returncode == 0, python_check.stderr
    assert 'VERSION="0.0.59"' in installer_source
    assert "@ag-ui/client@${VERSION}" in installer_source
    assert "@ag-ui/core@${VERSION}" in installer_source
    assert "@ag-ui/encoder@${VERSION}" in installer_source
    assert "new HttpAgent" in probe_source
    assert "AgentCapabilitiesSchema.parse" in probe_source
    assert 'fetch(`${normalizedBaseUrl}/ag-ui/capabilities`)' in probe_source
    assert "onStateSnapshotEvent" in probe_source
    assert "onStateDeltaEvent" in probe_source
    assert "isDeepStrictEqual(agent.state, expectedState)" in probe_source
    assert "agent.runAgent" in probe_source
    assert "onRunFinishedEvent" in probe_source
    assert "model_client=no_inference" in runner_source
    assert "complete_without_inference" in runner_source
    assert "clientProvided !== true" in client_tool_probe_source
    assert 'role: "tool"' in client_tool_probe_source
    assert "TOOL_CALL_START" in client_tool_probe_source
    assert "TOOL_CALL_RESULT" not in client_tool_probe_source
    assert "_NoInferenceClient" in client_tool_runner_source
    assert "model_client=no_inference" in client_tool_runner_source
    assert '"inference_allowed": False' in client_tool_runner_source
