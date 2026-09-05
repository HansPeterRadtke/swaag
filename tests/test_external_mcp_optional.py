from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from swaag.config import ExternalMcpServerConfig
from swaag.external_mcp import ExternalMcpClient, ExternalMcpError

pytestmark = pytest.mark.external_tool_integration


def _enabled() -> bool:
    return os.environ.get("SWAAG_RUN_EXTERNAL_TOOL_INTEGRATION") == "1"


def _repo_command(repo: Path, module: str) -> list[str]:
    external_python = Path("/data/venv/bin/python")
    if not external_python.exists():
        pytest.skip("shared external-tool Python environment is not installed")
    code = (
        f"import sys;sys.path.insert(0,{str(repo / 'src')!r});"
        f"from {module} import main;raise SystemExit(main())"
    )
    return [str(external_python), "-c", code]


@pytest.mark.skipif(not _enabled(), reason="optional external-tool integration suite disabled")
def test_real_aubro_mcp_server_is_discoverable() -> None:
    repo = Path("/data/src/github/devtests/aubro")
    if not repo.exists():
        pytest.skip("Aubro repository is not installed")
    client = ExternalMcpClient(
        "aubro",
        ExternalMcpServerConfig(
            enabled=True,
            optional=True,
            transport="stdio",
            command=_repo_command(repo, "aubro.mcp_server"),
            url="",
            header_env={},
            credential_command=[],
            credential_refresh_skew_seconds=30.0,
            timeout_seconds=20.0,
        ),
    )
    try:
        names = [tool.name for tool in client.list_tools()]
    except ExternalMcpError as exc:
        pytest.skip(f"Aubro MCP server unavailable: {exc}")
    assert names == ["aubro_search", "aubro_browse"]


@pytest.mark.skipif(not _enabled(), reason="optional external-tool integration suite disabled")
def test_real_all2text_mcp_capabilities_are_callable() -> None:
    repo = Path("/data/src/github/all2text")
    if not repo.exists():
        pytest.skip("all2text repository is not installed")
    client = ExternalMcpClient(
        "all2text",
        ExternalMcpServerConfig(
            enabled=True,
            optional=True,
            transport="stdio",
            command=_repo_command(repo, "all2text.mcp_server"),
            url="",
            header_env={},
            credential_command=[],
            credential_refresh_skew_seconds=30.0,
            timeout_seconds=30.0,
        ),
    )
    try:
        names = [tool.name for tool in client.list_tools()]
        result = client.call_tool("all2text_capabilities", {})
    except ExternalMcpError as exc:
        pytest.skip(f"all2text MCP server unavailable: {exc}")
    assert names == ["all2text_capabilities", "all2text_convert"]
    assert result.is_error is False
    assert result.structured_content


@pytest.mark.skipif(not _enabled(), reason="optional external-tool integration suite disabled")
def test_real_aubro_mcp_search_when_browser_environment_is_available() -> None:
    repo = Path("/data/src/github/devtests/aubro")
    if not repo.exists():
        pytest.skip("Aubro repository is not installed")
    client = ExternalMcpClient(
        "aubro",
        ExternalMcpServerConfig(
            enabled=True,
            optional=True,
            transport="stdio",
            command=_repo_command(repo, "aubro.mcp_server"),
            url="",
            header_env={},
            credential_command=[],
            credential_refresh_skew_seconds=30.0,
            timeout_seconds=90.0,
        ),
    )
    try:
        result = client.call_tool(
            "aubro_search", {"query": "Model Context Protocol", "limit": 1, "engine": "duckduckgo"}
        )
    except ExternalMcpError as exc:
        pytest.skip(f"Aubro browser environment unavailable: {exc}")
    if result.is_error:
        pytest.skip(
            "Aubro MCP server is installed but its browser/network environment is unavailable: "
            + str(result.structured_content.get("error", "external execution failed"))
        )
    assert result.structured_content.get("query") == "Model Context Protocol"
    assert isinstance(result.structured_content.get("results"), list)
