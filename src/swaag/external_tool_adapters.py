from __future__ import annotations

from swaag.config import AgentConfig
from swaag.external_mcp import ExternalMcpManager
from swaag.external_tools import RuntimeExternalToolManager


def build_runtime_external_tool_manager(config: AgentConfig) -> RuntimeExternalToolManager:
    """Build configured layer-three adapters outside core runtime semantics."""
    adapters = []
    if config.external_tools.mcp_servers:
        adapters.append(ExternalMcpManager(config.external_tools))
    return RuntimeExternalToolManager(adapters)
