"""Canonical foreground orchestrator must preserve an explicit tool-free policy."""

from swaag.communication import CommunicationService
from swaag.config import load_config
from swaag.runtime import AgentRuntime


def _config(tmp_path, *, tool_free: bool):
    env={
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__AGENT_DATA__ROOT": str(tmp_path / "agent-data"),
        "SWAAG__LOGGING__FILE_PATH": str(tmp_path / "events.jsonl"),
    }
    if not tool_free:
        return load_config(env=env)
    config_path=tmp_path / "noaction.toml"
    config_path.write_text(
        "[tools]\nenabled=[]\nstaged_discovery=false\n"
        "allow_stateful_tools=false\nallow_side_effect_tools=false\n",
        encoding="utf-8")
    return load_config([config_path],env=env)


def test_explicit_tool_free_is_not_replaced_by_orchestrator_defaults(tmp_path):
    service=CommunicationService.from_runtime(AgentRuntime(_config(tmp_path,tool_free=True)))
    try:
        c=service.orchestrator_runtime.config
        assert c.tools.enabled==[]
        assert c.tools.staged_discovery is False
        assert c.tools.allow_stateful_tools is False
        assert c.tools.allow_side_effect_tools is False
        assert service.orchestrator_runtime.tools.enabled_tools(c)==[]
        assert service.orchestrator_runtime.tools.prompt_tuples(c)==[]
    finally:
        service.close()


def test_ordinary_orchestrator_keeps_intended_production_tools(tmp_path):
    service=CommunicationService.from_runtime(AgentRuntime(_config(tmp_path,tool_free=False)))
    try:
        c=service.orchestrator_runtime.config
        assert c.tools.enabled
        assert c.tools.allow_stateful_tools is True
        assert c.tools.allow_side_effect_tools is True
        assert service.orchestrator_runtime.tools.enabled_tools(c)
    finally:
        service.close()
