from __future__ import annotations

from swaag.prompts import PromptBuilder
from swaag.tools.registry import ToolRegistry


def test_staged_tool_schemas_begin_with_only_loader(make_config):
    config = make_config(tools__staged_discovery=True)
    registry = ToolRegistry()
    specs = registry.staged_prompt_tuples(config, set())
    assert [item[0] for item in specs] == ["load_tools"]
    assert "read_file" in {name for name, _, _ in registry.capability_index(config)}


def test_staged_tool_schemas_load_only_selected_enabled_tools(make_config):
    config = make_config(tools__staged_discovery=True)
    registry = ToolRegistry()
    specs = registry.staged_prompt_tuples(config, {"read_file", "calculator"})
    names = [item[0] for item in specs]
    assert names[0] == "load_tools"
    assert set(names[1:]) == {"read_file", "calculator"}
    assert "shell_command" not in names


def test_action_prompt_has_compact_index_without_unloaded_schema(make_config):
    config = make_config(tools__staged_discovery=True)
    registry = ToolRegistry()
    builder = PromptBuilder(config)
    loaded = registry.staged_prompt_tuples(config, set())
    assembly = builder.build_agent_action_prompt(
        [],
        loaded,
        original_request="inspect a file",
        pending_user_messages=[],
        prompt_mode="standard",
        capability_index=registry.capability_index(config),
    )
    assert "read_file:" in assembly.prompt_text
    assert "Exact tool schemas available for this call:" in assembly.prompt_text
    loader_section = assembly.prompt_text.split("Exact tool schemas available for this call:", 1)[1]
    assert "- name: load_tools" in loader_section
    assert "- name: read_file" not in loader_section
