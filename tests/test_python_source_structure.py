"""Generic source-backed, non-executing AST inspection must respect tool gates."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from swaag.environment.filesystem import FilesystemError
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry
from swaag.types import SessionState
from tests.test_tools import _empty_state


def _config(make_config, workspace: Path):
    config = make_config()
    config.tools.read_roots = [workspace]
    config.tools.enabled = ["python_source_structure", "read_file"]
    return config


def test_ast_tool_counts_real_top_level_synchronous_async_and_decorated_functions_without_executing(
    make_config, tmp_path,
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    marker = tmp_path / "MUST_NOT_BE_CREATED"
    source = f'''@decorator
async def test_async_case():\n    return 1

def test_first():
    return 1

class Container:
    def test_method(self):
        return 2

    class Inner:
        def test_nested(self):
            return 3

def ordinary():
    def test_nested_function():
        return 4
    return 5

open({str(marker)!r}, 'w').write('danger')
'''
    path = workspace / "module.txt"  # Parse contents, not file extensions.
    path.write_text(source)
    config = _config(make_config, workspace)
    invocation, result = ToolRegistry().dispatch(
        "python_source_structure", {"path": "module.txt"}, config, _empty_state()
    )
    assert invocation.validated_input == {"path": "module.txt"}
    assert not marker.exists()  # AST parsed only; no top-level code executed.
    assert result.tool_name == "python_source_structure"
    assert result.output["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert result.output["source_size_chars"] == len(source)
    assert result.output["counts"] == {
        "top_level_functions": 3,
        "top_level_test_functions": 2,
        "top_level_classes": 1,
    }
    assert [part["name"] for part in result.output["top_level_functions"]] == [
        "test_async_case", "test_first", "ordinary"
    ]
    assert [part["async"] for part in result.output["top_level_functions"]] == [True, False, False]
    assert [part["name"] for part in result.output["top_level_classes"]] == ["Container"]
    assert {event.event_type for event in result.generated_events} == {
        "file_read_requested", "filesystem_read"
    }
    assert result.generated_events[0].payload["reason"] == "python_source_structure"
    raw = next(event.payload for event in result.generated_events if event.event_type == "filesystem_read")
    assert raw["text"] == source


def test_python_ast_tool_rejects_unenabled_tool_and_disabled_stateful(make_config, tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def test_a(): pass\n")
    registry = ToolRegistry()
    config = _config(make_config, tmp_path)
    config.tools.enabled = ["read_file"]
    with pytest.raises(PermissionError, match="not enabled"):
        registry.dispatch("python_source_structure", {"path": "module.py"}, config, _empty_state())
    config.tools.enabled = ["python_source_structure"]
    config.tools.allow_stateful_tools = False
    with pytest.raises(PermissionError, match="disabled by policy"):
        registry.dispatch("python_source_structure", {"path": "module.py"}, config, _empty_state())


def test_python_ast_tool_respects_read_root_and_symlink_escape(make_config, tmp_path):
    root = tmp_path / "allowed";root.mkdir()
    outside = tmp_path / "outside.py";outside.write_text("def test_outside(): pass\n")
    (root / "escaped.py").symlink_to(outside)
    config = _config(make_config, root)
    registry = ToolRegistry()
    for path in [str(outside), "escaped.py"]:
        with pytest.raises((FilesystemError, PermissionError, ValueError, OSError)):
            registry.dispatch("python_source_structure", {"path": path}, config, _empty_state())


def test_python_ast_tool_rejects_invalid_syntax_and_oversized_source(make_config, tmp_path):
    cfg=_config(make_config,tmp_path)
    p=tmp_path/"invalid.py";p.write_text("def missing(\n")
    with pytest.raises(ToolValidationError, match="syntax error"):
        ToolRegistry().dispatch("python_source_structure",{"path":"invalid.py"},cfg,_empty_state())
    p.write_text("#" + "x" * 2_000_001)
    with pytest.raises(ToolValidationError, match="size limit"):
        ToolRegistry().dispatch("python_source_structure",{"path":"invalid.py"},cfg,_empty_state())


def test_python_ast_tool_uses_actual_current_source_counts_not_hardcoded_fixture(make_config, tmp_path):
    root=tmp_path/"sources";root.mkdir()
    counts=[2,5,1,7]
    config = _config(make_config,root)
    registry = ToolRegistry()
    for i,count in enumerate(counts):
        text="\n".join(f"def test_case_{j}(): pass" for j in range(count)) + "\n"
        path=root/f"case_{i}.py";path.write_text(text)
        _,result=registry.dispatch("python_source_structure",{"path":path.name},config,_empty_state())
        assert result.output["counts"]["top_level_test_functions"] == count
        assert result.output["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_python_source_structure_is_model_discoverable_without_losing_tool_gates(make_config):
    from swaag.config import load_config
    from swaag.tools.base import _validate_schema_value
    cfg = make_config()
    registry = ToolRegistry()
    assert "python_source_structure" in registry.registered_names()
    assert "python_source_structure" not in {item[0] for item in registry.staged_prompt_tuples(cfg, [])}
    assert "python_source_structure" in {item[0] for item in registry.staged_prompt_tuples(
        cfg, ["python_source_structure"]
    )}
    names = {item[0] for item in registry.capability_index(cfg)}
    assert "python_source_structure" in names
    tool = registry.get("python_source_structure")
    assert tool.kind == "stateful"
    assert "without executing" in tool.description
    assert tool.output_schema["additionalProperties"] is False
    assert tool.input_schema["additionalProperties"] is False
    _validate_schema_value({"path": "source.py"}, tool.input_schema, path="test")
    with pytest.raises(ToolValidationError):
        _validate_schema_value({"path": "source.py", "execute": True}, tool.input_schema, path="test")
