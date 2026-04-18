from swaag.grammar import tool_input_contract
from swaag.tools.builtin import EditTextTool


def test_edit_text_tool_input_contract_requires_replacement_for_replace_pattern() -> None:
    contract = tool_input_contract("edit_text", EditTextTool.input_schema)
    schema = contract.json_schema
    assert schema is not None
    all_of = schema.get("allOf")
    assert isinstance(all_of, list)
    assert any(
        item.get("if", {}).get("properties", {}).get("operation", {}).get("enum") == ["replace_pattern_once", "replace_pattern_all"]
        and item.get("then", {}).get("required") == ["path", "operation", "pattern", "replacement"]
        for item in all_of
    )
