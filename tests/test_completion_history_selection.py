import json

import pytest

from swaag.runtime import AgentRuntime
from tests.test_agent_action_loop import FakeModelClient


@pytest.mark.parametrize('selected', [['E8'], []])
def test_selection_copies_only_model_selected_rows_exactly_and_keeps_source_access(make_config, selected):
    rows = [
        {'sequence': 8, 'role': 'user', 'content': '  Do not publish.\nKeep this exact.  '},
        {'sequence': 12, 'role': 'assistant', 'content': 'An unrelated old greeting.'},
        {'evidence_view': 'foreground_history_disclosure', 'notice': 'Fetch exact historical effects.'},
    ]
    original = json.loads(json.dumps(rows))
    def choose(payload):
        assert payload['contract'] == 'completion_history_selection'
        assert 'Do not publish.' in payload['prompt']
        assert 'An unrelated old greeting.' in payload['prompt']
        assert 'Select no rows when none adds needed evidence' in payload['prompt']
        return json.dumps({'selected_line_ids': selected})
    client = FakeModelClient([choose])
    runtime = AgentRuntime(make_config(model__context_limit=32000), model_client=client)
    state = runtime.create_or_load_session()
    result = runtime._select_foreground_completion_history(state, rows=rows,
        original_request='Summarize the report privately.', candidate='Here is the summary.', target_tokens=1000)
    assert result is not None
    decoded = json.loads(result[result.index('{'):])['ordered_history']
    assert decoded == ([[8, 'user', rows[0]['content']]] if selected else []) + [rows[2]]
    assert rows == original
    assert 'Unselected rows remain in the exact source catalogue' in result


def test_oversized_selection_declines_without_truncation(make_config):
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient([
        json.dumps({'selected_line_ids': ['E1']})]))
    state = runtime.create_or_load_session()
    rows = [{'sequence': 1, 'role': 'user', 'content': 'Do not publish. ' * 200}]
    result = runtime._select_foreground_completion_history(state, rows=rows,
        original_request='Continue.', candidate='Ready.', target_tokens=20)
    assert result is None
    assert rows[0]['content'] == 'Do not publish. ' * 200


def test_action_selection_preserves_current_authority_tool_receipts_and_unknown_metadata(make_config):
    from swaag.types import Message
    runtime = AgentRuntime(make_config(), model_client=FakeModelClient([
        json.dumps({'selected_line_ids': ['E1']})]))
    state = runtime.create_or_load_session()
    def message(role, sequence, text, **metadata):
        return Message(role=role, content=text, created_at='now', metadata={
            'source_message_event_sequence': sequence, **metadata})
    prior = message('user', 1, 'Do not publish.')
    irrelevant = message('assistant', 2, 'Hello from an older turn.')
    unknown = message('user', 3, 'Preserve extended authority.', future_authority=True)
    receipt = message('tool', 4, 'The earlier external effect failed.')
    current = message('user', 9, 'Continue checking privately.')
    correction = message('user', 11, 'Stop that check now.')
    messages = [prior, irrelevant, unknown, receipt, current, correction]
    selected = runtime._select_foreground_action_context(state, messages=messages,
        turn_sequence=10, original_request=current.content)
    assert selected == [prior, unknown, receipt, current, correction]
    assert messages == [prior, irrelevant, unknown, receipt, current, correction]


def test_capacity_splitting_examines_every_complete_row_and_unions_exact_selections(make_config, monkeypatch):
    from swaag.tools.base import SemanticCallContextOverflow
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    rows = [{'sequence': n, 'role': 'user', 'content': f'Complete constraint {n}. ' * n}
            for n in range(1, 9)]
    examined = []
    def select(_state, request):
        ids = request.contract.json_schema['properties']['selected_line_ids']['items']['enum']
        if len(ids) > 2:
            raise SemanticCallContextOverflow(None)
        examined.extend(ids)
        for key in ids:
            assert rows[int(key[1:]) - 1]['content'].strip() in request.components[-1].text
        return {'selected_line_ids': [key for key in ids if key in {'E1', 'E7'}]}
    monkeypatch.setattr(runtime, '_execute_tool_semantic_call', select)
    result = runtime._select_foreground_completion_history(state, rows=rows,
        original_request='Continue with applicable constraints.', candidate=None,
        target_tokens=0, return_rows=True)
    assert examined == [f'E{n}' for n in range(1, 9)]
    assert result == [rows[0], rows[6]]
