from dataclasses import asdict

import pytest

from swaag.runtime import AgentRuntime
from swaag.types import Message


@pytest.mark.parametrize('builder', ['standard', 'lean', 'capability', 'terminal'])
def test_question_resolution_can_cite_visible_user_message_sources(make_config, builder):
    runtime = AgentRuntime(make_config(), model_client=object())
    state = runtime.create_or_load_session()
    references = []
    for text in ['Wait for approval.', 'I approve the local publication.']:
        event = runtime.history.record_event(state, 'message_added', {
            'message': asdict(Message(role='user', content=text, created_at='now')),
        })
        references.append(event.sequence)
    messages = runtime.history.read_authoritative_messages(state.session_id)
    kwargs = dict(original_request='Wait for approval.', pending_user_messages=[])
    if builder in {'standard', 'lean'}:
        assembly = runtime.prompts.build_agent_action_prompt(messages, [], prompt_mode=builder, **kwargs)
    elif builder == 'capability':
        assembly = runtime.prompts.build_agent_capability_selection_prompt(messages, capability_index=[], **kwargs)
    else:
        assembly = runtime.prompts.build_agent_terminal_response_prompt(messages, **kwargs)
    for sequence in references:
        assert f'[USER MESSAGE SOURCE sequence={sequence}]' in assembly.prompt_text
    assert 'I approve the local publication.' in assembly.prompt_text
    assert 'Wait for approval.' in assembly.prompt_text


@pytest.mark.parametrize('role, metadata', [
    ('user', {}),
    ('user', {'source_message_event_sequence': True}),
    ('tool', {'source_message_event_sequence': 153}),
])
def test_user_citation_is_not_invented_or_assigned_to_tool_output(make_config, role, metadata):
    runtime = AgentRuntime(make_config(), model_client=object())
    message = Message(role=role, content='Quoted approval.', created_at='now', metadata=metadata)
    assert runtime.prompts.user_message_provenance(message) == ''
