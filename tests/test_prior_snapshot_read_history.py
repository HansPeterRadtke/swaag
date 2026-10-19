import copy
import json
from dataclasses import replace

from swaag.foreground_history import foreground_action_history
from swaag.types import Message


def test_prior_snapshot_read_is_recoverable_without_replaying_the_entire_value():
    output = {'observed_at': 'then', 'pointer': '', 'snapshot_hash': 'exact-hash',
              'snapshot_sequence': 4, 'value': {'old_detail': 'HUGE_OLD_SNAPSHOT_' * 10000}}
    message = Message(role='tool', name='runtime_snapshot_lookup', created_at='then',
        content=json.dumps(output), metadata={'output': output, 'context_updates': {},
        'source_message_event_sequence': 9, 'source_event_sequence': 8,
        'source_event_type': 'tool_result', 'source_event_hash': 'tool-event-hash',
        'raw_input': {'snapshot_sequence': 4, 'pointer': ''},
        'validated_input': {'snapshot_sequence': 4, 'pointer': ''}})
    original = copy.deepcopy(message)
    view, notice = foreground_action_history([message], [], 10)
    assert 'HUGE_OLD_SNAPSHOT_' not in view[0].content
    assert view[0].metadata['output']['value']['exact_tool_result_event'] == 8
    assert view[0].metadata['source_event_hash'] == 'tool-event-hash'
    assert view[0].metadata['raw_input'] == message.metadata['raw_input']
    assert 'read-only snapshot' in notice and 'history_window' in notice
    assert message == original

    # Current-turn evidence, effects, unknown extensions and user authority
    # cannot be hidden by this typed historical-read optimization.
    variants = [replace(message, role='user'), replace(message, name='write_file'),
        replace(message, metadata={**message.metadata, 'source_message_event_sequence': 11}),
        replace(message, metadata={**message.metadata, 'context_updates': {'changed': True}}),
        replace(message, metadata={**message.metadata, 'output': {**output, 'new_field': 'retain'}})]
    for variant in variants:
        kept, _ = foreground_action_history([variant], [], 10)
        assert kept == [variant]
