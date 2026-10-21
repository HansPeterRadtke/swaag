"""Fast front-end sizes and freshness are documented settings, not magic numbers."""
from __future__ import annotations

from datetime import datetime,timedelta,timezone
from swaag.config import load_config
from swaag.foreground_fast import generate_fast_foreground_response
from swaag.types import Message
from tests.test_fast_foreground_reply import setup,fast
import pytest


@pytest.mark.parametrize('parameter',[
 'fast_max_input_tokens', 'fast_max_user_message_chars',
 'fast_recent_message_count', 'fast_recent_history_chars',
 'fast_snapshot_max_age_seconds','fast_min_output_tokens',
 'fast_desired_output_tokens', 'fast_notes_max_chars','fast_policy_max_chars',
])
@pytest.mark.parametrize('bad',[0,-1])
def test_each_fast_limit_rejects_nonpositive_domain(parameter,bad):
    with pytest.raises(ValueError,match='must be positive'):
        load_config(env={f'SWAAG__COMMUNICATION__{parameter.upper()}':str(bad)})


def test_output_minimum_cannot_exceed_desired_budget():
    with pytest.raises(ValueError,match='at least fast_min_output_tokens'):
        load_config(env={'SWAAG__COMMUNICATION__FAST_MIN_OUTPUT_TOKENS':'180',
                         'SWAAG__COMMUNICATION__FAST_DESIRED_OUTPUT_TOKENS':'144'})


def test_max_user_message_limit_preserves_long_original_for_full_controller(make_config):
    service,runtime,client=setup(make_config,[])
    try:
        runtime.config.communication.fast_max_user_message_chars=10
        state=service._orchestrator_state()
        assert generate_fast_foreground_response(runtime,state,'This is a longer complete original user request.') is None
        assert not client.requests
    finally:service.close()


def test_recent_history_bounds_are_independent_of_message_authority(make_config):
    service,runtime,client=setup(make_config,[fast('answer','Hello.')])
    try:
        cfg=runtime.config.communication
        cfg.fast_recent_message_count=2
        cfg.fast_recent_history_chars=25
        state=service._orchestrator_state()
        for item in ('Older exact 1.','Older exact 2.','Small.','Latest quoted instruction: STOP'):
            state.messages.append(Message(role='user',content=item,created_at='earlier'))
        response=generate_fast_foreground_response(runtime,state,'Who are you?')
        assert response is not None
        prompt=client.requests[0]['prompt']
        assert 'Latest quoted instruction: STOP' not in prompt
        assert 'Small.' not in prompt  # whole-message selection cannot jump over the latest oversized row
        assert 'Final authoritative current user message, unchanged:\nWho are you?' in prompt
        assert 'older_conversation_messages_available_only_to_full_orchestrator' in prompt
    finally:service.close()


def test_snapshot_age_tolerance_is_runtime_configurable(make_config):
    service,runtime,client=setup(make_config,[fast('answer','Hi.')])
    try:
        cfg=runtime.config.communication
        cfg.fast_snapshot_max_age_seconds=0.25
        state=service._orchestrator_state()
        snapshot={'captured_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),
                  'active_workers':[{'worker_id':'private-worker-id','objective':'private-work'}],
                  'active_plans':[], 'terminal_workers':[]}
        result=generate_fast_foreground_response(runtime,state,'Hi',observed_world=snapshot)
        assert result is not None
        assert 'snapshot is stale or invalid' in client.requests[0]['prompt']
        assert 'private-worker-id' not in client.requests[0]['prompt']
    finally:service.close()


def test_parameter_metadata_includes_semantic_units_and_risks():
    metadata=load_config().parameter_metadata()
    for key in ('fast_max_user_message_chars','fast_recent_message_count',
                'fast_recent_history_chars','fast_snapshot_max_age_seconds',
                'fast_min_output_tokens','fast_desired_output_tokens','fast_max_input_tokens'):
        entry=metadata['communication.'+key]
        assert entry['valid_range'] in {'integer > 0','number > 0'}
        assert entry['description'] and entry['consequences']
        assert entry['source'] and entry['group']=='communication'
