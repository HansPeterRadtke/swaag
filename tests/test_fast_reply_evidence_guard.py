"""A short spoken reply is a claim, not proof that a device or tool acted."""
from __future__ import annotations

import pytest
from swaag.foreground_reply_evidence import normalize_fast_foreground_claim
from swaag.foreground_fast import generate_fast_foreground_response
from tests.test_fast_foreground_reply import setup,fast


@pytest.mark.parametrize('user,reply',[
 ('Can you hear me?','Yes, I can hear you.'),
 ('Can you hear me now?', 'I heard your voice loud and clear.'),
 ('Do you hear me?', 'Your microphone works perfectly.'),
])
def test_no_false_microphone_claim_for_text_only_chat(user,reply):
    mode,safe=normalize_fast_foreground_claim('answer',reply,user)
    assert mode=='answer'
    assert 'received' in safe.lower() and 'message' in safe.lower()
    assert 'can hear you' not in safe.lower()
    assert 'microphone works' not in safe.lower()


def test_quoted_audio_phrase_cannot_be_recast_as_measured_microphone_state():
    mode,safe=normalize_fast_foreground_claim('answer','I can hear you.',
                                              'Repeat exactly the quotation: "I can hear you."')
    assert mode=='delegate'  # preserve exactness via original deep controller
    assert 'heard' not in safe.lower()


@pytest.mark.parametrize('reply',[
    'Stopping all tasks as requested.',
    'Successfully deleted those files.',
    'I have started a new worker.',
    'The job was just submitted.',
    'I already ran the test suite.',
])
def test_delegated_first_speech_cannot_claim_an_effect_without_receipts(reply):
    mode,safe=normalize_fast_foreground_claim('delegate',reply,'Please perform the operation.')
    assert mode=='delegate'
    assert safe!=reply
    assert 'started' not in safe and 'deleted' not in safe


def test_genuine_future_intention_is_preserved():
    text='I will inspect the source files and check current progress.'
    assert normalize_fast_foreground_claim('delegate',text,'Inspect the files.') == ('delegate',text)


def test_impossibility_notices_about_unperformed_actions_need_full_controller():
    mode,safe=normalize_fast_foreground_claim('answer',
        'I am unable to check the status because the state is not supplied.',
        'Please check the current worker status.')
    assert mode=='delegate' and 'check' in safe.lower()


def test_mechanically_valid_exact_word_count_can_finish_without_second_llm():
    mode,reply=normalize_fast_foreground_claim('answer','one two three four five six seven',
        'Reply with exactly seven words: one two three four five six seven.')
    assert (mode,reply)==('answer','one two three four five six seven')
    mode,reply=normalize_fast_foreground_claim('answer','one two three',
        'Reply with exactly seven words: one two three four five six seven.')
    assert mode=='delegate'


def test_unrecognized_strict_word_count_fails_closed_without_guessing():
    mode,_=normalize_fast_foreground_claim('answer','four words for you',
        'Reply with exactly ninety-nine words: more words requested.')
    assert mode=='delegate'


def test_end_to_end_verified_fast_hearing_reply_remains_one_model_call(make_config):
    service,runtime,client=setup(make_config,[fast('answer','Yes, I can hear you.')])
    try:
        state=service._orchestrator_state()
        result=generate_fast_foreground_response(runtime,state,'Can you hear me?')
        assert result and result.mode=='answer' and 'received' in result.reply.lower()
        assert len(client.requests)==1
    finally:service.close()


@pytest.mark.parametrize('reply',[
    'Resetting the worker execution context.',
    'Deploying the fix now.',
    'Uploading the files.',
    'I cannot check CUDA memory directly. Please use your terminal.',
    'Please provide the repository path and files.',
    "I don't have access to the previous session's history.",
])
def test_unverified_delegated_claim_or_missing_access_does_not_shift_work_to_user(reply):
    mode,text=normalize_fast_foreground_claim('delegate',reply,
              'Please inspect my worker state and apply the change.')
    assert mode=='delegate'
    assert text != reply
    assert 'provide' not in text.lower() and 'resetting' not in text.lower()


def test_fast_system_description_explains_durable_receipts_for_simple_question(make_config):
    service,runtime,client=setup(make_config,[fast('answer','A control receipt records acceptance, not completion.')])
    try:
        state=service._orchestrator_state()
        r=generate_fast_foreground_response(runtime,state,'What is a tool receipt?')
        assert r and r.mode=='answer'
        prompt=client.requests[0]['prompt']
        assert 'accepted_control' in prompt and 'tool_result' in prompt
        assert 'verified_effect' in prompt
    finally:service.close()


def test_prompt_explicitly_distinguishes_new_observation_from_old_assistant_guess(make_config):
    from swaag.types import Message
    service,runtime,client=setup(make_config,[fast('answer','Current state shows the worker running.')])
    try:
        state=service._orchestrator_state()
        state.messages.append(Message(role='assistant',
             content='The worker is definitely done! All tasks completed!',
             created_at='older'))
        result=generate_fast_foreground_response(runtime,state,'Is the worker running?')
        assert result and result.mode=='answer'
        prompt=client.requests[0]['prompt']
        assert 'fresh observed state overrides unsupported prior assistant claims' in prompt
        assert 'The worker is definitely done' in prompt
    finally:service.close()


def test_currently_working_worker_cannot_be_reported_as_finished_without_evidence():
    live={'snapshot_complete':True,'active_worker_count':1,
          'workers':[{'id':'warehouse-worker','status':'working'}]}
    mode,text=normalize_fast_foreground_claim('answer',
        'The warehouse worker has finished its task and stopped.',
        'Is the warehouse worker still working?',current_world=live)
    assert mode=='delegate' and 'stopped' not in text.lower()
    mode,text=normalize_fast_foreground_claim('answer',
        'There are no active workers left.',
        'How many workers are active?',current_world=live)
    assert mode=='delegate'
    mode,text=normalize_fast_foreground_claim('answer',
        'The warehouse worker is currently working.',
        'Is the warehouse worker still working?',current_world=live)
    assert mode=='answer' and text=='The warehouse worker is currently working.'


def test_without_fresh_authoritative_worker_state_reply_is_not_misrepresented_as_verified():
    mode,text=normalize_fast_foreground_claim('answer',
        'The warehouse worker has stopped.', 'What does the history claim?')
    assert mode=='answer' and 'stopped' in text


def test_no_false_live_worker_if_fresh_inventory_contains_none():
    empty={'snapshot_complete':True,'active_worker_count':0,'workers':[]}
    mode,text=normalize_fast_foreground_claim('answer',
        'Your warehouse worker is currently working.',
        'Is a worker active now?',current_world=empty)
    assert mode=='delegate'
    mode,text=normalize_fast_foreground_claim('answer','There are no active workers.',
        'How many are active?',current_world=empty)
    assert mode=='answer'


def test_german_hearing_claim_is_bound_to_received_text_not_microphone():
    mode,text=normalize_fast_foreground_claim('answer','Ja, ich kann dich hören.',
        'Kannst du mich hören?')
    assert mode=='answer'
    assert 'angekommen' in text.lower()
    assert 'ich kann dich hören' not in text.lower()


def test_a_blocked_worker_cannot_be_reported_as_actively_computing():
    blocked={'snapshot_complete':True,'active_worker_count':1,
             'workers':[{'id':'worker-1','status':'input_required'}]}
    mode,_=normalize_fast_foreground_claim('answer',
        'Your warehouse worker is currently running.',
        'Is it working now?',current_world=blocked)
    assert mode=='delegate'
    mode,_=normalize_fast_foreground_claim('answer',
        'The worker is waiting for a required user answer.',
        'What is its status?',current_world=blocked)
    assert mode=='answer'
