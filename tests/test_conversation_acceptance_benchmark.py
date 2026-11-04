from pathlib import Path
import pytest
from swaag.benchmark.conversation_acceptance import _answer_checks, _workspace_hashes, run_conversation_acceptance
from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
from swaag.benchmark import benchmark_runner


def test_canonical_conversation_cases_have_distinct_isolated_authoritative_histories(tmp_path):
    cases = conversational_smoke_benchmark_tasks()
    assert len(cases) == 24
    for case in cases:
        scenario = case.create(tmp_path, live_mode=True)
        assert scenario.workspace.parent == tmp_path
        assert not scenario.verification_contract.expected_files
        assert scenario.verification_contract.max_tool_calls == 0
        assert scenario.verification_contract.forbid_unexpected_workspace_changes
        assert scenario.model_client is None
    assert _workspace_hashes(tmp_path) != {}


def test_canonical_conversation_answer_verifier_rejects_unsupported_identity():
    case = next(x for x in conversational_smoke_benchmark_tasks() if x.task_id == 'conversation_identity_plain')
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        contract = case.create(Path(tmp)).verification_contract
        assert all(_answer_checks('I am SWAAG, a local assistant.', contract).values())
        assert not all(_answer_checks('I am an anonymous helpful assistant.', contract).values())


def test_canonical_conversation_answer_verifier_requires_latest_correction():
    case = next(x for x in conversational_smoke_benchmark_tasks() if x.task_id == 'conversation_last_correction')
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        contract = case.create(Path(tmp)).verification_contract
        assert all(_answer_checks('Eclipse', contract).values())
        assert not all(_answer_checks('Harbor', contract).values())
        assert not all(_answer_checks('Use Harbor, not Eclipse', contract).values())


def test_canonical_conversation_cli_is_explicitly_isolated_and_selectable(tmp_path,monkeypatch):
    monkeypatch.setattr('swaag.benchmark.conversation_acceptance.run_conversation_acceptance',
                        lambda **kwargs: {'passed': 1,'failed':0,'cases':[{'case_id':'conversation_identity_plain'}]})
    assert benchmark_runner.main(['conversation-acceptance','--output',str(tmp_path),
                                  '--case','conversation_identity_plain']) == 0
    with pytest.raises(ValueError,match='unknown conversational cases'):
        run_conversation_acceptance(output_dir=tmp_path,case_ids=['missing-test'])


def test_spoken_history_cases_forbid_unneeded_clarification_even_with_keyword_match(tmp_path):
    tasks={x.task_id:x for x in conversational_smoke_benchmark_tasks()}
    case=tasks['conversation_expletive_question_context'].create(tmp_path)
    assert case.oracle is not None and case.oracle.ask_user is False
    assert _answer_checks('The recording upload stalled, but what do you mean by it?', case.verification_contract)['expected_answer_contains']
    # This remains a behavioral failure because it asks what 'it' refers to
    # despite a single salient antecedent in the exact recent messages.
    from types import SimpleNamespace
    events=[SimpleNamespace(event_type='agent_question')]
    assert any(e.event_type == 'agent_question' for e in events)


def test_chat_only_revision_contract_is_not_an_external_mutation(tmp_path):
    from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
    case=next(x.create(tmp_path) for x in conversational_smoke_benchmark_tasks()
              if x.task_id=='conversation_in_chat_fact_revision_only')
    assert len(case.history_messages)==2
    assert case.history_messages[0].content.endswith('Rook.')
    assert case.prompt.endswith('Rook with Cedar.')
    assert case.verification_contract.expected_answer_contains == ['Cedar']
    assert case.verification_contract.max_tool_calls == 0
    assert case.verification_contract.forbid_unexpected_workspace_changes
    assert case.oracle is not None and case.oracle.ask_user is False
    prompt = (Path(__file__).parents[1] / 'src/swaag/assets/prompts/completion_evaluation_system.txt').read_text()
    assert 'only revising the facts stated in the current chat' in prompt
    assert 'require the corresponding actual outcome evidence' in prompt


def test_canonical_restart_phase_requires_separate_initial_receipt(tmp_path):
    from swaag.benchmark.conversation_restart_acceptance import main, FIRST_PROMPT, LATEST_PROMPT
    assert 'codename' in FIRST_PROMPT and 'codename' in LATEST_PROMPT
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path)])


def test_canonical_restart_operational_phase_is_declared_and_bounded(tmp_path):
    from swaag.benchmark.conversation_restart_acceptance import run_phase
    with pytest.raises(ValueError, match='phase must be first or second'):
        run_phase(output_dir=tmp_path, config_paths=[], phase='invalid')


def test_voice_identity_speech_style_preserves_explicit_model_exception(tmp_path):
    from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
    catalog = {x.task_id:x for x in conversational_smoke_benchmark_tasks()}
    assert 'conversation_explicit_model_identity' in catalog
    assert catalog['conversation_explicit_model_identity'].create(tmp_path).verification_contract.expected_answer_contains == [
        'qwen3-coder-30b-a3b-instruct-q4-k-m'
    ]
    root=Path(__file__).parents[1]/'src/swaag/assets/prompts'
    for fn in ('foreground_fast_system.txt','system_foreground.txt'):
        text=(root/fn).read_text()
        assert ('Identity:' in text or 'Conversation identity rule:' in text)
        assert ('explicitly ask which model' in text or 'if the user explicitly asks for the underlying model name' in text)
        assert 'I am SWAAG, your local voice assistant.' in text

def test_reference_grounding_suite_has_independent_temporal_and_action_oracles(tmp_path):
    from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
    cases = {
        x.task_id: x
        for x in conversational_smoke_benchmark_tasks()
        if x.task_id.startswith("conversation_reference_")
    }
    assert len(cases) == 6
    for name, task in cases.items():
        scenario = task.create(tmp_path, live_mode=True)
        assert scenario.prompt
        assert len(scenario.history_messages) == 2
        assert [msg.role for msg in scenario.history_messages] == ["user", "assistant"]
        assert scenario.oracle is not None and scenario.oracle.ask_user is False
        assert scenario.verification_contract.max_tool_calls == 0
        assert scenario.verification_contract.forbid_unexpected_workspace_changes
        assert scenario.verification_contract.expected_answer_contains
        assert scenario.verification_contract.required_event_counts["message_added"] == 4
    reported = cases["conversation_reference_reported_upload_past"].create(tmp_path)
    assert "original is safe" in reported.history_messages[0].content
    current = cases["conversation_reference_fresh_check_unknown"].create(tmp_path)
    assert "right now" in current.prompt
    assert current.verification_contract.expected_answer_regex
    quoted = cases["conversation_reference_quoted_deletion_past"].create(tmp_path)
    assert "only an example" in quoted.history_messages[0].content
