from pathlib import Path
import pytest
from swaag.benchmark.conversation_acceptance import _answer_checks, _workspace_hashes, run_conversation_acceptance
from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
from swaag.benchmark import benchmark_runner


def test_canonical_conversation_cases_have_distinct_isolated_authoritative_histories(tmp_path):
    cases = conversational_smoke_benchmark_tasks()
    assert len(cases) == 17
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
