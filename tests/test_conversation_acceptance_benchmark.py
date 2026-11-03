from pathlib import Path
import pytest
from swaag.benchmark.conversation_acceptance import _answer_checks, _workspace_hashes, run_conversation_acceptance
from swaag.benchmark.task_definitions import conversational_smoke_benchmark_tasks
from swaag.benchmark import benchmark_runner


def test_canonical_conversation_cases_have_distinct_isolated_authoritative_histories(tmp_path):
    cases = conversational_smoke_benchmark_tasks()
    assert len(cases) == 6
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
