"""One-use local SWAAG developer-run guard tests: no model contact."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import hashlib
import pytest

ROOT=Path(__file__).resolve().parents[1]
RUN_SCRIPT=ROOT/'scripts/demos/run_fresh_reconnect_provenance_agent_once.py'
PREP_SCRIPT=ROOT/'scripts/demos/prepare_fresh_reconnect_provenance_agent_demo.py'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

RUN=load('wave66_one_use_noaction_model_runner',RUN_SCRIPT)
PREP=load('wave66_source_only_preparer_for_runner',PREP_SCRIPT)


def test_published_coder_and_exact_readonly_tool_contract():
    assert RUN.EXPECTED_ALIAS==PREP.MODEL_ALIAS
    assert RUN.EXPECTED_MODEL_URL==PREP.MODEL_BASE+'/v1/models'
    assert RUN.EXPECTED_FILES==tuple(PREP.ORIGINAL_SOURCES)
    assert RUN.EXPECTED_MISSING==PREP.BAD_PATH
    assert 'NOACTION' in RUN.OPTIN


def test_private_source_preflight_without_model_or_durable_turn(tmp_path):
    root=tmp_path/'private'
    PREP.prepare(ROOT,root)
    result=RUN.inspect_preflight(ROOT,root)
    assert result['tool_access']==PREP.TOOLS
    assert result['model']==PREP.MODEL_ALIAS
    assert result['source_revision']==PREP.subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert result['cfg_sha256']==RUN.digest(root/'read-only-config.toml')
    assert result['task_sha256']==RUN.digest(root/'fixed-new-reconnect-task.txt')
    assert not (root/'.one_use_started.json').exists()
    assert not list((root/'sessions').iterdir())
    assert not (root/'result.json').exists()


def test_marker_exclusively_claims_and_refuses_a_second_model_run(tmp_path):
    root=tmp_path/'private'
    PREP.prepare(ROOT,root)
    RUN.inspect_preflight(ROOT,root)
    marked=root/'.one_use_started.json'
    RUN.exclusive_json(marked,{'model_started':True,'any_prior_uncertain_effect_no_replay':True})
    with pytest.raises(ValueError,match='Prior one-use session activity'):
        RUN.inspect_preflight(ROOT,root)
    with pytest.raises(FileExistsError):
        RUN.exclusive_json(marked,{'overwritten':True})
    assert json.loads(marked.read_text())=={
        'model_started':True,'any_prior_uncertain_effect_no_replay':True}
    assert marked.stat().st_mode & 0o777==0o600


def test_forged_or_tampered_private_model_configuration_denied(tmp_path):
    root=tmp_path/'private'
    PREP.prepare(ROOT,root)
    config=root/'read-only-config.toml'
    config.write_text(config.read_text().replace('allow_side_effect_tools = false','allow_side_effect_tools = true'))
    with pytest.raises(ValueError,match='Frozen one-use task'):
        RUN.inspect_preflight(ROOT,root)
    assert not (root/'.one_use_started.json').exists()


def test_model_run_cannot_be_admitted_if_session_history_already_exists(tmp_path):
    root=tmp_path/'private'
    PREP.prepare(ROOT,root)
    logs=root/'sessions'/'session_fake'
    logs.mkdir()
    (logs/'complete_history.jsonl').write_text(json.dumps({
        'type':'model_request_sent','payload':{'kind':'action'}})+'\n')
    with pytest.raises(ValueError,match='Prior one-use session activity'):
        RUN.inspect_preflight(ROOT,root)
    assert not (root/'.one_use_started.json').exists()


def test_real_event_scanner_preserves_missing_file_failure_count_and_successes(tmp_path):
    root=tmp_path/'only-owned-synthetic'
    (root/'sessions'/'session_fake').mkdir(parents=True)
    p=root/'sessions'/'session_fake'/'complete_history.jsonl'
    rows=[]
    for n in range(3):
        rows += [{'type':'tool_called','payload':{'tool_name':'read_file','tool_input':{'path':RUN.EXPECTED_MISSING}}},
                 {'type':'tool_error','payload':{'tool_name':'read_file','error_type':'FilesystemError','tool_input':{'path':RUN.EXPECTED_MISSING}}}]
    rows += [{'type':'tool_called','payload':{'tool_name':'list_files','tool_input':{'path':'tests'}}},
             {'type':'tool_result','payload':{'tool_name':'list_files','validated_input':{'path':'tests'}}},
             {'type':'tool_invocation_failed_recoverably','payload':{'tool_name':'read_file'}},
             {'type':'model_request_sent','payload':{}},
             {'type':'model_response_received','payload':{}},
             {'type':'turn_finished','payload':{}}]
    p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    found=RUN.scan_journal(root)
    assert found['repeated_identical_missing_file_errors']==3
    assert found['real_tool_calls']==4
    assert found['actual_successful_tool_results']==1
    assert found['real_tool_errors']==3
    assert found['real_model_request_count']==1
    assert found['real_model_response_count']==1
    assert found['terminal_turn_events']==1
    assert found['recoverable_error_feedback_events']==1


def test_unknown_external_model_session_is_rejected_even_without_real_model_call(tmp_path,monkeypatch):
    root=tmp_path/'private';PREP.prepare(ROOT,root)
    # No opt-in: the --run path is blocked before a request or marker.
    monkeypatch.delenv(RUN.OPTIN,raising=False)
    with pytest.raises(ValueError,match='opt-in'):
        RUN.main(['--repo',str(ROOT),'--evidence',str(root),'--run'])
    assert not (root/'.one_use_started.json').exists()
    assert not list((root/'sessions').iterdir())


def test_secondary_session_directory_is_not_conflated_with_one_true_run(tmp_path):
    root=tmp_path/'private';(root/'sessions').mkdir(parents=True)
    for name in ('session_foo','session_bar'):
        (root/'sessions'/name).mkdir()
        (root/'sessions'/name/'complete_history.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='Mixed multiple sessions'):
        RUN.scan_journal(root)
