"""No-effect one-use admission for a private source-pinned restart test.

All mocked outputs in these tests are synthetic. The actual two unique
Jetson source-process restarts and their independent disk evidence are
separately preserved outside Git; never confuse a stub with real execution.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('oneuse_atomic_voice_restart',ROOT/'scripts/demos/verify_isolated_voice_atomic_draft_restart_once.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def synthetic_source_and_history(tmp_path,monkeypatch):
    src=tmp_path/'fake-server';src.mkdir()
    contents={n:(n+' synthetic source\n').encode() for n in M.BASE}
    for n,v in contents.items():(src/n).write_bytes(v)
    monkeypatch.setattr(M,'BASE',{n:M.sha(src/n) for n in contents})
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',M.sha(src/'cloudflare_ws_voice_server.py'))
    row=[{'status':'superseded','turn_id':'turn-old','asr_intent_epoch':1,'recognized_text':'old','submitted_text':'','assistant':''},
         {'status':'draft','turn_id':'turn-current','asr_intent_epoch':2,'recognized_text':'new','submitted_text':'','assistant':''}]
    original=tmp_path/'original.json';original.write_text(json.dumps({'dialogue_history':row}))
    monkeypatch.setattr(M,'ORIGINAL_DISK_SHA',M.sha(original))
    template=tmp_path/'public-config-test.json';template.write_text('{}')
    return src,original,template


def args_for(src,original,template,out):
    return ['--source',str(src),'--original-history',str(original),'--template',str(template),'--evidence',str(out)]


def test_exact_real_source_fingerprint_is_frozen():
    assert M.PATCHED_SERVER_SHA256=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    assert M.ORIGINAL_DISK_SHA=='bdde160a4996a1f38e8a88366bf2bb63e6ad3a65b2ebd26687326a03158102fc'
    assert M.SOURCE_CID=='private-w69-storage-race-A'


def test_preflight_authenticates_exact_pinned_both_history_rows(tmp_path,monkeypatch,capsys):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    out=tmp_path/'no-dir-created'
    assert M.main(args_for(src,original,template,out))==0
    d=json.loads(capsys.readouterr().out)
    assert d['status']=='private_durable_history_restart_preflight_no_network'
    assert d['distinct_test_server_child_processes_not_started']==2
    assert not out.exists()


def test_source_tamper_denied_even_if_history_original(tmp_path,monkeypatch):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    (src/'voice_ws_core.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source bytes differ'):
        M.main(args_for(src,original,template,tmp_path/'out'))


def test_history_reordering_or_unsubmitted_status_tamper_denied(tmp_path,monkeypatch):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    d=json.loads(original.read_text());d['dialogue_history'].reverse();original.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Original forced-race durable history changed'):
        M.main(args_for(src,original,template,tmp_path/'out'))


def test_source_or_history_symlinks_denied(tmp_path,monkeypatch):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    link=tmp_path/'symlink-original';link.symlink_to(original)
    with pytest.raises(ValueError,match='missing/symlinked'):
        M.safe_source(src,link,template)


def test_run_requires_explicit_optin_and_no_outcome_created_before_optin(tmp_path,monkeypatch):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    out=tmp_path/'untouched'
    monkeypatch.delenv(M.OPT_IN,raising=False)
    with pytest.raises(ValueError,match='opt-in'):
        M.main(args_for(src,original,template,out)+['--run'])
    assert not out.exists()


def test_existing_or_uncertain_original_result_refused_before_child_start(tmp_path,monkeypatch):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    out=tmp_path/'preexisting-attempt';out.mkdir();(out/'attempt.json').write_text('{"state":"unknown"}')
    monkeypatch.setenv(M.OPT_IN,'1')
    with pytest.raises(FileExistsError,match='Never repeat previous'):
        M.main(args_for(src,original,template,out)+['--run'])
    assert json.loads((out/'attempt.json').read_text())['state']=='unknown'


def test_mocked_pure_output_is_not_restart_evidence_but_is_oneuse(tmp_path,monkeypatch,capsys):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    out=tmp_path/'stubbed-test-only';monkeypatch.setenv(M.OPT_IN,'1')
    calls=[]
    def not_an_actual_server(source,original,hist,dest,mode):
        calls.append(mode)
        return {'mode':mode,'status':'source_pinned_real_server_restart_from_W69_durable_history_verified',
                'restored_active_review_count':1 if mode=='latest_new_draft' else 0,
                'historical_superseded_count':1,'private_owned_child_pid':999999}
    monkeypatch.setattr(M,'one_child',not_an_actual_server)
    args=args_for(src,original,template,out)+['--run']
    assert M.main(args)==0
    r=json.loads((out/'result.json').read_text());assert r['status']=='both_real_private_source_restart_draft_recovery_cases_passed'
    assert calls==['latest_new_draft','superseded_only']
    original_bytes=(out/'result.json').read_bytes()
    with pytest.raises(FileExistsError):M.main(args)
    assert (out/'result.json').read_bytes()==original_bytes


def test_failed_test_process_result_recorded_unknown_without_retry(tmp_path,monkeypatch):
    src,original,template=synthetic_source_and_history(tmp_path,monkeypatch)
    out=tmp_path/'failed-once';monkeypatch.setenv(M.OPT_IN,'1')
    def failing_child(*args):raise RuntimeError('synthetic server process failed')
    monkeypatch.setattr(M,'one_child',failing_child)
    args=args_for(src,original,template,out)+['--run']
    assert M.main(args)==3
    d=json.loads((out/'result.json').read_text());assert d['status']=='failed_or_unknown' and d['failure_type']=='RuntimeError'
    with pytest.raises(FileExistsError):M.main(args)
