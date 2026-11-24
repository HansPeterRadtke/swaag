"""Safe source pin and one-use admission tests for Jetson's disk race harness.

These tests have no network/audio/server/model effects, and never duplicate
an existing real Jetson one-use run. Real thread/fsync interleaving is proven
by the separate immutable Jetson result inspected by the source oracle.
"""
from __future__ import annotations
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
SPEC=importlib.util.spec_from_file_location('oneuse_real_private_atomic_draft_disk',ROOT/'scripts/demos/verify_isolated_voice_atomic_storage_race_once.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def dummy_files(tmp_path,monkeypatch):
    source=tmp_path/'fake-source';source.mkdir()
    raw={name:(name+' :: fixture source only').encode() for name in M.BASE}
    for name,body in raw.items(): (source/name).write_bytes(body)
    monkeypatch.setattr(M,'BASE',{name:M.digest(body) for name,body in raw.items()})
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',M.digest(raw['cloudflare_ws_voice_server.py']))
    config=tmp_path/'private-template.json';config.write_text('{}')
    return source,config


def test_actual_source_must_match_frozen_w69_sha_and_is_not_deployment_source():
    assert M.PATCHED_SERVER_SHA256=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    assert M.BASE['cloudflare_ws_voice_server.py']=='e0939fd23aefd11f6d3d8bfd0158508b7b2f0a4f75fb70467e67bced5d06f030'
    text=(ROOT/'scripts/demos/verify_isolated_voice_atomic_storage_race_once.py').read_text()
    for phrase in ('threading.Event','threading.Thread','core().atomic_write_json','old_write_accepted','superseded_by_intent_epoch','turn-current','turn-independent','expected_intent_epoch=epoch_a','_expected_intent_epoch=epoch_a','live_services_changed'):
        assert phrase in text
    assert 'systemctl restart' not in text and 'hf_hub_download' not in text


def test_source_only_preflight_no_evidence_directory_created(tmp_path,monkeypatch,capsys):
    src,template=dummy_files(tmp_path,monkeypatch)
    dest=tmp_path/'new-private-evidence'
    args=['--source',str(src),'--template',str(template),'--evidence',str(dest)]
    assert M.main(args)==0
    d=json.loads(capsys.readouterr().out)
    assert d['status']=='actual_source_storage_race_preflight_only'
    assert d['no_source_process_or_result_created']
    assert not dest.exists()


def test_unpinned_source_bytes_refused_before_any_run(tmp_path,monkeypatch):
    src,config=dummy_files(tmp_path,monkeypatch)
    (src/'voice_ws_config.py').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Not exact isolated patched module'):
        M.main(['--source',str(src),'--template',str(config),'--evidence',str(tmp_path/'result')])
    assert not (tmp_path/'result').exists()


def test_source_dir_symlink_refused(tmp_path,monkeypatch):
    src,config=dummy_files(tmp_path,monkeypatch)
    link=tmp_path/'link';link.symlink_to(src,target_is_directory=True)
    with pytest.raises(ValueError,match='Missing/symlinked'):
        M.main(['--source',str(link),'--template',str(config),'--evidence',str(tmp_path/'result')])


def test_only_explicit_optin_can_start_oneuse_real_thread_test(tmp_path,monkeypatch):
    src,config=dummy_files(tmp_path,monkeypatch)
    dest=tmp_path/'result'
    monkeypatch.delenv(M.OPTIN,raising=False)
    with pytest.raises(ValueError,match='opt-in'):
        M.main(['--source',str(src),'--template',str(config),'--evidence',str(dest),'--run'])
    assert not dest.exists()


def test_uncertain_or_completed_oneuse_result_must_never_be_replayed(tmp_path,monkeypatch):
    src,config=dummy_files(tmp_path,monkeypatch)
    dest=tmp_path/'already-started';dest.mkdir()
    (dest/'attempt.json').write_text(json.dumps({'attempted':True}))
    monkeypatch.setenv(M.OPTIN,'1')
    with pytest.raises(FileExistsError,match='must not be replayed'):
        M.main(['--source',str(src),'--template',str(config),'--evidence',str(dest),'--run'])
    assert json.loads((dest/'attempt.json').read_text())=={'attempted':True}


def test_completed_offline_mocked_result_cannot_be_replaced(tmp_path,monkeypatch,capsys):
    src,config=dummy_files(tmp_path,monkeypatch)
    dest=tmp_path/'oneuse-test-fixture'
    monkeypatch.setenv(M.OPTIN,'1')
    calls=[]
    def fake_execute(*args,**kwargs):
        calls.append('pure synthetic test case only')
        return {'status':'genuine_atomic_disk_commit_vs_new_manual_intent_race_passed',
           'one_use_wall_seconds':0.123,'durable_A_statuses':['superseded','draft'],
           'durable_B_turn_count':1,'other_conversation_elapsed_seconds':0.001}
    monkeypatch.setattr(M,'execute',fake_execute)
    args=['--source',str(src),'--template',str(config),'--evidence',str(dest),'--run']
    assert M.main(args)==0
    receipt=dest/'result.json';assert receipt.is_file()
    original=receipt.read_bytes();assert len(calls)==1
    assert json.loads((dest/'attempt.json').read_text())['one_use_non_network_real_core_storage'] is True
    assert receipt.stat().st_mode & 0o077==0
    with pytest.raises(FileExistsError):M.main(args)
    assert len(calls)==1 and receipt.read_bytes()==original


def test_failed_offline_stub_preserves_unknown_outcome_and_refuses_reuse(tmp_path,monkeypatch,capsys):
    src,config=dummy_files(tmp_path,monkeypatch)
    dest=tmp_path/'private-failed-attempt'
    monkeypatch.setenv(M.OPTIN,'1')
    def failure(*args,**kwargs):raise TimeoutError('synthetic disk stall')
    monkeypatch.setattr(M,'execute',failure)
    args=['--source',str(src),'--template',str(config),'--evidence',str(dest),'--run']
    assert M.main(args)==3
    outcome=json.loads((dest/'result.json').read_text())
    assert outcome['status']=='failed_or_unknown' and outcome['failure_type']=='TimeoutError'
    with pytest.raises(FileExistsError):M.main(args)
