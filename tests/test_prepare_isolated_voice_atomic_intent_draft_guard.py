"""No-action source admission, isolation and storage-fence contract checks.

The synthetic snippets below are NOT the actual original voice service. Its
seven original/private files and real tests are independently SHA-checked
on Jetson, not included in Git or reproduced in these unit fixtures.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/prepare_isolated_voice_atomic_intent_draft_guard.py'
SPEC=importlib.util.spec_from_file_location('atomic_server_source_private_patch',SCRIPT)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fake_repo(tmp_path,monkeypatch):
    src=tmp_path/'private-source-input';src.mkdir()
    original={n:('SYNTHETIC-SOURCE-TEST-FIXTURE-'+n+'\n').encode() for n in M.BASE}
    for n,v in original.items():(src/n).write_bytes(v)
    monkeypatch.setattr(M,'BASE',{n:hashlib.sha256(v).hexdigest() for n,v in original.items()})
    raw=original['cloudflare_ws_voice_server.py']
    new=raw+b'W69-atomic-TEST-ONLY-PATCH'
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',hashlib.sha256(new).hexdigest())
    monkeypatch.setattr(M,'modify',lambda inp:inp+b'W69-atomic-TEST-ONLY-PATCH')
    return src,original,new


def test_declared_actual_source_hashes_and_single_module_change():
    assert len(M.BASE)==7
    assert M.BASE['cloudflare_ws_voice_server.py']=='e0939fd23aefd11f6d3d8bfd0158508b7b2f0a4f75fb70467e67bced5d06f030'
    assert M.PATCHED_SERVER_SHA256=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    body=SCRIPT.read_text()
    for marker in ('conversation_intent_lock','core().conversation_lock','supersede_prior_speech','asr_intent_epoch','expected_intent_epoch','_expected_intent_epoch','core().atomic_write_json','with conversation_intent_lock(conversation)'):
        assert marker in body


def test_source_patch_rejects_unknown_or_drifted_upstream_content():
    with pytest.raises(ValueError,match='Exact immutable source anchor'):
        M.modify(b'# unexpected service content')


def test_original_source_preflight_is_no_side_effect(tmp_path,monkeypatch):
    src,data,_=fake_repo(tmp_path,monkeypatch)
    before={n:p.read_bytes() for n,p in ((p.name,p) for p in src.iterdir())}
    result=M.render(src,tmp_path/'not-created',run=False)
    assert result['exact_baseline_modules']==7
    assert result['changed_only']==['cloudflare_ws_voice_server.py']
    assert result['owner_live_service_or_phone_effects']==0
    assert result['model_provider_requests']==0
    assert not (tmp_path/'not-created').exists()
    assert {p.name:p.read_bytes() for p in src.iterdir()}==before


def test_any_original_support_module_change_refuses_to_generate_candidate(tmp_path,monkeypatch):
    src,_,_=fake_repo(tmp_path,monkeypatch)
    (src/'voice_ws_core.py').write_bytes(b'changed source')
    with pytest.raises(ValueError,match='source drift'):
        M.render(src,tmp_path/'out',run=True)
    assert not (tmp_path/'out').exists()


def test_symbolic_source_directory_rejected(tmp_path,monkeypatch):
    src,_,_=fake_repo(tmp_path,monkeypatch)
    link=tmp_path/'link';link.symlink_to(src,target_is_directory=True)
    with pytest.raises(ValueError,match='exact private source'):
        M.validate(link)


def test_symlinked_individual_source_file_refused(tmp_path,monkeypatch):
    src,_,_=fake_repo(tmp_path,monkeypatch)
    f=src/'voice_log_privacy.py';target=tmp_path/'source-outside';target.write_bytes(f.read_bytes());f.unlink();f.symlink_to(target)
    with pytest.raises(ValueError,match='Missing/symlinked'):
        M.validate(src)


def test_patched_sha_guard_rejects_buggy_or_unreviewed_source(tmp_path,monkeypatch):
    src,_,_=fake_repo(tmp_path,monkeypatch)
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256','0'*64)
    with pytest.raises(ValueError,match='Frozen atomic-intent candidate'):
        M.render(src,tmp_path/'out',run=True)
    assert not (tmp_path/'out').exists()


def test_only_a_new_private_benchmark_root_may_receive_candidate(tmp_path,monkeypatch):
    src,_,_=fake_repo(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='New private output only'):
        M.render(src,Path('/etc')/'voice-agent'/'unacceptable',run=True)
    with pytest.raises(ValueError,match='New private output only'):
        M.render(src,src,run=True)
    assert len(list(src.iterdir()))==7


def test_new_private_dir_is_mode_private_and_source_unmodified(tmp_path,monkeypatch):
    src,old,new=fake_repo(tmp_path,monkeypatch)
    # Unit-test basetemp resides under /data/var/swaag-benchmarks.
    target=tmp_path/'different-new-private-root';assert not target.exists()
    report=M.render(src,target,run=True)
    assert target.is_dir()
    assert target.stat().st_mode & 0o077 == 0
    assert {p.name:p.read_bytes() for p in src.iterdir()}==old
    for n in old:
        output=(target/n).read_bytes();assert output==(new if n=='cloudflare_ws_voice_server.py' else old[n])
        assert (target/n).stat().st_mode & 0o077 == 0
    manifest=json.loads((target/'source-manifest.json').read_text())
    assert manifest['new_atomic_server_sha256']==M.PATCHED_SERVER_SHA256
    assert manifest['changed_only']==['cloudflare_ws_voice_server.py']
    assert report['new_atomic_server_sha256']==M.PATCHED_SERVER_SHA256


def test_prohibit_unintentional_overwrite_of_existing_candidate(tmp_path,monkeypatch):
    src,_,_=fake_repo(tmp_path,monkeypatch)
    target=tmp_path/'already-here';target.mkdir()
    with pytest.raises(ValueError,match='New private output only'):
        M.render(src,target,run=True)


def test_source_does_not_embed_private_owner_audio_or_api_tokens():
    content=SCRIPT.read_text()
    assert '/etc/voice-agent/ws.token' not in content
    assert '/data/var/voice_agent_live_runtime/recordings/text/projects' not in content
    assert 'systemctl restart' not in content
    assert 'hf_hub_download' not in content
    assert 'subprocess.Popen' not in content
