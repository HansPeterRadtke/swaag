"""Pure source admission and permission boundaries for undeployed W71 patch.

All fixtures are synthetic; real Jetson coroutine and WebSocket outcomes
are independently source/log/disk verified outside the public repository.
"""
from __future__ import annotations
import hashlib,importlib.util,json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/demos/prepare_isolated_voice_stale_socket_intent_owner_guard.py'
SPEC=importlib.util.spec_from_file_location('exact_w71_source_patcher',SOURCE)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def fake_source(tmp_path,monkeypatch):
    src=tmp_path/'only-synthetic-seven-modules';src.mkdir()
    contents={name:(name+'\nsynthetic audio source, NOT actual Jetson Python code\n').encode() for name in M.BASE}
    for name,data in contents.items():(src/name).write_bytes(data)
    monkeypatch.setattr(M,'BASE',{name:hashlib.sha256(body).hexdigest() for name,body in contents.items()})
    patched=contents['cloudflare_ws_voice_server.py']+b'FIXTURE PATCH ONLY'
    monkeypatch.setattr(M,'NEW_SHA',hashlib.sha256(patched).hexdigest())
    monkeypatch.setattr(M,'patch',lambda x:x+b'FIXTURE PATCH ONLY')
    return src,contents,patched


def test_public_source_pins_to_exact_private_baseline_and_new_fixed_V2():
    assert len(M.BASE)==7
    assert M.BASE['cloudflare_ws_voice_server.py']=='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
    assert M.NEW_SHA=='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
    body=SOURCE.read_text()
    assert 'expected_active_session' in body
    assert 'expected_active_session.superseded' in body
    assert 'expected_active_session.closed' in body
    assert 'active is not expected_active_session' in body
    assert 'if self.intent_epoch is None: return' in body
    assert 'existing = await run_blocking(get_turn, self.conversation, turn_id)' in body
    assert 'An ACK-lost idempotent retry' in body
    assert 'systemctl restart' not in body


def test_unrelated_source_content_never_patched_by_guess():
    with pytest.raises(ValueError,match='Ambiguous changed source anchor'):
        M.patch(b'# unrelated source would not have expected original anchors')


def test_full_byte_pinned_private_source_preflight_has_no_write(tmp_path,monkeypatch):
    src,raw,patched=fake_source(tmp_path,monkeypatch)
    dest=tmp_path/'not-created'
    result=M.render(src,dest,run=False)
    assert result['new_undeployed_server_sha256']==M.NEW_SHA
    assert result['only_changed_file']=='cloudflare_ws_voice_server.py'
    assert result['same_conversation_lock_for_source_session_ownership_and_advance'] is True
    assert result['live_phone_user_provider_or_tools_effects']==0
    assert not dest.exists()
    assert {p.name:p.read_bytes() for p in src.iterdir()}==raw


def test_changed_original_module_denied_before_writing(tmp_path,monkeypatch):
    src,raw,_=fake_source(tmp_path,monkeypatch)
    p=src/'voice_ws_config.py';p.write_bytes(p.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='Unknown original private source'):
        M.render(src,tmp_path/'out',run=True)
    assert not (tmp_path/'out').exists()


def test_original_input_directory_symlink_denied(tmp_path,monkeypatch):
    src,_,_=fake_source(tmp_path,monkeypatch)
    link=tmp_path/'elsewhere';link.symlink_to(src,target_is_directory=True)
    with pytest.raises(ValueError,match='exact private original source'):
        M.source_files(link)


def test_original_support_file_symlink_denied(tmp_path,monkeypatch):
    src,_,_=fake_source(tmp_path,monkeypatch)
    file=src/'voice_ws_core.py';copy=tmp_path/'old';copy.write_bytes(file.read_bytes());file.unlink();file.symlink_to(copy)
    with pytest.raises(ValueError,match='Unknown original private source'):
        M.source_files(src)


def test_modified_candidate_sha_refuses_new_output(tmp_path,monkeypatch):
    src,_,_=fake_source(tmp_path,monkeypatch)
    monkeypatch.setattr(M,'NEW_SHA','0'*64)
    with pytest.raises(ValueError,match='reviewed exact source hash'):
        M.render(src,tmp_path/'output',run=True)
    assert not (tmp_path/'output').exists()


def test_new_source_may_only_be_created_in_exclusive_private_dir(tmp_path,monkeypatch):
    src,_,_=fake_source(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError,match='unused private'):
        M.render(src,Path('/etc/voice-agent/never-install'),run=True)
    with pytest.raises(FileExistsError,match='unused private'):
        M.render(src,src,run=True)
    other=tmp_path/'already-started';other.mkdir()
    with pytest.raises(FileExistsError,match='unused private'):
        M.render(src,other,run=True)


def test_new_private_source_has_only_one_changed_module_and_sha_manifest(tmp_path,monkeypatch):
    src,raw,new=fake_source(tmp_path,monkeypatch)
    dest=tmp_path/'private-only-output'
    report=M.render(src,dest,run=True)
    assert dest.stat().st_mode & 0o077==0
    for name,old in raw.items():
        changed=(dest/name)
        assert changed.read_bytes()==(new if name=='cloudflare_ws_voice_server.py' else old)
        assert changed.stat().st_mode&0o077==0
    manifest=json.loads((dest/'source-manifest.json').read_text())
    assert manifest['new_undeployed_server_sha256']==M.NEW_SHA
    assert manifest['live_phone_user_provider_or_tools_effects']==0
    assert manifest['only_changed_file']=='cloudflare_ws_voice_server.py'
    assert report['new_undeployed_server_sha256']==M.NEW_SHA
    assert {p.name:p.read_bytes() for p in src.iterdir()}==raw
