"""No-network no-phone source-pin and asyncio fairness source contract.

These tests contain only synthetic Python strings. Real Jetson process
receipts and original voice-audio files are verified separately and retained
outside this repository. A test fixture is never real model/source evidence.
"""
from __future__ import annotations
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/demos/prepare_isolated_voice_persisted_epoch_async_guard.py'
SPEC=importlib.util.spec_from_file_location('w70_exact_recovered_voice_source',P)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

PREFIX='''def advance_conversation_intent_epoch(conversation: str, *,
                                      supersede_prior_speech: bool = False) -> int:
    with conversation_intent_lock(conversation):
        revision = CONVERSATION_INTENT_EPOCH.get(conversation, 0) + 1
        CONVERSATION_INTENT_EPOCH[conversation] = revision
'''
ANCHOR='''                            and type(epoch) is int and epoch < revision
                            and not turn.get("submitted_text")):
'''

def fake_synthetic_code():
    code=PREFIX+'''        if supersede_prior_speech:
            if (True
'''+ANCHOR+'''                pass
        return revision
async def harmless_source_test_only(self):
'''
    code+='''    self.intent_epoch = advance_conversation_intent_epoch(self.conversation)
'''*6
    return code.encode()


def fake_private_modules(tmp_path,monkeypatch):
    root=tmp_path/'private-synthetic-input';root.mkdir()
    data={n:(fake_synthetic_code() if n=='cloudflare_ws_voice_server.py' else (n+'\n# only a fake test byte sequence\n').encode()) for n in M.BASE}
    for n,b in data.items():(root/n).write_bytes(b)
    monkeypatch.setattr(M,'BASE',{n:hashlib.sha256(b).hexdigest() for n,b in data.items()})
    newer=M.patch(data['cloudflare_ws_voice_server.py'])
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',hashlib.sha256(newer).hexdigest())
    return root,data,newer


def test_literal_real_frozen_original_and_new_sha_not_confused():
    assert len(M.BASE)==7
    assert M.BASE['cloudflare_ws_voice_server.py']=='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'
    assert M.PATCHED_SERVER_SHA256=='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'
    assert 'hf_hub_download' not in P.read_text()
    assert 'systemctl restart' not in P.read_text()


def test_synthetic_exact_source_contract_recovers_persisted_counter_and_offloads_six_async_handlers():
    out=M.patch(fake_synthetic_code()).decode()
    assert out.count('self.intent_epoch = await run_blocking(advance_conversation_intent_epoch, ')==6
    assert 'self.intent_epoch = advance_conversation_intent_epoch(' not in out
    assert 'latest = _recover_persisted_intent_epoch(conversation)' in out
    assert 'for key in ("asr_intent_epoch", "superseded_by_intent_epoch")' in out
    assert 'epoch is None or (type(epoch) is int and epoch < revision)' in out
    assert 'raise ValueError("stored speech revision malformed or unsupported")' in out
    ast.parse(out)


def test_unknown_voice_source_broken_or_anchor_missing_fails_closed():
    with pytest.raises(ValueError,match='not present'):
        M.patch(b'# unrelated source file')


def test_duplicate_or_already_partial_source_edit_refused():
    source=fake_synthetic_code().decode().replace(PREFIX,PREFIX+PREFIX)
    with pytest.raises(ValueError,match='not present'):
        M.patch(source.encode())
    source=fake_synthetic_code().decode().replace('self.intent_epoch = advance_conversation_intent_epoch(self.conversation)',
          'self.intent_epoch = await run_blocking(advance_conversation_intent_epoch, self.conversation)',1)
    with pytest.raises(ValueError,match='Expected six'):
        M.patch(source.encode())


def test_source_preflight_reads_all_supporting_files_without_writing(tmp_path,monkeypatch):
    src,files,_=fake_private_modules(tmp_path,monkeypatch)
    target=tmp_path/'output-should-not-be-created'
    d=M.render(src,target,run=False)
    assert d['only_changed_module']=='cloudflare_ws_voice_server.py'
    assert d['six_source_async_handlers_offload_mutex_and_fs_sync'] is True
    assert d['per_conversation_durable_draft_compare_and_write_remains_atomic'] is True
    assert d['owner_phone_account_production_or_tool_effects']==0
    assert not target.exists()
    assert {p.name:p.read_bytes() for p in src.iterdir()}==files


def test_changed_original_source_bytes_refused_even_in_docstring(tmp_path,monkeypatch):
    src,files,_=fake_private_modules(tmp_path,monkeypatch)
    p=src/'voice_ws_core.py';p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='content altered'):
        M.render(src,tmp_path/'output',run=True)
    assert not (tmp_path/'output').exists()


def test_any_symlinked_original_path_or_module_denied(tmp_path,monkeypatch):
    src,files,_=fake_private_modules(tmp_path,monkeypatch)
    bad=tmp_path/'alias';bad.symlink_to(src,target_is_directory=True)
    with pytest.raises(ValueError,match='private W69 source root'):
        M.validate(bad)
    f=src/'voice_ws_config.py';f.unlink();f.symlink_to(tmp_path/'dummy')
    with pytest.raises(ValueError,match='Untrusted original module'):
        M.validate(src)


def test_wrong_frozen_output_sha_must_not_create_candidate(tmp_path,monkeypatch):
    src,files,_=fake_private_modules(tmp_path,monkeypatch)
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256','0'*64)
    with pytest.raises(ValueError,match='reviewed source SHA changed'):
        M.render(src,tmp_path/'out',run=True)
    assert not (tmp_path/'out').exists()


def test_new_private_source_copy_is_single_owned_module_only(tmp_path,monkeypatch):
    src,files,new=fake_private_modules(tmp_path,monkeypatch)
    out=tmp_path/'distinct-private-copy'
    r=M.render(src,out,run=True)
    assert out.is_dir() and out.stat().st_mode&0o077==0
    for name,body in files.items():
        candidate=out/name
        assert candidate.read_bytes()==(new if name=='cloudflare_ws_voice_server.py' else body)
        assert candidate.stat().st_mode&0o077==0
    assert json.loads((out/'source-manifest.json').read_text())['new_server_sha256']==M.PATCHED_SERVER_SHA256
    assert r['new_server_sha256']==M.PATCHED_SERVER_SHA256
    assert {p.name:p.read_bytes() for p in src.iterdir()}==files


def test_private_only_path_rejects_existing_or_production_location(tmp_path,monkeypatch):
    src,_,_=fake_private_modules(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='can only create one new private'):
        M.render(src,Path('/etc/voice-agent/unsafe'),run=True)
    with pytest.raises(ValueError,match='can only create one new private'):
        M.render(src,src,run=True)
    p=tmp_path/'already-created';p.mkdir()
    with pytest.raises(ValueError,match='can only create one new private'):
        M.render(src,p,run=True)


def test_fake_historical_epoch_type_and_range_guards_are_part_of_source():
    s=M.patch(fake_synthetic_code()).decode()
    assert 'type(number) is not int or not 0 <= number <= 1_000_000_000' in s
    assert 'if latest >= 1_000_000_000:' in s
    assert 'raise ValueError("intent revision limit reached")' in s
    assert 'with conversation_intent_lock(conversation):' in s
