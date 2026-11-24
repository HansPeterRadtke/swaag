"""Pure refusal/semantic tests for the *unreleased* Jetson epoch-guard patch.

No production server code, user voice data, model, network, or GPU operation.
The separate real one-use Jetson tests verify the actual SHA-pinned source.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/prepare_isolated_voice_asr_intent_epoch_guard.py'
SPEC=importlib.util.spec_from_file_location('sha_pinned_epoch_patch',SCRIPT)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

# A tiny synthetic anchor-only copy, NOT the real source file. The actual
# seven immutable source blobs are separately verified on Nitro and Jetson.
MINI_SOURCE='''ACTIVE_BY_CONVERSATION: dict[str, "VoiceSession"] = {}
        self.input_revision = 0
        self.manual_pcm = bytearray()
    async def cancel_response(self, reason: str) -> None:
        self.generation += 1
        expected_generation = self.generation
        expected_revision = self.draft_revision
        expected_input = self.input_revision
        def current():
            return (expected_generation == self.generation
                    and expected_revision == self.draft_revision
                    and not self.draft_claimed
        if kind == "draft_discard":
            self.draft_revision += 1
        if kind == "draft_claim":
            self.draft_revision += 1
            if value.get("draft_claimed") is True:
                self.draft_revision += 1
        elif kind == "submit":
            self.draft_revision += 1
                self.had_voice = True
                self.input_revision += 1
'''


def fake_repository(tmp_path,monkeypatch):
    src=tmp_path/'immutable_original_code';src.mkdir()
    original={'cloudflare_ws_voice_server.py':MINI_SOURCE.encode()}
    for k in M.BASELINE:
        if k not in original:original[k]=(k+'\n# fixture only\n').encode()
    for k,data in original.items(): (src/k).write_bytes(data)
    monkeypatch.setattr(M,'BASELINE',{k:hashlib.sha256(v).hexdigest() for k,v in original.items()})
    new=M.rewrite(original['cloudflare_ws_voice_server.py'])
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256',hashlib.sha256(new).hexdigest())
    return src,new


def test_original_and_patched_exact_golden_sha_are_declared():
    assert len(M.BASELINE)==7
    assert M.BASELINE['cloudflare_ws_voice_server.py']=='cbcc77fb36ce8cf72cc6442ec0f35d84526aec4868a27b810b2c38bde52bb302'
    assert M.PATCHED_SERVER_SHA256=='e0939fd23aefd11f6d3d8bfd0158508b7b2f0a4f75fb70467e67bced5d06f030'
    assert SCRIPT.read_text().count('CONVERSATION_INTENT_EPOCH')>2


def test_patch_uses_single_conversation_epoch_across_all_new_user_intents():
    code=M.rewrite(MINI_SOURCE.encode()).decode()
    assert code.count('CONVERSATION_INTENT_EPOCH: dict[str, int] = {}')==1
    assert 'self.intent_epoch = CONVERSATION_INTENT_EPOCH.get(conversation, 0)' in code
    assert 'expected_intent_epoch = self.intent_epoch' in code
    assert 'expected_intent_epoch == CONVERSATION_INTENT_EPOCH.get(self.conversation, 0)' in code
    assert code.count('advance_conversation_intent_epoch(self.conversation)')>=5
    assert 'A reconnect' in code and 'alone does not advance this epoch' in code
    assert 'manual_pcm = bytearray()' in code


def test_rewrite_does_not_silently_patch_unknown_server_revision():
    with pytest.raises(ValueError,match='not exactly patchable'):
        M.rewrite(b'# unrelated Python service')


def test_rewrite_rejects_ambiguous_duplicate_patch_anchor():
    src=MINI_SOURCE.replace('        self.input_revision = 0\n        self.manual_pcm = bytearray()', '        self.input_revision = 0\n        self.manual_pcm = bytearray()\n        self.input_revision = 0\n        self.manual_pcm = bytearray()')
    with pytest.raises(ValueError,match='not exactly patchable'):
        M.rewrite(src.encode())


def test_offline_preflight_for_single_new_source_module(tmp_path,monkeypatch):
    src,patched=fake_repository(tmp_path,monkeypatch)
    assert len(M.preflight(src))==7
    before={p.name:p.read_bytes() for p in src.iterdir()}
    report=M.render(src,tmp_path/'not-created',run=False)
    assert report['baseline_modules']==7
    assert report['changed_modules']==['cloudflare_ws_voice_server.py']
    assert report['patched_SHA256']==hashlib.sha256(patched).hexdigest()
    assert {p.name:p.read_bytes() for p in src.iterdir()}==before
    assert not (tmp_path/'not-created').exists()


def test_corrupted_original_module_rejected_before_any_output(tmp_path,monkeypatch):
    src,_=fake_repository(tmp_path,monkeypatch)
    (src/'voice_ws_core.py').write_bytes(b'TAMPERED')
    with pytest.raises(ValueError,match='Baseline source drift'):
        M.render(src,tmp_path/'output',run=True)
    assert not (tmp_path/'output').exists()


def test_existing_or_symlinked_source_refused(tmp_path,monkeypatch):
    src,_=fake_repository(tmp_path,monkeypatch)
    (tmp_path/'other').mkdir()
    link=tmp_path/'symlink';link.symlink_to(src,target_is_directory=True)
    with pytest.raises(ValueError,match='private source root'):
        M.preflight(link)
    target=src/'voice_ws_config.py';target.unlink();target.symlink_to(tmp_path/'other',target_is_directory=True)
    with pytest.raises(ValueError,match='symlinked'):
        M.preflight(src)


def test_release_writes_can_only_target_new_private_benchmark_directory(tmp_path,monkeypatch):
    src,_=fake_repository(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Patched candidate cannot be installed'):
        M.render(src,Path('/tmp')/('w68-not-an-approved-private-voice-root-'+tmp_path.name),run=True)
    with pytest.raises(ValueError,match='Only a NEW private output'):
        M.render(src,src,run=True)
    assert len(list(src.iterdir()))==7


def test_patched_SHA_mismatch_refused_even_when_baseline_matches(tmp_path,monkeypatch):
    src,_=fake_repository(tmp_path,monkeypatch)
    monkeypatch.setattr(M,'PATCHED_SERVER_SHA256','0'*64)
    with pytest.raises(ValueError,match='tested pinned revision'):
        M.render(src,tmp_path/'out',run=True)
    assert not (tmp_path/'out').exists()
