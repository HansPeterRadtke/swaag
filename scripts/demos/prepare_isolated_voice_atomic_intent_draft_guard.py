#!/usr/bin/env python3
"""Reproducibly harden exact W68 private voice candidate storage transactions.

This follows the already verified per-conversation ASR epoch fence, but
closes its remaining check/async-filesystem-write gap by serializing the
intent epoch and speech draft commit under one per-conversation mutex. A
newer manual or typed turn retires older unsubmitted SPEECH drafts without
losing their durable audio or deleting user data. It only writes to an
exclusive new private directory; never to /etc/systemd, production source,
installed phone, network models or any owner service.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

BASE={
 'cloudflare_ws_voice_server.py':'e0939fd23aefd11f6d3d8bfd0158508b7b2f0a4f75fb70467e67bced5d06f030',
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
# Freeze exact output SHA after local review, before executing any real server.
PATCHED_SERVER_SHA256='3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f'


def digest(body:bytes)->str:return hashlib.sha256(body).hexdigest()


def modify(raw:bytes)->bytes:
    source=raw.decode('utf-8')
    changes=[
    ('''CONVERSATION_INTENT_EPOCH: dict[str, int] = {}


def advance_conversation_intent_epoch(conversation: str) -> int:
    revision = CONVERSATION_INTENT_EPOCH.get(conversation, 0) + 1
    CONVERSATION_INTENT_EPOCH[conversation] = revision
    return revision
''',
     '''CONVERSATION_INTENT_EPOCH: dict[str, int] = {}
# Each conversation independently serializes a user-intent revision and a
# durable speech-draft write. The mutex, not the GIL or an old socket's own
# draft_revision, is the linearization boundary across worker threads.
CONVERSATION_INTENT_LOCK_GUARD = threading.Lock()
CONVERSATION_INTENT_LOCKS: dict[str, threading.RLock] = {}


def conversation_intent_lock(conversation: str) -> threading.RLock:
    key = str(conversation)
    with CONVERSATION_INTENT_LOCK_GUARD:
        return CONVERSATION_INTENT_LOCKS.setdefault(key, threading.RLock())


def advance_conversation_intent_epoch(conversation: str, *,
                                      supersede_prior_speech: bool = False) -> int:
    with conversation_intent_lock(conversation):
        revision = CONVERSATION_INTENT_EPOCH.get(conversation, 0) + 1
        CONVERSATION_INTENT_EPOCH[conversation] = revision
        if supersede_prior_speech:
            # The explicit new manual/typed intent replaces prior unsubmitted
            # speech input. Preserve prior audio/transcript, but do not leave
            # an older draft in the phone's editable pending-draft state.
            with core().conversation_lock(conversation):
                data = _history_read(conversation)
                changed = False
                for turn in data.get("dialogue_history", []):
                    epoch = turn.get("asr_intent_epoch")
                    if (turn.get("status") == "draft"
                            and turn.get("input_source") == "speech"
                            and type(epoch) is int and epoch < revision
                            and not turn.get("submitted_text")):
                        turn["status"] = "superseded"
                        turn["superseded_by_intent_epoch"] = revision
                        turn["updated"] = time.time()
                        changed = True
                if changed:
                    core().atomic_write_json(core().conversation_path(conversation), data)
        return revision
'''),
    ('''def record_draft(conversation: str, turn_id: str, raw_text: str, recognized_text: str, user_audio_path: str, input_source: str = "speech") -> None:
    with core().conversation_lock(conversation):
''',
     '''def record_draft(conversation: str, turn_id: str, raw_text: str, recognized_text: str, user_audio_path: str, input_source: str = "speech", *, expected_intent_epoch: int | None = None) -> bool:
    # This is deliberately a single compare-and-write transaction shared by
    # the main-event-loop intent advancement and the executor's filesystem
    # writer. A newer intent cannot slip between validation and disk commit.
    with conversation_intent_lock(conversation):
        if (expected_intent_epoch is not None
                and CONVERSATION_INTENT_EPOCH.get(conversation, 0) != expected_intent_epoch):
            return False
        with core().conversation_lock(conversation):
'''),
    ('''        data = _history_read(conversation)
        history = data.setdefault("dialogue_history", [])
        history.append(
            {
                "user": recognized_text,''',
     '''            data = _history_read(conversation)
            history = data.setdefault("dialogue_history", [])
            history.append(
                {
                "user": recognized_text,'''),
    ('''                "input_source": str(input_source or "speech"),
            }
        )
        # The canonical transcript is durable; prompt and client windows are separate.
        data["dialogue_history"] = history
        core().atomic_write_json(core().conversation_path(conversation), data)


def update_turn(conversation: str, turn_id: str, **fields) -> bool:
''',
     '''                "input_source": str(input_source or "speech"),
                **({"asr_intent_epoch": expected_intent_epoch}
                   if expected_intent_epoch is not None else {}),
                }
            )
            # The canonical transcript is durable; prompt and client windows
            # are separate. The epoch mutex is held until the fsync/rename.
            data["dialogue_history"] = history
            core().atomic_write_json(core().conversation_path(conversation), data)
        return True


def update_turn(conversation: str, turn_id: str, **fields) -> bool:
'''),
    ('''def update_turn(conversation: str, turn_id: str, **fields) -> bool:
    found = False
    with core().conversation_lock(conversation):
        data = _history_read(conversation)
        history = data.setdefault("dialogue_history", [])
        for item in reversed(history):
            if item.get("turn_id") == turn_id:
                item.update(fields)
                item["updated"] = time.time()
                found = True
                break
        core().atomic_write_json(core().conversation_path(conversation), data)
    return found
''',
     '''def update_turn(conversation: str, turn_id: str, **fields) -> bool:
    expected_epoch = fields.pop("_expected_intent_epoch", None)
    found = False
    with conversation_intent_lock(conversation):
        if (expected_epoch is not None
                and CONVERSATION_INTENT_EPOCH.get(conversation, 0) != expected_epoch):
            return False
        with core().conversation_lock(conversation):
            data = _history_read(conversation)
            history = data.setdefault("dialogue_history", [])
            for item in reversed(history):
                if item.get("turn_id") == turn_id:
                    item.update(fields)
                    item["updated"] = time.time()
                    found = True
                    break
            if found:
                core().atomic_write_json(core().conversation_path(conversation), data)
    return found
'''),
    ('''        self.intent_epoch = advance_conversation_intent_epoch(self.conversation)
        self.generation += 1
''',
     '''        self.intent_epoch = advance_conversation_intent_epoch(
            self.conversation,
            supersede_prior_speech=reason in {"new_manual_utterance", "new_typed_turn"},
        )
        self.generation += 1
'''),
    ('''            await run_blocking(update_turn, self.conversation, turn_id, raw_asr=(previous_raw + " " + raw).strip(), recognized_text=combined, user=combined, user_audio_path=audio_path, status="draft")
        else:
            await run_blocking(record_draft, self.conversation, turn_id, raw, combined, audio_path, "speech")
        if not current():
''',
     '''            recorded = await run_blocking(update_turn, self.conversation, turn_id,
                raw_asr=(previous_raw + " " + raw).strip(), recognized_text=combined,
                user=combined, user_audio_path=audio_path, status="draft",
                _expected_intent_epoch=expected_intent_epoch)
        else:
            recorded = await run_blocking(record_draft, self.conversation, turn_id,
                raw, combined, audio_path, "speech",
                expected_intent_epoch=expected_intent_epoch)
        if not recorded or not current():
'''),
    ('''            requires_review=manual_review_required,
        )
        if not current():
            return
        self.pending_turn_id = turn_id
''',
     '''            requires_review=manual_review_required,
            _expected_intent_epoch=expected_intent_epoch,
        )
        if not current():
            return
        self.pending_turn_id = turn_id
'''),
    ]
    for old,new in changes:
        hits=source.count(old)
        if hits!=1:
            raise ValueError('Exact immutable source anchor no longer unique: '+repr(old[:100])+': '+str(hits))
        source=source.replace(old,new)
    return source.encode('utf-8')


def validate(source:Path)->dict[str,bytes]:
    if source.is_symlink() or not source.is_dir():raise ValueError('Not an exact private source directory')
    result={}
    for name,want in BASE.items():
        path=source/name
        if not path.is_file() or path.is_symlink():raise ValueError('Missing/symlinked source '+name)
        raw=path.read_bytes()
        if digest(raw)!=want:raise ValueError('Existing private speaker-review source drift '+name)
        result[name]=raw
    return result


def render(source:Path,target:Path,run:bool=False)->dict:
    baseline=validate(source)
    updated=modify(baseline['cloudflare_ws_voice_server.py'])
    measured=digest(updated)
    if measured!=PATCHED_SERVER_SHA256:
        raise ValueError('Frozen atomic-intent candidate source hash mismatch')
    data={'prior_server_sha256':BASE['cloudflare_ws_voice_server.py'],
      'new_atomic_server_sha256':measured,'exact_baseline_modules':7,
      'changed_only':['cloudflare_ws_voice_server.py'],
      'other_module_hashes':{name:value for name,value in BASE.items() if name!='cloudflare_ws_voice_server.py'},
      'atomic_storage_compare_and_write':True,
      'newer_manual_or_typed_intent_supersedes_only_older_unsubmitted_speech_drafts':True,
      'owner_live_service_or_phone_effects':0,'model_provider_requests':0,
      'source_is_not_deployed':True}
    if not run:return data
    if (not target.is_absolute() or target.exists() or target.is_symlink()
        or not str(target).startswith('/data/var/swaag-benchmarks/') or target==source):
        raise ValueError('New private output only, refusing existing or deployed destination')
    target.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(target,0o700)
    for name,raw in baseline.items():
        body=updated if name=='cloudflare_ws_voice_server.py' else raw
        p=target/name
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(body);f.flush();os.fsync(f.fileno())
        if digest(p.read_bytes())!= (measured if name=='cloudflare_ws_voice_server.py' else BASE[name]):
            raise RuntimeError('Private atomic-intent code copy corrupted')
    manifest=target/'source-manifest.json'
    fd=os.open(manifest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump({**data,'created_at_utc':datetime.now(timezone.utc).isoformat()},f,indent=2);f.flush();os.fsync(f.fileno())
    return data


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--private-output',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    print(json.dumps(render(args.source,args.private_output,args.run),sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
