#!/usr/bin/env python3
"""Reproducibly derive private W70 voice server from exact W69 source only.

Restores the maximum valid persisted speech/user-intent epoch after a server
restart so new manual input supersedes the old unsubmitted speech draft.
Moves potentially slow per-conversation epoch acquisition and disk mutation
out of the asyncio event loop without reducing atomicity. Production Jetson,
user accounts, model endpoints and prior evidence are never touched.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

BASE={
 'cloudflare_ws_voice_server.py':'3d4e6e029aa4a467684c618dc31cfd60d53d77939843f547d394b04a979bad2f',
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
PATCHED_SERVER_SHA256='f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996'

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def patch(original:bytes)->bytes:
    s=original.decode('utf-8')
    old='''def advance_conversation_intent_epoch(conversation: str, *,
                                      supersede_prior_speech: bool = False) -> int:
    with conversation_intent_lock(conversation):
        revision = CONVERSATION_INTENT_EPOCH.get(conversation, 0) + 1
        CONVERSATION_INTENT_EPOCH[conversation] = revision
'''
    new='''def _recover_persisted_intent_epoch(conversation: str) -> int:
    # The live server is one process, but a process restart clears the RAM
    # dictionary. Always hydrate from actual durable history *before* a new
    # user intent supersedes a recorded speech draft. Older records without
    # an epoch are intentionally treated as pre-epoch speech; they may be
    # retired by an explicit new manual/typed input, never by idle reconnect.
    # Run only within conversation_intent_lock AND Core's storage lock.
    value = 0
    for turn in _history_read(conversation).get("dialogue_history", []):
        for key in ("asr_intent_epoch", "superseded_by_intent_epoch"):
            number = turn.get(key)
            if number is None:
                continue
            if type(number) is not int or not 0 <= number <= 1_000_000_000:
                raise ValueError("stored speech revision malformed or unsupported")
            value = max(value, number)
    return value


def advance_conversation_intent_epoch(conversation: str, *,
                                      supersede_prior_speech: bool = False) -> int:
    with conversation_intent_lock(conversation):
        latest = CONVERSATION_INTENT_EPOCH.get(conversation)
        if latest is None:
            with core().conversation_lock(conversation):
                latest = _recover_persisted_intent_epoch(conversation)
        if latest >= 1_000_000_000:
            raise ValueError("intent revision limit reached")
        revision = latest + 1
        CONVERSATION_INTENT_EPOCH[conversation] = revision
'''
    if s.count(old)!=1:raise ValueError('Expected exact W69 process restart revision bug source not present')
    s=s.replace(old,new)
    old='''                            and type(epoch) is int and epoch < revision
                            and not turn.get("submitted_text")):
'''
    new='''                            and (epoch is None or (type(epoch) is int and epoch < revision))
                            and not turn.get("submitted_text")):
'''
    if s.count(old)!=1:raise ValueError('Missing exact earlier guarded draft supersession anchor')
    s=s.replace(old,new)
    prefix='self.intent_epoch = advance_conversation_intent_epoch('
    hits=s.count(prefix)
    if hits!=6:raise ValueError('Expected six actual direct event-loop source epoch calls, got '+str(hits))
    # Includes cancel_response(), draft_discard/claim, client-state claim,
    # text submit, and confirmed acoustic voice onset, all async contexts.
    s=s.replace(prefix,'self.intent_epoch = await run_blocking(advance_conversation_intent_epoch, ')
    if s.count('self.intent_epoch = await run_blocking(advance_conversation_intent_epoch, ')!=6:
        raise ValueError('Not every same-conversation mutex caller moved to an executor')
    if 'self.intent_epoch = advance_conversation_intent_epoch(' in s:
        raise ValueError('Blocking event-loop source epoch mutation remains')
    return s.encode('utf-8')


def validate(src:Path)->dict[str,bytes]:
    if not src.is_dir() or src.is_symlink():raise ValueError('Not an exact private W69 source root')
    files={}
    for n,want in BASE.items():
        f=src/n
        if not f.is_file() or f.is_symlink():raise ValueError('Untrusted original module missing '+n)
        data=f.read_bytes()
        if sha(data)!=want:raise ValueError('W69 source content altered '+n)
        files[n]=data
    return files


def render(src:Path,out:Path,run=False)->dict:
    inp=validate(src)
    modified=patch(inp['cloudflare_ws_voice_server.py'])
    newer=sha(modified)
    if newer!=PATCHED_SERVER_SHA256:
        raise ValueError('W70 reviewed source SHA changed')
    meta={'baseline_exact_W69_server_sha256':BASE['cloudflare_ws_voice_server.py'],
        'new_server_sha256':newer,'support_modules_sha256':{k:v for k,v in BASE.items() if k!='cloudflare_ws_voice_server.py'},
        'only_changed_module':'cloudflare_ws_voice_server.py',
        'persisted_epochs_recovered_after_real_process_restart':True,
        'per_conversation_durable_draft_compare_and_write_remains_atomic':True,
        'six_source_async_handlers_offload_mutex_and_fs_sync':True,
        'unsubmitted_legacy_speech_draft_without_epoch_superseded_on_explicit_new_input':True,
        'owner_phone_account_production_or_tool_effects':0,
        'candidate_not_deployed_to_live_server':True}
    if not run:return meta
    if not out.is_absolute() or not str(out).startswith('/data/var/swaag-benchmarks/') or out.exists() or out.is_symlink() or out==src:
        raise ValueError('Source patch can only create one new private benchmark directory')
    out.mkdir(mode=0o700,parents=True,exist_ok=False);os.chmod(out,0o700)
    for n,b in inp.items():
        content=modified if n=='cloudflare_ws_voice_server.py' else b
        p=out/n
        fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
        assert sha(p.read_bytes())==sha(content)
    p=out/'source-manifest.json'
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump({**meta,'recorded_utc':datetime.now(timezone.utc).isoformat()},f,indent=2);f.flush();os.fsync(f.fileno())
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','private-output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run',action='store_true')
    x=p.parse_args(argv)
    print(json.dumps({'status':'NEW_PRIVATE_GENERATION' if x.run else 'NO_SIDE_EFFECT_PREVIEW',
                     **render(x.source,x.private_output,run=x.run)},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
