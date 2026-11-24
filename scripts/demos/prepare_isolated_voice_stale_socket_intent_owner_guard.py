#!/usr/bin/env python3
"""Derive exact private W71 guarded Jetson source, not a live unit update.

An already-buffered control message from a predecessor WebSocket may await a
thread executor while the new client's microphone/new manual input becomes
the only active owner. Validate ownership INSIDE the per-conversation mutex,
immediately before modifying the version or durable speech drafts. A closed,
superseded predecessor must never invalidate legitimate newer utterances.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

BASE={
 'cloudflare_ws_voice_server.py':'f3283c43a196595c4b1fade69e6f2ef0f23943dc6d0b91c5b5ab63f16c17c996',
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
NEW_SHA='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def patch(source:bytes)->bytes:
    s=source.decode('utf-8')
    changes=[
    ('''def advance_conversation_intent_epoch(conversation: str, *,
                                      supersede_prior_speech: bool = False) -> int:
    with conversation_intent_lock(conversation):
        latest = CONVERSATION_INTENT_EPOCH.get(conversation)
''',
    '''def advance_conversation_intent_epoch(conversation: str, *,
                                      supersede_prior_speech: bool = False,
                                      expected_active_session=None) -> int | None:
    with conversation_intent_lock(conversation):
        if expected_active_session is not None:
            active = ACTIVE_BY_CONVERSATION.get(conversation)
            # A cancelled/superseded socket may have entered async control()
            # BEFORE network replacement, yet its executor may only reach
            # this point AFTER the replacement's newer manual recording.
            # The check must be INSIDE the same lock as the disk/epoch write.
            if (expected_active_session.superseded or expected_active_session.closed
                    or (active is not None and active is not expected_active_session)):
                return None
        latest = CONVERSATION_INTENT_EPOCH.get(conversation)
'''),
    ('''            supersede_prior_speech=reason in {"new_manual_utterance", "new_typed_turn"},
        )
        self.generation += 1''',
    '''            supersede_prior_speech=reason in {"new_manual_utterance", "new_typed_turn"},
            expected_active_session=self,
        )
        if self.intent_epoch is None:
            return
        self.generation += 1'''),
    ('''    async def control(self, message: str) -> None:
        try:
            value = json.loads(message)
''',
    '''    async def control(self, message: str) -> None:
        # This guard covers incoming buffered old control messages that have
        # not yet been dispatched. Already-dispatched controls are separately
        # checked by advance_conversation_intent_epoch under its lock.
        active = ACTIVE_BY_CONVERSATION.get(self.conversation)
        if self.superseded or self.closed or (active is not None and active is not self):
            return
        try:
            value = json.loads(message)
'''),
    ('''        elif kind == "begin_manual_utterance":
            await self.cancel_response("new_manual_utterance")
            self.awaiting_manual = False''',
    '''        elif kind == "begin_manual_utterance":
            await self.cancel_response("new_manual_utterance")
            if self.superseded or self.closed:
                return
            self.awaiting_manual = False'''),
    ('''            await self.cancel_response("new_typed_turn")
            self.draft_claimed = False''',
    '''            await self.cancel_response("new_typed_turn")
            if self.superseded or self.closed:
                return
            self.draft_claimed = False'''),
    ('''        await self.cancel_response("semantic_barge_in")
        self.utterance_pcm = bytearray(preserved_pcm)''',
    '''        await self.cancel_response("semantic_barge_in")
        if self.superseded or self.closed:
            return
        self.utterance_pcm = bytearray(preserved_pcm)'''),
    ('''                existing = await run_blocking(get_turn, self.conversation, turn_id)
                if existing is not None:
                    if existing.get("input_source") != "typed" or str(''',
    '''                existing = await run_blocking(get_turn, self.conversation, turn_id)
                # A connection can lose ownership while get_turn is awaited.
                # An ACK-lost idempotent retry may reach submit_turn directly,
                # bypassing the ordinary epoch guard. Never let the old socket
                # cross the agent submission boundary after replacement.
                active = ACTIVE_BY_CONVERSATION.get(self.conversation)
                if self.superseded or self.closed or (active is not None and active is not self):
                    return
                if existing is not None:
                    if existing.get("input_source") != "typed" or str('''),
    ]
    for old,new in changes:
        if s.count(old)!=1:raise ValueError('Ambiguous changed source anchor '+repr(old[:86])+': '+str(s.count(old)))
        s=s.replace(old,new)
    # Preserve the actual nested source indentation separately for each of
    # the five async-control/voice onset sites. We refuse unknown changes.
    import re
    pattern=re.compile(r'(?P<indent>^ *)self\.intent_epoch = await run_blocking\(advance_conversation_intent_epoch, self\.conversation\)$',re.M)
    matches=list(pattern.finditer(s))
    if len(matches)!=5:
        raise ValueError('Expected five exact line-delimited async source sites, got '+str(len(matches)))
    def guard_line(match):
        indent=match.group('indent')
        return (indent+'self.intent_epoch = await run_blocking(advance_conversation_intent_epoch, '
               'self.conversation, expected_active_session=self)\n'
               +indent+'if self.intent_epoch is None: return')
    s=pattern.sub(guard_line,s)
    import ast
    try:ast.parse(s)
    except SyntaxError as err:raise ValueError('Ambiguous async indentation during W71 source rewrite: '+str(err))
    if s.count('expected_active_session=self')!=6:raise ValueError('Not all async callbacks guarded')
    return s.encode('utf-8')


def source_files(root:Path)->dict[str,bytes]:
    if root.is_symlink() or not root.is_dir():raise ValueError('No exact private original source')
    out={}
    for name,expected in BASE.items():
        p=root/name
        if not p.is_file() or p.is_symlink() or sha(p.read_bytes())!=expected:
            raise ValueError('Unknown original private source bytes '+name)
        out[name]=p.read_bytes()
    return out


def render(original:Path,dest:Path,run:bool)->dict:
    files=source_files(original)
    body=patch(files['cloudflare_ws_voice_server.py'])
    measured=sha(body)
    if measured!=NEW_SHA:
        raise ValueError('Proposed patch content differs from reviewed exact source hash')
    record={'previous_undeployed_server_sha256':BASE['cloudflare_ws_voice_server.py'],
        'new_undeployed_server_sha256':measured,'other_six_supporting_modules_unchanged':True,
        'only_changed_file':'cloudflare_ws_voice_server.py',
        'same_conversation_lock_for_source_session_ownership_and_advance':True,
        'already_dispatched_superseded_control_fails_closed_before_mutation':True,
        'unknown_speaker_auto_submit_still_denied':True,
        'live_phone_user_provider_or_tools_effects':0}
    if not run:return record
    if dest.exists() or dest.is_symlink() or not dest.is_absolute() or not str(dest).startswith('/data/var/swaag-benchmarks/') or dest==original:
        raise FileExistsError('Only unused private benchmark root may contain new code')
    dest.mkdir(mode=0o700,parents=True,exist_ok=False);os.chmod(dest,0o700)
    for name,previous in files.items():
        b=body if name=='cloudflare_ws_voice_server.py' else previous
        p=dest/name;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        assert sha(p.read_bytes())==sha(b)
    manifest=dest/'source-manifest.json'
    fd=os.open(manifest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump({**record,'at_utc':datetime.now(timezone.utc).isoformat()},f,indent=2);f.flush();os.fsync(f.fileno())
    return record


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--private-output',required=True,type=Path);p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    d=render(a.source,a.private_output,a.run)
    print(json.dumps({'status':'NEW_PRIVATE_SOURCE_CREATED' if a.run else 'SOURCE_ONLY_NO_EFFECT_PREFLIGHT',**d},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
