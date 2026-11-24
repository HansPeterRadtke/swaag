#!/usr/bin/env python3
"""Reproducibly patch EXACT undeployed 7-module Jetson speaker-review candidate.

A delayed ASR from a superseded socket used to resurrect a stale speech draft
after the new mobile socket began an unrelated manual capture. Preserve
pending ASR across idle reconnects but fence every new user intent across all
sockets for a conversation. This script creates only a NEW private code tree;
it never deploys to Jetson systemd, changes /etc, transcribes, runs models, or
modifies a running user voice server.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

BASELINE={
 'cloudflare_ws_voice_server.py':'cbcc77fb36ce8cf72cc6442ec0f35d84526aec4868a27b810b2c38bde52bb302',
 'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
 'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
 'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
 'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
 'voice_swaag_transport.py':'87c6040493e4e6d1c99ef6f2ee1a125337f257f44fd90424d0896d09565dd1f9',
 'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
}
# Filled after independent exact local source render; source code MUST match.
PATCHED_SERVER_SHA256='e0939fd23aefd11f6d3d8bfd0158508b7b2f0a4f75fb70467e67bced5d06f030'

def digest(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def rewrite(raw:bytes)->bytes:
    src=raw.decode('utf-8')
    changes=[
       ('''ACTIVE_BY_CONVERSATION: dict[str, "VoiceSession"] = {}\n''',
        '''ACTIVE_BY_CONVERSATION: dict[str, "VoiceSession"] = {}\n# User input intention changes across replacement WebSockets. A reconnect\n# alone does not advance this epoch: pending ASR may still finish if the new\n# client remains idle. Any later user cancellation, draft edit, manual capture\n# or acoustic turn supersedes the old in-flight ASR, even if the new socket\n# itself disconnects before that older recognizer returns.\nCONVERSATION_INTENT_EPOCH: dict[str, int] = {}\n\n\ndef advance_conversation_intent_epoch(conversation: str) -> int:\n    revision = CONVERSATION_INTENT_EPOCH.get(conversation, 0) + 1\n    CONVERSATION_INTENT_EPOCH[conversation] = revision\n    return revision\n'''),
       ('''        self.input_revision = 0\n        self.manual_pcm = bytearray()''',
        '''        self.input_revision = 0\n        self.intent_epoch = CONVERSATION_INTENT_EPOCH.get(conversation, 0)\n        self.manual_pcm = bytearray()'''),
       ('''    async def cancel_response(self, reason: str) -> None:\n        self.generation += 1''',
        '''    async def cancel_response(self, reason: str) -> None:\n        # This can be invoked on a *new* client while the replaced client's\n        # final ASR is in a thread. Fence that earlier accepted speech before\n        # any await, including if this new socket disconnects next.\n        self.intent_epoch = advance_conversation_intent_epoch(self.conversation)\n        self.generation += 1'''),
       ('''        expected_generation = self.generation\n        expected_revision = self.draft_revision\n        expected_input = self.input_revision\n        def current():''',
        '''        expected_generation = self.generation\n        expected_revision = self.draft_revision\n        expected_input = self.input_revision\n        expected_intent_epoch = self.intent_epoch\n        def current():'''),
       ('''            return (expected_generation == self.generation\n                    and expected_revision == self.draft_revision\n                    and not self.draft_claimed''',
        '''            return (expected_generation == self.generation\n                    and expected_revision == self.draft_revision\n                    and expected_intent_epoch == CONVERSATION_INTENT_EPOCH.get(self.conversation, 0)\n                    and not self.draft_claimed'''),
       ('''        if kind == "draft_discard":\n            self.draft_revision += 1''',
        '''        if kind == "draft_discard":\n            self.intent_epoch = advance_conversation_intent_epoch(self.conversation)\n            self.draft_revision += 1'''),
       ('''        if kind == "draft_claim":\n            self.draft_revision += 1''',
        '''        if kind == "draft_claim":\n            self.intent_epoch = advance_conversation_intent_epoch(self.conversation)\n            self.draft_revision += 1'''),
       ('''            if value.get("draft_claimed") is True:\n                self.draft_revision += 1''',
        '''            if value.get("draft_claimed") is True:\n                self.intent_epoch = advance_conversation_intent_epoch(self.conversation)\n                self.draft_revision += 1'''),
       ('''        elif kind == "submit":\n            self.draft_revision += 1''',
        '''        elif kind == "submit":\n            self.intent_epoch = advance_conversation_intent_epoch(self.conversation)\n            self.draft_revision += 1'''),
       ('''                self.had_voice = True\n                self.input_revision += 1''',
        '''                self.had_voice = True\n                self.intent_epoch = advance_conversation_intent_epoch(self.conversation)\n                self.input_revision += 1'''),
    ]
    for original,updated in changes:
        count=src.count(original)
        if count!=1:
            raise ValueError('Original source not exactly patchable at anchor '+repr(original[:90])+': '+str(count))
        src=src.replace(original,updated)
    output=src.encode('utf-8')
    if len(output)<len(raw)+500 or b'CONVERSATION_INTENT_EPOCH' not in output:
        raise ValueError('The hardened input epoch guard is missing')
    return output


def preflight(src:Path)->dict:
    if not src.is_dir() or src.is_symlink():raise ValueError('Not the exact private source root')
    modules={}
    for name,want in BASELINE.items():
        path=src/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Source missing, symlinked, or unreadable: '+name)
        raw=path.read_bytes()
        if digest(raw)!=want:raise ValueError('Baseline source drift: '+name)
        modules[name]=raw
    return modules


def render(src:Path,out:Path,*,run:bool=False)->dict:
    baseline=preflight(src)
    new=rewrite(baseline['cloudflare_ws_voice_server.py'])
    measured=digest(new)
    if measured!=PATCHED_SERVER_SHA256:
        raise ValueError('Exact patched source changed from tested pinned revision')
    info={'original_SHA256':BASELINE['cloudflare_ws_voice_server.py'],
          'patched_SHA256':measured,
          'baseline_modules':len(baseline),'changed_modules':['cloudflare_ws_voice_server.py'],
          'unchanged_modules_sha256':{k:v for k,v in BASELINE.items() if k!='cloudflare_ws_voice_server.py'},
          'safe_older_ASR_when_new_client_idle':True,
          'user_cancel_manual_capture_draft_edit_or_new_voice_supersedes_old_ASR':True,
          'production_service_changes':0,'provider_or_user_actions':0,
          'new_code_test_only':True}
    if not run:return info
    if not out.is_absolute() or out.exists() or out.is_symlink() or out==src or out in src.parents:
        raise ValueError('Only a NEW private output directory is supported')
    if not str(out).startswith('/data/var/swaag-benchmarks/'):
        raise ValueError('Patched candidate cannot be installed to systemd/prod path')
    out.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(out,0o700)
    for name,body in baseline.items():
        data=new if name=='cloudflare_ws_voice_server.py' else body
        file=out/name
        fd=os.open(file,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if digest(file.read_bytes())!=(measured if name=='cloudflare_ws_voice_server.py' else BASELINE[name]):
            raise RuntimeError('Private candidate copy corrupted')
    manifest=out/'source-manifest.json'
    fd=os.open(manifest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump({**info,'created_utc':datetime.now(timezone.utc).isoformat()},f,indent=2)
        f.flush();os.fsync(f.fileno())
    return info


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--private-output',required=True,type=Path)
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    result=render(args.source,args.private_output,run=args.run)
    print(json.dumps({'status':'private_source_hardening_created' if args.run else 'source_patch_offline_preflight_only',**result},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
