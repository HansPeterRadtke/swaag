#!/usr/bin/env python3
"""Prepare a private successor to the exact W71 voice source; never deploy.

Persist an acknowledged explicit media stop before notifying the client.
This is separate from an agent cancellation and from lost playback ACK.
Only the versioned source derivation is changed; no live service is touched.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

from verify_isolated_voice_release_server import EXPECTED_SOURCES, _write_unique

BASE = '873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
PREFIX = '/data/var/swaag-benchmarks/owner-serial-000100-20261125/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patch(raw):
    if sha(raw)!=BASE:
        raise ValueError('unknown W71 base source')
    s=raw.decode('utf-8')
    changes=[
        ('''    expected_epoch = fields.pop("_expected_intent_epoch", None)
    found = False''',
         '''    expected_epoch = fields.pop("_expected_intent_epoch", None)
    require_unstopped_audio = fields.pop("_require_unstopped_audio", False)
    found = False'''),
        ('''                if item.get("turn_id") == turn_id:
                    item.update(fields)
                    item["updated"] = time.time()
                    found = True
                    break''',
         '''                if item.get("turn_id") == turn_id:
                    if item.get("audio_stopped") is True:
                        if require_unstopped_audio:
                            return False
                        # Audio completion is independent of an answer being
                        # generated. Late transport bookkeeping must not
                        # resurrect a deliberately stopped delivery or report
                        # playback that the user explicitly declined.
                        if fields.get("status") in {"ready", "speaking"} and (
                            fields.get("assistant_audio_path") or item.get("assistant_audio_path")
                        ):
                            fields["status"] = "completed"
                        if "audio_transmitted" in fields:
                            fields["audio_transmitted"] = False
                        if "audio_completed" in fields:
                            fields["audio_completed"] = False
                    item.update(fields)
                    item["updated"] = time.time()
                    found = True
                    break'''),
        ('''def get_turn(conversation: str, turn_id: str) -> dict | None:''',
         '''def persist_audio_stop(conversation, turn_id, stream_id, generation, expected_session):
    # An old socket must not stop a replacement socket's newer stream.
    # Serialize with all durable turn updates, including late answer/media
    # status transitions, before acknowledging the stop to the client.
    with conversation_intent_lock(conversation):
        active = ACTIVE_BY_CONVERSATION.get(conversation)
        if (active is not expected_session or expected_session.closed
                or expected_session.superseded):
            return False
        with core().conversation_lock(conversation):
            data = _history_read(conversation)
            for item in reversed(data["dialogue_history"]):
                if item.get("turn_id") != turn_id:
                    continue
                if (not stream_id or item.get("audio_stream_id") != stream_id
                        or item.get("audio_generation") != generation):
                    return False
                if item.get("audio_completed") is True:
                    return True  # Never undo a genuine completed receipt.
                item.update(audio_stopped=True, audio_stopped_stream_id=stream_id,
                            audio_transmitted=False, audio_completed=False,
                            updated=time.time())
                if (item.get("status") in {"ready", "speaking"}
                        and item.get("assistant_audio_path")):
                    item["status"] = "completed"
                core().atomic_write_json(core().conversation_path(conversation), data)
                return True
    return False


def get_turn(conversation: str, turn_id: str) -> dict | None:'''),
        ('''        if item.get("status") == "ready" and item.get("assistant") and item.get("assistant_audio_path"):
            return dict(item)''',
         '''        if (item.get("status") == "ready" and not item.get("audio_stopped")
                and item.get("assistant") and item.get("assistant_audio_path")):
            return dict(item)'''),
        ('''                "audio_completed": bool(item.get("audio_completed")),
                "requires_review":''',
         '''                "audio_completed": bool(item.get("audio_completed")),
                "audio_stopped": bool(item.get("audio_stopped")),
                "requires_review":'''),
        ('''        self.output_stream_id = b""
        self.output_offset = 0''',
         '''        self.output_stream_id = b""
        self.output_audio_turn_id = None
        self.output_audio_generation = None
        self.output_offset = 0'''),
        ('''                self.output_stream_id = uuid.uuid4().bytes
                self.output_offset = 0''',
         '''                self.output_stream_id = uuid.uuid4().bytes
                self.output_audio_turn_id = (payload.get("turn_id")
                    if payload.get("type") == "audio" else None)
                self.output_audio_generation = payload.get("generation")
                self.output_offset = 0'''),
        ('''                    await run_blocking(update_turn,self.conversation,turn_id,
                        audio_stream_id=payload["stream_id"],audio_generation=payload.get("generation"),
                        audio_transmitted=False,audio_completed=False)''',
         '''                    if not await run_blocking(update_turn,self.conversation,turn_id,
                        audio_stream_id=payload["stream_id"],audio_generation=payload.get("generation"),
                        audio_transmitted=False,audio_completed=False,_require_unstopped_audio=True):
                        self.output_muted = True
                        return'''),
        ('''        if not turn or turn.get("status") != "ready":
            return
        reply = str(turn.get("assistant") or "")''',
         '''        if not turn or turn.get("status") != "ready" or turn.get("audio_stopped"):
            return
        reply = str(turn.get("assistant") or "")'''),
        ('''        await run_blocking(update_turn, self.conversation, turn_id, status="speaking", audio_started=True, audio_completed=False)
        bytes_per_second''',
         '''        if not await run_blocking(update_turn, self.conversation, turn_id,
                status="speaking", audio_started=True, audio_completed=False,
                _require_unstopped_audio=True):
            self.speaking = False
            self.active_turn_id = None
            return
        bytes_per_second'''),
        ('''            self.active_turn_id = None
            await self.send_event({"type": "audio", "event": "end", "generation": generation, "turn_id": turn_id})
            await self.send_event({"type": "state", "state": "listening", "generation": generation})''',
         '''            self.active_turn_id = None
            if not self.output_muted:
                await self.send_event({"type": "audio", "event": "end", "generation": generation, "turn_id": turn_id})
            await self.send_event({"type": "state", "state": "listening", "generation": generation})'''),
        ('''            if not self.closed:
                try:
                    await self.send_event({"type":"audio","event":"end","generation":generation,"turn_id":turn_id})''',
         '''            if not self.closed and not self.output_muted:
                try:
                    await self.send_event({"type":"audio","event":"end","generation":generation,"turn_id":turn_id})'''),
        ('''            if (turn and turn.get("audio_transmitted")
                and value.get("generation") == turn.get("audio_generation")''',
         '''            if (turn and turn.get("audio_transmitted") and not turn.get("audio_stopped")
                and value.get("generation") == turn.get("audio_generation")'''),
        ('''        if kind == "stop_audio":
            self.output_muted = True
            if self.replay_task and not self.replay_task.done():
                self.replay_task.cancel()
            await self.emit_event({"type": "audio", "event": "cancel", "generation": self.generation,
                                   "reason": "playback_stopped"})
            return''',
         '''        if kind == "stop_audio":
            self.output_muted = True
            if self.replay_task and not self.replay_task.done():
                self.replay_task.cancel()
            turn_id = self.output_audio_turn_id
            stream_id = self.output_stream_id.hex()
            if turn_id:
                try:
                    saved = await run_blocking(persist_audio_stop,
                        self.conversation, turn_id, stream_id,
                        self.output_audio_generation, self)
                except Exception as exc:
                    saved = False
                    log("playback_stop_storage_error", severity="WARNING",
                        conversation=self.conversation, error_type=type(exc).__name__)
                if not saved:
                    if not self.closed and not self.superseded:
                        await self.send_event({"type":"error","stage":"playback_stop",
                            "message":"playback_stop_not_durable","turn_id":turn_id})
                    return
            # Never route an old socket's stop through emit_event to the
            # new owner of the conversation. Manual replay remains separate.
            if not self.closed and not self.superseded:
                await self.send_event({"type": "audio", "event": "cancel",
                    "generation": self.generation, "reason": "playback_stopped",
                    "turn_id": turn_id})
            return'''),
    ]
    for old,new in changes:
        if s.count(old)!=1:
            raise ValueError('changed/ambiguous source anchor: '+old[:100])
        s=s.replace(old,new)
    ast.parse(s)
    return s.encode('utf-8')


def prepare(source, output):
    if source.is_symlink() or output.is_symlink() or output.exists():
        raise ValueError('source or output not admissible')
    if not str(output.resolve()).startswith(PREFIX):
        raise ValueError('private new output root required')
    content={}
    for n,h in EXPECTED_SOURCES.items():
        p=source/n;want=BASE if n=='cloudflare_ws_voice_server.py' else h
        if p.is_symlink() or not p.is_file() or sha(p.read_bytes())!=want:
            raise ValueError('source hash mismatch: '+n)
        content[n]=p.read_bytes()
    content['cloudflare_ws_voice_server.py']=patch(content['cloudflare_ws_voice_server.py'])
    output.mkdir(mode=0o700,parents=False)
    for n,raw in content.items():_write_unique(output/n,raw)
    record={'original_server_sha256':BASE,'candidate_server_sha256':sha(content['cloudflare_ws_voice_server.py']),
            'files':{n:sha(raw) for n,raw in content.items()},'deployed':False}
    _write_unique(output/'source-manifest.json',json.dumps(record,indent=2).encode())
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
