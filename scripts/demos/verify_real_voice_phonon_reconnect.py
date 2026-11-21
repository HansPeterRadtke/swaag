#!/usr/bin/env python3
"""Reproduce a staged, REAL no-action voice/TTS/reconnect acceptance.

The canonical SWAAG reply is a prior durable, verified NO-TOOLS result. This
script DOES NOT start a new SWAAG turn or authorize any user command. It calls
the existing private loopback Phonon TTS and real Thor resident Whisper through
an isolated Jetson WebSocket fixture, then proves durable replay/ACK.

Run only with explicit --run and SWAAG_PHONON_VOICE_NOACTION=1, after checking
shared service health and the Raspi resource broker. Nothing touches the
production phone or production voice session.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
import unittest
import wave

ORIGINAL_AUDIO_SHA256 = "2b29bead44440ee10117f8dc6f2f20be552aa2ffbb39b36f10c854dc165376ee"
CANONICAL_REPLY_SHA256 = "78c4dd833853f3b881806d7ec76dd572ffa153bcd09bdc72fe7334da7731dbab"
CANONICAL_JOURNAL_SHA256 = "1e7fd9076c7c0364d32494dbbfdd95d280ac9071e11daf5c59e268e99b28aaa4"
VOICE_FILES_SHA256 = {
    "cloudflare_ws_voice_server.py": "5fd9cefd0eece90f8c7861f188601307445a6030ca2b7a23e1e1bcce7a8d5c96",
    "test_voice_real_asr_reconnect.py": "09d275ead552fbef3a08942fc3423a9e6c13fff409c177623d95bd5bb0e585fc",
    "test_voice_complete_noaction_e2e.py": "119c895e13af087eaf7061b4bfeb35288532d5e1a6addc539028108aee653328",
}
REQUIRED_ACCEPTANCE = (
    "reply_pcm_after_reconnect_exact", "playback_ack_persisted", "fourth_socket_no_replay",
    "manual_confirmation_received", "one_durable_turn", "no_lost_ack_simulated",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_source(proof_path: Path) -> str:
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    text = proof.get("canonical_reply")
    if not isinstance(text, str) or digest(text.encode()) != CANONICAL_REPLY_SHA256:
        raise ValueError("Reply does not match verified canonical no-action source")
    if (proof.get("reply_sha256") != CANONICAL_REPLY_SHA256
            or proof.get("source_journal_sha256") != CANONICAL_JOURNAL_SHA256
            or proof.get("source_verified_no_tools") is not True
            or proof.get("source_finished") is not True
            or proof.get("not_fresh_voice_response") is not True
            or proof.get("not_authorized_for_external_action") is not True):
        raise ValueError("Missing durable canonical-source authority and scope evidence")
    return text


def check_voice_source(source_root: Path) -> dict:
    source_root = source_root.resolve()
    for filename, expected in VOICE_FILES_SHA256.items():
        path = source_root / filename
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError("Voice source revision mismatch: " + filename)
    # The vendor-side test has a pinned original source WAV and does not submit
    # actual SWAAG work or permit newly generated tool calls. Use it unchanged.
    recording = Path("/data/var/swaag-benchmarks/target-speaker-20261111/target.wav")
    manifest = recording.parent / "source.json"
    if not manifest.is_file() or not recording.is_file():
        raise FileNotFoundError("Recorded source and provenance are mandatory")
    data = recording.read_bytes()
    original_manifest = json.loads(manifest.read_text())
    if digest(data) != ORIGINAL_AUDIO_SHA256 or original_manifest.get("sha256") != ORIGINAL_AUDIO_SHA256:
        raise ValueError("Recorded original WAV does not match pinned source provenance")
    return {"voice_source_path": str(source_root), "voice_test_sha256": VOICE_FILES_SHA256,
            "original_source_wav_sha256": ORIGINAL_AUDIO_SHA256}


def safe_loopback_tts_url(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost"}
            or parts.port != 15305 or parts.path != "/synthesize"
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("TTS must use the existing Jetson-local Phonon Easton tunnel")
    return url


def inspect_wav(raw: bytes) -> dict:
    if not raw.startswith(b"RIFF") or not 44 <= len(raw) <= 2_000_000:
        raise ValueError("TTS did not return a bounded RIFF WAV")
    try:
        with wave.open(io.BytesIO(raw), "rb") as audio:
            rate,channels,width,frames = (audio.getframerate(),audio.getnchannels(),
                                           audio.getsampwidth(),audio.getnframes())
            pcm = audio.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid TTS WAV header") from exc
    if (rate,channels,width) != (16000,1,2) or frames <= rate or frames >= rate*30:
        raise ValueError("TTS WAV not a bounded 16 kHz mono PCM16 response")
    if len(pcm) != frames*2 or not any(pcm):
        raise ValueError("TTS WAV truncated or silent")
    return {"sample_rate":rate,"channels":channels,"sample_width":width,
            "frames":frames,"duration_seconds":round(frames/rate,3),
            "pcm_bytes":len(pcm),"wav_sha256":digest(raw)}


class RejectHttpRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise RuntimeError("TTS HTTP redirect forbidden: must remain on private loopback")


def validate_pinned_swaag_paths(source_root: str, config_path: str,
                                source_commit: str, config_sha256: str) -> tuple[str,str]:
    """Allow only fixed-shaped on-host SWAAG working trees and private config.

    Every argument is separately bound to the verified on-disk Git revision and
    private configuration SHA before any model call. No user text is interpolated
    into a shell command, and paths cannot contain shell metacharacters.
    """
    source_re=r'/data/src/worktrees/[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}'
    config_re=r'/data/var/swaag-benchmarks/[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}/noaction-config\.toml'
    if type(source_root) is not str or re.fullmatch(source_re,source_root) is None:
        raise ValueError('Canonical SWAAG source must be a named /data/src/worktrees path')
    if type(config_path) is not str or re.fullmatch(config_re,config_path) is None:
        raise ValueError('Fresh noaction config must be under the private SWAAG benchmark root')
    for value in (source_commit,config_sha256):
        if type(value) is not str or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',value) is None:
            raise ValueError('Expected source revision and config SHA-256 must be hex digests')
    if len(source_commit)!=40 or len(config_sha256)!=64:
        raise ValueError('Expected source revision must be SHA-1 and config must be SHA-256')
    return source_root,config_path


def validate_short_voice_clarification(text: str) -> str:
    """Reject unhelpful meta, repeated or unfinished fresh no-action answers.

    This is a narrow acceptance gate for an *unverified reading* where the
    correct user-facing response is exactly one natural clarification question.
    It does not invent, shorten, or semantically rewrite the model's answer.
    """
    if type(text) is not str or not text.strip():
        raise ValueError('No usable user-facing answer')
    plain=' '.join(text.split())
    if (len(plain)>170 or len(plain.split())>28 or plain.count('?')!=1
            or not plain.endswith('?') or '\n' in text):
        raise ValueError('Voice answer is not one short completed clarification question')
    if re.search(r"(?:\b(?:tools?|permissions?|system prompt|language model|I cannot|I can't|I won't|AI assistant)\b|[`#*])",plain,re.I):
        raise ValueError('Voice answer contains internal capability or formatting meta')
    return plain


def synthesize(text: str, endpoint: str) -> tuple[bytes,dict]:
    endpoint = safe_loopback_tts_url(endpoint)
    # Never honor proxy environment variables or HTTP redirects for owner
    # speech. The posted text must remain on the existing local-only tunnel.
    opener = build_opener(ProxyHandler({}), RejectHttpRedirects())
    health_url = endpoint.rsplit('/',1)[0] + '/health'
    with opener.open(health_url,timeout=5) as health_response:
        health = json.loads(health_response.read(4096))
    if (health.get('ok') is not True or health.get('service') != 'phonon_easton_voice'
            or health.get('engine') != 'phonon_31m_easton_cpu'
            or health.get('speaker') != 'Easton'):
        raise RuntimeError('Installed Phonon Easton service identity was not verified')
    request = Request(endpoint,data=json.dumps({"text":text,"sample_rate":16000}).encode(),
                      headers={"Content-Type":"application/json"},method="POST")
    started = time.monotonic()
    with opener.open(request,timeout=65) as response:
        if response.status != 200 or response.headers.get("Content-Type") != "audio/wav":
            raise RuntimeError("Live Phonon synthesis did not complete as audio/wav")
        data = response.read(2_000_001)
    metadata = inspect_wav(data)
    metadata["http_wall_seconds"] = round(time.monotonic()-started,3)
    metadata["engine"] = "installed Nitro Phonon Easton via Jetson's existing loopback tunnel"
    return data,metadata


def run_isolated_voice_acceptance(source_root: Path,reply_wav: Path) -> tuple[dict,str]:
    # Import only the checksum-pinned original, which creates a NEW ephemeral
    # loopback WebSocket, private VoiceCore state and test bearer credential.
    # Its guards prohibit all live SWAAG/orchestrator/TTS actions in the server;
    # this orchestrated demo has made exactly one explicit TTS request above.
    sys.path.insert(0,str(source_root.resolve()))
    os.environ["SWAAG_FULL_VOICE_NOACTION"] = "1"
    import test_voice_complete_noaction_e2e as previous
    previous.REPLY = reply_wav
    suite=unittest.defaultTestLoader.loadTestsFromName(
        "FullVoiceNoActionAcceptance.test_original_voice_real_asr_manual_review_safe_answer_playback_and_reconnect",
        previous,
    )
    report=io.StringIO()
    with redirect_stdout(report):
        verdict=unittest.TextTestRunner(stream=report,verbosity=2).run(suite)
    trace=report.getvalue()
    if not verdict.wasSuccessful() or verdict.testsRun != 1 or verdict.skipped:
        raise RuntimeError("Actual four-generation no-action WebSocket acceptance failed: "+trace[-600:])
    match=re.search(r"CONTINUOUS_NOACTION_VOICE_ACCEPTANCE (\{[^\n]*\})",trace)
    if match is None:
        raise RuntimeError("No durable observed acceptance receipt from voice integration")
    metrics=json.loads(match.group(1))
    if any(metrics.get(item) is not True for item in REQUIRED_ACCEPTANCE):
        raise RuntimeError("Missing verified reconnect/ACK/manual-review evidence")
    if (metrics.get("connection_generations") != [1,2,3,4]
            or metrics.get("Whisper_largev3_calls") != 1
            or metrics.get("real_swaag_or_tts_calls") != 0
            or metrics.get("prerecorded_reply_wav_sha256") != digest(reply_wav.read_bytes())
            or metrics.get("user_wav_sha256") != ORIGINAL_AUDIO_SHA256):
        raise RuntimeError("Voice test cannot prove this exact TTS WAV and original speech replay")
    return metrics,trace


def main(argv=None) -> int:
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reply-proof",type=Path,required=True)
    cli.add_argument("--voice-source",type=Path,required=True)
    cli.add_argument("--evidence",type=Path,required=True)
    cli.add_argument("--tts-url",default="http://127.0.0.1:15305/synthesize")
    cli.add_argument("--run",action="store_true",help="Actual resident Whisper, TTS and localhost voice stream")
    args=cli.parse_args(argv)
    text=load_verified_source(args.reply_proof)
    source=check_voice_source(args.voice_source)
    safe_loopback_tts_url(args.tts_url)
    if not args.run:
        print(json.dumps({"status":"source_preflight_only","would_execute":False,
                          "no_live_model_called":True,"input":source}))
        return 0
    if os.environ.get("SWAAG_PHONON_VOICE_NOACTION")!="1":
        cli.error("Explicit SWAAG_PHONON_VOICE_NOACTION=1 opt-in required")
    old_umask=os.umask(0o077)
    try:
        args.evidence.mkdir(mode=0o700,parents=False,exist_ok=False)
    finally:os.umask(old_umask)
    wav_path=args.evidence/"real-phonon-canonical-reply.wav"
    record={"status":"started","source":source,"reply_source_journal_sha256":CANONICAL_JOURNAL_SHA256,
            "canonical_reply_sha256":CANONICAL_REPLY_SHA256,
            "scope":"staged integration: PRIOR canonical answer, fresh real Phonon synthesis, real Whisper and four WS generations",
            "not_fresh_orchestrator_execution":True,"physical_phone_test":False,"external_actions_authorized":False}
    try:
        audio,metadata=synthesize(text,args.tts_url)
        wav_path.write_bytes(audio);wav_path.chmod(0o600)
        result,trace=run_isolated_voice_acceptance(args.voice_source,wav_path)
        trace_path=args.evidence/"voice-test.log"
        trace_path.write_text(trace,encoding="utf-8")
        trace_path.chmod(0o600)
        record.update({"status":"verified_staged_live_noaction","phonon_tts":metadata,
                       "actual_isolated_voice_receipt":result,"new_TTS_requests_outside_voice_fixture":1,
                       "SWAAG_tool_calls":0,"original_audio_unchanged":True,
                       "production_voice_service_not_reconfigured":True})
    except BaseException as exc:
        record.update({"status":"failed_or_incomplete","error":type(exc).__name__+": "+str(exc)[:500]})
        result_path=args.evidence/"result.json"
        result_path.write_text(json.dumps(record,indent=2)+"\n")
        result_path.chmod(0o600)
        raise
    result_path=args.evidence/"result.json"
    result_path.write_text(json.dumps(record,indent=2)+"\n")
    result_path.chmod(0o600)
    print("REAL_STAGED_VOICE_PHONON_RECONNECT",json.dumps({
        "status":record["status"],"tts_seconds":record["phonon_tts"]["http_wall_seconds"],
        "voice_generations":result["connection_generations"],"replay_exact":result["reply_pcm_after_reconnect_exact"],
        "ack":result["playback_ack_persisted"],"external_actions":0,
        "evidence_root":str(args.evidence)}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
