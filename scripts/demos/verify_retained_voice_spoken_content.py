#!/usr/bin/env python3
"""Independent, opt-in spoken-content check for an existing real SWAAG reply.

A prior one-use no-action SWAAG -> Phonon -> reconnect run must have
already finished and retained the exact WAV. This checker never synthesizes
speech or invokes SWAAG. On explicit --run it sends only that existing WAV
to the already resident, fixed local Thor Whisper verifier, compares the
actual words to the prior source-backed answer, and saves private evidence.
It does not establish owner-speaker identity, real handset playback, or a
human acoustic quality judgment.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener
import wave

ASR_URL = 'http://thor:13484/v1/voice/transcribe-verify'
OPT_IN = 'SWAAG_RETAINED_TTS_ASR_NOACTION'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', text.casefold())


def word_edits(expected: list[str], observed: list[str]) -> int:
    previous = list(range(len(observed) + 1))
    for i, a in enumerate(expected, 1):
        row = [i]
        for j, b in enumerate(observed, 1):
            row.append(min(previous[j] + 1, row[j-1] + 1,
                           previous[j-1] + (a != b)))
        previous = row
    return previous[-1]


def inspect_completed_turn(receipt: Path, audio: Path) -> tuple[dict, bytes, str]:
    if receipt.is_symlink() or audio.is_symlink():
        raise ValueError('A source evidence symlink is not admitted')
    if receipt.stat().st_size > 200_000:
        raise ValueError('Unbounded original receipt')
    data = receipt.read_bytes()
    obj = json.loads(data)
    answer = obj.get('canonical_fresh_reply')
    if (obj.get('status') != 'real_fast_swaag_direct_reply_phonon_reconnect_noaction'
        or obj.get('canonical_answer_not_fresh') is not False
        or obj.get('fast_foreground_direct_answer') is not True
        or obj.get('fresh_canonical_response_calls') != 1
        or obj.get('model_action_calls') != 0
        or obj.get('played_pcm_recovery_exact') is not True
        or obj.get('normal_ack') is not True
        or obj.get('no_replay_after_ack') is not True
        or not isinstance(answer, dict)
        or not isinstance(answer.get('reply'), str)
        or not 5 <= len(answer['reply']) <= 350
        or not words(answer['reply'])
        or sha(answer['reply'].encode('utf-8')) != answer.get('stdout_sha256')):
        raise ValueError('Missing verified original no-action answer/reconnect receipts')
    if audio.stat().st_size > 2_000_000:
        raise ValueError('Unbounded original generated WAV')
    wav = audio.read_bytes()
    stored = obj.get('preserved_generated_audio') or {}
    if (not isinstance(stored, dict) or len(wav) > 2_000_000
        or sha(wav) != obj.get('tts_wav_sha256')
        or sha(wav) != stored.get('retained_wav_sha256')
        or len(wav) != stored.get('retained_wav_bytes')):
        raise ValueError('Retained WAV differs from original generated reply')
    try:
        with wave.open(io.BytesIO(wav), 'rb') as w:
            rate,channels,width,frames=(w.getframerate(),w.getnchannels(),
                                        w.getsampwidth(),w.getnframes())
            pcm=w.readframes(frames)
    except (wave.Error,EOFError) as exc:
        raise ValueError('Unparseable generated audio') from exc
    if ((rate,channels,width)!=(16000,1,2) or not 16000<frames<16000*30
        or len(pcm)!=frames*2 or not any(pcm)):
        raise ValueError('Generated audio is not bounded 16-kHz speech PCM16')
    return obj,wav,sha(data)


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('No redirects allowed for private ASR input')


def actual_resident_asr(wav: bytes) -> str:
    # Fixed authenticated private-network destination; no arbitrary URL option,
    # proxy, redirect, remote provider, service restart or agent invocation.
    opener=build_opener(ProxyHandler({}),RejectRedirect())
    request=Request(ASR_URL,data=wav,method='POST',headers={
        'Content-Type':'audio/wav', 'X-Voice-Language':'en', 'X-Voice-Prompt':''})
    with opener.open(request,timeout=35) as response:
        if response.status!=200:
            raise RuntimeError('Resident Whisper returned a non-200 status')
        body=response.read(200_001)
    if len(body)>200_000:
        raise ValueError('Unbounded Whisper response')
    data=json.loads(body)
    text=data.get('text') if isinstance(data,dict) else None
    if (not isinstance(data,dict) or data.get('ok') is False
        or not isinstance(text,str) or len(text)>3000 or not text.strip()):
        raise ValueError('Invalid resident Whisper transcription')
    return ' '.join(text.split())


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    obj,wav,receipt_sha=inspect_completed_turn(args.receipt,args.audio)
    reply=obj['canonical_fresh_reply']['reply']
    base={'source_receipt_sha256':receipt_sha,'audio_sha256':sha(wav),
          'spoken_text_expected_sha256':sha(reply.encode()),
          'input_wav_bytes':len(wav), 'expected_words':len(words(reply)),
          'no_swaag_tts_or_tool_calls':True,
          'endpoint':'local_resident_Thor_Whisper_verify',
          'limits':'Not a physical phone, owner speaker or human hearing test'}
    if not args.run:
        print(json.dumps({'status':'preflight_only_no_service_contact',**base},sort_keys=True))
        return 0
    if os.environ.get(OPT_IN)!='1':
        raise ValueError('Real ASR test requires explicit environment opt-in')
    if args.evidence.is_symlink() or args.evidence.exists():
        raise FileExistsError('Evidence path already exists; refuse uncertain replay')
    args.evidence.mkdir(parents=True,mode=0o700,exist_ok=False)
    os.chmod(args.evidence,0o700)
    marker=args.evidence/'attempt.json'
    # Claim before outbound ASR. Never retry an uncertain external request.
    marker.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'external_request_limit':1,'source_receipt_sha256':receipt_sha,
        'audio_sha256':sha(wav),'retry':False}),encoding='utf-8')
    marker.chmod(0o600)
    start=time.monotonic()
    report={**base,'utc':datetime.now(timezone.utc).isoformat(),
            'actual_asr_requests_attempted':1}
    try:
        heard=actual_resident_asr(wav)
        exp,got=words(reply),words(heard)
        edits=word_edits(exp,got)
        report.update({'status':'verified' if edits==0 else 'content_mismatch',
          'actual_whisper_text':heard,'expected_canonical_text':reply,
          'whisper_word_count':len(got),'word_edit_distance':edits,
          'word_error_rate':round(edits/max(len(exp),1),4),
          'exact_normalized_word_match':edits==0})
    except Exception as exc:
        report.update({'status':'asr_error','error_type':type(exc).__name__})
    report['asr_seconds']=round(time.monotonic()-start,3)
    destination=args.evidence/'result.json'
    fd=os.open(destination,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())
    # Ordinary stdout exposes only source/score metadata, not private text.
    print(json.dumps({k:report[k] for k in ('status','audio_sha256',
          'word_error_rate','word_edit_distance','asr_seconds') if k in report}))
    return 0 if report['status']=='verified' else 2


if __name__=='__main__':
    sys.exit(main())
