"""Preserve an actually synthesized voice WAV in immutable private evidence.

Pure local evidence helper; no model, user actions or outbound requests.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import wave

def preserve_new_generated_audio(source: Path, destination: Path) -> dict:
    """Preserve the exact generated PCM/WAV before disposable fixture teardown.

    The private output directory must already exist and be owned by this test;
    never overwrite an existing result or symlink, even after an interrupted
    previous run. This is an evidence copy, not a new TTS/model invocation.
    """
    raw=source.read_bytes()
    if not 44<=len(raw)<=2_000_000 or not raw.startswith(b'RIFF'):
        raise ValueError('Generated WAV is missing, truncated or over its cap')
    try:
        with wave.open(io.BytesIO(raw),'rb') as stream:
            rate,channels,width,n=(stream.getframerate(),stream.getnchannels(),
                                   stream.getsampwidth(),stream.getnframes())
            pcm=stream.readframes(n)
    except (wave.Error,EOFError) as exc:
        raise ValueError('Generated response has an invalid WAV header') from exc
    if ((rate,channels,width)!=(16000,1,2) or n<=rate or n>16000*30
            or len(pcm)!=2*n or not any(pcm)):
        raise ValueError('Generated response is not bounded non-silent PCM16')
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError('Private evidence output root is unavailable')
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb') as output:
            output.write(raw);output.flush();os.fsync(output.fileno())
    except BaseException:
        # Retain even a partial copy after an uncertain failure for diagnosis.
        raise
    digest=hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Retained generated WAV is not byte-exact')
    return {'retained_wav_sha256':digest,'retained_wav_bytes':len(raw),
            'retained_pcm_bytes':len(pcm),'retained_wav_sample_rate':rate,
            'retained_wav_name':destination.name}
