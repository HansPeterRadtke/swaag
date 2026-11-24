#!/usr/bin/env python3
"""Make EXACT W73 new real public recorded-speech reconnect test WAV.

Read one already existing SHA-pinned PUBLIC LibriSpeech recording; append
exactly 600ms of digital silence in private storage. The original recorded
WAV is never modified or downloaded. Source is public recorded speech, not
owner microphone audio. Dry-run computes bytes entirely in memory, never
writes. Full creation requires distinct unused private project output path;
no models, phone, provider or SWAAG agent tools are invoked.
"""
from __future__ import annotations
import argparse,hashlib,io,os,wave
from pathlib import Path

ORIGINAL_SOURCE_SHA='473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1'
EXPECTED_NEW_AUDIO_SHA='03cedb42d9dabeab954793edea1b823f2bd9687df6af683a262cb09816c30d25'
SILENCE_SAMPLES=9600
DEST_BASENAME='w73-new-third-public-recital-padded.wav'


def make(source:Path)->bytes:
    if not source.is_file() or source.is_symlink():raise ValueError('Missing/gone public recording')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=ORIGINAL_SOURCE_SHA:
        raise ValueError('Previously reviewed PUBLIC source recording changed')
    with wave.open(str(source),'rb') as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,16000):
            raise ValueError('Original recording not mono 16 kHz 16-bit PCM')
        frames=f.readframes(f.getnframes())
        if not 4.5<f.getnframes()/16000<5.2:
            raise ValueError('Original WAV source duration is not the expected 4.815 seconds')
    out=io.BytesIO()
    with wave.open(out,'wb') as target:
        target.setnchannels(1);target.setsampwidth(2);target.setframerate(16000)
        target.writeframes(frames+b'\0\0'*SILENCE_SAMPLES)
    result=out.getvalue()
    if hashlib.sha256(result).hexdigest()!=EXPECTED_NEW_AUDIO_SHA:
        raise ValueError('The original new third-attempt WAV construction cannot be reproduced byte-for-byte')
    return result


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--public-wav',type=Path,required=True)
    a.add_argument('--private-output',type=Path,required=True)
    a.add_argument('--run',action='store_true')
    x=a.parse_args(argv)
    b=make(x.public_wav)
    if not x.run:
        print('REPRODUCIBLE_PUBLIC_WAV_DRYRUN_NO_WRITE',EXPECTED_NEW_AUDIO_SHA,len(b))
        return 0
    out=x.private_output
    if (not out.is_absolute() or not str(out).startswith('/data/var/swaag-benchmarks/')
        or out.name!=DEST_BASENAME or out.exists() or out.is_symlink()):
        raise FileExistsError('Never overwrite or replay an original real recorded speech test input')
    fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert hashlib.sha256(out.read_bytes()).hexdigest()==EXPECTED_NEW_AUDIO_SHA
    print('NEW_ONLY_PUBLIC_WAV_CREATED_SHA_PINNED',out,EXPECTED_NEW_AUDIO_SHA)
    return 0

if __name__=='__main__':raise SystemExit(main())
