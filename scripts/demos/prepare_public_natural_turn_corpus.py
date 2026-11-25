#!/usr/bin/env python3
"""Extract the exact pinned public, labeled English/German general validation rows.

Only public audio, row identity, time spans and labels enter the private
evaluation directory. No transcripts, account data, models or network calls.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import wave

DATA_REVISION='ae834af9e9bddf73d1f550c190aaeb99bf31f523'
DATASET='livekit/eot-bench-data'
FILES={
 'de':('de.parquet','82c6d47d218107023fecc175c92535b39c8f6d69f5d77715e9e6b27c1b2364e9',109307809,707),
 'en':('en-first.parquet','52bf7ff71e83d0214d744693839f8a6b14e78e0f26db20dfdb7f589453a50103',109936364,850),
}
PRIVATE_PREFIX='/data/var/swaag-benchmarks/owner-serial-000103-20261125/'


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    if p.is_symlink() or not p.is_file():
        raise ValueError('unsafe source file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):
            h.update(b)
    return h.hexdigest()


def validate_row(r,language):
    # Use the publisher's explicit labels, never infer from a final gap
    # or from a transcript. Validate every boundary against actual samples.
    if r['language']!=language or r['cohort']!='general':
        raise ValueError('wrong corpus stratum')
    ident=r['id']
    if not isinstance(ident,str) or not ident or len(ident)>160:
        raise ValueError('invalid row ID')
    raw=r['audio']['bytes']
    if not isinstance(raw,bytes) or not 44<len(raw)<10_000_000:
        raise ValueError('invalid audio bytes')
    with wave.open(io.BytesIO(raw),'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):
            raise ValueError('unexpected native audio format')
        frames=w.getnframes()
        if len(w.readframes(frames))!=frames*2:
            raise ValueError('truncated PCM')
    if not math.isfinite(r['duration']) or abs(r['duration']-frames/16000)>0.002:
        raise ValueError('duration differs from actual audio')
    spans=r['silence_spans']
    if not isinstance(spans,list) or not spans or len(spans)>100:
        raise ValueError('invalid annotated spans')
    result=[]
    prev=0
    for i,s in enumerate(spans):
        a,b=s['start'],s['end']
        if (not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or
                not math.isfinite(a) or not math.isfinite(b) or a<prev-1e-6 or
                b-a<0.099999 or b>frames/16000+0.002):
            raise ValueError('invalid pause boundary')
        if s['label']!=('eot' if i==len(spans)-1 else 'hold'):
            raise ValueError('unexpected source pause label')
        start=round(a*16000)
        end=min(frames,round(b*16000))
        if end<=start:
            raise ValueError('empty pause')
        result.append({'index':i,'start_sample':start,'end_sample':end,'label':s['label']})
        prev=b
    return {'id':ident,'language':language,'frames':frames,'wav_sha256':sha_bytes(raw),
            'spans':result},raw


def prepare(source,out):
    import pyarrow.parquet as pq
    if source.is_symlink() or out.exists() or out.is_symlink():
        raise ValueError('new private corpus required')
    if not str(out.resolve()).startswith(PRIVATE_PREFIX):
        raise ValueError('private output root required')
    for _,(name,digest,size,_) in FILES.items():
        p=source/name
        if p.stat().st_size!=size or sha_file(p)!=digest:
            raise ValueError('pinned parquet mismatch')
    out.mkdir(mode=0o700)
    rows=[]
    seen=set()
    for lang,(name,_,_,expected_hold) in FILES.items():
        f=pq.ParquetFile(source/name)
        if f.metadata.num_rows!=400:
            raise ValueError('unexpected general validation row count')
        group=[]
        for batch in f.iter_batches(batch_size=8,columns=['id','audio','language','duration','cohort','silence_spans']):
            for row in batch.to_pylist():
                meta,raw=validate_row(row,lang)
                if meta['id'] in seen:
                    raise ValueError('duplicate public turn ID')
                seen.add(meta['id'])
                # Derive the local file name, not a publisher-controlled path.
                name=sha_bytes(meta['id'].encode())+'.wav'
                meta['file']=name
                with (out/name).open('xb') as file:file.write(raw)
                group.append(meta)
        if len(group)!=400 or sum(s['label']=='hold' for r in group for s in r['spans'])!=expected_hold:
            raise ValueError('label count differs from exact source')
        rows.extend(group)
    manifest={'dataset':DATASET,'revision':DATA_REVISION,'license':'CC-BY-4.0',
              'scope':'all 400 general de and all 400 general en rows in pinned shards',
              'source_files':FILES,'rows':rows}
    data=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    with (out/'manifest.json').open('xb') as f:f.write(data)
    print(json.dumps({'rows':len(rows),'holds':sum(s['label']=='hold' for r in rows for s in r['spans']),
           'manifest_sha256':sha_bytes(data),'source_files_verified':True,'no_model_calls':True}))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.umask(0o077);prepare(a.source,a.output)
    return 0

if __name__=='__main__':raise SystemExit(main())
