#!/usr/bin/env python3
"""Independent original-Parquet provenance audit; no models or network.

Reconstruct every audio checksum, annotation, and causal text prefix from
the original publisher's immutable public shards. Raw text stays private.
Can verify the reproducibility of extracted WAV copies before their removal.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import wave

from verify_p99_natural_turn_evidence import CORPUS, check, digest, read
from verify_natural_turn_fusion_evidence import TEXT_SHA

PINNED={
 'de':('de.parquet',109307809,'82c6d47d218107023fecc175c92535b39c8f6d69f5d77715e9e6b27c1b2364e9'),
 'en':('en-first.parquet',109936364,'52bf7ff71e83d0214d744693839f8a6b14e78e0f26db20dfdb7f589453a50103'),
}


def sha_file(path):
    check(path.is_file() and not path.is_symlink(),'unsafe public source')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def gold_prefix(words,cutoff):
    def admissible(w):
        a,b=w.get('start'),w.get('end')
        return (type(a) in (int,float) and type(b) in (int,float) and
                math.isfinite(a) and math.isfinite(b) and 0<=a<=b
                and b*16000<=cutoff+1e-7 and isinstance(w.get('word'),str))
    return ' '.join(w['word'].strip() for w in itertools.takewhile(admissible,words)
                    if w['word'].strip())


def verify(root,verify_copies=False):
    import pyarrow.parquet as pq
    m=read(root,'corpus/manifest.json')
    t=read(root,'causal-text-manifest.json')
    check(digest(m)==CORPUS and digest(t)==TEXT_SHA,'corpus/word manifest drift')
    rows=json.loads(m)['rows']
    texts=json.loads(t)['cases']
    row_by_id={r['id']:r for r in rows}
    text_by_id={(x['id'],x['span_index']):x for x in texts}
    check(len(row_by_id)==len(rows)==800 and len(text_by_id)==len(texts)==1730,'duplicate ID')
    expected_names={'manifest.json'}
    extracted_bytes=0;checked_spans=0;checked_prefixes=0;checked_rows=0
    for lang,(name,size,h) in PINNED.items():
        p=root/name
        check(p.stat().st_size==size and sha_file(p)==h,'public shard changed')
        f=pq.ParquetFile(p)
        check(f.metadata.num_rows==400,'wrong public stratum size')
        for batch in f.iter_batches(batch_size=8,columns=['id','audio','language','duration',
                                                        'silence_spans','words','cohort']):
            for original in batch.to_pylist():
                row=row_by_id.pop(original['id'])
                raw=original['audio']['bytes']
                with wave.open(io.BytesIO(raw),'rb') as w:
                    frames=w.getnframes()
                    check((w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
                          and len(w.readframes(frames))==2*frames,'invalid original audio')
                check(row['language']==original['language']==lang and original['cohort']=='general'
                      and row['frames']==frames and abs(original['duration']-frames/16000)<=.002
                      and row['wav_sha256']==digest(raw),'wrong original audio identity')
                n=digest(original['id'].encode())+'.wav'
                check(row['file']==n,'wrong extracted file')
                expected_names.add(n)
                if verify_copies:
                    check(read(root,'corpus/'+n,10_000_000)==raw,'extracted audio differs')
                    extracted_bytes+=len(raw)
                check(len(row['spans'])==len(original['silence_spans']),'missing original pause')
                for i,(g,s) in enumerate(zip(original['silence_spans'],row['spans'])):
                    start,end=round(g['start']*16000),min(frames,round(g['end']*16000))
                    check(s=={'index':i,'start_sample':start,'end_sample':end,'label':g['label']},
                          'annotation changed')
                    checked_spans+=1
                    # Full native chunks only; no zero padding into future.
                    eligible=any(start+3200<=end_sample<=end
                                 for end_sample in range(2560,frames+1,2560))
                    if eligible:
                        text=text_by_id.pop((row['id'],i))
                        prefix=gold_prefix(original['words'],start)
                        check(text=={'id':row['id'],'language':lang,'span_index':i,
                              'label':g['label'],'cutoff_sample':start,'text':prefix,
                              'text_sha256':digest(prefix.encode()),'has_prefix':bool(prefix)},
                              'causal prefix differs from original words')
                        checked_prefixes+=1
                checked_rows+=1
    check(not row_by_id and not text_by_id,'missing/extra source case')
    if verify_copies:
        check({p.name for p in (root/'corpus').iterdir()}==expected_names,
              'unknown extracted files; cleanup not safe')
    return {'status':'independent_original_public_parquet_source_verified',
            'rows':checked_rows,'labeled_pauses':checked_spans,
            'eligible_causal_prefixes':checked_prefixes,
            'all_extracted_waveforms_match_original':verify_copies,
            'reproducible_extracted_bytes':extracted_bytes,
            'no_model_or_network_calls':True,'release_authorized':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify-copies',action='store_true')
    a=p.parse_args();print(json.dumps(verify(a.root,a.verify_copies),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
