#!/usr/bin/env python3
"""Prepare causal transcript prefixes from the pinned, public annotated corpus.

This supplies ideal word-aligned text, not measured recognition output.
Only local public dataset words ending by each labeled pause start enter
the prefix. Prior conversation and later words never enter model input.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from prepare_public_natural_turn_corpus import FILES, PRIVATE_PREFIX, sha_file, sha_bytes
from run_p99_natural_turn_cpu_once import MANIFEST_SHA, step_endpoints, unique


def prefix(words, start_sample):
    if not isinstance(words,list):
        raise ValueError("invalid aligned words")
    parts=[]
    # Only consume a contiguous prefix. If a word crosses the cutoff,
    # or its timing is invalid, do not peek at later words.
    for w in words:
        a,b=w.get('start'),w.get('end')
        if (type(a) not in (float,int) or type(b) not in (float,int)
                or not math.isfinite(a) or not math.isfinite(b)
                or a<0 or b<a or b*16000>start_sample+1e-7):
            break
        t=w.get('word')
        if not isinstance(t,str):
            break
        if t.strip():parts.append(t.strip())
    return ' '.join(parts)


def prepare(root,out):
    import pyarrow.parquet as pq
    if out.exists() or out.is_symlink() or not str(out.resolve()).startswith(PRIVATE_PREFIX):
        raise ValueError('fresh private transcript manifest required')
    source=root/'corpus/manifest.json'
    if sha_file(source)!=MANIFEST_SHA:raise ValueError('corpus manifest drift')
    rows=json.loads(source.read_text())['rows']
    byid={r['id']:r for r in rows}
    cases=[]
    for lang,(name,sha,size,_) in FILES.items():
        path=root/name
        if path.stat().st_size!=size or sha_file(path)!=sha:raise ValueError('parquet drift')
        f=pq.ParquetFile(path)
        for batch in f.iter_batches(batch_size=16,columns=['id','language','cohort','words','silence_spans']):
            for r in batch.to_pylist():
                row=byid.pop(r['id'])
                if row['language']!=lang or r['language']!=lang or r['cohort']!='general':
                    raise ValueError('wrong source stratum')
                if len(row['spans'])!=len(r['silence_spans']):raise ValueError('pause count mismatch')
                for s,g in zip(row['spans'],r['silence_spans']):
                    if (s['label']!=g['label'] or s['start_sample']!=round(g['start']*16000)
                            or s['end_sample']!=min(row['frames'],round(g['end']*16000))):
                        raise ValueError('annotation mismatch')
                    if not step_endpoints(s):continue
                    text=prefix(r['words'],s['start_sample'])
                    cases.append({'id':r['id'],'language':lang,'span_index':s['index'],'label':s['label'],
                        'cutoff_sample':s['start_sample'],'text':text,'text_sha256':sha_bytes(text.encode()),
                        'has_prefix':bool(text)})
    if byid:raise ValueError('missing source rows')
    if len(cases)!=len({(c['id'],c['span_index']) for c in cases}):
        raise ValueError('duplicate pause')
    raw=json.dumps({'source_manifest_sha256':MANIFEST_SHA,
        'text_source':'public word-aligned ideal transcript prefix at pause start',
        'no_current_model_or_phone':True,'cases':cases},
        sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    unique(out,raw)
    print(json.dumps({'cases':len(cases),'empty_prefix':sum(not c['has_prefix'] for c in cases),
        'manifest_sha256':sha_bytes(raw),'no_inference':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.umask(0o077);prepare(a.root,a.output);return 0

if __name__=='__main__':raise SystemExit(main())
