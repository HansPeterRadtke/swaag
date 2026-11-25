#!/usr/bin/env python3
"""Prepare exact public WAVs and first causal decision points from W103.

No model inference, remote calls, private user speech or action authority.
The previously frozen pause corpus and actual source audio are reused
unchanged; the model family is new. One decision per eligible pause.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from prepare_public_natural_turn_corpus import FILES, sha_file, sha_bytes, validate_row
from run_p99_natural_turn_cpu_once import MANIFEST_SHA, step_endpoints, unique

ROOT=Path('/data/var/swaag-benchmarks/owner-serial-000104-20261125')
OLD=Path('/data/var/swaag-benchmarks/owner-serial-000103-20261125')


def points(row):
    result=[]
    for s in row['spans']:
        eligible=step_endpoints(s)
        result.append({**s,'decision_sample':eligible[0] if eligible else None})
    return result


def prepare(old,root):
    import pyarrow.parquet as pq
    if root.resolve()!=ROOT or root.is_symlink() or old.resolve()!=OLD or old.is_symlink():
        raise ValueError('only exact private roots')
    source=old/'corpus/manifest.json'
    if sha_file(source)!=MANIFEST_SHA:raise ValueError('old public manifest changed')
    rows=json.loads(source.read_text())['rows']
    expected={r['id']:r for r in rows}
    out=root/'input-wavs'
    if out.exists() or out.is_symlink():raise FileExistsError('do not replace input copies')
    out.mkdir(mode=0o700)
    manifest=[]
    for language,(name,digest,size,_) in FILES.items():
        p=old/name
        if p.stat().st_size!=size or sha_file(p)!=digest:raise ValueError('original shard drift')
        f=pq.ParquetFile(p)
        for batch in f.iter_batches(batch_size=8,columns=['id','audio','language','duration','cohort','silence_spans']):
            for r in batch.to_pylist():
                meta,raw=validate_row(r,language)
                gold=expected.pop(meta['id'])
                if meta!={k:v for k,v in gold.items() if k!='file'}:
                    raise ValueError('original row does not match W103')
                path=gold['file']
                if path!=sha_bytes(meta['id'].encode())+'.wav':
                    raise ValueError('unsafe filename')
                unique(out/path,raw)
                manifest.append({**gold,'spans':points(gold)})
    if expected or len(manifest)!=800:raise ValueError('missing input row')
    spans=[s for row in manifest for s in row['spans']]
    if len(spans)!=2357 or sum(s['decision_sample'] is not None for s in spans)!=1730:
        raise ValueError('wrong pause coverage')
    raw=json.dumps({'old_manifest_sha256':MANIFEST_SHA,'decision_rule':'first completed native 160ms chunk at least 200ms after annotated pause start, never after pause ends','rows':manifest},sort_keys=True,separators=(',',':')).encode()
    unique(root/'inputs.json',raw)
    print(json.dumps({'rows':len(manifest),'pauses':len(spans),'eligible':1730,'manifest_sha256':sha_bytes(raw)}))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--old',type=Path,default=OLD)
    a=p.parse_args();os.umask(0o077);prepare(a.old,a.root);return 0


if __name__=='__main__':raise SystemExit(main())
