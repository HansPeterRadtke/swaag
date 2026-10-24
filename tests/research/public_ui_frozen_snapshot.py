#!/usr/bin/env python3
"""Acquire only independently labeled REAL-page UIJudgeBench v0.4.0 snapshots.

No train/model work. Reports are in /data/var and cannot contaminate project's Git.
Does not repeat completed images, checks byte hashes. No untrusted code execution.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import urllib.request, json, hashlib, datetime
ROOT=Path('/data/var/perceptual-independent-20261024/uijudge-v0.4.0')
LABELS=ROOT/'items.jsonl'; OUT=ROOT/'images';MANIFEST=ROOT/'real_ui_frozen_manifest.json'
BASE='https://huggingface.co/datasets/gojiberries/uijudge-bench/resolve/v0.4.0/corpus/real/'
def candidates():
 rows=[json.loads(s) for s in LABELS.read_text().splitlines() if s]
 cases=[r for r in rows if r.get('task_level')=='L1' and r.get('page_id','').startswith('real-') and ((r.get('track')=='layout') or (r.get('criterion_code')=='wcag:1.4.3')) and r.get('ground_truth') in ('yes','no')]
 # Include entire externally frozen train/dev and test natural-page split; not a handpicked subset.
 return cases
def dl(page):
 file=OUT/(page+'.png');u=BASE+page+'/screenshot_desktop.png?download=true'
 try:
  if not file.is_file():
   req=urllib.request.Request(u,headers={'User-Agent':'perceptual-eval-readonly/1.0'})
   with urllib.request.urlopen(req,timeout=12) as response:
    declared=response.headers.get('Content-Length')
    data=response.read(3000001)
   if len(data)>3000000:raise ValueError('snapshot exceeds 3 MB cap')
   if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('non PNG response')
   file.write_bytes(data)
  data=file.read_bytes();return page,dict(path=str(file),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return page,dict(error=type(e).__name__+': '+str(e)[:160])
def main():
 OUT.mkdir(parents=True,exist_ok=True);cases=candidates();pages=sorted({r['page_id'] for r in cases}); prior={}
 if MANIFEST.is_file():prior=json.loads(MANIFEST.read_text()).get('pages',{})
 # 3 simultaneous small public GETs; no GUI/model/GPU/production activity.
 with ThreadPoolExecutor(max_workers=3) as pool:
  jobs={pool.submit(dl,p):p for p in pages}
  for task in as_completed(jobs):
   k,v=task.result();prior[k]=v
 report={'source':'gojiberries/uijudge-bench revision v0.4.0','ground_truth_is_browser_verified_not_human':True,'filter':'public real page L1 layout and WCAG 1.4.3 contrast; both dev and test; no mutation-based selection','date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'items':cases,'pages':prior}
 MANIFEST.write_text(json.dumps(report,indent=2)+'\n')
 print('ITEMS',len(cases),'PAGES',len(pages),'READY',sum('sha256' in v for v in prior.values()),'FAILED',[k for k,v in prior.items() if 'error' in v][:12],flush=True)
 if len(prior)!=len(pages):raise SystemExit(2)
if __name__=='__main__':main()
