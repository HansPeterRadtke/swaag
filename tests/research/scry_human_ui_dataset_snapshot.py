#!/usr/bin/env python3
"""Fetch pinned 311-pair human-reviewed mobile UI diff dataset outside Git.

The 16 shards are obtained by official Hugging Face dataset API metadata.
Bound download size and worker count; never overwrite existing shard files.
Screenshots/contents remain owned by original apps, DO NOT COMMIT them.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime,hashlib,json,urllib.request
ROOT=Path('/data/var/perceptual-independent-20261024/scry311')
API='https://huggingface.co/api/datasets/Scrymore/scry-design-diff-eval/tree/main?recursive=true'
BASE='https://huggingface.co/datasets/Scrymore/scry-design-diff-eval/resolve/main/'
MAX_SHARD=20_000_000
MAX_TOTAL=250_000_000

def fetch(x):
 name=x['path'];dst=ROOT/Path(name).name
 if dst.exists():
  if dst.stat().st_size!=x['size']:return {'path':name,'status':'size_mismatch_existing','size':dst.stat().st_size}
  return {'path':name,'status':'existing_verified_size','bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
 try:
  req=urllib.request.Request(BASE+name+'?download=true',headers={'User-Agent':'perceptual-scientific-evaluation/1.0'})
  with urllib.request.urlopen(req,timeout=35) as response:data=response.read(MAX_SHARD+1)
  if len(data)!=x['size'] or len(data)>MAX_SHARD:raise ValueError('declared/actual size mismatch or limit exceeded')
  if data[:4]!=b'PAR1' or data[-4:]!=b'PAR1':raise ValueError('not a parquet file')
  dst.write_bytes(data)
  return {'path':name,'status':'new_download_ok','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 except Exception as e:return {'path':name,'status':'error','error':type(e).__name__+': '+str(e)[:200]}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 req=urllib.request.Request(API,headers={'User-Agent':'perceptual-scientific-evaluation/1.0'})
 with urllib.request.urlopen(req,timeout=12) as response: files=json.load(response)
 shards=sorted((x for x in files if x['path'].startswith('data/test-') and x['path'].endswith('.parquet')),key=lambda x:x['path'])
 assert len(shards)==16 and all(x['size']<=MAX_SHARD for x in shards) and sum(x['size'] for x in shards)<=MAX_TOTAL
 receipts=[]
 with ThreadPoolExecutor(max_workers=3) as ex:
  jobs={ex.submit(fetch,x):x for x in shards}
  for job in as_completed(jobs):
   r=job.result();receipts.append(r)
   print('FILE',r['path'],r['status'],r.get('bytes'),flush=True)
 receipt={'dataset':'Scrymore/scry-design-diff-eval main observed 2026-10-24','source_API':API,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'num_shards':len(shards),'expected_total_bytes':sum(x['size'] for x in shards),'sources_not_relicensed_or_committed':True,'files':sorted(receipts,key=lambda x:x['path'])}
 (ROOT/'shard_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('COMPLETE_SHARDS',sum(r['status'] in ('new_download_ok','existing_verified_size') for r in receipts),'EXPECTED',len(shards),'BYTES',sum(r.get('bytes',0) for r in receipts))
if __name__=='__main__':main()
