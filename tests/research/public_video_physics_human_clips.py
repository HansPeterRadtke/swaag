#!/usr/bin/env python3
"""Download selected original real-vs-generated physics clips, bounded read-only.

Only 24 compact MP4s from 12 externally human-compared scenarios. No dataset
scripts or browser invocation, no production state, no video re-encode, and
never overwrite an existing clip. Third-party videos are NOT put in Git.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import urllib.request,urllib.parse,hashlib,json,datetime
ROOT=Path('/data/var/perceptual-independent-20261024/rapidata-physics-video')
SOURCES=ROOT/'frozen_human_pair_source_manifest.json'
MEDIA=ROOT/'clips';OUT=ROOT/'selected_human_pair_clip_receipts.json'
MAX_CLIP=5_000_000

def get(info,kind):
 u=info[kind+'_url'];parsed=urllib.parse.urlparse(u)
 if parsed.scheme!='https' or parsed.hostname!='assets.rapidata.ai' or not parsed.path.endswith('.mp4'):return {'status':'unsafe_url_refused','source_prompt_sha256':info['scenario_prompt_sha256'],'kind':kind}
 name=hashlib.sha256(u.encode()).hexdigest()[:22]+'.mp4';path=MEDIA/name
 try:
  if path.exists():data=path.read_bytes();status='existing_not_redownloaded'
  else:
   req=urllib.request.Request(u,headers={'User-Agent':'perceptual-research/1.0'})
   with urllib.request.urlopen(req,timeout=20) as f:data=f.read(MAX_CLIP+1)
   if len(data)>MAX_CLIP or len(data)<1000 or data[4:8]!=b'ftyp':raise ValueError('bad or oversize MP4')
   path.write_bytes(data);status='downloaded'
  return {'kind':kind,'prompt_sha256':info['scenario_prompt_sha256'],'source_url':u,'file':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':status}
 except Exception as e:return {'kind':kind,'prompt_sha256':info['scenario_prompt_sha256'],'status':'failed','error':type(e).__name__+': '+str(e)[:180]}

def main():
 if OUT.exists():print('EXISTING_OUTPUT_NOT_REPLAYED');return
 data=json.loads(SOURCES.read_text());assert len(data['samples'])==12
 MEDIA.mkdir(exist_ok=True)
 rec=[]
 with ThreadPoolExecutor(max_workers=3) as ex:
  calls=[ex.submit(get,x,k) for x in data['samples'] for k in ('ground_truth','generated')]
  for future in as_completed(calls):
   result=future.result();rec.append(result)
   print('VIDEO',len(rec),'/',len(calls),result['kind'],result['status'],result.get('bytes'),flush=True)
 ordered=sorted(rec,key=lambda x:(x.get('prompt_sha256',''),x['kind']))
 stat={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_sha256':hashlib.sha256(SOURCES.read_bytes()).hexdigest(),'model_pair':['Ground Truth','Cosmos 2.5'],'clips':ordered,'total_bytes':sum(x.get('bytes',0) for x in rec),'source':'Rapidata Physics-IQ public clip URLs, original user-generated video rights unchanged, no redistribution','clip_size_max_bytes':MAX_CLIP}
 OUT.write_text(json.dumps(stat,indent=2)+'\n')
 print('SOURCE_CLIPS',len(rec),'OK',sum(x['status'] in ('downloaded','existing_not_redownloaded') for x in rec),'TOTAL_MB',round(stat['total_bytes']/1000000,2))
if __name__=='__main__':main()
