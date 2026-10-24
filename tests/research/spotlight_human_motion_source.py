#!/usr/bin/env python3
"""Externally annotated real generated-video error benchmark, bounded sample.

Source UBC-ViL Spotlight ECCV 2026, CC BY-NC 4.0. Select 4 prompt families
with >=2/3 motion-labeled generated versions and 4 with 0/3 motion labels.
Choose lexicographic SHA256(prompt_id) within each group (never visual feature
or threshold-driven selection). Require each of three source videos <=12 MB.
24 original MP4s stay ONLY in /data/var; never commit licensed source videos.
No release code execution, no user production workload, no model inference.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import collections,hashlib,json,urllib.request,datetime
ROOT=Path('/data/var/perceptual-independent-20261024/spotlight-videogen')
REPO='UBC-ViL/Spotlight-VideoGen-Errors'
PIN='4165ea17ece2fda7b6b0df52bcbf07348a31253e'
API='https://huggingface.co/api/datasets/'+REPO+'/tree/main/test/spotlight/'
BASE='https://huggingface.co/datasets/'+REPO+'/resolve/'+PIN+'/test/'
MANIFEST=ROOT/'motion_human_24_video_manifest.json'
GENERATORS=['ltx2','seedance','veo3']
MAX_BYTES=12_000_000

def index_gen(gen):
 req=urllib.request.Request(API+gen+'?recursive=false',headers={'User-Agent':'visual-scientific-evaluation/1.0'})
 with urllib.request.urlopen(req,timeout=16) as f:objects=json.load(f)
 assert len(objects)==200
 return gen,{x['path']:x['size'] for x in objects}

def acquire(meta):
 name=meta['source_file_name'];expected=meta['declared_bytes'];path=ROOT/name
 assert name.startswith('spotlight/') and name.endswith('.mp4') and expected<=MAX_BYTES
 try:
  if path.exists():
   if path.stat().st_size!=expected:return {'status':'existing_size_mismatch_not_overwritten','filename':name,'existing_bytes':path.stat().st_size}
   body=path.read_bytes();status='existing_not_redownloaded'
  else:
   path.parent.mkdir(parents=True,exist_ok=True)
   u=BASE+name+'?download=true';req=urllib.request.Request(u,headers={'User-Agent':'visual-scientific-evaluation/1.0'})
   with urllib.request.urlopen(req,timeout=40) as f:body=f.read(MAX_BYTES+1)
   if len(body)!=expected or body[4:8]!=b'ftyp':raise ValueError('size discrepancy or invalid MP4')
   path.write_bytes(body);status='downloaded'
  return {'status':status,'filename':name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'path':str(path)}
 except Exception as e:return {'status':'error','filename':name,'error':type(e).__name__+': '+str(e)[:200]}

def main():
 if MANIFEST.exists():
  d=json.loads(MANIFEST.read_text());print('PREVIOUS_COMPLETE_MANIFEST_NOT_REPEATED',d.get('video_count'));return
 ROOT.mkdir(exist_ok=True)
 src=ROOT/'metadata.jsonl';content=src.read_bytes();metadata=[json.loads(x) for x in content.decode().splitlines()]
 assert len(metadata)==600
 with ThreadPoolExecutor(max_workers=3) as ex: files=dict(ex.map(index_gen,GENERATORS))
 by_prompt=collections.defaultdict(dict)
 for item in metadata:
  gen=item['video_from'];path='test/'+item['file_name'];ann=json.loads(item['annotation'])
  by_prompt[item['pid']][gen]={'id':item['id'],'pid':item['pid'],'source_file_name':item['file_name'],'declared_bytes':files[gen][path],'motion_annotations':[x for x in ann if x['type']=='motion'],'other_annotations':[x for x in ann if x['type']!='motion'],'all_annotations_count':len(ann),'generator':gen}
 valid={pid:grp for pid,grp in by_prompt.items() if len(grp)==3 and all(grp[g]['declared_bytes']<=MAX_BYTES for g in GENERATORS)}
 order=lambda ids:sorted(ids,key=lambda p:hashlib.sha256(p.encode()).hexdigest())
 positive=order([pid for pid,grp in valid.items() if sum(bool(grp[g]['motion_annotations']) for g in GENERATORS)>=2])[:4]
 negative=order([pid for pid,grp in valid.items() if all(not grp[g]['motion_annotations'] for g in GENERATORS)])[:4]
 assert len(positive)==len(negative)==4 and not (set(positive)&set(negative))
 choices=[]
 for group,ids in [('motion_positive_prompt',positive),('no_motion_label_prompt',negative)]:
  for pid in ids:
   for gen in GENERATORS:
    row=by_prompt[pid][gen];choices.append({**row,'selection_group':group})
 assert len(choices)==24
 print('FROZEN_SELECTED_PROMPTS',positive,negative,'MOTION_VIDEOS',sum(bool(x['motion_annotations']) for x in choices),'ANNOTATION_EMPTY',sum(x['all_annotations_count']==0 for x in choices),'TOTAL_NOMINAL_MB',round(sum(x['declared_bytes'] for x in choices)/1e6,1),flush=True)
 receipts={}
 with ThreadPoolExecutor(max_workers=3) as ex:
  fut={ex.submit(acquire,x):x for x in choices}
  for f in as_completed(fut):
   item=fut[f];result=f.result();receipts[item['id']]=result
   print('VIDEO',len(receipts),'/',len(choices),item['generator'],item['pid'],result['status'],result.get('bytes'),flush=True)
 manifest={'dataset':REPO,'revision':PIN,'license':'cc-by-nc-4.0','date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_sha256':hashlib.sha256(content).hexdigest(),'video_count':len(choices),'selected_prompt_ids':{'motion_positive_prompt':positive,'no_motion_label_prompt':negative},'selection':'SHA256 prompt ID first four of >=2 of 3 motion-labeled and first four of 0 of 3 motion-labeled among prompt triples <=12MB per source. Balanced enrichment in annotated motion, NOT a population random sample. Non-motion annotation is not proof a clip is clean!','source_annotated_motion_video_count':sum(bool(x['motion_annotations']) for x in choices),'cases':[dict(x,**{'receipt':receipts[x['id']]}) for x in choices],'limitations':'Human region/time annotations are not pixel-level frame-freeze labels; no motion annotation DOES NOT mean physically correct or defect-free. Presence labels selected before feature computation; class prevalence is balanced artificially. Source datasets and videos not licensable for commercial use without review.'}
 MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
 print('COMPLETE_RECEIPTS',sum(x['status'] in ('downloaded','existing_not_redownloaded') for x in receipts.values()),'of',len(choices),'SOURCE_MB',round(sum(x.get('bytes',0) for x in receipts.values())/1e6,1))
if __name__=='__main__':main()
