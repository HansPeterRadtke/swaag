#!/usr/bin/env python3
"""New prompt-disjoint 72 ORIGINAL video source sample, fixed previous config.

24 distinct prompts disjoint from all 8 previously consumed pilot prompt IDs:
12 prompts with human motion error in >=2 of 3 generated models; 12 with no
motion error label in any of 3. Choose by SHA256 prompt ID sort, not image
features / score / difficulty; source generation families LTX2, Seedance, Veo3.
All original clips <=12MB; size checks and SHA256. No video modified.
External data CC BY-NC 4.0 stays outside Git; motion-unlabeled is not clean.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import collections,hashlib,json,urllib.request,datetime
BASE=Path('/data/var/perceptual-independent-20261024/spotlight-videogen')
OUT=BASE/'holdout72_prompt_disjoint_source_20261025.json'
INDEX='https://huggingface.co/api/datasets/UBC-ViL/Spotlight-VideoGen-Errors/tree/main/test/spotlight/'
DOWNLOAD='https://huggingface.co/datasets/UBC-ViL/Spotlight-VideoGen-Errors/resolve/4165ea17ece2fda7b6b0df52bcbf07348a31253e/test/'
GENERATORS=('ltx2','seedance','veo3')
MAX_BYTES=12_000_000

def indexed(gen):
 with urllib.request.urlopen(urllib.request.Request(INDEX+gen+'?recursive=false',headers={'User-Agent':'visual-research/1.0'}),timeout=12) as f:d=json.load(f)
 assert len(d)==200
 return gen,{x['path']:x['size'] for x in d}

def dl(row):
 fname=row['source_file'];path=BASE/fname
 try:
  if path.exists():
   if path.stat().st_size!=row['declared_bytes']:
    return row['id'],{'status':'existing_size_mismatch','path':str(path),'actual_size':path.stat().st_size}
   data=path.read_bytes();status='previously_present_not_redownloaded'
  else:
   path.parent.mkdir(parents=True,exist_ok=True)
   with urllib.request.urlopen(urllib.request.Request(DOWNLOAD+fname+'?download=true',headers={'User-Agent':'visual-research/1.0'}),timeout=40) as resp:data=resp.read(MAX_BYTES+1)
   if len(data)!=row['declared_bytes'] or len(data)>MAX_BYTES or data[4:8]!=b'ftyp':raise ValueError('invalid video size or header')
   path.write_bytes(data);status='downloaded'
  return row['id'],{'status':status,'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 except Exception as e:return row['id'],{'status':'error','reason':type(e).__name__+': '+str(e)[:170]}

def main():
 if OUT.is_file():print('EXISTING_COMPLETE_MANIFEST_NOT_REUSED',OUT);return
 original=(BASE/'metadata.jsonl').read_bytes();a=[json.loads(s) for s in original.decode().splitlines() if s]
 prior=json.loads((BASE/'motion_human_24_video_manifest.json').read_text())
 consumed=set(prior['selected_prompt_ids']['motion_positive_prompt']+prior['selected_prompt_ids']['no_motion_label_prompt'])
 assert len(consumed)==8 and len(a)==600
 with ThreadPoolExecutor(max_workers=3) as pool: file_sizes=dict(pool.map(indexed,GENERATORS))
 by=collections.defaultdict(dict)
 for row in a:
  g=row['video_from'];src='test/'+row['file_name'];ann=json.loads(row['annotation']);size=file_sizes[g][src]
  by[row['pid']][g]={'id':row['id'],'prompt_id':row['pid'],'generator':g,'source_file':row['file_name'],'declared_bytes':size,'motion_annotations':[z for z in ann if z['type']=='motion'],'other_annotation_types':sorted({z['type'] for z in ann if z['type']!='motion'}),'all_annotation_count':len(ann)}
 eligible={pid:grp for pid,grp in by.items() if pid not in consumed and len(grp)==3 and all(grp[g]['declared_bytes']<=MAX_BYTES for g in GENERATORS)}
 def select(condition):return sorted([k for k,v in eligible.items() if condition(v)],key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:12]
 pos=select(lambda v:sum(bool(v[g]['motion_annotations']) for g in GENERATORS)>=2)
 unlabeled=select(lambda v:all(not v[g]['motion_annotations'] for g in GENERATORS))
 assert len(pos)==len(unlabeled)==12 and not set(pos)&set(unlabeled)
 selected=[{**by[pid][g],'selection_group':'motion_label_enriched' if pid in pos else 'no_motion_label_group'} for pid in pos+unlabeled for g in GENERATORS]
 assert len(selected)==72
 print('SELECTED',len(selected),'PROMPTS',len(set(x['prompt_id'] for x in selected)),'POS_VIDEOS',sum(bool(x['motion_annotations']) for x in selected),'TOTAL_NOMINAL_MB',round(sum(x['declared_bytes'] for x in selected)/1e6,1),'OLDER_PROMPT_OVERLAP',len(set(pos+unlabeled)&consumed),flush=True)
 receipts={}
 with ThreadPoolExecutor(max_workers=3) as pool:
  tasks={pool.submit(dl,row):row['id'] for row in selected}
  for f in as_completed(tasks):
   key,receipt=f.result();receipts[key]=receipt
   print('SOURCE',len(receipts),'/',len(selected),'ID',key,receipt['status'],'bytes',receipt.get('bytes'),flush=True)
 result={'source':'UBC-ViL/Spotlight-VideoGen-Errors pinned original human annotations CC-BY-NC-4.0','revision':'4165ea17ece2fda7b6b0df52bcbf07348a31253e','metadata_sha256':hashlib.sha256(original).hexdigest(),'excluded_prior_prompt_ids':sorted(consumed),'selected_prompt_ids':{'motion_label_enriched':pos,'no_motion_label_group':unlabeled},'selection':'12 SHA256 prompt IDs with >=2 motion-labeled of 3 and 12 SHA256 IDs with zero motion-labeled of 3, disjoint from earlier 8 prompts; size cap 12MB per video; models ltx2, seedance, veo3. No model features used in selection. No-motion-label NOT independently clean.', 'items':[dict(row,receipt=receipts[row['id']]) for row in selected],'source_file_total_bytes':sum(x.get('bytes',0) for x in receipts.values()),'n':len(selected),'all_completed':all(r['status'] in ('downloaded','previously_present_not_redownloaded') for r in receipts.values()),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('CHECKPOINT',OUT,'COMPLETE',result['all_completed'],'SOURCE_BYTES',result['source_file_total_bytes'],'ERRORS',sum(r['status']=='error' for r in receipts.values()))
if __name__=='__main__':main()
