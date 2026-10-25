#!/usr/bin/env python3
"""Acquire all 129 ORIGINAL human-judged golden GLBs for external evaluation.

No model inference, no Blender or GPU, no file export to Git. Source author
labels already frozen by 3D-DefectBench; 10 previously used examples are left
untouched and will be excluded from heldout metric. Each downloaded glTF must
match upstream byte size and magic, cached files never overwritten. Results
stay under /data/var and retain source SHA and CC BY-NC provenance.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import urllib.request,json,csv,hashlib,datetime,collections
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench')
DEST=ROOT/'golden-glbs'
OUT=ROOT/'all129_glb_source_manifest_20261025.json'
REPO='zzhao0500/3D-DefectBench';API='https://huggingface.co/api/datasets/'+REPO
PREVIOUS_IDS={'81','82','87','90','878','879','992','993','1094','1155'}

def download(src,rev):
 name=src['path'];fname=name.split('/')[-1];p=DEST/fname
 try:
  if p.is_file():
   data=p.read_bytes()
   if len(data)!=src['size'] or data[:4]!=b'glTF':return name,{'status':'existing_mismatch_NOT_overwritten','bytes':len(data),'expected_bytes':src['size']}
   result='preexisting_verified_no_download'
  else:
   url=f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{name}?download=true'
   req=urllib.request.Request(url,headers={'User-Agent':'visual-research/1.0'})
   with urllib.request.urlopen(req,timeout=36) as f:data=f.read(8_000_001)
   if len(data)!=src['size'] or len(data)>8_000_000 or data[:4]!=b'glTF':raise RuntimeError('size cap or invalid source glb')
   p.write_bytes(data);result='new_download_verified'
  return name,{'status':result,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'path':str(p)}
 except Exception as e:return name,{'status':'download_or_source_error','error':type(e).__name__+': '+str(e)[:190]}

def main():
 if OUT.is_file():print('EXISTING_COMPLETED_MANIFEST_PRESERVED');return
 label_bytes=(ROOT/'golden_labels.csv').read_bytes()
 labels=list(csv.DictReader(label_bytes.decode().splitlines()))
 ids={r['object_id'] for r in labels}
 assert len(ids)==len(labels)==129
 with urllib.request.urlopen(API,timeout=10) as f:d=json.load(f)
 pin=d['sha'];assert d.get('gated') is False and pin
 with urllib.request.urlopen(API+'/tree/main/glb/golden?recursive=false',timeout=12) as f:tree=json.load(f)
 listing={x['path']:x for x in tree if x['path'].endswith('.glb')}
 assert len(listing)==129 and {Path(x).stem for x in listing}==ids
 assert max(x['size'] for x in listing.values())<8_000_000
 DEST.mkdir(exist_ok=True)
 done={}
 with ThreadPoolExecutor(max_workers=3) as ex:
  jobs={ex.submit(download,v,pin):key for key,v in listing.items()}
  for fut in as_completed(jobs):
   key,result=fut.result();done[key]=result
   if len(done)%15==0 or result['status']=='download_or_source_error':
    print('GOLDEN_SOURCE',len(done),'/',len(listing),'status',result['status'],'bytes',result.get('bytes'),flush=True)
 table={}
 for record in labels:
  ident=record['object_id'];item=listing[f'glb/golden/{ident}.glb'];receipt=done[item['path']]
  table[ident]={'label_source_expert_votes':record['num_labels'],'gold_missing_parts_majority':int(record['q_missing_parts_majority_vote']),'gold_missing_parts_agreement':float(record['q_missing_parts_agreement_rate']),'gold_fused_incomplete':int(record['q_fused_incomplete_majority_vote']),'gold_surface_form':int(record['q_form_surface_majority_vote']),'previously_selected_before_this_holdout':ident in PREVIOUS_IDS,'source':receipt}
 out={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'https://huggingface.co/datasets/'+REPO,'source_revision':pin,'license':'CC BY-NC 4.0 noncommercial original generated meshes','gold_labels_sha256':hashlib.sha256(label_bytes).hexdigest(),'selection':'ALL 129 golden human-majority assets, no cherry picking. Previous fixed-rule pilot IDs omitted from heldout metrics. No parameter tuning after previous 10 examples.','total_cases':len(table),'holdout_n_not_previously_selected':sum(not r['previously_selected_before_this_holdout'] for r in table.values()),'labels':table,'all_source_content_verified':all(r['source']['status'] in ('preexisting_verified_no_download','new_download_verified') for r in table.values()),'total_source_bytes':sum(r['source'].get('bytes',0) for r in table.values())}
 OUT.write_text(json.dumps(out,indent=2)+'\n')
 print('GOLDEN_COMPLETE',out['total_cases'],'NEW_HELDOUT_N',out['holdout_n_not_previously_selected'],'TOTAL_MB',round(out['total_source_bytes']/1e6,1),'ALL_OK',out['all_source_content_verified'],'HOLDOUT_POS',sum(r['gold_missing_parts_majority'] for r in table.values() if not r['previously_selected_before_this_holdout']))
if __name__=='__main__':main()
