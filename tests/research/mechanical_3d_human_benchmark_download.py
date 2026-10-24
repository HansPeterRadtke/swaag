#!/usr/bin/env python3
"""Frozen narrow human-labeled 3D defectbench golden mechanical slice.

Select fixed original asset IDs with source-human q_missing_parts annotations;
never write or commit licensed GLBs. Labels never fed into vision predictor.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,json,hashlib,urllib.request,datetime
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench')
MODEL='zzhao0500/3D-DefectBench'
BASE='https://huggingface.co/datasets/'+MODEL+'/resolve/main/glb/golden/'
IDS=['81','82','87','90','878','879','992','993','1094','1155']

def download(i):
 p=ROOT/'golden-glbs'/(i+'.glb')
 try:
  if not p.exists():
   with urllib.request.urlopen(urllib.request.Request(BASE+i+'.glb?download=true',headers={'User-Agent':'perceptual-scientific-research/1.0'}),timeout=22) as f:data=f.read(8000000)
   if len(data)>7500000 or not data.startswith(b'glTF'):raise ValueError('unexpected GLB or >7.5MB cap')
   p.write_bytes(data)
  return i,{'filename':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'available'}
 except Exception as e:return i,{'filename':p.name,'status':'error','error':type(e).__name__+': '+str(e)[:130]}

def main():
 source=list(csv.DictReader((ROOT/'golden_labels.csv').open()))
 annotated={r['object_id']:r for r in source}
 assert all(i in annotated for i in IDS)
 (ROOT/'golden-glbs').mkdir(exist_ok=True)
 values={}
 with ThreadPoolExecutor(max_workers=3) as pool:
  jobs=[pool.submit(download,i) for i in IDS]
  for job in as_completed(jobs):
   i,result=job.result();values[i]=result
   print('GOLDEN',i,result['status'],result.get('bytes'),flush=True)
 cases=[]
 for i in IDS:
  label=annotated[i]
  case={'object_id':i,'prompt':label['prompt'],'reference_model_version':label['model_version'],'num_independent_human_votes':int(label['num_labels']),'gold_missing_parts':int(label['q_missing_parts_majority_vote']),'gold_geometry_form':int(label['q_form_surface_majority_vote']),'gold_fused_incomplete':int(label['q_fused_incomplete_majority_vote']),'agreement_missing_parts':float(label['q_missing_parts_agreement_rate']),'file':values[i]}
  cases.append(case)
 report={'source':'3D-DefectBench CC-BY-NC 4.0 golden human majority labels, public 3D assets','source_labels_sha256':hashlib.sha256((ROOT/'golden_labels.csv').read_bytes()).hexdigest(),'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'Fixed ten mechanical/machinery-like objects chosen to include both human-judged missing parts and human-judged non-missing, not random, not class-representative; source labels available publicly (possible training contamination).','cases':cases,'status':'source_only_no_model_accuracy'}
 (ROOT/'mechanical_golden_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COUNT',len(cases),'READY',sum(x['file']['status']=='available' for x in cases),'HUMAN_MISSING_YES',sum(x['gold_missing_parts'] for x in cases),'NO',sum(not x['gold_missing_parts'] for x in cases))
if __name__=='__main__':main()
