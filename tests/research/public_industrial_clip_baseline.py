#!/usr/bin/env python3
"""Actual real industrial photographs vs frozen CLIP B/32, CPU only.

All image labels originate from MVTec LOCO AD externally, not generated.
Two zero-shot text prompts and the threshold are fixed before scoring.
No image names or labels are given to model. Normal-only reference test is separate.
"""
from __future__ import annotations
from pathlib import Path
import json,time,datetime,os,resource
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-subset')
MODEL='/data/models/rag_tests/vision/openai_clip-vit-base-patch32'
OUTPUT=ROOT/'clip_zeroshot_predictions.json'
PROMPTS=[
 'A photograph of a yellow organizer tray holding pushpins, with the correct number of pins in their correct compartments, no missing walls, a normal undamaged arrangement.',
 'A photograph of a yellow organizer tray holding pushpins, with a visibly wrong arrangement, missing pushpins, misplaced pins, or missing separating compartment walls, an abnormal damaged assembly.',
]

def main():
 import torch,numpy as np
 from transformers import CLIPModel,CLIPProcessor
 from PIL import Image
 from sklearn.metrics import roc_auc_score
 torch.set_num_threads(1);t0=time.monotonic()
 data=json.loads((ROOT/'original_manifest.json').read_text())['cases']
 model=CLIPModel.from_pretrained(MODEL,local_files_only=True).eval().cpu()
 processor=CLIPProcessor.from_pretrained(MODEL,local_files_only=True)
 with torch.no_grad():
  tok=processor(text=PROMPTS,return_tensors='pt',padding=True,truncation=True)
  text=model.get_text_features(**tok)
  text=torch.nn.functional.normalize(text,dim=-1)
  results=[];vectors={}
  for i,r in enumerate(data):
   start=time.monotonic()
   im=Image.open(ROOT/r['file']).convert('RGB')
   values=processor(images=im,return_tensors='pt')
   out=model.get_image_features(**values);out=torch.nn.functional.normalize(out,dim=-1)
   scores=(out@text.T)[0].cpu().numpy();prob=float(torch.softmax(torch.tensor(scores),dim=0)[1])
   vectors[r['file']]=out[0].cpu().numpy()
   results.append({'id':str(i),'image_sha256':r['sha256'],'seconds':round(time.monotonic()-start,3),'zero_shot_anomaly_score':round(prob,6),'zero_shot_predicted_anomaly':prob>=.5,'gold_anomaly':r['gold']=='anomaly'})
   print('IMAGE',i+1,'/',len(data),'SECONDS',round(time.monotonic()-start,2),flush=True)
  # Frozen reference subset: first 4 normal originals by numeric filename.
  normal=sorted([r for r in data if r['gold']=='good'],key=lambda x:x['file'])
  anchors=normal[:4];anchor_vec=[vectors[r['file']] for r in anchors]
  def nearest(v,others):return max(float(np.dot(v,x)) for x in others)
  train_leave_one=[1-nearest(vectors[r['file']],[vectors[q['file']] for q in anchors if q['file']!=r['file']]) for r in anchors]
  threshold=max(train_leave_one)
  test_names={r['file'] for r in data if r not in anchors}
  nearest_results=[]
  for r in data:
   if r['file'] not in test_names:continue
   score=1-nearest(vectors[r['file']],anchor_vec)
   nearest_results.append({'image_sha256':r['sha256'],'gold_anomaly':r['gold']=='anomaly','score':round(score,6),'predicted_anomaly':score>threshold})
 def score(rows,key):
  tp=sum(r['gold_anomaly'] and r[key] for r in rows)
  fp=sum(not r['gold_anomaly'] and r[key] for r in rows)
  fn=sum(r['gold_anomaly'] and not r[key] for r in rows)
  tn=sum(not r['gold_anomaly'] and not r[key] for r in rows)
  return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
 auc=float(roc_auc_score([r['gold_anomaly'] for r in results],[r['zero_shot_anomaly_score'] for r in results]))
 nearest_auc=float(roc_auc_score([r['gold_anomaly'] for r in nearest_results],[r['score'] for r in nearest_results]))
 output={'model':'openai/clip-vit-base-patch32 local torch fp32 CPU','image_set':'18 original-color real MVTec LOCO AD pushpins sample converted JPEG, CC BY-NC-SA noncommercial','source_sha_manifest':'original_manifest.json','prompts_predeclared':PROMPTS,'time_total_seconds':round(time.monotonic()-t0,2),'memory_kb_maxrss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'zero_shot':{'cases':results,'scores':score(results,'zero_shot_predicted_anomaly'),'AUROC':auc},'one_class_closest_normal':{'calibration_good_count':4,'held_out_good_count':4,'held_out_anomaly_count':10,'calibration_max_leave_one_distance':threshold,'cases':nearest_results,'scores':score(nearest_results,'predicted_anomaly'),'AUROC':nearest_auc},'caveats':'small public curated single-object benchmark subset, not machinery render realism; zero-shot prompt chosen before visual cases, no threshold tuning; normal reference trained only on four normal exemplars'}
 OUTPUT.write_text(json.dumps(output,indent=2)+'\n');print('ZERO_SHOT',output['zero_shot']['scores'],'AUC',auc);print('ONE_CLASS',output['one_class_closest_normal']['scores'],'AUC',nearest_auc);print('RUNTIME',output['time_total_seconds'])
if __name__=='__main__':main()
