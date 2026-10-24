#!/usr/bin/env python3
"""Frozen CPU patch-nearest-normal diagnostic on real MVTec LOCO pushpins JPEGs.

NOT official PatchCore: uses pretrained CLIP patch tokens, not WideResNet pooled
multiscale features or a coreset; no defect-label training. Compare to earlier
frozen global CLIP on exactly same four calibration normals and 14 test photos.
"""
from __future__ import annotations
import json,os,time,hashlib,resource
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-subset')
SOURCE=ROOT/'original_manifest.json'
MODEL='/data/models/rag_tests/vision/openai_clip-vit-base-patch32'
RESULT=ROOT/'clip_patch_memory_20261024.json'

def main():
 if RESULT.exists(): print('EXISTING COMPLETE FILE PRESERVED');return
 start=time.monotonic()
 import torch
 from transformers import CLIPModel,CLIPProcessor
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 model=CLIPModel.from_pretrained(MODEL,local_files_only=True).eval().cpu()
 proc=CLIPProcessor.from_pretrained(MODEL,local_files_only=True)
 manifest=json.loads(SOURCE.read_text())['cases']
 normals=sorted((r for r in manifest if r['gold']=='good'),key=lambda r:r['file']);cal=normals[:4]
 test=[r for r in manifest if r not in cal]
 assert len(test)==14 and sum(r['gold']=='anomaly' for r in test)==10
 vec={}
 with torch.no_grad():
  for row in manifest:
   image=Image.open(ROOT/row['file']).convert('RGB')
   inputs=proc(images=image,return_tensors='pt')
   hidden=model.vision_model(pixel_values=inputs.pixel_values).last_hidden_state[0,1:,:]
   v=torch.nn.functional.normalize(hidden,dim=-1).cpu().numpy().astype(np.float32)
   vec[row['file']]=v

 def score(v,bank):
  candidates=np.concatenate(bank,axis=0)
  d=1-np.max(v@candidates.T,axis=1)
  return float(np.percentile(d,95))
 # Threshold entirely from held-out-by-normal calibration leave-one-out.
 train=[score(vec[r['file']],[vec[q['file']] for q in cal if q['file']!=r['file']]) for r in cal]
 threshold=max(train)
 rows=[]
 for r in test:
  v=score(vec[r['file']],[vec[q['file']] for q in cal])
  rows.append({'source_image_sha256':r['sha256'],'gold_anomaly':r['gold']=='anomaly','score':round(v,8),'predicted_anomaly':v>threshold})
 tp=sum(r['gold_anomaly'] and r['predicted_anomaly'] for r in rows)
 fp=sum(not r['gold_anomaly'] and r['predicted_anomaly'] for r in rows)
 fn=sum(r['gold_anomaly'] and not r['predicted_anomaly'] for r in rows)
 tn=sum(not r['gold_anomaly'] and not r['predicted_anomaly'] for r in rows)
 from sklearn.metrics import roc_auc_score
 roc=float(roc_auc_score([r['gold_anomaly'] for r in rows],[r['score'] for r in rows]))
 out={'method':'CLIP vision last_hidden_state patch tokens 95th percentile one-class nearest normal; NOT PatchCore','model':'openai/clip-vit-base-patch32 CPU fp32','source':'18 real MVTec LOCO pushpins test JPEGs, 8 normal, 10 logical/structural anomalies; CC BY-NC-SA noncommercial','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'training_normal_images':[r['sha256'] for r in cal],'calibration_leave_one_out_distances':train,'threshold':threshold,'sample_n':len(rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'AUROC':roc,'rows':rows,'runtime_seconds':round(time.monotonic()-start,3),'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':'ONLY four normal samples for memory bank; not official PatchCore, highly domain-specific and correlated photos, no CAD physics gold. Results are not transferable to rendered mechanical objects.'}
 RESULT.write_text(json.dumps(out,indent=2)+'\n')
 print('RESULT',{'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':out['precision'],'recall':out['recall'],'AUROC':roc,'seconds':out['runtime_seconds']})
if __name__=='__main__':main()
