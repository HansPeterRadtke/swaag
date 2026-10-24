#!/usr/bin/env python3
"""Blind image-only CLIP baseline on externally judged real website target crops.

Freeze two image/text prompts before observing any case labels. No filename,
DOM, OCR, fixture label, or selector reaches image encoder. CPU-only local model.
This is not a specialized UI model and cannot establish screenshot QA reliability.
"""
from __future__ import annotations
import hashlib,json,resource,time,os
from pathlib import Path
from PIL import Image
import numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/visible-ui-real')
MANIFEST=ROOT/'anchored_manifest.json'
OUT=ROOT/'clip_zeroshot_result.json'
MODEL='/data/models/rag_tests/vision/openai_clip-vit-base-patch32'
PROMPTS=[
 'A screenshot of a normal readable software user interface with fully visible text and correctly placed buttons and labels.',
 'A screenshot of a broken software user interface with cut-off or clipped text, overlapping labels, or elements extending beyond the window edge.',
]

def main():
 if OUT.exists():print('EXISTING_RESULTS_NOT_REPEATED');return
 import torch
 from transformers import CLIPModel,CLIPProcessor
 torch.set_num_threads(1)
 t0=time.monotonic()
 model=CLIPModel.from_pretrained(MODEL,local_files_only=True).cpu().eval()
 proc=CLIPProcessor.from_pretrained(MODEL,local_files_only=True)
 manifest=json.loads(MANIFEST.read_text())['cases']
 assert len(manifest)==32 and all(v['status']=='ok' for v in manifest.values())
 raw=[]
 with torch.no_grad():
  text=proc(text=PROMPTS,return_tensors='pt',padding=True,truncation=True)
  embeddings=torch.nn.functional.normalize(model.get_text_features(**text),dim=-1)
  for i,(key,v) in enumerate(sorted(manifest.items())):
   t=time.monotonic()
   im=Image.open(v['image_path']).convert('RGB')
   features=torch.nn.functional.normalize(model.get_image_features(**proc(images=im,return_tensors='pt')),dim=-1)
   similarity=(features@embeddings.T)[0]
   prob=float(torch.softmax(similarity,dim=0)[1])
   raw.append({'source_item_sha256':hashlib.sha256(key.encode()).hexdigest(),'image_sha256':v['image_sha256'],'anomaly_probability':round(prob,7),'predicted_anomaly':prob>=0.5,'seconds':round(time.monotonic()-t,3)})
   print('IMAGE',i+1,'/',len(manifest),'s',round(time.monotonic()-t,2),flush=True)
 # Finally open independent author labels, never passed to CLIP.
 joined=[]
 for (key,v),r in zip(sorted(manifest.items()),raw):
  assert hashlib.sha256(key.encode()).hexdigest()==r['source_item_sha256']
  joined.append({**r,'gold_violation':v['gold_violation'],'criterion':v['criterion'],'split':v['split']})
 outcomes={}
 for split in ['dev','test']:
  for cr in ['redecheck:element-protrusion','redecheck:viewport-protrusion']:
   subset=[v for v in joined if v['split']==split and v['criterion']==cr]
   tp=sum(v['predicted_anomaly'] and v['gold_violation'] for v in subset)
   fp=sum(v['predicted_anomaly'] and not v['gold_violation'] for v in subset)
   fn=sum(not v['predicted_anomaly'] and v['gold_violation'] for v in subset)
   tn=sum(not v['predicted_anomaly'] and not v['gold_violation'] for v in subset)
   outcomes[split+'_'+cr]={'n':len(subset),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
 result={'model':'OpenAI CLIP-ViT-B/32 pretrained local, fp32 CPU','prompts_predefined':PROMPTS,'status':'COMPLETE_IMAGE_ONLY_SCREENSHOT_DIAGNOSTIC_NOT_ACCEPTED','source':'UIJudgeBench v0.4.0 real sites, capture target within viewport, labels externally browser-verified','cases':joined,'scores':outcomes,'run_seconds':round(time.monotonic()-t0,2),'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':'not UI-trained model; four paired website families correlated, many viewport protrusions not inferable from crop; results classify authors DOM labels not necessarily visibly evident defect pixels. No label data input to feature encoder, no prompt tuning.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('SCORES',outcomes,flush=True)
 print('RUN_SECONDS',result['run_seconds'],flush=True)
if __name__=='__main__':main()
