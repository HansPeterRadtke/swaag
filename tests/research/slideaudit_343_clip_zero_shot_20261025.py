#!/usr/bin/env python3
"""ACTUAL image model CLIP ViT-B/32 on real human-rated 343 slide flaws.

Frozen six zero-shot prompts form 3 positive/negative visual-property pairs:
content cut-off/overflow, overlapping or occluded elements, text/background
contrast. The image encoder receives the slide pixels ONLY (no human label,
bbox, original metadata/source). 343 sampled images include strong-agreement
human positive/negative classes matched on slide origin+alteration as source.

Historical pretrained OpenAI CLIP ViT-B/32 local PyTorch weights; logits are
cosine similarity to two predeclared text prompts, softmax positive side. No
training or prompt retuning on the measured human labels. Freeze model outputs
BEFORE reading source human annotations and score separately. CPU one thread,
no shared GPU, source image and weights kept outside project Git.
"""
from pathlib import Path
import hashlib,json,time,datetime,resource,os
import torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
ROOT=Path('/data/var/perceptual-independent-20261024')
INPUT=ROOT/'slideaudit-public/data/images'
WEIGHTS=Path('/data/models/rag_tests/vision/openai_clip-vit-base-patch32')
OUT=ROOT/'slideaudit_343_clip_image_only_20261025.json'
PROMPTS=[
 'A screenshot of a presentation slide where text or graphics are visibly cut off or overflowing past the slide edges.',
 'A screenshot of a well laid out presentation slide with all text and graphics fully visible inside the slide boundaries.',
 'A screenshot of a presentation slide where overlapping text and graphics obscure or hide other content.',
 'A screenshot of a clear presentation slide with separated unobstructed text and graphic elements.',
 'A screenshot of a presentation slide with faint low contrast text that is difficult to read against its background.',
 'A screenshot of a presentation slide with sharply readable high contrast text and a clear background.',
]
TASKS={'Content Overflow/Cut-off':(0,1),'Occluded Content':(2,3),'Insufficient Color Contrast for Readability':(4,5)}
CFG={'original_model':'openai/clip-vit-base-patch32','local_checkpoint':'/data/models/rag_tests/vision/openai_clip-vit-base-patch32/pytorch_model.bin','model_type':'Vision-language contrastive pretrained CLIP ViT-B/32, NOT specialized defect classifier','prompts':PROMPTS,'tasks':TASKS,'input_source':'343 SlideAudit UIST 2025 original/synthetic slide raster images matched on manual target label','max_images':343,'preprocessing':'official pretrained CLIPProcessor resized 224x224','decision':'positive if P(prompt_defect)>0.5; no threshold fit to human data','inference':'CPU PyTorch single thread nice18, CUDA_VISIBLE_DEVICES empty; output is confidence proxy not calibrated truth','source_repo_revision':'642d490b7c1d2e78a50a631bfd359433397f3ecf'}

def main():
 pathlist=sorted(INPUT.glob('slide_*.png'))
 assert len(pathlist)==CFG['max_images']
 if OUT.exists():
  state=json.loads(OUT.read_text());
  if state['status']=='complete':print('PREVIOUS_MODEL_SCORES_PRESERVED_NO_REINFERENCE');return
  raise RuntimeError('Prior interrupted model inference checkpoint exists, do not replay ambiguous effects')
 torch.set_num_threads(1)
 state={'config':CFG,'status':'starting_model','predictions':{},'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_weights_sha256':hashlib.sha256((WEIGHTS/'pytorch_model.bin').read_bytes()).hexdigest()}
 OUT.write_text(json.dumps(state,indent=2)+'\n')
 start=time.monotonic()
 processor=CLIPProcessor.from_pretrained(str(WEIGHTS),local_files_only=True)
 model=CLIPModel.from_pretrained(str(WEIGHTS),local_files_only=True,torch_dtype=torch.float32).eval().cpu()
 with torch.inference_mode():
  tokens=processor(text=PROMPTS,return_tensors='pt',padding=True,truncation=True)
  text_vectors=torch.nn.functional.normalize(model.get_text_features(**tokens),dim=-1)
 state['status']='in_progress';state['model_load_wall_seconds']=round(time.monotonic()-start,2)
 OUT.write_text(json.dumps(state,indent=2)+'\n')
 print('LOADED_CLIP_CPU',state['model_load_wall_seconds'],'PROMPTS',len(PROMPTS),flush=True)
 for i,p in enumerate(pathlist,1):
  t=time.monotonic();source_sha=hashlib.sha256(p.read_bytes()).hexdigest()
  row={'source_sha256':source_sha,'status':'pending_or_uncertain'}
  state['predictions'][p.stem]=row;OUT.write_text(json.dumps(state,indent=2)+'\n')
  try:
   with Image.open(p) as im:
    rgb=im.convert('RGB');inp=processor(images=rgb,return_tensors='pt')
   with torch.inference_mode():
    vector=torch.nn.functional.normalize(model.get_image_features(**inp),dim=-1)
    sims=(vector@text_vectors.T).squeeze(0)
   probs={name:round(float(torch.softmax(sims[list(pair)],dim=0)[0]),7) for name,pair in TASKS.items()}
   row.update(status='complete',positive_prompt_probability=probs,input_image_size=list(rgb.size),seconds=round(time.monotonic()-t,3))
  except Exception as e:
   row.update(status='failed_or_uncertain_no_retry',error=type(e).__name__+': '+str(e)[:170],seconds=round(time.monotonic()-t,3))
  OUT.write_text(json.dumps(state,indent=2)+'\n')
  if i%12==0 or row['status']!='complete':print('CLIP_REAL_HUMAN_SLIDE',i,'/',len(pathlist),row['status'],'SECONDS',row['seconds'],flush=True)
 assert len(state['predictions'])==len(pathlist)
 state['status']='complete';state['total_cpu_wall_seconds']=round(time.monotonic()-start,2)
 state['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 OUT.write_text(json.dumps(state,indent=2)+'\n')
 print('CLIP_343_REAL_IMAGE_RESULTS_FROZEN',OUT,'COMPLETE',sum(v['status']=='complete' for v in state['predictions'].values()),'TOTAL_S',state['total_cpu_wall_seconds'],'RSS_KB',state['peak_rss_kb'],flush=True)
if __name__=='__main__':main()
