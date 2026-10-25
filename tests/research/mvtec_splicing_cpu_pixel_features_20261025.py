#!/usr/bin/env python3
"""Extract actual industrial pixel-only features on 134 ORIGINAL MVTec LOCO images.

Different product category from all prior SWAAG pushpin tests: real splicing
connectors, original official train/validation/test splits, pristine PNG, CC
BY-NC-SA 4.0 research only. Frozen official source paths and bytes verified by
SHA before every image. Does NOT expose heldout good/logical/structural label to
model: only image content and anonymized hash enter image feature extraction.
No GPU (CUDA_VISIBLE_DEVICES empty), CPU one PyTorch thread/nice 19, uses an
already installed frozen OpenAI CLIP ViT-B/32 578MB local fp32 weights.

Features per image: (1) CLIP global 512D unit embedding; (2) all 49 CLIP last
hidden patch tokens 768D unit normal embeddings; (3) HSV 2x4 spatial tile
histograms with 8x4x4 bins and L2-normalized vector, no fitting. Save per-image
compressed numeric arrays out of Git and per-source SHA receipts. Subsequent
normal-only training and calibration performed separately, AFTER features
frozen; threshold and feature choices predeclared before using anomaly gold.

This is not full Anomalib EfficientAD or PatchCore, nor MVTec-source-trained
specialist. CLIP ViT32 may miss tiny logical displacements and physical support
errors. Training with 24 original normal images cannot prove reliable factory
inspection or commercial license. Some upstream CLIP pretrain exposure unknown.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time,datetime,resource,io,collections
import numpy as np,cv2,torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-splicing-20261025')
MAN=ROOT/'source_frozen_pre_analysis_selection_20261025.json'
RECEIPT=ROOT/'original_source_image_134_SHA256_receipts_20261025.json'
FEATURES=ROOT/'frozen_original_image_only_embeddings'
STATE=ROOT/'134_original_image_visual_feature_capture_state_20261025.json'
MODEL=Path('/data/models/rag_tests/vision/openai_clip-vit-base-patch32')
CFG={'date':'2026-10-25','model':'OpenAI CLIP ViT-B/32 local pinned model fp32 pretrained weights (model path external)', 'CPU_only':True,'torch_threads':1,'image_input_preprocess':'pretrained official CLIPProcessor 224x224 LANCZOS crop','clip_global_image_representation':'CLIPModel.get_image_features 512D normalized','clip_patch_visual_tokens':'CLIPModel.vision_model(pixel_values).last_hidden_state[:,1:] 49x768 L2 unit embeddings','HSV_baseline':'OpenCV original RGB pixels downscaled to 448x224; 2 rows x4 columns, HSV H=8 S=4 V=4 normalized frequencies per tile, concatenate and L2 normalize','no_gold_labels_during_pixel_feature_extraction':True,'original_png_license':'CC BY-NC-SA4.0 research only, images not Git','training_and_calibration_after_frozen_feature_extraction':True,'comparison':'One-class kNN CLIP/global, CLIP patch bank and HSV histogram bank with exactly SAME 24 source-normal images and same 20 source-normal validation threshold. Separate score/eval script.'}

def hs(img):
 a=np.asarray(img.resize((448,224),Image.Resampling.BILINEAR).convert('RGB'))
 hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
 vec=[]
 for i in range(2):
  for j in range(4):
   tile=hsv[i*112:(i+1)*112,j*112:(j+1)*112]
   hist=cv2.calcHist([tile],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]).reshape(-1).astype(np.float32)
   hist/=max(1,float(hist.sum()))
   vec.append(hist)
 v=np.concatenate(vec).astype(np.float32)
 return v/max(1e-9,float(np.linalg.norm(v)))

def digest(p):
 h=hashlib.sha256();total=0
 with p.open('rb') as f:
  while b:=f.read(1048576):total+=len(b);h.update(b)
 return total,h.hexdigest()

def main():
 torch.set_num_threads(1);cv2.setNumThreads(1)
 manifest=json.loads(MAN.read_text());receipt=json.loads(RECEIPT.read_text())
 assert len(manifest['selected'])==134 and len(receipt['completed'])==134
 if STATE.exists():
  state=json.loads(STATE.read_text())
  if state['status']=='complete':print('PREVIOUS_COMPLETE_IMAGE_ONLY_MODEL_FEATURES_PRESERVED_NO_REPLAY');return
 else:state={'created':'2026-10-25','status':'starting_model','config':CFG,'source_selection_sha256':hashlib.sha256(MAN.read_bytes()).hexdigest(),'source_file_receipts_SHA256_at_start':hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),'source_model_weights_sha256':hashlib.sha256((MODEL/'pytorch_model.bin').read_bytes()).hexdigest(),'cases':{}}
 assert state['config']==CFG
 FEATURES.mkdir(parents=True,exist_ok=True)
 STATE.write_text(json.dumps(state,indent=2)+'\n')
 start=time.monotonic()
 processor=CLIPProcessor.from_pretrained(str(MODEL),local_files_only=True)
 model=CLIPModel.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=torch.float32).eval().cpu()
 state['model_load_s']=round(time.monotonic()-start,3);state['status']='in_progress';STATE.write_text(json.dumps(state,indent=2)+'\n')
 print('NEW_SPICING_CLIP_MODEL_LOADED_ON_NITRO_CPU',state['model_load_s'],'MODEL_PARAM_MB',round((MODEL/'pytorch_model.bin').stat().st_size/1e6,1),flush=True)
 for idx,item in enumerate(manifest['selected'],1):
  key=item['source_image_path'];ref=receipt['completed'][key]
  ident=hashlib.sha256(key.encode()).hexdigest()
  if ident in state['cases']:continue
  imgpath=ROOT/ref['source_path'];size,sha=digest(imgpath)
  assert size==item['source_HF_file_size_bytes'] and sha==ref['source_photo_SHA256']
  dest=FEATURES/(ident+'.npz')
  if dest.exists():raise RuntimeError('Uncertain existing prior feature file without receipt, no overwrite')
  state['cases'][ident]={'status':'pending_or_uncertain','input_original_SHA256':sha}
  STATE.write_text(json.dumps(state,indent=2)+'\n')
  begin=time.monotonic()
  try:
   with Image.open(imgpath) as im:
    rgb=im.convert('RGB');baseline=hs(rgb);inputs=processor(images=rgb,return_tensors='pt')
   with torch.inference_mode():
    globalv=model.get_image_features(**inputs).detach().cpu().numpy().reshape(-1).astype(np.float32)
    hidden=model.vision_model(pixel_values=inputs['pixel_values']).last_hidden_state[:,1:,:].detach().cpu().numpy().reshape(49,768).astype(np.float32)
   globalv/=max(np.linalg.norm(globalv),1e-9)
   hidden/=np.maximum(np.linalg.norm(hidden,axis=1,keepdims=True),1e-9)
   assert globalv.shape==(512,) and hidden.shape==(49,768) and baseline.shape==(1024,)
   # Direct-file creation without overwrite; incomplete bytes remain unknown.
   with dest.open('xb') as f:np.savez_compressed(f,global_embedding=globalv,patch_embeddings=hidden,hsv_spatial_histogram=baseline)
   state['cases'][ident]={'status':'complete','source_image_sha256':sha,'local_feature_file':str(dest.relative_to(ROOT)),'feature_file_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'global_dims':len(globalv),'patch_shape':[49,768],'HSV_spatial_dims':len(baseline),'elapsed_seconds':round(time.monotonic()-begin,2)}
  except Exception as e:
   state['cases'][ident]={'status':'error_or_uncertain_no_retry','source_image_sha256':sha,'exception':type(e).__name__+': '+str(e)[:190]}
  STATE.write_text(json.dumps(state,indent=2)+'\n')
  if idx%7==0 or state['cases'][ident]['status']!='complete':print('IMAGE_ONLY_MVTEC_SOURCE_FEATURES',idx,'/134','STATUS',state['cases'][ident]['status'],'SEC',state['cases'][ident].get('elapsed_seconds'),flush=True)
 assert len(state['cases'])==134
 state['status']='complete';state['wall_s']=round(time.monotonic()-start,2);state['rss_peak_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 STATE.write_text(json.dumps(state,indent=2)+'\n')
 print('ORIGINAL_134_IMAGE_ONLY_PIXELS_FROZEN',STATE,'STATUS_COUNTS',dict(collections.Counter(x['status'] for x in state['cases'].values())),'SECONDS',state['wall_s'],'PEAK_RSS_KB',state['rss_peak_kb'],flush=True)
if __name__=='__main__':main()
