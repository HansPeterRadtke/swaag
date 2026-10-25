#!/usr/bin/env python3
"""Frozen pixel features for a NEW actual industrial screw-bag category.

134 original MVTec LOCO AD noncommercial research photos: 24 source train
normal, 20 source validation normal, and 30 each independent original test
normal/logical/structural. Source selection frozen by hash BEFORE images were
seen, copied to /data/var only. Image feature extraction sees ONLY pixels and
SHA item IDs and never reads normal/defective test labels. Research entirely
Nitro CPU nice19/1 thread; no GPU/model services or production assets.

Extract official ImageNet1K pretrained torchvision ResNet18 (verified source
SHA256 f37072fd...). Two PREDECLARED pretrained image feature variants:
  - 512D global average pool ResNet18 full image (cosine one-class nearest).
  - PaDiM-inspired spatial features: concatenate layer1 (64 channels at 56x56
    averaged 28x28), layer2 (128 @28x28), layer3 (256 @14x14 upsample 28x28),
    choose 32/448 channels with fixed seed unrelated to test ground truth.
    Train spatial per-position multivariate Gaussian with 24 source normal
    image features, shrink covariance +0.01 I, score 95th percentile per
    position Mahalanobis distance. This is not official PaDiM, not trained
    specialized logical-constraint model and no anomaly labels used in fit.
  - Simple HSV 2x4 tiles, 8x4x4 histogram normalized, no pretrained weights.
Threshold is fixed 95th percentile of only 20 normal VAL source images for
all methods, full raw test scores frozen BEFORE source test labels join.

Copyright original MVTec images CC BY-NC-SA 4.0, industrial noncommercial
research only; all original images, features, weights outside Git. No flags
from source names, annotated masks, color defect tables or prior results.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,time,resource,collections
import numpy as np,torch,torch.nn.functional as F,cv2
from PIL import Image
from torchvision.models import resnet18,ResNet18_Weights
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-screwbag-20261025')
MAN=ROOT/'source_frozen_pre_analysis_selection_20261025.json'
RECEIPT=ROOT/'original_source_image_134_SHA256_receipts_20261025.json'
WEIGHT=ROOT/'torchvision_resnet18_imagenet1k_v1_f37072fd.pth'
FOLDER=ROOT/'frozen_actual_image_only_features'
STATE=ROOT/'134_source_pixelonly_resnet18_imagenet_hsv_features_20261025.json'
CHANNEL_SEED=20261025
RNG=np.random.default_rng(CHANNEL_SEED)
CHANNELS=sorted(RNG.choice(448,32,replace=False).tolist())
CFG={'model_version':'torchvision 0.20.1 official ResNet18 ImageNet1K V1 pretrained weights ResNet18_Weights.DEFAULT f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec','preprocessor':'torchvision ResNet18_Weights.IMAGENET1K_V1.transforms resize256 centerCrop224 RGB normalize','spatial_feature':'layer1 avgpool 28x28 + layer2 28x28 + layer3 bilinear 28x28, channel select exact 32/448 with deterministic seed20261025 unrelated to labels','spatial_channel_indexes':CHANNELS,'global_feature':'layer4 adaptive avgpool512 unit norm','hsv_baseline':'2x4 tile HSV hist each 8x4x4 bins source RGB full resize448x224, per-tile frequency normalize then concatenate and unit L2 normalization','data':'original MVTec LOCO AD screw_bag 134 fixed source photos pinned Rafa612/mvtec_loco revision839fee8178c10d2e50afddc671238ba85d9975d9','privacy_license':'MVTec CC BY-NC-SA4.0 original external industrial photos/weights/features stay /data/var only','GPU_allocated':False,'torch_num_threads':1,'source_labels_read_for_pixel_features':False,'threshold_training':'only train normal 24 and val normal20 after features frozen, no test scores or labels used to choose parameters'}

def sha(path):
 d=hashlib.sha256();n=0
 with path.open('rb') as f:
  while b:=f.read(1048576):d.update(b);n+=len(b)
 return n,d.hexdigest()

def hsv_features(im):
 raw=np.asarray(im.resize((448,224),Image.Resampling.BILINEAR).convert('RGB'))
 vals=cv2.cvtColor(raw,cv2.COLOR_RGB2HSV)
 vec=[]
 for i in range(2):
  for j in range(4):
   hist=cv2.calcHist([vals[i*112:(i+1)*112,j*112:(j+1)*112]],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]).reshape(-1).astype(np.float32)
   hist/=max(hist.sum(),1.)
   vec.append(hist)
 result=np.concatenate(vec)
 return result/max(1e-9,np.linalg.norm(result))

def extract(model,tensor):
 with torch.inference_mode():
  x=model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
  l1=model.layer1(x)
  l2=model.layer2(l1)
  l3=model.layer3(l2)
  l4=model.layer4(l3)
  v=F.adaptive_avg_pool2d(l4,1).flatten(1)[0].cpu().numpy().astype(np.float32)
  l1=F.adaptive_avg_pool2d(l1,(28,28))
  l3=F.interpolate(l3,size=(28,28),mode='bilinear',align_corners=False)
  spatial=torch.cat([l1,l2,l3],dim=1)[0,CHANNELS].permute(1,2,0).cpu().numpy().reshape(784,32).astype(np.float32)
 v/=max(1e-9,np.linalg.norm(v))
 return v,spatial

def main():
 torch.set_num_threads(1);cv2.setNumThreads(1)
 assert sha(WEIGHT)[1]==CFG['model_version'].split()[-1]
 source=json.loads(MAN.read_text());record=json.loads(RECEIPT.read_text())
 assert len(source['selected'])==len(record['completed'])==134
 if STATE.exists():
  state=json.loads(STATE.read_text())
  if state['status']=='complete':print('RESNET18_134_ALREADY_FULLY_FROZEN_NOT_REINFERRED');return
 else:state={'date':'2026-10-25','status':'starting_model','config':CFG,'source_selection_sha256':sha(MAN)[1],'original_media_receipts_sha256_at_start':sha(RECEIPT)[1],'official_pretrained_weights_sha256':sha(WEIGHT)[1],'results':{}}
 assert state['config']==CFG and state['source_selection_sha256']==sha(MAN)[1]
 FOLDER.mkdir(exist_ok=True,parents=True)
 STATE.write_text(json.dumps(state,indent=2)+'\n')
 t=time.monotonic()
 model=resnet18(weights=None)
 model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
 model.eval().cpu();transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
 state['model_load_s']=round(time.monotonic()-t,3);state['status']='in_progress';STATE.write_text(json.dumps(state,indent=2)+'\n')
 print('OFFICIAL_RESNET18_CPU_IMAGENET_LOADED',state['model_load_s'],'CANONICAL_CHANNEL_COUNT',len(CHANNELS),flush=True)
 for i,row in enumerate(source['selected'],1):
  path=row['source_image_path'];key=hashlib.sha256(path.encode()).hexdigest()
  if key in state['results']:continue
  ref=record['completed'][path];p=ROOT/ref['source_path'];n,digest=sha(p)
  assert n==ref['source_bytes']==row['source_HF_file_size_bytes'] and digest==ref['source_photo_SHA256']
  target=FOLDER/(key+'.npz')
  if target.exists():raise RuntimeError('ambiguous prior feature file without receipt no overwrite')
  state['results'][key]={'status':'pending_or_uncertain','original_pixel_sha256':digest};STATE.write_text(json.dumps(state,indent=2)+'\n')
  start=time.monotonic()
  try:
   with Image.open(p) as f:
    rgb=f.convert('RGB');hsv=hsv_features(rgb);batch=transform(rgb).unsqueeze(0)
   glob,spatial=extract(model,batch)
   assert glob.shape==(512,) and spatial.shape==(784,32) and hsv.shape==(1024,)
   with target.open('xb') as f:np.savez_compressed(f,global_resnet_embedding=glob,padim_spatial_selected_32ch=spatial,hsv_tile_histogram=hsv)
   state['results'][key]={'status':'complete','original_pixel_sha256':digest,'feature_relpath':str(target.relative_to(ROOT)),'feature_file_sha256':sha(target)[1],'elapsed_sec':round(time.monotonic()-start,3)}
  except Exception as e:state['results'][key]={'status':'source_unscorable_no_retry','exception':type(e).__name__+': '+str(e)[:160],'original_pixel_sha256':digest}
  STATE.write_text(json.dumps(state,indent=2)+'\n')
  if i%8==0 or state['results'][key]['status']!='complete':print('ORIGINAL_SCREWBAG_REAL_2D_IMAGE_FEATURES',i,'/134','STATUS',state['results'][key]['status'],'SECONDS',state['results'][key].get('elapsed_sec'),flush=True)
 assert len(state['results'])==134
 state['status']='complete';state['wall_seconds']=round(time.monotonic()-t,2);state['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 STATE.write_text(json.dumps(state,indent=2)+'\n')
 print('FROZEN_134_NEW_INDUSTRIAL_SOURCE_PIXEL_FEATURES',STATE,'OUTCOMES',dict(collections.Counter(x['status'] for x in state['results'].values())),'WALL_S',state['wall_seconds'],'PEAK_RSS_KB',state['peak_rss_kb'])
if __name__=='__main__':main()
