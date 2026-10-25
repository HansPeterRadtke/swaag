#!/usr/bin/env python3
"""Fixed source image selection/download for NEW industrial visual inspection.

MVTec LOCO AD original source-labelled screw bags with logical and structural defects
(official scientific source; public HF Rafa612/mvtec_loco repo pinned
839fee8178c10d2e50afddc671238ba85d9975d9). CC BY-NC-SA 4.0 research
only. Prior SWAAG MVTec work inspected 18 pushpin source images: this chooses a
DIFFERENT REAL industrial product category and previously unseen test photos.

Selection is frozen BEFORE decoding images: 24 original train/good (normal-only
model training), 20 original validation/good (normal-only threshold), and 30
original heldout TEST images from each: good/logical_anomalies/
structural_anomalies. All from official original splits, random SHA256 ranking
by fixed 20261025 seed; no class balancing after seeing model predictions.
Never use label/path to infer pixel scores in subsequent frozen feature work.

Download only 134 selected images ~194MB, not MVTec 6GB entire release;
source photo pixels, source labels and provenance only under /data/var,
never source Git. No user production images or GPU use; preserve .part files
and do not silently retry an uncertain or failed fetch. API original SHA path,
LFS hash if available, size, original content sha checked; no credential data.
"""
from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,os,time,datetime,collections,argparse
DATA=Path('/data/var/perceptual-independent-20261024/mvtec-loco-screwbag-20261025')
MAN=DATA/'source_frozen_pre_analysis_selection_20261025.json'
RECEIPT=DATA/'original_source_image_134_SHA256_receipts_20261025.json'
REPO='Rafa612/mvtec_loco';REV='839fee8178c10d2e50afddc671238ba85d9975d9'
QUOTAS={'train/good':24,'validation/good':20,'test/good':30,'test/logical_anomalies':30,'test/structural_anomalies':30}
SEED='SWAAG-SCREW-BAG-PADIM-INDEPENDENT-CATEGORY-20261025-v1'

def listing(folder):
 path='screw_bag/'+folder
 u=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/'+urllib.parse.quote(path,safe='/')+'?limit=500'
 with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'SWAAG-research/1.0'}),timeout=22) as f:data=json.load(f)
 images=[x for x in data if x['path'].endswith('.png') and x['type']=='file']
 assert len(images)>=QUOTAS[folder]
 assert len({x['path'] for x in images})==len(images)
 return images

def freeze():
 if MAN.exists():print('EXISTING_FROZEN_SELECTION_PRESERVED_NO_RESAMPLE');return
 DATA.mkdir(mode=0o700,parents=True,exist_ok=True)
 source=[];stats={}
 for folder,n in QUOTAS.items():
  imgs=listing(folder)
  pick=sorted(imgs,key=lambda x:hashlib.sha256((SEED+'|'+x['path']).encode()).hexdigest())[:n]
  for p in pick:
   assert p['size']<4_000_000
   source.append({'source_image_path':p['path'],'source_HF_file_size_bytes':int(p['size']),'upstream_sha1':p.get('oid'),'upstream_lfs_data':p.get('lfs'),'selection_stratum':folder,'source_normal_only_train':folder=='train/good','source_normal_only_val':folder=='validation/good','test_trust_reference_label':folder if folder.startswith('test/') else None})
  stats[folder]={'full_original_source_images':len(imgs),'selected_source_images':len(pick),'selected_original_source_download_bytes':sum(x['size'] for x in pick)}
 assert len(source)==134 and len({r['source_image_path'] for r in source})==134
 selection={'timestamp':'2026-10-25','source_repo':REPO,'official_replica_revision':REV,'original_licence':'MVTec LOCO AD CC BY-NC-SA4.0, not commercial; keep photographs outside Git','source_original':'MVTec LOCO AD original 2022 IJCV manufacturer inspection real images, logical vs structural defect labels independent of our model scores','prior_overlap':'none: older local visual model benchmark used PUSH PINS category only, not screw bags','split_purpose':'24 normal-only training/20 normal-only validation/30 each independent test good, logical, structural','selection_seed':SEED,'selection_no_pixel_or_label_optimizing':True,'quotas':QUOTAS,'original_inventory':stats,'n_unique_source_original_photos':len(source),'total_original_download_MB':round(sum(r['source_HF_file_size_bytes'] for r in source)/1e6,2),'selected':source,'planned_CPU_inference':'official pretrained torchvision ResNet18 PaDiM-style spatial Gaussian + ResNet18 global cosine + HSV baseline, normal-only training and calibration CPU, no GPU'}
 MAN.write_text(json.dumps(selection,indent=2)+'\n')
 print('FROZEN_NEW_INDUSTRIAL_ORIGINAL_SOURCE_SELECTION',len(source),'IMAGE_MB',selection['total_original_download_MB'],'ORIGINAL_SOURCE_COUNTS',stats,'REV',REV,flush=True)

def sha_local(p):
 h=hashlib.sha256();bytes_count=0
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b);bytes_count+=len(b)
 return bytes_count,h.hexdigest()

def acquire(start,count):
 m=json.loads(MAN.read_text());assert len(m['selected'])==134
 if RECEIPT.exists():d=json.loads(RECEIPT.read_text());assert d['selection_SHA256']==hashlib.sha256(MAN.read_bytes()).hexdigest()
 else:d={'selection_SHA256':hashlib.sha256(MAN.read_bytes()).hexdigest(),'completed':{}}
 for item in m['selected'][start:start+count]:
  key=item['source_image_path']
  if key in d['completed']:continue
  target=DATA/'original_media'/key
  expected=item['source_HF_file_size_bytes']
  if target.exists():
   size,sha=sha_local(target)
   if size!=expected:raise RuntimeError('Existing source mismatch, cannot repeat or overwrite')
   status='verified_existing_original_length_and_sha_not_redownloaded'
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   tmp=target.with_suffix('.png.part')
   if tmp.exists():raise RuntimeError('uncertain prior .part source download; no blind rerun')
   u=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/'+urllib.parse.quote(key,safe='/')+'?download=true'
   t=time.monotonic();h=hashlib.sha256();total=0
   with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'SWAAG-independent-perception/1.0'}),timeout=55) as r,tmp.open('xb') as out:
    while b:=r.read(1<<20):
     total+=len(b)
     if total>expected+1_000_000:raise RuntimeError('Unexpected excess source bytes')
     out.write(b);h.update(b)
    out.flush();os.fsync(out.fileno())
   if total!=expected:raise RuntimeError('Source image length mismatch')
   sha=h.hexdigest();size=total;tmp.rename(target);status='new_original_source_image_length_SHA256_verified'
   print('NEW_INDUSTRIAL_SOURCE_PHOTO',len(d['completed'])+1,'/',134,'CATEGORY',item['selection_stratum'],'BYTES',total,'WALL_SEC',round(time.monotonic()-t,2),flush=True)
  d['completed'][key]={'source_path':str(target.relative_to(DATA)),'source_bytes':size,'source_photo_SHA256':sha,'status':status}
  RECEIPT.write_text(json.dumps(d,indent=2)+'\n')
 print('SOURCE_ORIGINAL_MEDIA_RECEIPT',len(d['completed']),'of134','NEW_BOUNDED_BATCH',len(m['selected'][start:start+count]),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['plan','fetch']);parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=10)
 a=parser.parse_args();freeze() if a.mode=='plan' else acquire(a.start,a.count)
