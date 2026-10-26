#!/usr/bin/env python3
"""Previously unseen 72 SlideAudit 3/3 human-negative originals: deployment false-alarm study.

No prior selected 343 case is replayed. Freeze chooses 6 from EACH 12
source_type/alteration stratum among 1102 unused unanimously negative across
clipping/occlusion/contrast. Predictors consume only frozen image IDs and
original JPEG/PNG pixels. Join label files only after ALL predictions frozen.
Original third-party images, text, annotations and model weights stay /data/var.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess

BASE=Path('/data/var/perceptual-independent-20261024')
SRC=BASE/'slideaudit-public'
OUT=BASE/'slideaudit_unseen_triple_neg_20261026'
PREVIOUS=BASE/'slideaudit_selection_20261025.json'
REV='642d490b7c1d2e78a50a631bfd359433397f3ecf'
CATS=('Content Overflow/Cut-off','Occluded Content','Insufficient Color Contrast for Readability')
DESIGN={
 'date':'2026-10-26','source':'SlideAudit UIST 2025 source pinned 642d490b7c1d2e78a50a631bfd359433397f3ecf',
 'gold':'The original authors have 3 Prolific trained human annotators per slide; has_strong_agreement=True indicates unanimous 3/3. For ALL THREE requested categories all three annotators marked defect absent.',
 'source_truth_scope':'human unanimous *absence of those three categories only*; other possible defects and actual native GUI correctness NOT certified',
 'stratification':'6 images each of 12 source_type x alteration strata from 1102 unused 3-property unanimous-negative original slides, ranked SHA256 of ID and fixed salt',
 'source_version':'official local source repository git rev must match pin, original annotation and source metadata hashes recorded at freeze',
 'no_contamination':'exclude exact 343 previously selected source IDs, freeze before decoding or model inference; old published ROC/threshold rules unchanged',
 'pixel_CV':'Tesseract PSM11/OpenCV original code and original fixed thresholds, clipping <=2% outer border, OCR word IoU>=0.15, 35% glyph patches <4.5 grayscale',
 'CLIP':'same original pretrained openai/clip-vit-base-patch32 ViT-B/32, same 6 frozen prompts, positive if pair softmax >0.5; CPU one thread, GPU disabled',
 'source_geometry':'same original source rectangle check, text object >2% off slide canvas, any object overlap >=10% smaller box. This tests source contract proxies, NOT user-visible occlusion or contrast.',
 'null':'always no flaw => zero false proposals on these unanimity negatives, but no positive recall',
 'positive_comparator':'previously frozen 343 selected actual images with all strong positive labels per category; no new positive human gold exists outside prior selection',
 'timing':'all three candidate detectors timed wall-clock on these exact 72 original image files; no GPU, no VLM service interruption',
 'output_actions':'Review proposal on any model flag, never automatic fail from pixels or ambiguous source geometry; explicit source checked property can auto-fail only when final reference/output and intent verified',
 'limitation':'Already published source dataset, 6 per stratum not population-representative and 3 target categories only. No original owner app/3D/video human-rated data.',
}

def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(path,obj):
 if path.exists():raise RuntimeError('source/frozen result already exists, never overwrite '+str(path))
 b=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_suffix('.tmpfreeze');tmp.write_bytes(b);os.replace(tmp,path)

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 if any((OUT/f).exists() for f in ('protocol.json','selection.json','pixel_features.json','clip_features.json','source_features.json','scored.json')):raise RuntimeError('Earlier experimental state exists')
 revision=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
 assert revision==REV
 previous=json.loads(PREVIOUS.read_text());assert previous['official_source_commit']==REV and len(previous['items'])==343
 previously_used=set(int(x) for x in previous['items'])
 metadata={int(r['id']):r for r in csv.DictReader((SRC/'data/metadata.csv').open())};assert len(metadata)==2400
 strata=defaultdict(list);source_count=Counter();all_label_sha=[]
 for path in sorted((SRC/'data/annotations').glob('slide_*.json')):
  b=path.read_bytes();info=json.loads(b);i=int(info['slide_id'])
  all_label_sha.append((i,sha(b)))
  if i in previously_used:continue
  tags={t['design_deficiency']:t for t in info['annotations']}
  if all(not bool(tags[k]['response']) and bool(tags[k]['has_strong_agreement']) for k in CATS):
   m=metadata[i];strata[(m['source_type'],m['alteration'])].append(i)
  source_count['annotated_total']+=1
 assert sum(len(v) for v in strata.values())==1102
 assert len(strata)==12
 selection=[]
 for st,v in sorted(strata.items()):
  assert len(v)>=6
  ranked=sorted(v,key=lambda i:sha(f'new-locked-unanimous-negative-20261026|{i}'.encode()))
  for i in ranked[:6]:
   p=SRC/'data/annotations'/f'slide_{i:04}.json';desc=SRC/'data/descriptions'/f'slide_{i:04}.json'
   assert p.exists() and desc.exists()
   selection.append({'id':i,'img_name':f'slide_{i:04}.png','annotation_sha256':file_sha(p),'source_description_sha256':file_sha(desc),'stratum':'|'.join(st)})
 assert len(selection)==72 and len({c['id'] for c in selection})==72
 assert not {c['id'] for c in selection}.intersection(previously_used)
 save(OUT/'protocol.json',DESIGN)
 save(OUT/'selection.json',{'protocol_sha256':file_sha(OUT/'protocol.json'),'source_revision':REV,'source_all_annotation_chain_sha256':sha(''.join(h for _,h in sorted(all_label_sha)).encode()),'source_metadata_sha256':file_sha(SRC/'data/metadata.csv'),'previous_evaluated_343_source_selection_sha256':file_sha(PREVIOUS),'num_unseen_fully_negative_available':1102,'num_selected':len(selection),'per_stratum':6,'selected':selection})
 print('UNANIMOUS_NEGATIVE_HOLDOUT_FROZEN',len(selection),'SHA',file_sha(OUT/'selection.json'),'source_unanimous_pool',1102)

def fetch_one(item):
 path=OUT/'media'/item['img_name']
 if path.exists():return item['id'],file_sha(path),path.stat().st_size,'existing'
 import requests
 url=f'https://raw.githubusercontent.com/zhuohaouw/SlideAudit/{REV}/data/images/{item["img_name"]}'
 r=requests.get(url,timeout=35);r.raise_for_status()
 data=r.content
 if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('not source PNG')
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
 return item['id'],sha(data),len(data),'downloaded'

def download():
 if (OUT/'download_receipts.json').exists():raise RuntimeError('Already downloaded and frozen, refuse rerun')
 selected=json.loads((OUT/'selection.json').read_text())['selected']
 t=time.monotonic();rows={}
 with ThreadPoolExecutor(max_workers=6) as exe:
  futures=[exe.submit(fetch_one,item) for item in selected]
  for job in as_completed(futures):
   ident,digest,size,status=job.result();rows[str(ident)]={'sha256':digest,'bytes':size,'origin':status}
 assert len(rows)==72
 save(OUT/'download_receipts.json',{'selection_sha256':file_sha(OUT/'selection.json'),'source_original_files_sha256':rows,'source_bytes':sum(x['bytes'] for x in rows.values()),'wall_s':round(time.monotonic()-t,3),'all_72_original_source_images_verified':True})
 print('HOLDOUT_NEW_SOURCES_DOWNLOADED',len(rows),'MB',round(sum(x['bytes'] for x in rows.values())/1e6,2),'sec',round(time.monotonic()-t,2),'sha',file_sha(OUT/'download_receipts.json'))


def pixel():
 import cv2
 from slideaudit_pixels_only_visual_baselines_20261025 import estimate,CFG
 if (OUT/'pixel_features.json').exists():raise RuntimeError('Pixel scores frozen: not rerunning')
 selection=json.loads((OUT/'selection.json').read_text()); downloads=json.loads((OUT/'download_receipts.json').read_text())
 assert downloads['selection_sha256']==file_sha(OUT/'selection.json')
 cv2.setNumThreads(1)
 from collections import Counter
 import resource
 t=time.monotonic(); rows={}
 for ii,case in enumerate(selection['selected'],start=1):
  path=OUT/'media'/case['img_name'];ident=case['id'];digest=file_sha(path)
  assert digest==downloads['source_original_files_sha256'][str(ident)]['sha256']
  r=estimate(path)
  if r['source_image_sha256']!=digest:raise RuntimeError('source changed in read')
  rows[str(ident)]={'source_sha256':digest,'status':r['status'],'elapsed_s':r['elapsed_s'],'clip_ocr_proposal':r['predicted_clipped_text_near_canvas_boundary'],'overlap_ocr_proposal':r['predicted_ocr_box_overlap'],'contrast_ocr_proposal':r['predicted_low_contrast'],'text_word_conf25_count':r['ocr_word_count_conf25']}
  if ii%12==0:print('CV_OCR_72',ii,'elapsed',round(time.monotonic()-t,1),'s',flush=True)
 save(OUT/'pixel_features.json',{'source_sha256':file_sha(OUT/'download_receipts.json'),'config_sha256':sha(json.dumps(CFG,sort_keys=True).encode()),'stage':'pixel predictions source blind','status_counts':dict(Counter(x['status'] for x in rows.values())),'wall_s':round(time.monotonic()-t,3),'peak_rss_KB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rows':rows})
 print('CV_OCR_PREDICTIONS_FROZEN',len(rows),'sec',round(time.monotonic()-t,2),'sha',file_sha(OUT/'pixel_features.json'))

def geometry():
 from slideaudit_geometry_descriptions_pixel_human_20261025 import analyze,CONFIG
 import resource
 if (OUT/'source_features.json').exists():raise RuntimeError('Source geometry frozen')
 selection=json.loads((OUT/'selection.json').read_text());t=time.monotonic();rows={}
 for case in selection['selected']:
  ident=case['id'];p=SRC/'data/descriptions'/f'slide_{ident:04}.json';assert file_sha(p)==case['source_description_sha256']
  objects=json.loads(p.read_text())['elements'];feat=analyze(objects)
  rows[str(ident)]={'source_description_sha256':case['source_description_sha256'],'source_off_slide_textbox_proposal':feat['source_predicted_overflow'],'source_object_overlap_proposal':feat['source_predicted_any_element_overlap'],'source_elements':feat['source_element_count'],'valid_geometry':feat['source_valid_geometry_count'],'invalid_geometry':feat['source_invalid_geometry_count']}
 save(OUT/'source_features.json',{'selection_sha256':file_sha(OUT/'selection.json'),'config_sha256':sha(json.dumps(CONFIG,sort_keys=True).encode()),'source_only_rectangles_not_final_visual_truth':True,'wall_s':round(time.monotonic()-t,3),'peak_rss_KB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rows':rows})
 print('SOURCE_GEOMETRY_FROZEN',len(rows),'seconds',round(time.monotonic()-t,2),'sha',file_sha(OUT/'source_features.json'))


def clip():
 import torch,resource
 from transformers import CLIPModel, CLIPProcessor
 from slideaudit_343_clip_zero_shot_20261025 import PROMPTS, TASKS, WEIGHTS
 from PIL import Image
 if (OUT/'clip_features.json').exists():raise RuntimeError('CLIP scores frozen: not repeating')
 if not (OUT/'pixel_features.json').exists():raise RuntimeError('Freeze all cheaper pixel-only features first')
 torch.set_num_threads(1)
 model_sha=file_sha(WEIGHTS/'pytorch_model.bin')
 selection=json.loads((OUT/'selection.json').read_text()); downloads=json.loads((OUT/'download_receipts.json').read_text())
 start=time.monotonic()
 processor=CLIPProcessor.from_pretrained(str(WEIGHTS),local_files_only=True)
 model=CLIPModel.from_pretrained(str(WEIGHTS),local_files_only=True,torch_dtype=torch.float32).eval().cpu()
 with torch.inference_mode():
  tok=processor(text=PROMPTS,return_tensors='pt',padding=True,truncation=True)
  tv=torch.nn.functional.normalize(model.get_text_features(**tok),dim=-1)
 loadtime=time.monotonic()-start
 rows={}
 for idx,case in enumerate(selection['selected'],start=1):
  t=time.monotonic();p=OUT/'media'/case['img_name'];ident=case['id'];digest=file_sha(p)
  assert digest==downloads['source_original_files_sha256'][str(ident)]['sha256']
  with Image.open(p) as im:inp=processor(images=im.convert('RGB'),return_tensors='pt')
  with torch.inference_mode():
   vec=torch.nn.functional.normalize(model.get_image_features(**inp),dim=-1)
   sims=(vec@tv.T).squeeze(0)
  probs={task:float(torch.softmax(sims[list(pair)],dim=0)[0]) for task,pair in TASKS.items()}
  rows[str(ident)]={'source_sha256':digest,'probs':probs,'elapsed_s':round(time.monotonic()-t,4),'status':'complete'}
  if idx%12==0:print('LOCAL_CLIP_72',idx,'sec',round(time.monotonic()-start,1),flush=True)
 save(OUT/'clip_features.json',{'selection_sha256':file_sha(OUT/'selection.json'),'original_local_model_weights_sha256':model_sha,'same_frozen_CLIP_prompts':PROMPTS,'prompt_sha256':sha(json.dumps(PROMPTS).encode()),'model_load_seconds':round(loadtime,3),'total_wall_seconds':round(time.monotonic()-start,3),'peak_rss_KB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_inference_used':False,'rows':rows})
 print('CLIP_PREDICTIONS_FROZEN',len(rows),'sec',round(time.monotonic()-start,2),'sha',file_sha(OUT/'clip_features.json'))


def score():
 if (OUT/'scored.json').exists():raise RuntimeError('Already scored, not replaying')
 sel=json.loads((OUT/'selection.json').read_text());cfg=json.loads((OUT/'protocol.json').read_text())
 cv=json.loads((OUT/'pixel_features.json').read_text());geo=json.loads((OUT/'source_features.json').read_text());clip=json.loads((OUT/'clip_features.json').read_text());download=json.loads((OUT/'download_receipts.json').read_text())
 assert len(cv['rows'])==len(clip['rows'])==len(geo['rows'])==len(sel['selected'])==72
 assert all(cv['rows'][str(r['id'])]['source_sha256']==clip['rows'][str(r['id'])]['source_sha256']==download['source_original_files_sha256'][str(r['id'])]['sha256'] for r in sel['selected'])
 comparisons={name:{'ocr':0,'CLIP':0,'source_geometry':None} for name in CATS};comparisons['Content Overflow/Cut-off']['source_geometry']=0;comparisons['Occluded Content']['source_geometry']=0
 flagged={'ocr':set(),'CLIP':set(),'source_geometry':set()}
 missed_gold=[];cases=[]
 for item in sel['selected']:
  sid=item['id'];key=str(sid)
  ann=SRC/'data/annotations'/f'slide_{sid:04}.json';assert file_sha(ann)==item['annotation_sha256']
  human=json.loads(ann.read_text());t={x['design_deficiency']:x for x in human['annotations']}
  if not all(t[c]['has_strong_agreement'] and not t[c]['response'] for c in CATS):missed_gold.append(sid)
  c=cv['rows'][key];v=clip['rows'][key];g=geo['rows'][key]
  cr={'Content Overflow/Cut-off':c['clip_ocr_proposal'],'Occluded Content':c['overlap_ocr_proposal'],'Insufficient Color Contrast for Readability':c['contrast_ocr_proposal']}
  gr={'Content Overflow/Cut-off':g['source_off_slide_textbox_proposal'],'Occluded Content':g['source_object_overlap_proposal']}
  clf={k:(v['probs'][k]>.5) for k in CATS}
  if c['status']!='complete':continue
  for name in CATS:
   comparisons[name]['ocr']+=int(cr[name]);comparisons[name]['CLIP']+=int(clf[name]);
   if name in gr:comparisons[name]['source_geometry']+=int(gr[name])
  if any(cr.values()):flagged['ocr'].add(sid)
  if any(clf.values()):flagged['CLIP'].add(sid)
  if any(gr.values()):flagged['source_geometry'].add(sid)
  cases.append({'slide_id':sid,'stratum':item['stratum'],'source_SHA':c['source_sha256'],'verified_unanimous_negative':True,'cv':cr,'CLIP':clf,'source_geometry':gr,'status':c['status'],'OCR_cpu_s':c['elapsed_s'],'CLIP_inference_cpu_s':v['elapsed_s']})
 assert not missed_gold and len(cases)==72
 scores={'source_dataset_version':REV,'human_annotation_scope':'unanimous three-human absence of all three target design defect categories, NOT any flaw-free slide or live GUI','predeclared_selection_SHA256':file_sha(OUT/'selection.json'),'num_strong_negative_source':1102,'new_never_before_seen_72':True,'methods_same_original_source_images':['OCR/OpenCV pixels','CLIP pixels','source rectangles'],'binary_thresholds_from_previous_run':True,'frozen_inputs_sha256':{f:file_sha(OUT/f) for f in ('protocol.json','selection.json','download_receipts.json','pixel_features.json','clip_features.json','source_features.json')},'unanimous_negative_n':len(cases),'false_alert_by_target':comparisons,'false_alert_screen_ANY':{k:len(v) for k,v in flagged.items()},'constant_null_any_false_alert_count':0,'ocr_timeout_count':sum(x['status']!='complete' for x in cv['rows'].values()),'cost':{'pixel_OCR_CV_total_wall_s':cv['wall_s'],'pixel_OCR_CV_peak_RSS_KB':cv['peak_rss_KB'],'CLIP_total_wall_s':clip['total_wall_seconds'],'CLIP_load_wall_s':clip['model_load_seconds'],'CLIP_peak_RSS_KB':clip['peak_rss_KB'],'source_geometry_total_wall_s':geo['wall_s'],'download_wall_s':download['wall_s'],'download_MB':round(download['source_bytes']/1000000,2),'GPU_used':False},'cases':cases, 'deployment_choice':'All 72 exact three-category negative images still require REVIEW if tested in absence of property-verified source facts; flagged cases are human review candidates, not automatic failures','no_real_GUI_owner_gold':True}
 save(OUT/'scored.json',scores)
 print('THREE_CATEGORY_HUMAN_NEGATIVE_SCORE',json.dumps({k:v for k,v in scores.items() if k not in ('cases',)},sort_keys=True)[:4700],flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('freeze','download','pixel','geometry','clip','score'));x=ap.parse_args().stage;globals()[x]()
