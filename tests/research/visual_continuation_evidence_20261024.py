#!/usr/bin/env python3
"""Verify independently persisted results and publish only aggregated measurements.

No model inference, network access, source mutation or file replay. Licensed media
remains in /data/var; published records include SHA of source+result manifests.
"""
from __future__ import annotations
import json,hashlib,collections
from pathlib import Path
VAR=Path('/data/var/perceptual-independent-20261024')
OWN=Path(__file__).resolve().parents[2]
DEST=OWN/'docs/research/perceptual-visual-continuation-evidence-2026-10-24.json'

def read(relative):
 p=VAR/relative;body=p.read_bytes()
 return json.loads(body),hashlib.sha256(body).hexdigest()

def main():
 rapid,rapid_hash=read('uijudge-v0.4.0/rapidocr_immutable_predictions.json')
 dom,dom_hash=read('uijudge-v0.4.0/firefox_dom_independent.json')
 vid,vid_hash=read('real_render_video_continuity/20261024_real_video_results.json')
 dia,dia_hash=read('diagram_mmu/diagram_source_labels_ocr_results.json')
 patch,patch_hash=read('mvtec-loco-subset/clip_patch_memory_20261024.json')
 assert len(rapid['images'])==70 and len(rapid['scores'])==4 and all(x['status']=='ok' for x in rapid['images'].values())
 assert len(dom['cases'])==32 and len(dom['scores'])==2 and all(x['status']=='ok' for x in dom['cases'].values())
 assert len(vid['cases'])==48 and len(vid['inputs'])==4
 assert len(dia['samples'])==37 and sum(len(x['gold']) for x in dia['samples'])==134
 assert patch['sample_n']==14 and patch['TP']==1 and patch['FN']==9
 assert sum(r['variant']=='intentional_hold' for r in vid['cases'])==8
 assert sum(r['variant']=='freeze_three' for r in vid['cases'])==8
 # Prevent confusion between 4 delivered variants and genuinely independent scenes.
 assert vid['parameters']['independent_media_families']==2
 bycrit={}
 for split in ('dev','test'):
  for criterion in ('redecheck:element-protrusion','redecheck:viewport-protrusion'):
   rows=[x for x in dom['cases'].values() if x['split']==split and x['criterion']==criterion]
   scores={k:0 for k in ['TP','FP','TN','FN']}
   for x in rows:
    truth=x['gold']=='violation';pred=x['predicted_violation']
    scores['TP' if truth and pred else 'FP' if pred else 'FN' if truth else 'TN']+=1
   bycrit[split+'_'+criterion]=scores
 assert sum(v['TP'] for v in bycrit.values())==11
 assert sum(v['FN'] for v in bycrit.values())==5
 assert sum(v['FP'] for v in bycrit.values())==0
 # Check all unintended and approved holds have identical pixel evidence; source
 # sequence frames are real actual rendering outputs but manipulations are controlled.
 identical_hold_pairs=0
 for group in {(x['source'],x['window_index']) for x in vid['cases']}:
  f=next(x for x in vid['cases'] if (x['source'],x['window_index'])==group and x['variant']=='freeze_three')
  h=next(x for x in vid['cases'] if (x['source'],x['window_index'])==group and x['variant']=='intentional_hold')
  assert f['pixel_hash_stream_sha256']==h['pixel_hash_stream_sha256']
  identical_hold_pairs+=1
 assert identical_hold_pairs==8
 summary={
  'task_receipt_date':'2026-10-24',
  'conclusion':'NO universally qualified GUI/diagram/3D/video visual acceptance model; several narrow instrumented deterministic methods promising but require genuine independent verification',
  'raw_local_evidence_sha256':{'rapidocr':rapid_hash,'firefox_dom':dom_hash,'real_video_frame_continuity':vid_hash,'independent_diagram_source_ocr':dia_hash,'CLIP_patch_memory':patch_hash},
  'rapidocr_vs_tesseract':{'rapidocr_version':rapid['config'],'evaluated_screenshots':len(rapid['images']),'cpu_wall_s':round(sum(x['seconds'] for x in rapid['images'].values()),3),'scores':{k:{'counts':x['counts'],'precision':x['precision'],'recall':x['recall']} for k,x in rapid['scores'].items()},'prior_warning':'benchmark labels often refer to offscreen full-page DOM; scores are not screenshot-visible classification reliability'},
  'firefox_dom_real_site_l1':{'browser':'Firefox 146/geckodriver 0.36.0, real static archived UIJudgeBench website snapshots, locator given to predictor','cases':len(dom['cases']),'split_scores':dom['scores'],'criterion_scores':bycrit,'model_style':'Browser computed-scroll geometry, no vision model','failure':'documentElement.scrollWidth misses target protrusion if ancestor clips it; element getBoundingClientRect proves those lengths, but post-hoc rule not heldout verified','not_image_only':True},
  'diagram_mmu_literal_labels':{'source':'Diagram-MMU testmini, 300 independently curated source/code diagrams in 6 categories','syntax_eligible_diagrams':len(dia['samples']),'gold_literal_node_labels':134,'psm6':{k:v for k,v in dia['results']['6'].items() if k!='misses'},'psm11':{k:v for k,v in dia['results']['11'].items() if k!='misses'},'limit':'Source code parsed with strict simple-label matcher; labels may not be legible at rendered resolution; not an entire diagram correctness test'},
  'real_render_video_controlled_frame_edits':{'source_media_families':2,'delivery_encodings':4,'windows':8,'cases':len(vid['cases']),'approved_hold_vs_freeze_identical_pixel_pairs':identical_hold_pairs,'scores':vid['results'],'source_receipts':[{k:x[k] for k in ('name','sha256','bytes','decoded_window_offsets')} for x in vid['inputs']],'ground_truth':'known exact deliberate frame edits plus explicitly approved editorial exceptions, NOT human judgements of authentic quality','runtime_s':vid['wall_seconds']},
  'clip_local_patches_on_real_industrial_photos':{'vision_checkpoint':patch['model'],'technique':patch['method'],'calibration_normals':4,'heldout_normals':4,'anomalies':10,'TP':patch['TP'],'FP':patch['FP'],'TN':patch['TN'],'FN':patch['FN'],'precision':patch['precision'],'recall':patch['recall'],'AUROC':patch['AUROC'],'runtime_seconds':patch['runtime_seconds'],'limit':'not official PatchCore; CLIP patches with four reference samples, and MVTec one category not rendered 3D mechanical physical correctness'},
  'primary_evidence':[{'source':'ScreenParse ICML 2026','url':'https://proceedings.mlr.press/v306/gurbuz26a.html','status':'author-reported; ScreenVLM untested locally'},{'source':'Diagram-MMU ECCV 2026','url':'https://vi-ocean.github.io/projects/diagram-mmu/index.html','status':'benchmark results author-reported'},{'source':'PatchCore CVPR 2022','url':'https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html','status':'official method untested here'},{'source':'EfficientAD WACV 2024','url':'https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html','status':'official method untested here'},{'source':'CGVQM+D Intel 2025','url':'https://www.intel.com/content/www/us/en/developer/articles/technical/cgvqm-d-computer-graphics-video-quality.html','status':'matched-reference perceptual rendered video metric untested here'},{'source':'NR frame freezing LIVE Mobile 2019','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6864624/','status':'subjective video evidence not downloaded; original dataset requires request'}],
  'data_retention':'External licensed/raw images and videos are not included in this Git commit. Input SHA stored, rerun scripts independent of published report.'}
 DEST.write_text(json.dumps(summary,indent=2)+'\n')
 print('AGGREGATE VERIFIED',DEST,'BYTES',DEST.stat().st_size)
 for key,val in bycrit.items():print(key,val)
if __name__=='__main__':main()
