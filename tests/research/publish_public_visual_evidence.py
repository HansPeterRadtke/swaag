#!/usr/bin/env python3
"""Materialize ONLY aggregate results and source hashes; never publish licensed media."""
from __future__ import annotations
from pathlib import Path
import json,hashlib,datetime,statistics,collections
from PIL import Image
VAR=Path('/data/var/perceptual-independent-20261024')
PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'docs/research/perceptual-independent-evidence-2026-10-24.json'

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cm(rows):
 return {'TP':sum(x['gold_anomaly'] and x['zero_shot_predicted_anomaly'] for x in rows),
         'FP':sum(not x['gold_anomaly'] and x['zero_shot_predicted_anomaly'] for x in rows),
         'TN':sum(not x['gold_anomaly'] and not x['zero_shot_predicted_anomaly'] for x in rows),
         'FN':sum(x['gold_anomaly'] and not x['zero_shot_predicted_anomaly'] for x in rows)}
def main():
 ui=VAR/'uijudge-v0.4.0';industrial=VAR/'mvtec-loco-subset'
 dataset=ui/'real_ui_frozen_manifest.json';pred=ui/'tesseract_immutable_predictions.json';images=industrial/'original_manifest.json';clip=industrial/'clip_zeroshot_predictions.json';rapid=ui/'rapidocr_immutable_predictions.json'
 u=json.loads(pred.read_text());c=json.loads(clip.read_text());m=json.loads(dataset.read_text());ind=json.loads(images.read_text())
 assert len(u['images'])==70 and len(m['pages'])==70 and len(m['items'])==89
 assert len(c['zero_shot']['cases'])==18 and len(ind['cases'])==18
 hash_groups=collections.defaultdict(list)
 for page,info in m['pages'].items():hash_groups[info['sha256']].append(page)
 assert all('sha256' in info for info in m['pages'].values()) and len(hash_groups)==69
 duplicate_groups=[names for names in hash_groups.values() if len(names)>1]
 assert len(duplicate_groups)==1
 assert len({x['sha256'] for x in ind['cases']})==18
 expected={'dev_layout':(7,6,7,6,0),'test_layout':(3,3,0,0,0),'dev_contrast':(2,23,14,2,0),'test_contrast':(2,10,3,0,1)}
 parsed={}
 for key,values in expected.items():
  v=u['scores'][key];ct=v['counts'];got=tuple(ct[x] for x in ('TP','FP','TN','FN','ABSTAIN'))
  if got!=values:raise AssertionError(f'{key}: {got} != {values}')
  parsed[key]={'n_items':v['n_items'],'n_pages':v['n_distinct_pages'],'counts':ct,'precision':v['precision'],'recall':v['recall']}
 assert cm(c['zero_shot']['cases'])=={'TP':2,'FP':3,'TN':5,'FN':8}
 assert c['one_class_closest_normal']['scores']=={'TP':3,'FP':1,'FN':7,'TN':3,'precision':0.75,'recall':0.3}
 layout_defects=[v for v in m['items'] if v['track']=='layout' and v['task_level']=='L1' and v['ground_truth']=='no']
 visibility={'fully_outside_png':0,'at_least_partly_in_png':0,'unknown_bounds':0,'outside_examples':[]}
 for item in layout_defects:
  receipt=item.get('receipt',{});bbox=receipt.get('bbox') or (receipt.get('measured') or {}).get('bbox')
  if not bbox or len(bbox)!=4:visibility['unknown_bounds']+=1;continue
  with Image.open(m['pages'][item['page_id']]['path']) as im:width,height=im.size
  x,y,bw,bh=bbox
  intersects=(x<width and y<height and x+bw>0 and y+bh>0)
  if intersects:visibility['at_least_partly_in_png']+=1
  else:
   visibility['fully_outside_png']+=1
   visibility['outside_examples'].append({'page_id':item['page_id'],'split':item['split'],'criterion':item['criterion_code'],'bbox':bbox,'image_size':[width,height]})
 assert len(layout_defects)==16 and visibility['fully_outside_png']==12
 r=json.loads(rapid.read_text()) if rapid.exists() else {'images':{},'scores':{}}
 # Rapid run may still be ongoing. Re-record only incomplete status without claiming result.
 summary={
  'source_revisions':{'uijudgebench':'v0.4.0','mvtec_loco_subset':'saiphyohein/spec2inspect-images main as observed 2026-10-24'},
  'scope':'new independent real-site screenshots, externally labeled machine photographs; no real annotated 3D render or video defect evaluation',
  'inputs_are_in_local_var_not_this_git_commit':True,
  'sha256_manifest':{'uijudge_real_websites':h(dataset),'uijudge_raw_labels':h(ui/'items.jsonl'),'uijudge_tesseract_predictions':h(pred),'mvtec_loco_18_image_receipts':h(images),'clip_raw_predictions':h(clip)},
  'image_modality_audit':{'unique_web_png_hashes':len(hash_groups),'duplicate_hash_page_groups':duplicate_groups,'externally_labeled_defect_anchor_visibility':visibility,'warning':'most labeled L1 full-page layout defect anchors lie entirely outside the input screenshot; scores below cannot legitimately be interpreted as standalone screenshot-visible defect classifier accuracy'},
  'sample_sizes':{'website_pngs':70,'external_ui_labels':89,'industrial_photos':18,'industrial_normal':8,'industrial_anomalies':10},
  'uijudge_tesseract_psm11_vs_pixel_contrast':{'version':'Tesseract 5.3.4','image_only':True,'single_core':True,'total_seconds':round(sum(row.get('seconds',0) for row in u['images'].values()),2),'scores':parsed},
  'clip_vision_frozen_zero_shot':{'model':'openai/clip-vit-base-patch32 PyTorch CPU','images':18,'scores':c['zero_shot']['scores'],'AUROC':c['zero_shot']['AUROC'],'wall_seconds':c['time_total_seconds']},
  'clip_normal_exemplar_anomaly':{'calibration_normal':4,'heldout_normal':4,'heldout_anomaly':10,'scores':c['one_class_closest_normal']['scores'],'AUROC':c['one_class_closest_normal']['AUROC']},
  'rapidocr_status':{'completed_page_records_at_this_snapshot':len(r.get('images',{})),'full_scores_ready':bool(r.get('scores')),'note':'intermediate status is not accuracy evidence; another owned-process termination attempt was explicitly denied'},
  'caveat':'LABEL-IN-IMAGE MISMATCH: 12/16 layout anomalies not in frozen PNG, and two different mutated pages share identical screenshot pixels. These full-page gold labels are NOT complete screenshot-only ground truth. Multiple items per source webpage; class imbalance; public (non-private) benchmark; labels created by independent dataset authors, most deliberate real-site mutations; 18 photos one industrial subcategory only; no licensed third-party images in Git; no model chosen as accepted for production'
 }
 OUT.write_text(json.dumps(summary,indent=2)+'\n');print('VALIDATED',OUT,'BYTES',OUT.stat().st_size,'RAPID',summary['rapidocr_status'])
if __name__=='__main__':main()
