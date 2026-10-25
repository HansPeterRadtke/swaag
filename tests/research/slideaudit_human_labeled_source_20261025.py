#!/usr/bin/env python3
"""Preselect REAL visually-human-reviewed slide GUI-like defects, no image peeking.

SlideAudit UIST 2025, 2400 original slide images, 3 trained human annotators
per slide, 19 visual-defect categories. Parent source pinned GitHub commit
642d490b7c1d2e78a50a631bfd359433397f3ecf, public CC BY4.

Select ALL strongly agreed positives for three specific unanswered visual
properties: content clipping/cut-off, content occlusion/overlap, and low visual
contrast/readability. For each, select equally many strongly agreed negatives
matched (source_type, alteration) where possible; deterministic SHA256 order,
source ground labels inspected BEFORE algorithm parameters/evaluation. Preserve
weak/disagreed labels as uncertainty, never convert them to ground truth.

Samples include actual and intentionally altered authored slides, NOT mobile
or web app GUIs. Image pixels kept separately from human labels; model gets
only the image. Dataset source image/annotation licensed CC-BY4 remains out of
project Git; only methods, numbers, hashes, aggregate metrics published.
"""
from pathlib import Path
import collections,csv,hashlib,json,datetime
ROOT=Path('/data/var/perceptual-independent-20261024/slideaudit-public')
DATA=ROOT/'data'
OUT=Path('/data/var/perceptual-independent-20261024/slideaudit_selection_20261025.json')
SPARSE=Path('/data/var/perceptual-independent-20261024/slideaudit_sparse_patterns_20261025.txt')
LABELS=['Content Overflow/Cut-off','Occluded Content','Insufficient Color Contrast for Readability']
REV='642d490b7c1d2e78a50a631bfd359433397f3ecf'

def hashkey(r):return hashlib.sha256(('slideaudit-holdout-20261025|'+str(r['slide_id'])).encode()).hexdigest()

def main():
 if OUT.exists():
  print('ORIGINAL_SELECTION_PRESERVED_NOT_RESELECTED');return
 assert (DATA/'annotations').is_dir()
 assert (ROOT/'.git').is_dir()
 metadata={int(r['id']):r for r in csv.DictReader((DATA/'metadata.csv').open())}
 assert len(metadata)==2400
 rows={}
 for p in sorted((DATA/'annotations').glob('slide_*.json')):
  d=json.loads(p.read_text());ident=int(d['slide_id']);assert ident not in rows
  cats={v['design_deficiency']:v for v in d['annotations']}
  assert len(cats)==19 and all(label in cats for label in LABELS)
  meta=metadata[ident];rows[ident]={'slide_id':ident,'source_type':meta['source_type'],'alteration':meta['alteration'],'width':int(meta['image_width']),'height':int(meta['image_height']),'label_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'labels':{name:{'positive':bool(cats[name]['response']),'strong':bool(cats[name]['has_strong_agreement']),'boxes':cats[name].get('bounding_boxes',[])} for name in LABELS}}
 assert len(rows)==2400
 selected=set();categories={}
 for label in LABELS:
  positive=[r for r in rows.values() if r['labels'][label]['positive'] and r['labels'][label]['strong']]
  negative=[r for r in rows.values() if not r['labels'][label]['positive'] and r['labels'][label]['strong']]
  assert len(negative)>=len(positive)
  positives=sorted(positive,key=hashkey)
  neg_by=collections.defaultdict(list)
  for r in negative:neg_by[(r['source_type'],r['alteration'])].append(r)
  for k in neg_by:neg_by[k].sort(key=hashkey)
  chosen=[];control_fallback=[]
  for pos in positives:
   k=(pos['source_type'],pos['alteration']);ls=neg_by.get(k,[])
   if ls:
    c=ls.pop(0)
   else:
    remaining=[c for v in neg_by.values() for c in v]
    c=min(remaining,key=hashkey)
    neg_by[(c['source_type'],c['alteration'])].remove(c)
    control_fallback.append({'positive_id':pos['slide_id'],'control_id':c['slide_id']})
   chosen.append((pos,c));selected.add(pos['slide_id']);selected.add(c['slide_id'])
  categories[label]={'strong_positive_count':len(positives),'matched_negative_count':len(chosen),'source_strata_matched_without_fallback':len(chosen)-len(control_fallback),'fallback_controls':control_fallback,'selected_positive_ids':[p['slide_id'] for p,c in chosen],'selected_negative_ids':[c['slide_id'] for p,c in chosen],'original_total_positive_strong_or_not':sum(x['labels'][label]['positive'] for x in rows.values()),'original_weak_positive_not_adjudicated':sum(x['labels'][label]['positive'] and not x['labels'][label]['strong'] for x in rows.values()),'original_negative_strong_count':len(negative),'original_weak_negative_not_adjudicated':sum(not x['labels'][label]['positive'] and not x['labels'][label]['strong'] for x in rows.values())}
 assert len(selected)>=300 and len(selected)<=414
 # For a given label, retain only strong-agreement cases; all other cases are
 # explicitly ABSTAIN for that binary category. This selection is not a random
 # population sample; it enriches rarer positives to evaluate miss/false rates.
 subset={str(i):rows[i] for i in sorted(selected)}
 result={'source':'SlideAudit UIST 2025 2400 original human-annotated real/synthetic presentation slide images','official_source_commit':REV,'dataset_license':'CC-BY4','source_labels':'3 independent Prolific trained crowd annotators and consensus response/has_strong_agreement','source_input_2400_annotations_sha256':hashlib.sha256(''.join(rows[i]['label_source_sha256'] for i in sorted(rows)).encode()).hexdigest(),'task_labels':LABELS,'sampling':'ALL strongly agreed positive labels per task, equal number strongly agreed negative controls matched by source_type and alteration when possible; source IDs ranked by fixed SHA256, deduplicated across labels; manual label independent of image feature. Therefore estimates are conditional on balanced tested groups, not deployment prevalence.','n_selected_unique':len(selected),'task_counts':categories,'items':subset,'selected_full_id_sha256':hashlib.sha256(','.join(map(str,sorted(selected))).encode()).hexdigest(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'image_source_paths_expected':[f'data/images/slide_{i:04d}.png' for i in sorted(selected)],'visual_prediction_must_not_read_label_manifest':True,'allowed_metrics':['image-only OCR bounding text contrast','image-only OCR bounding box clipping near screen edge','generic visual complexity as negative-control baseline'],'scientific_notes':'Slide content altered for controlled deficits: source_type Gemini/gdcdataset/Google, alteration nojitter/texture/layout/alignment. Not an original Android GUI screenshot. Controls may show other defects; no-tagged target strong negatives only. Strong positive contrast sample small. Bboxes only source annotations, NEVER given to model/predictor.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 SPARSE.write_text('/data/metadata.csv\n/data/annotations/*\n'+'\n'.join('/data/images/slide_'+str(i).zfill(4)+'.png' for i in sorted(selected))+'\n')
 print('SLIDEAUDIT_SELECTION_LOCKED',OUT,'N',len(selected),'SOURCE_SHA',result['source_input_2400_annotations_sha256'])
 for label,v in categories.items():print('LABEL',label,'POS',v['strong_positive_count'],'NEG',v['matched_negative_count'],'MATCHED_STRATA',v['source_strata_matched_without_fallback'],'FALLBACK',len(v['fallback_controls']),'WEAK_POS_EXCLUDED',v['original_weak_positive_not_adjudicated'])
 print('SPARSE_PATTERNS',SPARSE,'N_LINES',len(selected)+2)
if __name__=='__main__':main()
