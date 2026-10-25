#!/usr/bin/env python3
"""Post-hoc DIAGNOSTIC (NOT untouched test): continuous node-distance control.

Original TopoBench 69 image-only OCR and Hough pixel predictions were already
frozen and original source graph labels joined in previous evaluation. We now
compute a stronger NO-LINE-PIXEL continuous source-name-localization baseline
on the EXACT same node pairs, using reciprocal normalized Euclidean distance
between two OCR-anchored source label centers, without redoing any image OCR,
image inference, model or moving source data. Unlike earlier predeclared binary
nearest-neighbor control, this is post-hoc after seeing Hough results and MAY
alter the recommendation. It must not be portrayed as a final untouched
model-selection supertest. Bootstrap paired by actual diagram, 1500 draws.
"""
from pathlib import Path
import json,hashlib,math,statistics,collections
import numpy as np,pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from topobench_69_real_arrow_connection_eval_20261025 import anchors
ROOT=Path('/data/var/perceptual-independent-20261024/topobench-independent-20261025')
OUT=ROOT/'69_real_original_diagram_continuous_distance_null_POSTHOC_20261025.json'

def normkey(s):
 import unicodedata
 s=unicodedata.normalize('NFKC',str(s)).casefold()
 return ' '.join(''.join(c if c.isalnum() else ' ' for c in s).split())

def run():
 if OUT.exists():print('PREVIOUS_POSTHOC_SOURCE_CONTROL_PRESERVED_NOT_REPEATED');return
 raw=json.loads((ROOT/'original_69_raster_OCR_line_geometry_frozen_20261025.json').read_text())
 score=json.loads((ROOT/'69_real_diagram_source_annotated_directed_edge_recovery_20261025.json').read_text())
 t=pq.read_table(ROOT/'official_topobench_162_test_original.parquet',columns=['sample_id','annotation','has_image'])
 sources={r['sample_id']:json.loads(r['annotation']) for r in t.to_pylist() if r['has_image']}
 assert len(sources)==69
 allresult={}
 for mode in ('11','6'):
  sample=[]
  for ident,ann in sources.items():
   source=score['PSM_image_only_methods'][mode]['per_source_paired_prediction_labels'][ident]
   if source['status']!='complete':continue
   feat=raw['source_original_69_results'][ident]
   unique,_=anchors(feat['raw_tesseract_ocr'][mode]['words'],list(ann))
   diag=math.hypot(*feat['working_size_wh'])
   loc={hashlib.sha256(normkey(k).encode()).hexdigest():np.array(v['center'],dtype=float) for k,v in unique.items()}
   per=[]
   for pair in source['pairs']:
    a=pair['a_sha'];b=pair['b_sha'];assert a in loc and b in loc
    distance=float(np.linalg.norm(loc[a]-loc[b]))
    row={'original_diagram_source_id':ident,'gold':pair['gold_undirected_edge'],'score_hough':pair['Hough_score'],'score_nearest_binary':int(pair['nearest_one_way_any']),'score_geometry_distance':-distance/diag}
    per.append(row)
   sample.append({'id':ident,'rows':per})
  rows=[v for x in sample for v in x['rows']]
  y=[int(x['gold']) for x in rows];assert len(set(y))==2
  aucs={k:float(roc_auc_score(y,[x[k] for x in rows])) for k in ['score_hough','score_nearest_binary','score_geometry_distance']}
  rng=np.random.default_rng(20261025);deltas=[]
  nonzero=[x for x in sample if x['rows']]
  for _ in range(1500):
   test=[r for i in rng.integers(0,len(nonzero),size=len(nonzero)) for r in nonzero[i]['rows']]
   yy=[int(x['gold']) for x in test]
   if len(set(yy))<2:continue
   deltas.append(float(roc_auc_score(yy,[x['score_hough'] for x in test])-roc_auc_score(yy,[x['score_geometry_distance'] for x in test])))
  data={'original_diagrams_with_pair_sources':len(nonzero),'original_node_pairs_both_anchors':len(rows),'actual_annotated_source_positive_undirected_edges':sum(y),'source_negative_non_edges':len(y)-sum(y),'AUROC':aucs,'Hough_AUROC_minus_CONTINUOUS_distance_AUROC':aucs['score_hough']-aucs['score_geometry_distance'],'source_diagram_bootstrap_95pct_delta':[float(np.quantile(deltas,.025)),float(np.quantile(deltas,.975))],'no_model_training_or_threshold_selection':True}
  allresult[mode]=data
  print('POSTHOC_CONTINUOUS_SPATIAL_CONTROL_REAL_DIAGRAMS',mode,data,flush=True)
 output={'date':'2026-10-25','source':'TopoBench pinned 69 public source diagram images and externally curated graph edges, existing already frozen image-only OCR/Hough output','posthoc_diagnostic_NOT_predeclared_supertest':True,'all_69_actual_images_already_extracted_no_duplicate_processing':True,'source_sha256':{'original_source_curated_graph_scores':hashlib.sha256((ROOT/'69_real_diagram_source_annotated_directed_edge_recovery_20261025.json').read_bytes()).hexdigest(),'original_source_frozen_pixel_ocr_hough':hashlib.sha256((ROOT/'original_69_raster_OCR_line_geometry_frozen_20261025.json').read_bytes()).hexdigest()},'method':'Continuous baseline negative Euclidean distance / diagonal between OCR boxes of separately supplied source node labels; source labels used only to match original frozen OCR, not to fit distances or reference labels','results':allresult,'limitations':'Post-hoc improved baseline chosen after seeing earlier Hough score; consumed external source benchmark, not fresh final holdout. Data source node labels may be incomplete or spatially misleading. Continuous distance has no arrowhead direction and is deliberately not a trained or VLM model.'}
 OUT.write_text(json.dumps(output,indent=2)+'\n')
 print('POSTHOC_STRONGER_CONTINUOUS_BASELINE_FROZEN',OUT)
if __name__=='__main__':run()
