#!/usr/bin/env python3
"""Evaluate SOURCE-BLIND OCR+line evidence on independently annotated REAL diagrams.

TopoBench 69 redistributed diagrams, 54 AI2D and 15 public licensed network
images, 162 source annotations in total but 93 must NOT be downloaded via
link-only image URLs with unresolved redistribution rights. Original graph
node names and directed relation source labels are joined only AFTER 69
Tesseract OCR TSV outputs and Hough line segments were frozen from pixels.

No trained vision model, no source graph passed to image feature extractor.
The scorer uses AUTHORITATIVE SOURCE NAMES ONLY to match text boxes in the
already frozen OCR output, to ask a narrower relevant engineering question:
with names localized in true rendered pixels, is an actual drawn line/arrow
recoverable reliably as a graph edge? Unknown OCR node anchors ABSTAIN and
reduce end-to-end coverage, never count as accepted no-edge cases.

Two fixed candidate OCR PSM 6/11 modes, one line-alignment detector, and
same-budget spatial nearest-neighbor and no-edge controls were predeclared.
No threshold selected on labels; score endpoint/orientation separately.

LINE SCORE: max over Hough segments from actual pixels of cosine alignment
>=.82, segment projection overlap on node-center baseline, perpendicular
error relative image diagonal with exponential scale .06; event if score>.32.
Baseline k=1 neighbor undirected if either OCR grounded node names the other
as closest in center distance, no pixels except text locations. The no-edge
baseline catches dataset class imbalance. Directed direction-of-reading
baseline (left-right when dx dominates, top-down otherwise) and symmetric
unknown arrowhead inference require explicit limits; DO NOT infer direction
from Hough line which has no intrinsic arrowhead without independent proof.

Independent target: original annotated unique directed source relations, and
all absent unordered node pairs as negative controls. A pair is scorable only
when both endpoint labels were located uniquely from image OCR; ambiguous or
missing names reduce coverage. Reciprocally directed edges evaluated as
undirected connections but NOT falsely declared single-direction arrows.

Actual source graph annotation is a curated external reference, not guaranteed
100% human-verified pixel truth or fresh untouched final product; source
network diagrams vary in style and may include invisible or unlabeled edges.
No media/GPU/render state modified. Errors and input-source SHA preserved.
"""
from pathlib import Path
import json,hashlib,collections,math,re,unicodedata,statistics,itertools
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/data/var/perceptual-independent-20261024/topobench-independent-20261025')
PARQUET=ROOT/'official_topobench_162_test_original.parquet'
RAW=ROOT/'original_69_raster_OCR_line_geometry_frozen_20261025.json'
OUT=ROOT/'69_real_diagram_source_annotated_directed_edge_recovery_20261025.json'
CONFIG={'source':'TopoBench 69 legally redistributed original real diagram test images with source node and directed relation annotations, no other 93 link-only images fetched','author_pinned_revision':'df366de6a2c8aaf83c0a95ba8bfb1a28e231cb14','ocr_psm_predeclared':[11,6],'OCR_unique_anchors':'exact normalized full source node name token sequence in one Tesseract block/paragraph/line, conf >=30, deduplicate matching text locations and ABSTAIN if multiple/no matches','OCR_source_normalization':'NFKC casefold and non-letter-digit -> spaces, exact consecutive normalized tokens, no fuzzy Levenshtein tuning','line_rule':'Hough segments length >=2.4pct diagonal from original pixels, maximum cosine>0.82, max of cosine * segment-projection-overlap-fraction * exp(-mean_endpoint_perp_distance/(.06*diag)), threshold >0.32','nearest_neighbor_baseline':'If either source node is the nearest text-anchor neighbor of the other in actual OCR box-center pixel coordinates, mark undirected connection regardless of line pixels','no_link_baseline':'Always classify no source edge, valid for measuring false-positive class imbalance, not real detection','directed_null':'Only after predicted undirected link, assume arrow direction left-to-right when horizontal dominance or top-to-bottom otherwise. This is a layout shortcut, not arrowhead vision. Exact directed edge scoring reported separately, not conflated with undirected edge recognition.','source_negative_controls':'Every unique unordered pair of uniquely OCR-grounded source node names with NO annotated relationship in either direction','independent_source_labels':'Curated third-party graph labels; human reliability/exhaustiveness not independently certified, ensure this caveat','threshold_fit_to_labels':False,'unit_of_grouped_bootstrap':'original distinct diagram ID, 1500 replicates seed 20261025'}


def norm(s):
 s=unicodedata.normalize('NFKC',str(s)).casefold()
 return ' '.join(''.join(c if c.isalnum() else ' ' for c in s).split())

def anchors(raw_words,names):
 groups=collections.defaultdict(list)
 for item in raw_words:
  if item['conf']<30:continue
  text=norm(item['text']);toks=text.split()
  if not toks:continue
  groups[(item['page'],item['block'],item['paragraph'],item['line'])].append((int(item['x']),item,toks))
 sequences=[]
 for key,words in groups.items():
  words.sort(key=lambda z:z[0])
  seq=[]
  for _,obj,tok in words:
   for t in tok:seq.append((t,obj))
  sequences.append(seq)
 boxes={};multi=0
 for name in names:
  target=norm(name).split()
  if not target:continue
  matches=[]
  for seq in sequences:
   for i in range(len(seq)-len(target)+1):
    if [x[0] for x in seq[i:i+len(target)]]!=target:continue
    src=[x[1] for x in seq[i:i+len(target)]]
    x0=min(o['x'] for o in src);y0=min(o['y'] for o in src)
    x1=max(o['x']+o['w'] for o in src);y1=max(o['y']+o['h'] for o in src)
    matches.append((x0,y0,x1,y1))
  matches=list(set(matches))
  if len(matches)==1:
   x0,y0,x1,y1=matches[0];boxes[name]={'x0':x0,'y0':y0,'x1':x1,'y1':y1,'center':[(x0+x1)/2,(y0+y1)/2]}
  elif len(matches)>1:multi+=1
 return boxes,multi


def geomscore(a,b,segs,diag):
 x1,y1=a['center'];x2,y2=b['center'];dx=x2-x1;dy=y2-y1
 L=math.hypot(dx,dy)
 if L<max(3,.007*diag):return 0.
 u=np.array([dx/L,dy/L],dtype=float);v=np.array([-u[1],u[0]])
 seg=np.asarray(segs,dtype=float)
 if len(seg)==0:return 0.
 endpoints=seg.reshape(-1,2,2)
 delta=endpoints[:,1,:]-endpoints[:,0,:];lens=np.linalg.norm(delta,axis=1)
 good=lens>max(3,.018*diag)
 if not np.any(good):return 0.
 proj_cos=np.abs(delta@u)/(lens+1e-9)
 good&=(proj_cos>=.82)
 if not np.any(good):return 0.
 e=endpoints[good]
 # projection of candidate image line relative to node center baseline
 q=(e-np.array([x1,y1])).reshape(-1,2,2)
 projections=q@u
 overlap=np.minimum(L,np.max(projections,axis=1))-np.maximum(0,np.min(projections,axis=1))
 overlap=np.maximum(0,overlap)/L
 perpa=np.abs((np.array([x1,y1])-e[:,0,:])@v)
 perpb=np.abs((np.array([x2,y2])-e[:,0,:])@v)
 attenuation=np.exp(-((perpa+perpb)/2)/(.06*diag))
 vals=proj_cos[good]*overlap*attenuation
 return float(vals.max()) if len(vals) else 0.


def evaluate_binary(rows,key):
 tp=sum(r['gold_undirected_edge'] and bool(r[key]) for r in rows)
 fp=sum(not r['gold_undirected_edge'] and bool(r[key]) for r in rows)
 fn=sum(r['gold_undirected_edge'] and not bool(r[key]) for r in rows)
 tn=sum(not r['gold_undirected_edge'] and not bool(r[key]) for r in rows)
 return {'scored_pairs':len(rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'precision':tp/(tp+fp) if tp+fp else None,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn and tn+fp) else None}


def pair_results(entry,features,psm):
 anno=json.loads(entry['annotation']);names=list(anno)
 name2norm={norm(n):n for n in names if norm(n)}
 if len(name2norm)!=len(names):return {'status':'ambiguous_original_source_node_names','node_total':len(names),'unique_anchor_nodes':0,'pairs':[],'missed_gold_relations_due_unanchored':entry['relation_count']}
 raw=features['raw_tesseract_ocr'][str(psm)]
 if raw['status']!='complete':return {'status':'ocr_abstain','node_total':len(names),'unique_anchor_nodes':0,'pairs':[],'missed_gold_relations_due_unanchored':entry['relation_count']}
 boxes,multi=anchors(raw['words'],names)
 # relations may contain equivalent nodes as written, even with distinct case.
 edges=set();bad=0;reltypes=collections.Counter()
 for node,obj in anno.items():
  for r in obj.get('relations',[]):
   src=name2norm.get(norm(r['source']));dst=name2norm.get(norm(r['target']))
   if src is None or dst is None:bad+=1;continue
   if src==dst:continue
   edges.add((src,dst));reltypes[str(r.get('type','unknown'))]+=1
 # dedup (u,v) to evaluate binary connection, plus directed edge list
 ground=set(frozenset(x) for x in edges)
 loc=list(boxes)
 point={key:np.array(boxes[key]['center'],dtype=float) for key in loc}
 nearest={}
 for a in loc:
  neighbors=sorted([b for b in loc if b!=a],key=lambda b:float(np.linalg.norm(point[a]-point[b])))
  nearest[a]=neighbors[0] if neighbors else None
 diag=math.hypot(*features['working_size_wh'])
 matches=[]
 for a,b in itertools.combinations(loc,2):
  h=geomscore(boxes[a],boxes[b],features['hough_line_segments'],diag)
  image_link=h>.32
  nnear=nearest[a]==b or nearest[b]==a
  real=frozenset([a,b]) in ground
  # Layout-only directed null uses location ordering, *not* true arrow evidence.
  diff=point[b]-point[a]
  prefer_ab=(diff[0]>0) if abs(diff[0])>=.4*abs(diff[1]) else (diff[1]>0)
  predicted=(a,b) if prefer_ab else (b,a)
  matches.append({'a_sha':hashlib.sha256(norm(a).encode()).hexdigest(),'b_sha':hashlib.sha256(norm(b).encode()).hexdigest(),'gold_undirected_edge':real,'gold_a_to_b':(a,b) in edges,'gold_b_to_a':(b,a) in edges,'Hough_score':round(h,6),'Hough_flag':image_link,'nearest_one_way_any':nnear,'never_link':False,'Hough_layout_direction_guess_is_correct':predicted in edges if image_link else None,'Hough_layout_direction_guess':image_link,'has_unambiguous_directed_truth':((a,b) in edges)!=((b,a) in edges),'layout_direction_guess_positive':image_link and predicted in edges})
 unseen=sum(frozenset(x) not in {frozenset([a,b]) for a,b in itertools.combinations(loc,2)} for x in ground)
 return {'status':'complete','source_node_total':len(names),'original_distinct_directed_edges':len(edges),'original_distinct_undirected_links':len(ground),'original_relation_type_counts':dict(reltypes),'bad_source_relation_refs':bad,'unique_ocr_grounded_node_count':len(loc),'ambiguous_label_count':multi,'image_scorable_unordered_node_pairs':len(matches),'positive_edge_pairs_ungrounded_or_unknown':unseen,'pairs':matches}


def pooled(records,key):
 rows=[r for rec in records.values() if rec['status']=='complete' for r in rec['pairs']]
 stats=evaluate_binary(rows,key)
 score=[r['Hough_score'] for r in rows];label=[r['gold_undirected_edge'] for r in rows]
 if len(set(label))==2:stats['Hough_AUROC_same_pairs']=float(roc_auc_score(label,score))
 return stats


def boot(records,mode='11',n=1500):
 # Bootstrap entire original independent source diagrams; no per-edge IID.
 cases=[v for v in records.values() if v['status']=='complete' and v['pairs']]
 if len(cases)<10:return None
 rng=np.random.default_rng(20261025);vals={'Hough_balanced_accuracy':[],'nearest_balanced_accuracy':[],'Hough_AUROC':[],'Hough_AUROC_minus_layout_nearest_AUROC':[]}
 for i in range(n):
  batch=[x for idx in rng.integers(0,len(cases),size=len(cases)) for x in cases[idx]['pairs']]
  try:
   y=[r['gold_undirected_edge'] for r in batch]
   if len(set(y))<2:continue
   hu=evaluate_binary(batch,'Hough_flag')['balanced_accuracy'];nn=evaluate_binary(batch,'nearest_one_way_any')['balanced_accuracy']
   if hu is None or nn is None:continue
   roc=float(roc_auc_score(y,[r['Hough_score'] for r in batch]))
   null=float(roc_auc_score(y,[int(r['nearest_one_way_any']) for r in batch]))
   vals['Hough_balanced_accuracy'].append(hu);vals['nearest_balanced_accuracy'].append(nn);vals['Hough_AUROC'].append(roc);vals['Hough_AUROC_minus_layout_nearest_AUROC'].append(roc-null)
  except ValueError:continue
 return {'resampling':'source diagrams, 1500 samples, seed20261025','source_diagrams':len(cases),'CI95':{k:[round(float(np.quantile(v,q)),4) for q in [.025,.975]] if len(v)>n//2 else None for k,v in vals.items()}}


def run():
 if OUT.exists():print('ALREADY_PERFORMED_ORIGINAL_TOPOBENCH_ANALYSIS_PRESERVED');return
 raw=json.loads(RAW.read_text());assert raw['status']=='complete' and len(raw['source_original_69_results'])==69
 tbl=pq.read_table(PARQUET,columns=['sample_id','annotation','domain','node_count','relation_count','has_image','license_status','redistribution'])
 rows=[x for x in tbl.to_pylist() if x['has_image']]
 assert len(rows)==69 and {x['sample_id'] for x in rows}==set(raw['source_original_69_results'])
 allcases={}
 for psm in CONFIG['ocr_psm_predeclared']:
  each={}
  for row in rows:
   key=row['sample_id'];e=pair_results(row,raw['source_original_69_results'][key],psm)
   e['source_domain']=row['domain'];e['original_declared_node_count']=row['node_count'];e['original_declared_relation_count']=row['relation_count'];each[key]=e
  vals=[pair for c in each.values() for pair in c.get('pairs',[])]
  annotated=[c for c in each.values() if c['status']=='complete']
  if not vals:raise RuntimeError('all OCR grounded pairs unavailable, record abstain rather than claim success')
  totals={'raster_diagrams':len(each),'ocr_parsed_diagrams':len(annotated),'total_original_nodes':sum(c.get('source_node_total',0) for c in each.values()),'unique_OCR_grounded_nodes':sum(c.get('unique_ocr_grounded_node_count',0) for c in each.values()),'total_annotated_undirected_links':sum(c.get('original_distinct_undirected_links',0) for c in each.values()),'total_unscorable_links_due_unrecognized_nodes':sum(c.get('positive_edge_pairs_ungrounded_or_unknown',0) for c in each.values()),'total_canonical_pairs_both_endpoints_grounded':len(vals),'source_positive_links_both_grounded':sum(r['gold_undirected_edge'] for r in vals),'source_negative_links_both_grounded':sum(not r['gold_undirected_edge'] for r in vals),'unambiguous_one_way_ground_pairs':sum(r['has_unambiguous_directed_truth'] for r in vals),'hough_geometry':evaluate_binary(vals,'Hough_flag'),'source_assisted_OCR_label_nearest_neighbor':evaluate_binary(vals,'nearest_one_way_any'),'no_link_baseline':evaluate_binary(vals,'never_link'),'Hough_score_AUROC':float(roc_auc_score([r['gold_undirected_edge'] for r in vals],[r['Hough_score'] for r in vals])) if len({r['gold_undirected_edge'] for r in vals})==2 else None,'same_source_diagram_CI':boot(each),'by_source_domain':{domain:{'n_diagrams':sum(v['source_domain']==domain for v in each.values()),'Hough':evaluate_binary([r for v in each.values() if v['source_domain']==domain for r in v.get('pairs',[])],'Hough_flag'),'nearest_baseline':evaluate_binary([r for v in each.values() if v['source_domain']==domain for r in v.get('pairs',[])],'nearest_one_way_any')} for domain in ['ai2d_web','network']}}
  allcases[str(psm)]={'metrics':totals,'per_source_diagram_summary':{k:{z:v[z] for z in ['status','source_domain','source_node_total','original_distinct_directed_edges','original_distinct_undirected_links','unique_ocr_grounded_node_count','ambiguous_label_count','positive_edge_pairs_ungrounded_or_unknown','image_scorable_unordered_node_pairs'] if z in v} for k,v in each.items()},'per_source_paired_prediction_labels':each}
  print('TOPOBENCH_69_REAL_PIXEL_ARROW_EDGE_PSM',psm,'SOURCE_ANNOTATED_IMAGE_PAIRS',len(vals),'ACTUAL_GRAPH_POSITIVE',totals['source_positive_links_both_grounded'],'NEGATIVE',totals['source_negative_links_both_grounded'],'RECOGNIZED_NODES',totals['unique_OCR_grounded_nodes'],'OF',totals['total_original_nodes'],'HOUGH',totals['hough_geometry'],'BASELINE',totals['source_assisted_OCR_label_nearest_neighbor'],'AUC',totals['Hough_score_AUROC'],flush=True)
 out={'date':'2026-10-25','source':'WayneGuo0011/TopoBench-180 69 redistributed real diagram images, original curated source node directed graphs; public pinned HF parquet','original_parquet_sha256':hashlib.sha256(PARQUET.read_bytes()).hexdigest(),'image_only_frozen_features_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'source_annotation_counts':{'full_test_source_entries':162,'source_images_redistributed':69,'link_only_excluded':93,'domains':dict(collections.Counter(r['domain'] for r in rows)),'source_license_statuses':dict(collections.Counter(r['license_status'] for r in rows))},'config':CONFIG,'PSM_image_only_methods':allcases,'scope_limits':['Image-only OCR+Hough receives no graph relationships until predictions are frozen; matching source node names to already recognized words is source-assisted verification, not generic zero-shot topology interpretation.','Source curated graph annotations may not be perfect human-certified source-vs-image truth; 15 network and 54 AI2D are stylistically different; original IDs within a source family correlated.','Only OCR unique endpoint pairs are scored. Missing or ambiguous text labels are ABSTAINS not correct negatives, so conditional edge precision can substantially overestimate entire diagram success.','Line segments detect graphical strokes, but cannot infer arrowhead direction; reading-order baseline is NOT arrow semantics or connection proof.','No image/annotation used to tune OCR, Hough or nearest baseline; post-hoc choosing best would consume this test set.','No image, raw OCR text, PDF screenshot or restricted link-only image published to Git, only compact aggregates and reproducible code.','When graph contains both A->B and B->A, unordered link truth treats them as one connection, and arrow-direction cannot be judged from Hough alignment alone.','No source label correspondences to original intended 3D physical/GUI/financial diagram output; this is a separate diagram topology source benchmark.']}
 OUT.write_text(json.dumps(out,indent=2)+'\n')
 print('SAVED_REAL_TOPOBENCH_69_ANNOTATED_GRAPH_EVIDENCE',OUT,'BYTES',OUT.stat().st_size,flush=True)
if __name__=='__main__':run()
