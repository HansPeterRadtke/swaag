#!/usr/bin/env python3
"""Exact text-label recovery on independently sourced 2026 Diagram-MMU TikZ figures.

Selected by PREDEFINED strict source token syntax, without observing OCR output.
Labels come from published diagram source code, not predicted screenshot text.
This evaluates legibility/text OCR, not full diagram semantic correctness.
Do not interpret OCR extra words as false positives: gold parser incomplete.
"""
from __future__ import annotations
import pyarrow.parquet as pq
import subprocess, tempfile, pathlib, re, json,hashlib,time,statistics,os,collections
ROOT=pathlib.Path('/data/var/perceptual-independent-20261024/diagram_mmu')
SOURCE=ROOT/'testmini-diagrams.parquet';OUT=ROOT/'diagram_source_labels_ocr_results.json'
DOMAIN_ORDER=['graph_structures','planar_geometry','circuit_diagrams','3d_shapes','charts','chemistry']
RE_LABEL=re.compile(r'\\node\b[^;]*?\{\$?([A-Za-z0-9]{1,20})\$?\}\s*;',re.S)
RE_WORD=re.compile(r'[A-Za-z0-9]+')
PSMS=[6,11]

def cases():
 data=pq.read_table(SOURCE,columns=['diagram_id','domain','source_code','image']).to_pylist()
 data.sort(key=lambda r:(DOMAIN_ORDER.index(r['domain']),r['diagram_id']))
 for d in data:
  gold=sorted({x.lower() for x in RE_LABEL.findall(d['source_code']) if len(x)>=1})
  if gold:
   yield {'id':d['diagram_id'],'domain':d['domain'],'source_sha256':hashlib.sha256(d['source_code'].encode()).hexdigest(),'image_sha256':hashlib.sha256(d['image']['bytes']).hexdigest(),'expected_literal_node_labels':gold,'image_bytes':d['image']['bytes']}

def predict(image,mode):
 with tempfile.TemporaryDirectory(prefix='diagram-cv-',dir='/data/var/tmp') as td:
  p=pathlib.Path(td)/'image.png';p.write_bytes(image)
  try:
   start=time.monotonic()
   pr=subprocess.run(['tesseract',str(p),'stdout','--psm',str(mode)],capture_output=True,text=True,timeout=7,env={**os.environ,'OMP_THREAD_LIMIT':'1'})
   return {'status':'ok' if pr.returncode==0 else 'error','seconds':round(time.monotonic()-start,3),'raw_output':pr.stdout[:6000],'return_code':pr.returncode,'stderr':pr.stderr[:160] if pr.returncode else ''}
  except subprocess.TimeoutExpired:
   return {'status':'timeout','seconds':7}

def eval_scores(rows,psm):
 recall=[];totaltp=totallabels=0;allfound=0;n=0;miss=[];times=[]
 for x in rows:
  r=x['ocr'][str(psm)]
  if r['status']!='ok':continue
  pred={w.lower() for w in RE_WORD.findall(r['raw_output'])}
  truth=set(x['gold'])
  tp=len(truth&pred);n+=1;totaltp+=tp;totallabels+=len(truth)
  recall.append(tp/len(truth));allfound+=int(tp==len(truth));times.append(r['seconds'])
  if tp<len(truth):miss.append({'diagram_id':x['id'],'domain':x['domain'],'missing_labels':sorted(truth-pred),'expected_n':len(truth)})
 return {'sampled_diagrams':len(rows),'scored_diagrams':n,'source_label_instances':totallabels,'exact_labels_recovered':totaltp,'micro_recall':totaltp/totallabels if totallabels else None,'all_labels_found_diagram_fraction':allfound/n if n else None,'mean_per_diagram_recall':statistics.mean(recall) if recall else None,'total_inference_seconds':round(sum(times),2),'misses':miss}

def main():
 if OUT.is_file():print('PREVIOUS COMPLETE RESULTS EXIST NOT REPEATED');return
 manifest=[];beg=time.monotonic()
 for i,row in enumerate(cases()):
  case={k:v for k,v in row.items() if k!='image_bytes'}
  case['gold']=row['expected_literal_node_labels'];case['ocr']={}
  for psm in PSMS:case['ocr'][str(psm)]=predict(row['image_bytes'],psm)
  manifest.append(case)
  print('DIAGRAM',i+1,'DOMAIN',row['domain'],'GOLD_COUNT',len(case['gold']),'PSM',[(p,case['ocr'][str(p)]['status'],case['ocr'][str(p)]['seconds']) for p in PSMS],flush=True)
 final={'source':'AIGrounding/Diagram-MMU testmini canonical source 300 independent diagrams across six domains','source_parquet_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'task':'OCR source-code literal node label recovery only; not general semantic diagram correctness','predeclared_selection':'all \\node occurrences with exactly one simple ASCII alphanumeric text label in braces, math $...$ accepted; no OCR-driven sampling','predeclared_psms':PSMS,'samples':manifest,'results':{str(p):eval_scores(manifest,p) for p in PSMS},'total_wall_seconds':round(time.monotonic()-beg,2),'limitations':'Labels derived from source TikZ and strict subset parser; node text may be converted to curves or indistinguishable at displayed resolution; parser incomplete and excludes rich LaTeX. No false-positive rate can be soundly calculated from this incomplete label inventory; source code ground truth may differ from pixel visibility at a given render. PSM comparison on same data is development, not unbiased model selection.'}
 OUT.write_text(json.dumps(final,indent=2)+'\n')
 for p in PSMS:
  s=final['results'][str(p)];print('SCORE_PSM',p,s['scored_diagrams'],'LABELS',s['source_label_instances'],'FOUND',s['exact_labels_recovered'],'MICRO_RECALL',s['micro_recall'],'ALL_FRAC',s['all_labels_found_diagram_fraction'],'OCR_SECONDS',s['total_inference_seconds'],flush=True)
 print('COMPLETED',OUT,'TOTAL_SECONDS',final['total_wall_seconds'],flush=True)
if __name__=='__main__':main()
