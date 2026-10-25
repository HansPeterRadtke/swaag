#!/usr/bin/env python3
"""Actual OCR coverage of human entity labels in all 154 REAL architecture diagrams.

SADU source full 154 diagrams + independently author-annotated graph JSON, not
32 generated icon/letter figures or source metadata passed to detector.
Only original PNG pixels enter Tesseract 5.3.4 PSM11 (max long side 4096 px),
no model/GPU and no remote script. Recognized OCR text is frozen per diagram
BEFORE joining original graph names. Human ground truth includes annotated
entities that may not be legible at specific rendered resolutions; no OCR
precision is measurable because relation/legend text also appears. Missing
expected names => false negatives under a strict pixel/text reference.

Do not reopen any ZIP personal_token paths. Strictly whitelist Dataset/SAD
Diagram and json_object paths. Third-party image and OCR remains under /data/var.
Checkpoint attempts BEFORE executing OCR: unknown/timeout never automatically
retried. Only verified aggregate research files become Git changes.
"""
from pathlib import Path
import hashlib,io,json,time,tempfile,subprocess,zipfile,statistics,collections,re,unicodedata,datetime,os
from PIL import Image
ROOT=Path('/data/var/perceptual-independent-20261024/sadu-diagrams')
ZIP=ROOT/'SADU_ASE_submission.zip'
RAW=ROOT/'sadu_154_original_png_frozen_ocr_psm11_20261025.json'
SCORE=ROOT/'sadu_154_image_only_ocr_vs_graph_labels_20261025.json'
CONFIG={'source':'Zenodo SADU 19339991 154 REAL architecture diagrams, source graph labels independently annotated by authors','ocr_engine':'Tesseract 5.3.4 PSM11, standard installed English language','max_image_long_side':4096,'min_token_length':1,'ocr_seconds_max_per_diagram':16,'local_cpu_threads':1,'clip_image':'no crop, whole original chart, scaling down only when longer than 4096 pixels','text_normalization':'Unicode NFKC, casefold, underscores -> spaces, strip other punctuation, collapse spaces','ground_truth_nothing_seen_by_ocr':True,'file_selection':'every one of 154 source Diagram images in ZIP, no model output-driven filtering','limitation':'Cannot score OCR false positives against entity-only labels: many legitimate relation labels and axis/legend text excluded from entity list. Native diagrams include font/compression unreadability. Not a graph topology understanding test.'}

def normalize(txt):
 s=unicodedata.normalize('NFKC',txt).casefold().replace('_',' ')
 s=''.join(ch if ch.isalnum() else ' ' for ch in s)
 return ' '.join(s.split())

def images(z):
 names=set(z.namelist());root='SADU-main/dataset/SAD/'
 result=[]
 for n in sorted(names):
  if not n.startswith(root) or '/Diagram/' not in n or not n.endswith(('.png','.jpg','.jpeg')):continue
  parts=n.split('/');domain=parts[3];base=parts[-1].rsplit('.',1)[0]
  struct=f'{root}{domain}/json_object/{base}.json'
  assert struct in names
  result.append((n,struct,domain,base))
 assert len(result)==154 and len({(k[2],k[3]) for k in result})==154
 return result

def predict_image(data):
 with Image.open(io.BytesIO(data)) as source:
  original=source.size
  img=source.convert('RGB')
  scale=min(1.,CONFIG['max_image_long_side']/max(img.size))
  if scale<1:img=img.resize(tuple(max(1,int(v*scale)) for v in img.size),Image.Resampling.LANCZOS)
  size=img.size
  with tempfile.TemporaryDirectory(prefix='SADU-diagram-read-',dir='/data/var/tmp') as tmp:
   file=Path(tmp)/'diagram.png';img.save(file)
   req=['tesseract',str(file),'stdout','--psm','11']
   t=time.monotonic()
   try:
    p=subprocess.run(req,capture_output=True,text=True,timeout=CONFIG['ocr_seconds_max_per_diagram'],env={**os.environ,'OMP_THREAD_LIMIT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    if p.returncode:raise RuntimeError('OCR process returned '+str(p.returncode)+' '+p.stderr[-110:])
    return {'status':'complete','raw_ocr_text':p.stdout[:40000],'ocr_elapsed_s':round(time.monotonic()-t,3),'original_source_image_size_wh':list(original),'ocr_image_size_wh':list(size)}
   except subprocess.TimeoutExpired:
    return {'status':'timed_out_no_retry','ocr_elapsed_s':round(time.monotonic()-t,3),'original_source_image_size_wh':list(original),'ocr_image_size_wh':list(size)}


def freeze():
 if RAW.exists():
  result=json.loads(RAW.read_text())
  if result['status']=='complete':print('PREVIOUS_COMPLETE_REAL_DIAGRAM_OCR_PRESERVED');return
 state=json.loads(RAW.read_text()) if RAW.exists() else {'source_zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'date':'2026-10-25','config':CONFIG,'source_image_count':154,'status':'in_progress','predictions':{}}
 assert json.dumps(state['config'],sort_keys=True)==json.dumps(CONFIG,sort_keys=True)
 t=time.monotonic()
 with zipfile.ZipFile(ZIP) as z:
  for idx,(image,graph,domain,base) in enumerate(images(z)):
   key=domain+'|'+base
   if key in state['predictions']:continue
   content=z.read(image)
   row={'image_sha256':hashlib.sha256(content).hexdigest(),'domain':domain,'source_filename':base,'status':'pending_or_uncertain'}
   state['predictions'][key]=row;RAW.write_text(json.dumps(state,indent=2)+'\n')
   try:row.update(predict_image(content))
   except Exception as e:row.update(status='technical_error_no_retry',exception_type=type(e).__name__,error=str(e)[:180])
   RAW.write_text(json.dumps(state,indent=2)+'\n')
   print('DIAGRAM_OCR',idx+1,'/',154,key,row['status'],'seconds',row.get('ocr_elapsed_s'),'source_size',row.get('original_source_image_size_wh'),flush=True)
 assert len(state['predictions'])==154
 state['status']='complete';state['wall_clock_inference_s_this_iteration']=round(time.monotonic()-t,2)
 RAW.write_text(json.dumps(state,indent=2)+'\n')
 print('FROZEN_DIAGRAM_OCR_SOURCE_LABELS_NOT_YET_SCORED',RAW,'TOTAL_S',state['wall_clock_inference_s_this_iteration'])


def score():
 if SCORE.exists():print('PREVIOUS_DIAGRAM_OCR_LABEL_COMPARISON_PRESERVED');return
 a=json.loads(RAW.read_text());assert a['status']=='complete' and len(a['predictions'])==154
 records=[]
 with zipfile.ZipFile(ZIP) as z:
  for _,graph,domain,base in images(z):
   key=domain+'|'+base
   values=a['predictions'][key]
   d=json.loads(z.read(graph));entity=d['entity'];assert isinstance(entity,dict)
   names=[str(row.get('name','')) for row in entity.values() if row.get('name')]
   expected=sorted({normalize(x) for x in names if normalize(x)})
   seen=normalize(values.get('raw_ocr_text',''))
   # Exact contiguous *whole words* of each entity name, not fuzzy matching.
   found=[x for x in expected if re.search(r'(?<!\w)'+re.escape(x)+r'(?!\w)',seen)]
   # Optimistic per-token reference coverage. Can appear from unrelated legend.
   source_tokens={token for name in expected for token in name.split() if len(token)>2}
   found_tokens={token for token in source_tokens if re.search(r'(?<!\w)'+re.escape(token)+r'(?!\w)',seen)}
   records.append({'domain':domain,'source_diagram_sha256':values['image_sha256'],'human_entity_names':len(names),'human_unique_name_labels':len(expected),'image_OCR_status':values['status'],'entity_names_recovered_exactly':len(found),'entity_names_missed':len(expected)-len(found),'expected_reference_tokens':len(source_tokens),'recognized_reference_tokens':len(found_tokens),'ocr_time_s':values.get('ocr_elapsed_s'),'image_size':values.get('original_source_image_size_wh'),'ocr_screen_size':values.get('ocr_image_size_wh')})
 def summary(items):
  eligible=[r for r in items if r['image_OCR_status']=='complete']
  gold=sum(x['human_unique_name_labels'] for x in eligible);hit=sum(x['entity_names_recovered_exactly'] for x in eligible)
  words=sum(x['expected_reference_tokens'] for x in eligible);got=sum(x['recognized_reference_tokens'] for x in eligible)
  return {'total_diagrams':len(items),'scored_diagrams':len(eligible),'ocr_abstain_or_timeout':len(items)-len(eligible),'author_annotated_unique_entity_labels':gold,'exact_labels_recognized':hit,'exact_entity_name_recall':hit/gold if gold else None,'reference_unique_words':words,'individual_reference_word_coverage_upper_bound':got/words if words else None,'entire_diagrams_with_all_entity_names':sum(x['entity_names_missed']==0 for x in eligible),'total_cpu_ocr_s':round(sum(x['ocr_time_s'] for x in eligible),2)}
 groups={k:summary([r for r in records if r['domain']==k]) for k in ('behavior','structural','ER')}
 report={'source':'SADU 154 original manually structured and REAL rendered architectural diagrams; 1842 raw source entities, 154 original graph JSON files','source_zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'original_frozen_pixel_OCR_SHA256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'ocr_config':CONFIG,'n_diagrams':len(records),'overall':summary(records),'by_diagram_type':groups,'cases':records,'limits':'Strict label recall only. Human graph entity names may omit visible legend text and connectors, so extra OCR words cannot be treated as false-positive names. No relation or connection directions judged, and OCR output on too-small text can fail even for semantically correct diagram. Source supports failure-rate assessment for expected text, not complete diagram understanding. Dataset is public rather than unseen from model pretraining.'}
 SCORE.write_text(json.dumps(report,indent=2)+'\n')
 print('154_ORIGINAL_DIAGRAM_OCR_SCORING',json.dumps({'overall':report['overall'],'by_category':groups},indent=2))
if __name__=='__main__':freeze();score()
