#!/usr/bin/env python3
"""Independently score published 11 VLM outputs on 154 real SADU diagrams.

Primary data: Ouyang et al. SADU Zenodo record 19339991 (CC BY 4.0), pinned
SADU_ASE_submission.zip MD5 e1d920fd43b36442427c8c36a2e7582e.
154 real architecture diagrams, 154 graph JSON source annotations, 154 manually
curated QA files, and 11 already-generated commercial/local VLM response logs.

No live VLM inference, no model/GPU use, no network or source script execution.
Source ZIP includes a 'personal_token' directory: NEVER open, list contents,
extract or report anything within it. Only explicit whitelisted dataset/SAD
QA/Diagram/json_object entries and response_result/default + evaluated author
judge log entries are read. Raw original images and responses stay in /data/var.

Independent evaluation: exact normalized reference QA answer matches, discrete
numeric count over/under, retrieved string/pair-set elements precision/recall,
parse/format errors, and where author LLM-as-judge agrees with numeric exact
and textual set equality. Grouped bootstrap by source DIAGRAM (not QA row).
No threshold tuning, no synonym learned judge, and no claim model captions
verify correctness of diagrams generated from arbitrary user specification.
"""
from pathlib import Path
import zipfile,json,hashlib,re,unicodedata,collections,datetime,math,statistics
import numpy as np

VAR=Path('/data/var/perceptual-independent-20261024/sadu-diagrams')
SRC=VAR/'SADU_ASE_submission.zip'
RAW=VAR/'sadu_11_models_independent_qa_decisions_20261025.json'
SUMMARY=VAR/'sadu_11_models_aggregate_20261025.json'
MODELS=('claude-haiku-4-5','claude-sonnet-4-5','gemini-2.5-flash-lite','gemini-2.5-flash','gemini-3-flash-preview','gemini-3.1-flash-lite-preview','gpt-4o-mini','gpt-5-nano','qwen_2.5-VL-32B','qwen_2.5-VL-3B','qwen_2.5-VL-7B')
ROOT='SADU-main/'
COUNT_SUBTYPES={'total_entity_number','total_cluster_number','total_relation_number','how_many_entities_not_in_clusters','how_many_clusters_have_clusters_within','how_many_relation_types','how_many_relations_have_label','entities_have_more_than_one_relations','how_many_attributes_certain_entity_has','how_many_entities_does_certain_cluster_have','how_many_methods_certain_entity_has'}
RELATION_SUBTYPES={'relation_statement','type_on_the_relation','relation_source','relation_target','which_relation_has_certain_label','label_on_the_relation','label_on_the_relation_source','label_on_the_relation_target'}
SRC_MD5='e1d920fd43b36442427c8c36a2e7582e'
BOOT=2000
SEED=20261025


def norm(s):
 # No fuzzy string/synonym equivalence. A visually detected entity can differ by
 # case or underscore vs spaces, but different named parts are distinct.
 s=unicodedata.normalize('NFKC',str(s)).casefold().replace('_',' ')
 return ' '.join(s.split())


def decode_json_response(txt):
 """Use last valid JSON {answer} object, preferring official [start] markers.

 Reject malformed/no-answer data rather than interpreting free-form explanation
 as a valid model output. Some small-model responses use ```json blocks or an
 array of a single object instead of official answer dict; count format laxness.
 """
 if not isinstance(txt,str):return None,'no_text',False
 original=txt
 markers=bool(re.search(r'\[start\]',original,re.I) and re.search(r'\[end\]',original,re.I))
 chunks=[]
 if markers:
  start=list(re.finditer(r'\[start\]',txt,re.I))[-1].end()
  end=re.search(r'\[end\]',txt[start:],re.I)
  if end:chunks.append(txt[start:start+end.start()])
 chunks.append(txt)
 dec=json.JSONDecoder()
 all_candidates=[]
 for region in chunks:
  local=[]
  for i,c in enumerate(region):
   if c not in '{[':continue
   try:obj,off=dec.raw_decode(region,i)
   except (ValueError,TypeError):continue
   if isinstance(obj,dict) and 'answer' in obj:local.append((i,obj['answer']))
   elif isinstance(obj,list) and len(obj)==1 and isinstance(obj[0],dict) and 'answer' in obj[0]:local.append((i,obj[0]['answer']))
  if local:
   all_candidates=local;break
 if not all_candidates:return None,'unparseable',markers
 return all_candidates[-1][1],'parsed',markers


def normalize_gold(gold):
 if not isinstance(gold,list) or not gold:return 'unknown',None
 if len(gold)==1 and isinstance(gold[0],(int,float)) and not isinstance(gold[0],bool):return 'numeric',float(gold[0])
 if all(isinstance(x,str) for x in gold):return 'text',tuple(sorted(set(norm(x) for x in gold)))
 if all(isinstance(x,list) and len(x)==2 and all(isinstance(y,str) for y in x) for x in gold):
  return 'oriented_pairs',tuple(sorted(set(tuple(norm(y) for y in x) for x in gold)))
 return 'unhandled_gold',None


def make_tokens(value,typ):
 if typ=='numeric':
  if isinstance(value,list) and len(value)==1:value=value[0]
  if isinstance(value,(int,float)) and not isinstance(value,bool):return 'ok',float(value)
  # A model emitting the numeric count as a string is format-inexact;
  # record but do not convert to valid number silently.
  return 'wrong_type_numeric',None
 if typ=='text':
  if isinstance(value,str):value=[value]
  if isinstance(value,list) and all(isinstance(x,str) for x in value):return 'ok',tuple(sorted(set(norm(x) for x in value)))
  return 'wrong_type_text',None
 if typ=='oriented_pairs':
  if isinstance(value,list) and len(value)==2 and all(isinstance(x,str) for x in value):
   # Permissive alternate vector coding, not symmetric order: source,target.
   value=[value]
  if isinstance(value,list) and all(isinstance(x,list) and len(x)==2 and all(isinstance(y,str) for y in x) for x in value):
   return 'ok',tuple(sorted(set(tuple(norm(y) for y in x) for x in value)))
  return 'wrong_type_pairs',None
 return 'unhandled_gold',None


def score(raw_response,gold):
 typ,expected=normalize_gold(gold)
 pred,fmt,markers=decode_json_response(raw_response)
 if fmt!='parsed' or typ=='unhandled_gold':return {'gold_type':typ,'parse_status':fmt,'expected_count':len(expected) if isinstance(expected,tuple) else None,'is_correct':False,'format_marker':markers,'numeric_true':expected if typ=='numeric' else None,'numeric_pred':None,'numeric_error':None,'set_tp':0,'set_fp':0,'set_fn':len(expected) if isinstance(expected,tuple) else 0}
 validity,canonical=make_tokens(pred,typ)
 if validity!='ok':return {'gold_type':typ,'parse_status':validity,'expected_count':len(expected) if isinstance(expected,tuple) else None,'is_correct':False,'format_marker':markers,'numeric_true':expected if typ=='numeric' else None,'numeric_pred':None,'numeric_error':None,'set_tp':0,'set_fp':0,'set_fn':len(expected) if isinstance(expected,tuple) else 0}
 if typ=='numeric':
  return {'gold_type':typ,'parse_status':'ok','is_correct':canonical==expected,'format_marker':markers,'numeric_true':expected,'numeric_pred':canonical,'numeric_error':canonical-expected,'set_tp':0,'set_fp':0,'set_fn':0}
 E=set(expected);P=set(canonical)
 return {'gold_type':typ,'parse_status':'ok','is_correct':E==P,'format_marker':markers,'numeric_true':None,'numeric_pred':None,'numeric_error':None,'set_tp':len(E&P),'set_fp':len(P-E),'set_fn':len(E-P),'expected_count':len(E),'predicted_count':len(P)}


def load_model(z,name,kind):
 directory=f'{ROOT}experiment_result/{kind}/default/'
 suffix='_response_' if kind=='response_result' else '_judge_'
 names=[n for n in z.namelist() if n.startswith(directory+name+suffix) and n.endswith('.log')]
 assert len(names)==1,(name,kind,names)
 text=z.read(names[0]).decode('utf8','replace')
 arr=[json.loads(s) for s in text.splitlines() if s]
 assert len(arr)==2431 and len({int(r['index']) for r in arr})==2431
 return {int(r['index']):r for r in arr}


def verify_original_qa(z,one):
 # Cross-check all model evaluation labels with separately archived source QA
 # documents, and verify associated diagram files/graph JSON are present.
 root=f'{ROOT}dataset/SAD/'
 names=set(z.namelist())
 sources={}
 for n in sorted(name for name in names if name.startswith(root) and '/QA/' in name and name.endswith('_QA.json')):
  sections=n.split('/');family=sections[3];filename=sections[-1][:-8]
  qa=json.loads(z.read(n));key=(family,filename)
  assert key not in sources
  # normalized textual question exact plus gold answer JSON canonical
  vals=collections.Counter((x['question'],json.dumps(x['answer'],sort_keys=True,ensure_ascii=False),x['metadata']['sub_type']) for x in qa)
  sources[key]=vals
  graph=f'{root}{family}/json_object/{filename}.json'
  image_prefix=f'{root}{family}/Diagram/{filename}'
  assert graph in names
  assert any((image_prefix+ext) in names for ext in ['.png','.jpg','.jpeg']),image_prefix
 assert len(sources)==154
 counted=collections.Counter()
 for r in one.values():
  key=(r['diagram_type'],r['file'])
  gold=(r['question'],json.dumps(r['answer'],sort_keys=True,ensure_ascii=False),r['metadata']['sub_type'])
  assert sources[key][gold]>0,(r['index'],key,'source QA mismatch')
  counted[(key,gold)]+=1
  assert counted[(key,gold)]<=sources[key][gold]
 assert sum(sum(a.values()) for a in sources.values())==2431
 assert sum(counted.values())==2431
 return {'human_diagram_groups':len(sources),'QAs_matched_to_original_source_json':2431,'categories':dict(collections.Counter(k[0] for k in sources))}


def record():
 if RAW.exists():
  old=json.loads(RAW.read_text())
  if old.get('status')=='complete':print('ARCHIVED_QA_MODEL_PREVIOUSLY_SCORED_NO_REPLAY');return
 assert SRC.stat().st_size==60967413 and hashlib.md5(SRC.read_bytes()).hexdigest()==SRC_MD5
 rows={};first_keys=None;source_key_to_canonical_index=None;qa_note=None;index_integrity={}
 with zipfile.ZipFile(SRC) as z:
  zinfo=set(z.namelist());assert len(zinfo)>=890
  for name in MODELS:
   data=load_model(z,name,'response_result')
   author=load_model(z,name,'evaluation_result')
   if first_keys is None:
    first_keys={index:(r['diagram_type'],r['file'],r['question'],json.dumps(r['answer'],sort_keys=True,ensure_ascii=False),r['metadata']['sub_type']) for index,r in data.items()}
    qa_note=verify_original_qa(z,data)
    source_key_to_canonical_index={value:idx for idx,value in first_keys.items()}
    assert len(source_key_to_canonical_index)==2431
   else:
    assert set(first_keys)==set(data)
    # Authors' three Qwen logs reused per-model ordinal index ordering, not
    # canonical QA ordering. Source (type,file,question,gold,subtype) multiset
    # is identical; rejoin by exact SOURCE question identity, never by row ID.
    observed={(r['diagram_type'],r['file'],r['question'],json.dumps(r['answer'],sort_keys=True,ensure_ascii=False),r['metadata']['sub_type']) for r in data.values()}
    assert len(observed)==2431 and observed==set(first_keys.values()),(name,'questions/source mismatch')
   answers=[]
   reassigned=0
   for index in sorted(data):
    item=data[index];source_key=(item['diagram_type'],item['file'],item['question'],json.dumps(item['answer'],sort_keys=True,ensure_ascii=False),item['metadata']['sub_type'])
    canonical_idx=source_key_to_canonical_index[source_key]
    reassigned+=int(index!=canonical_idx)
    x=score(item['response'],item['answer']);src_judge=author[index]
    assert src_judge['response_obj']['index']==index
    judge=src_judge['judge_response'].lstrip()
    truth_by_author=judge.startswith('True') if judge.startswith(('True','False')) else None
    answers.append({'index':canonical_idx,'diagram_type':item['diagram_type'],'diagram_id':item['file'],'source_group_sha256':hashlib.sha256((item['diagram_type']+'|'+item['file']).encode()).hexdigest(),'task_subtype':item['metadata']['sub_type'],'author_judge_positive':truth_by_author,**x})
   assert len({r['index'] for r in answers})==2431
   rows[name]=sorted(answers,key=lambda r:r['index'])
   index_integrity[name]={'source_question_ordinals_differ_from_canonical':reassigned,'source_qa_multiset_complete':True,'join_method':'exact diagram_type+file+question+gold+subtype, not positional index'}
   print('SADU_MODEL_REANALYZED',name,'source_QA',len(answers),'numeric_count',sum(r['gold_type']=='numeric' for r in answers),'correct_count',sum(r['is_correct'] for r in answers),'format_invalid',sum(r['parse_status']!='ok' for r in answers),'judge_yes',sum(r['author_judge_positive'] is True for r in answers),flush=True)
 raw={'source':'Zenodo 19339991 official SADU dataset CC BY4, DOI 10.5281/zenodo.19339991','source_original_ZIP_MD5':SRC_MD5,'source_original_ZIP_SHA256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'date':'2026-10-25','dataset_verified':qa_note,'models_archived_not_new_inference':list(MODELS),'index_integrity':index_integrity,'scoring':{'parse':'last valid JSON answer object, strict numeric, normalized whitespace/case/underscore for entity names, set equality on retrieval and oriented pairs','retrieval_pair':'tolerates flat 2-string pair for one expected relation, preserves edge direction','grouping':'154 original diagrams, source QA and structured graph independently checked','author_judge':'compare to original LLM judge True/False; labels on numeric answers can be independently decided exactly'},'status':'complete','cases_by_model':rows,'important_caveat':'Source VLM responses were already generated by authors; no GPUs, current API endpoints, or live prompts used. Strict normalization may reject synonymous valid text that independent human adjudicator could accept. LLM-as-judge may also accept invalid sets; compare both and do not conflate.'}
 RAW.write_text(json.dumps(raw,indent=2)+'\n')
 print('SOURCE_QA_FROZEN_EVALUATED',RAW,'MODEL_COUNT',len(rows),'DIAGRAMS',qa_note['human_diagram_groups'])


def summary_group(rows):
 n=len(rows);correct=sum(r['is_correct'] for r in rows)
 typ=collections.Counter(r['gold_type'] for r in rows)
 num=[r for r in rows if r['gold_type']=='numeric']
 numeric_present=[r for r in num if r['numeric_pred'] is not None]
 nonzero=[r for r in numeric_present if r['numeric_error']!=0]
 retrieval=[r for r in rows if r['gold_type'] in ('text','oriented_pairs')]
 tp=sum(r['set_tp'] for r in retrieval);fp=sum(r['set_fp'] for r in retrieval);fn=sum(r['set_fn'] for r in retrieval)
 return {'n':n,'exact_correct':correct,'exact_accuracy':correct/n if n else None,'malformed_or_wrong_answer_type':sum(r['parse_status']!='ok' for r in rows),'answer_type_n':dict(typ),'numerical_n':len(num),'numeric_correct_n':sum(r['is_correct'] for r in num),'numeric_exact_accuracy':sum(r['is_correct'] for r in num)/len(num) if num else None,'numeric_underestimate_count':sum(r['numeric_error']<0 for r in nonzero),'numeric_overestimate_count':sum(r['numeric_error']>0 for r in nonzero),'numeric_mean_signed_error':float(statistics.mean(r['numeric_error'] for r in numeric_present)) if numeric_present else None,'numeric_mean_absolute_error':float(statistics.mean(abs(r['numeric_error']) for r in numeric_present)) if numeric_present else None,'retrieved_expected_elements_TP':tp,'retrieved_spurious_elements_FP':fp,'retrieved_missing_elements_FN':fn,'retrieval_micro_precision':tp/(tp+fp) if tp+fp else None,'retrieval_micro_recall':tp/(tp+fn) if tp+fn else None,'author_llm_judge_positive':sum(r['author_judge_positive'] is True for r in rows),'author_llm_judge_positive_rate':sum(r['author_judge_positive'] is True for r in rows)/n if n else None,'author_vs_exact_mismatch':sum(r['author_judge_positive']!=r['is_correct'] for r in rows)}


def analyze():
 if SUMMARY.exists():print('PREVIOUS_SADU_AGGREGATE_PRESERVED');return
 raw=json.loads(RAW.read_text());assert raw['status']=='complete' and len(raw['cases_by_model'])==len(MODELS)==11
 agg={'source':'154 real SADU software diagrams, 2431 human original QA, 11 archived VLM responses, all base default runs','source_archive_sha256':raw['source_original_ZIP_SHA256'],'original_diagram_sources_crosschecked':raw['dataset_verified'],'evaluation_version':'Deterministic source QA exact count or case/space/underscore normalized set comparison (not trained judge)','models':{},'scope':'models already run by external authors; this is independent replay of their prediction text against source QA, NOT user owner images or new Nitro model inference','bootstrap':'2000 draws clustered by source diagram to account for 10-20 correlated questions per image; per-model uncertainty, NOT a fair tuned same-benchmark model selection supertest'}
 for name,rows in raw['cases_by_model'].items():
  if len(rows)!=2431 or len({r['source_group_sha256'] for r in rows})!=154:raise ValueError('incomplete model source')
  cases=collections.defaultdict(list)
  for row in rows:cases[row['source_group_sha256']].append(row)
  diagrams=sorted(cases)
  rng=np.random.default_rng(SEED)
  probs=[];rel_probs=[];num_probs=[]
  for _ in range(BOOT):
   group=rng.choice(diagrams,size=len(diagrams),replace=True)
   batch=[r for grp in group for r in cases[grp]]
   probs.append(sum(r['is_correct'] for r in batch)/len(batch))
   relations=[r for r in batch if r['task_subtype'] in RELATION_SUBTYPES]
   rel_probs.append(sum(r['is_correct'] for r in relations)/len(relations) if relations else None)
   numbers=[r for r in batch if r['gold_type']=='numeric']
   num_probs.append(sum(r['is_correct'] for r in numbers)/len(numbers) if numbers else None)
  bytype={typ:summary_group([r for r in rows if r['diagram_type']==typ]) for typ in ('behavior','structural','ER')}
  bysub={typ:summary_group([r for r in rows if r['task_subtype']==typ]) for typ in sorted({r['task_subtype'] for r in rows})}
  tot=summary_group(rows)
  relation=[r for r in rows if r['task_subtype'] in RELATION_SUBTYPES]
  count=[r for r in rows if r['task_subtype'] in COUNT_SUBTYPES]
  judge_discrep={}
  for typ in ('numeric','text','oriented_pairs'):
   a=[r for r in rows if r['gold_type']==typ]
   judge_discrep[typ]={'n':len(a),'LLM_judge_says_correct_but_deterministic_wrong':sum(r['author_judge_positive'] is True and not r['is_correct'] for r in a),'deterministic_correct_but_LLM_judge_false':sum(r['author_judge_positive'] is False and r['is_correct'] for r in a)}
  intervals=lambda values:[round(float(np.quantile([v for v in values if v is not None],q)),4) for q in (.025,.975)]
  agg['models'][name]={'total':tot,'by_diagram_type':bytype,'by_subtype':bysub,'relation_retrieval':summary_group(relation),'numeric_count_tasks':summary_group(count),'strict_format_marker_fraction':sum(r['format_marker'] for r in rows)/len(rows),'source_cluster_bootstrap_CI95':{'overall':intervals(probs),'numeric':intervals(num_probs),'relation':intervals(rel_probs)},'author_judge_comparison_by_gold_type':judge_discrep}
  print('SADU_SCORE',name,'ALL',round(tot['exact_accuracy'],4),'NUMERIC',round(agg['models'][name]['numeric_count_tasks']['exact_accuracy'],4),'RELATION',round(agg['models'][name]['relation_retrieval']['exact_accuracy'],4),'RETRIEVAL_PREC',round(tot['retrieval_micro_precision'],4),'RECALL',round(tot['retrieval_micro_recall'],4),'JUDGE_DISAGREEMENT',tot['author_vs_exact_mismatch'],flush=True)
 SUMMARY.write_text(json.dumps(agg,indent=2)+'\n')
 print('SADU_COMPLETE_AGGREGATE',SUMMARY,'BYTES',SUMMARY.stat().st_size,'MODEL_COUNT',len(agg['models']))
if __name__=='__main__':record();analyze()
