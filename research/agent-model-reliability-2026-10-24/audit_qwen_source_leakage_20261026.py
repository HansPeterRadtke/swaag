#!/usr/bin/env python3
"""Read-only, source-pinned audit of completed Qwen SWAAG relevance evidence.

Never sends a model request; never resumes/replays incomplete trials. Requires
published local original project document hashes and pre-existing research
journals to match. Gold is explicitly researcher-authored and PID-leaking.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import math
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
EVIDENCE=HERE/'evidence/context-grounding'
VERSION='swaag-archived-qwen-source-leakage-audit-v1-20261026'
HASHES={
 'context_grounding_v1.json':'74e9e1bcd1e4381971374d11f73bffb0145e7af35294134b85b5c50fc0ceb321',
 'build_context_grounding_v1.py':'e5144d70204546d70e12a67c8c708a403e7edbd32301d6b112b40a2a77d40542',
 'evidence/context-grounding/observations-final.jsonl':'431a4677f184c9a52e543995a16963ec990d3dbbdbe3218bf18e6b5d4bbc5ffe',
 'evidence/context-grounding/observations.jsonl':'7273cb92d1b3835fb98a14884d7efffa00ab3fa3a1a4cea7fd577b74c0d48325',
 'evidence/context-grounding/observations-overnight1.jsonl':'e71a5471b997b6d831071203b546c6b7788f739d1b4292f496fe6e842a4f6b93',
 'evidence/context-grounding/old_history_authority_coder_wrong_20261025.jsonl.gz':'776b47b0da36fcc35f70c9a050faec34e5c1e196420f074aac50377addb7fd51',
 'evidence/context-grounding/overnight4_two_qwen_completed.jsonl.gz':'2e1b246f4faa650f98c7b8076df844d54e6e31884e90a9974fe0b2a3d45550f6',
 'evidence/context-grounding/three_partial_selectors_no_answer_20261025.jsonl.gz':'f868ae7c7c41957593ffec3fa49d9b4d75f30a3df3542a95fe6be7069b33fd9a',
}
PORT_MODELS={
 14833:'qwen3-coder',
 14835:'qwen38',
}
CONDITIONS=('full','gold_select','model_select')


def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset():
    for rel,h in HASHES.items():
        actual=hashfile(HERE/rel)
        if actual!=h:raise ValueError(f'archive changed: {rel}: {actual} != {h}')
    source=json.loads((HERE/'context_grounding_v1.json').read_text())
    cases=source['cases']
    if len(cases)!=8 or len({c['case_id'] for c in cases})!=8:raise ValueError('case manifest changed')
    original_files={}
    for c in cases:
        passages=c['passages']
        if len(passages)!=5 or len({p['pid'] for p in passages})!=5:
            raise ValueError('case passage cardinality or IDs changed')
        if not set(c['support_ids'])<={p['pid'] for p in passages}:
            raise ValueError('missing support')
        for p in passages:
            relative=Path(p['file'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe source path')
            path=PROJECT/relative
            actual_hash=hashfile(path)
            if actual_hash!=p['file_sha256']:raise ValueError('source hash mismatch '+str(relative))
            if path.read_text(encoding='utf8')[p['start']:p['end']]!=p['text']:
                raise ValueError('source passage span mismatch')
            original_files[str(relative)]=actual_hash
    return cases,original_files


def deduplicate(cases):
    planned={f"ground-v1:{c['case_id']}:{port}:{variant}:s17"
             for c in cases for port in PORT_MODELS for variant in CONDITIONS}
    by_key={}; source_files=defaultdict(set); count=0; duplicates=0
    for relative in HASHES:
        if not relative.startswith('evidence/context-grounding/'):
            continue
        path=HERE/relative
        op=gzip.open if path.suffix=='.gz' else open
        with op(path,'rt',encoding='utf8') as f:
            for line in f:
                if not line.strip():continue
                e=json.loads(line)
                if e.get('event') not in ('begin','request_start','response','completed','stopped'):
                    raise ValueError('unknown event')
                tid=e.get('trial_id')
                if tid not in planned:raise ValueError('unknown or unrelated trial: '+str(tid))
                key=(tid,e['event'],e.get('stage'))
                # Preserve the original event verbatim and reject divergent copies.
                signature=json.dumps(e,sort_keys=True,ensure_ascii=False)
                if key in by_key:
                    if by_key[key]!=signature:raise ValueError('conflicting duplicated original event '+str(key))
                    duplicates+=1
                else:by_key[key]=signature
                source_files[tid].add(relative)
                count+=1
    sessions=defaultdict(dict)
    for (tid,event,stage),txt in by_key.items():
        sessions[tid][event+(':'+str(stage) if stage is not None else '')]=json.loads(txt)
    if not set(sessions)<=planned:raise ValueError('unexpected session')
    assert count==len(by_key)+duplicates
    return sessions,{'planned':len(planned),'observed_trial_ids':len(sessions),
                     'original_event_records':count,'unique_events':len(by_key),
                     'byte_identical_overlaps':duplicates,
                     'trial_ids_only_in_unattempted_plan':len(planned)-len(sessions)},source_files


def words(text):return re.findall(r"[a-z0-9_]+",text.lower())


def bm25_ids(question,passages,k):
    # Frozen lexical baseline: BM25 k1=1.2,b=0.75, questions only, no IDs or
    # researcher answer-options/supports in the ranking query.
    docs=[words(p['text']) for p in passages];q=set(words(question))
    n=len(docs);meanlen=sum(map(len,docs))/n
    df=Counter(token for doc in docs for token in set(doc))
    weights=[]
    for i,doc in enumerate(docs):
        tf=Counter(doc);length=len(doc);score=0.0
        for token in q:
            freq=tf[token]
            if freq==0:continue
            idf=math.log1p((n-df[token]+0.5)/(df[token]+0.5))
            score+=idf*freq*(1.2+1)/(freq+1.2*(1-0.75+0.75*length/meanlen))
        # ID-blind tie key: immutable original file and span, never pid/support.
        weights.append((score,passages[i]['file'],passages[i]['start'],i))
    ranked=sorted(weights,key=lambda x:(-x[0],x[1],x[2]))
    return [passages[i]['pid'] for _,_,_,i in ranked[:k]]


def confusion(c,ids):
    allids={p['pid'] for p in c['passages']}; chosen=set(ids);gold=set(c['support_ids'])
    if not chosen<=allids or len(chosen)!=len(ids):raise ValueError('selector invalid IDs')
    return {'TP':len(chosen&gold),'FP':len(chosen-gold),
            'FN':len(gold-chosen),'TN':len(allids-(chosen|gold))}


def cumulative(items):
    counts=Counter()
    for row in items:counts.update(row)
    positive=counts['TP']+counts['FN'];negative=counts['FP']+counts['TN']
    return {**{k:counts[k] for k in ['TP','FP','FN','TN']},
            'support_recall':round(counts['TP']/positive,6) if positive else None,
            'unsupported_source_FPR':round(counts['FP']/negative,6) if negative else None,
            'selection_precision':round(counts['TP']/(counts['TP']+counts['FP']),6) if counts['TP']+counts['FP'] else None}


def audit():
    cases,source_files=dataset(); lookup={c['case_id']:c for c in cases}
    sessions,archive_counts,provenance=deduplicate(cases)
    direct=[]; gold_selection=[]; selection=[]; unfinished=[]; preflight=[]
    for tid in sorted(sessions):
        e=sessions[tid]
        # Parse original random-order schedule identifier but do not replay it.
        _,case_id,port,variant,seed=tid.split(':');port=int(port)
        if seed!='s17' or variant not in CONDITIONS:raise ValueError('bad trial ID')
        begin=e.get('begin')
        if begin is None:
            if set(e)!={'stopped'} or e['stopped'].get('event')!='stopped':
                raise ValueError('unstarted trial contains returned model output')
            preflight.append({'trial_id':tid,'reason_category':'admission_refused_before_inference'})
            continue
        if begin['case']!=lookup[case_id] or begin['variant']!=variant or begin['seed']!=17:
            raise ValueError('frozen trial/case/seed mismatch')
        if begin['ground_truth']!=lookup[case_id]['gold_answer']:
            raise ValueError('researcher expectation drift')
        model=begin['model'];expected=PORT_MODELS[port]
        if expected not in model.lower():raise ValueError('model route mismatch')
        complete=e.get('completed');stopped=e.get('stopped')
        r_answer=e.get('response:answer');r_selector=e.get('response:selector')
        if complete is not None:
            if stopped is not None or r_answer is None or r_selector is not None:
                raise ValueError('inconsistent recorded completed inference')
            if complete['status']!='scored' or complete['correct']!=(complete['answer']==lookup[case_id]['gold_answer']):
                raise ValueError('original final scoring differs from authored expected answer')
            row={'trial_id':tid,'case_id':case_id,'model_id':model,'model_family':expected,
                 'variant':variant,'answer':complete['answer'],'matches_researcher_answer':complete['correct'],
                 'tokens':complete['total_tokens'],'wall_seconds':complete['total_wall_seconds'],
                 'source_count':len(complete['selected_ids'])}
            if variant=='full':
                if set(complete['selected_ids'])!={p['pid'] for p in lookup[case_id]['passages']}:
                    raise ValueError('full source omitted original passage')
                direct.append(row)
            elif variant=='gold_select':
                if set(complete['selected_ids'])!=set(lookup[case_id]['support_ids']):
                    raise ValueError('gold-select did not receive privileged label')
                gold_selection.append(row)
            else:raise ValueError('unexpected complete model_select')
        elif stopped is not None:
            if variant!='model_select' or r_selector is None or r_answer is not None:
                raise ValueError('unknown stop state; require manual audit')
            ids=r_selector['decoded'].get('ids')
            if not isinstance(ids,list) or not 1<=len(ids)<=3:
                raise ValueError('selector invalid source IDs')
            c=lookup[case_id]
            measured={'trial_id':tid,'case_id':case_id,'model_id':model,'model_family':expected,
                      'selected_ids':ids,'tokens':r_selector['usage']['total_tokens'],
                      'wall_seconds':r_selector['wall_seconds'],'source_count':len(ids),
                      'source_confusion':confusion(c,ids)}
            # Predeclare actual same-source, equal number-of-source-slots
            # comparators, not model-vs-model reasoning equivalence.
            measured['bm25_ids_same_source_budget']=bm25_ids(c['question'],c['passages'],len(ids))
            measured['bm25_confusion']=confusion(c,measured['bm25_ids_same_source_budget'])
            # Zero-inference trivial exploit of stable hint identifiers.
            measured['role_id_constant_ids']=['r1','r2']  # fixed without reading researcher gold
            measured['role_id_constant_confusion']=confusion(c,measured['role_id_constant_ids'])
            selection.append(measured)
            unfinished.append({'trial_id':tid,'status':'selector_completed_answer_never_dispatched',
                               'reason_category':'supervisor_admission_refusal_after_stage_one'})
        else:raise ValueError('begin without terminal event: UNKNOWN, do not replay')

    # Completeness accounting is the core antidote to completed-only selection bias.
    observed=archive_counts['observed_trial_ids'];complete=len(direct)+len(gold_selection)
    assert observed==len(preflight)+len(unfinished)+complete
    assert complete==13 and len(direct)==9 and len(gold_selection)==4
    assert len(selection)==8 and len(preflight)==10 and observed==31
    assert len(cases)*len(PORT_MODELS)*len(CONDITIONS)==48
    assert all(c['support_ids']==['r1','r2'] for c in cases), 'leakage properties changed'
    by_case_model={(x['case_id'],x['model_family']):x for x in direct}
    matching=[]
    for s in selection:
        d=by_case_model.get((s['case_id'],s['model_family']))
        if not d:continue
        matching.append({'case_id':s['case_id'],'model_family':s['model_family'],
                         'full_direct_correct_against_author_label':d['matches_researcher_answer'],
                         'full_direct_tokens':d['tokens'],'first_selector_tokens':s['tokens'],
                         'full_direct_wall_seconds':d['wall_seconds'],
                         'first_selector_wall_seconds':s['wall_seconds'],
                         'selector_minus_full_tokens':s['tokens']-d['tokens'],
                         'selector_minus_full_wall_seconds':round(s['wall_seconds']-d['wall_seconds'],6)})
    total_selected=sum(s['source_count'] for s in selection)
    compare={
        '8_existing_Qwen_first_stage_calls':cumulative(s['source_confusion'] for s in selection),
        'zero_inference_pid_constant_selector':cumulative(s['role_id_constant_confusion'] for s in selection),
        'pid_blind_BM25_same_selected_passage_counts':cumulative(s['bm25_confusion'] for s in selection),
    }
    for cond in compare.values():
        assert sum(cond[k] for k in ['TP','FP','FN','TN'])==len(selection)*5
    costs={
       'first_selector_calls':len(selection),
       'all_first_selector_evaluated_tokens':sum(s['tokens'] for s in selection),
       'all_first_selector_wall_seconds':round(sum(s['wall_seconds'] for s in selection),6),
       'same_case_model_paired_full_vs_selector_first_stage_count':len(matching),
       'paired_full_direct_evaluated_tokens':sum(m['full_direct_tokens'] for m in matching),
       'paired_first_selector_evaluated_tokens':sum(m['first_selector_tokens'] for m in matching),
       'paired_full_direct_wall_seconds':round(sum(m['full_direct_wall_seconds'] for m in matching),6),
       'paired_first_selector_wall_seconds':round(sum(m['first_selector_wall_seconds'] for m in matching),6),
       'paired_selector_stage_already_slower_count':sum(m['first_selector_wall_seconds']>m['full_direct_wall_seconds'] for m in matching),
       'reliable_total_model_select_final_cost_available':False,
       'llm_model_requests_in_this_audit':0,
    }
    completed_by_family=Counter(x['model_family'] for x in direct)
    correct_by_family=Counter(x['model_family'] for x in direct if x['matches_researcher_answer'])
    summary={
        **archive_counts,
        'unique_case_ids':len(cases),'authored_support_labels_per_case':2,'authored_support_label_identity':'r1,r2 for every case',
        'unchanged_source_passage_spans_verified':sum(len(c['passages']) for c in cases),
        'full_planned_condition_cells':48,
        'completed_full_context_direct':len(direct),
        'completed_researcher_oracle_gold_selection':len(gold_selection),
        'completed_real_two_call_selection_then_answer':0,
        'completed_model_source_selection_first_stage_only':len(selection),
        'admission_refused_before_model_call':len(preflight),
        'not_attempted_any_recorded_call':48-observed,
        'direct_source_answer_author_anchor_correct':sum(x['matches_researcher_answer'] for x in direct),
        'direct_source_answer_author_anchor_incorrect':sum(not x['matches_researcher_answer'] for x in direct),
        'full_direct_by_model':{m:{'observed':completed_by_family[m],'matched_researcher_answer':correct_by_family[m]} for m in sorted(PORT_MODELS.values())},
        'missing_full_direct_cells':16-len(direct),
        'worst_case_correct_full_direct_of_16_if_all_missing_wrong':sum(x['matches_researcher_answer'] for x in direct),
        'best_case_correct_full_direct_of_16_if_all_missing_correct':sum(x['matches_researcher_answer'] for x in direct)+(16-len(direct)),
        'compared_selector_ids_total':total_selected,
    }
    return {
      'version':VERSION,'source_checksums':HASHES,'original_project_source_checksums':source_files,
      'ground_truth_type':'authored_source_question_and_r1_r2_labels, not blinded human nor executable end_to_end effect gold',
      'evidence_provenance':'actual previously recorded Qwen model messages and usage; no new model call',
      'limitations':[
        'Every source gold support ID is r1+r2 by construction; this is label leakage and invalidates these task labels as an independent semantic selection test.',
        'The four researcher-oracle gold_select answers already receive the handpicked required sources and cannot be counted as automated selection successes.',
        '8 model-selected first stages completed, zero second-stage answers; admission refusals are not semantic errors.',
        'No independently human-adjudicated relevance gold, no full actual SWAAG worker tool/effect receipts, no GPU queue, per-request VRAM, or training FLOP figures.',
        'Retrospectively only 9 of 16 planned full-context direct cells completed in this available archive; completed-only accuracy is selection-biased.',
        'Equivalent source-slot budgets do not equal model inference compute budgets: Qwen uses a full extra LLM call; lexical and constant baselines use none.',
        'Original model-run wall times may include variable backend contention; no uncontended paired randomized inference control exists.',
      ],
      'summary':summary,'selection_confusions':compare,'costs':costs,
      'matched_direct_vs_first_selector':matching,
      'source_selector_records':selection,
      'direct_final_records':direct,
      'privileged_gold_source_final_records':gold_selection,
      'unfinished_selector_records':unfinished,
      'preflight_refusal_records':preflight,
    }


def main():
    out=audit()
    target=HERE/'evidence/source-leakage/archived_qwen_source_and_cost_v1.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print('SOURCE/LABEL/COST AUDIT')
    print(json.dumps({'summary':out['summary'],'selection_confusions':out['selection_confusions'],'costs':out['costs']},indent=2))
    print('evidence',target)

if __name__=='__main__':main()
