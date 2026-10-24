#!/usr/bin/env python3
"""Compute descriptive trial totals only. No model-judging or significance claims."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

def summarize(rows):
    groups=defaultdict(list);stops=[]
    for r in rows:
        if 'steps' not in r:
            stops.append({'trial':r['trial'],'reason':r.get('reason','other')});continue
        if r.get('incomplete'):
            stops.append({'trial':r['trial'],'reason':r.get('blocked_after_first_stage','incomplete_stages')});continue
        groups[(r['kind'],r['model'],r['encoding'])].append(r)
    completed=[r for v in groups.values() for r in v]
    by_encoding={}
    by_model={}
    for r in completed:
        if r['kind']=='relevance':
            by_encoding.setdefault((r['input']['id'],r['model'],r['seed']),{})[r['encoding']]=r
        by_model.setdefault((r['input']['id'],r['encoding'],r['seed']),{})[r['model']]=r
    matched_enc=[]
    for key,v in by_encoding.items():
        if 'number' in v and 'category' in v:
            matched_enc.append({'case':key[0],'model':key[1],'seed':key[2],
                'number_score':v['number']['canonical_answer'], 'category_score':v['category']['canonical_answer'],
                'gold':v['number']['input']['gold'], 'number_correct':v['number']['correct'],'category_correct':v['category']['correct']})
    matched_models=[]
    for key,v in by_model.items():
        if len(v)>=2:
            vals=list(v.values())
            matched_models.append({'case':key[0],'encoding':key[1],'seed':key[2],
                'models':[{'model':x['model'],'answer':x['canonical_answer'],'correct':x['correct']} for x in vals],
                'both_wrong':all(not x['correct'] for x in vals),'both_correct':all(x['correct'] for x in vals)})
    return {'completed':sum(map(len,groups.values())), 'capacity_or_transport_stops':len(stops),
        'relevance_matched_encoding_pairs':matched_enc,'cross_model_matched_input_pairs':matched_models,
        'stops':stops,
        'groups':[{'task_kind':k[0],'model':k[1],'encoding':k[2],'n':len(v),'valid':sum(bool(x['valid']) for x in v),
            'correct':sum(bool(x['correct']) for x in v),'mean_latency_seconds':round(statistics.mean(x['total_elapsed_seconds'] for x in v),3),
            'judgments':[{'trial':x['trial'],'answer':x['canonical_answer'],'gold':x['input']['gold'],'correct':x['correct']} for x in v]} for k,v in sorted(groups.items())]}

def main():
    p=argparse.ArgumentParser();p.add_argument('journals',nargs='+',type=Path);a=p.parse_args();rows=[]
    for journal in a.journals:
        rows += [json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
    print(json.dumps(summarize(rows),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
