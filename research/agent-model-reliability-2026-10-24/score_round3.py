#!/usr/bin/env python3
"""Descriptive independent-oracle evaluation of append-only real-model events.

Errors and omissions are separate from source inference correctness. The
journal is never changed by this scorer; every gold is recalculated separately.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
from logic_oracle import load


def score(journal, cases):
    reference={c['id']:c for c in cases}
    grouped=defaultdict(list)
    for row in journal:
        grouped[row['trial']].append(row)
    records=[]
    for name,events in grouped.items():
        begins=[x for x in events if x['event']=='begin']
        terminal=[x for x in events if x['event'] in ('stopped','completed')]
        if len(begins)>1 or len(terminal)>1:
            raise ValueError('duplicate_trial_or_terminal')
        if begins:
            base=begins[0]
            c=base['case'];oracle=reference[c['id']]['gold']
            if c['source']!=reference[c['id']]['source'] or c['question']!=reference[c['id']]['question'] or c['gold']!=oracle:
                raise ValueError('invalid_case_provenance')
            method=base['method'];model=base['model'];port=base['port'];seed=base['seed'];style=base['style']
        else:
            method=model=port=seed=style=None;oracle=None
        responses=[x for x in events if x['event']=='response']
        terminal_status=(terminal[0].get('status') or terminal[0]['event']) if terminal else 'incomplete_unknown'
        if terminal_status=='scored':
            if len(responses) != (2 if method=='split' else 1):
                raise ValueError('invalid_stage_count')
            if terminal[0]['gold']!=oracle:
                raise ValueError('oracle_mismatch')
            if terminal[0]['correct']!=(terminal[0]['answer']==oracle):
                raise ValueError('incorrect_scoring')
        records.append({'trial':name,'case_id':c['id'] if begins else None,'gold':oracle,
                        'model':model,'port':port,'seed':seed,'style':style,'method':method,
                        'status':terminal_status,'answer':terminal[0].get('answer') if terminal else None,
                        'correct':terminal[0].get('correct') if terminal else None,
                        'latency':terminal[0].get('total_elapsed_seconds') if terminal else None,
                        'tokens':terminal[0].get('total_tokens') if terminal else None,
                        'response_stages':len(responses)})
    successful=[r for r in records if r['status']=='scored']
    bycond=defaultdict(list)
    for r in successful:bycond[(r['model'],r['method'],r['style'])].append(r)
    pairs=defaultdict(dict)
    models=defaultdict(dict)
    for r in successful:
        pairs[(r['case_id'],r['model'],r['seed'],r['style'])][r['method']]=r
        models[(r['case_id'],r['method'],r['seed'],r['style'])][r['model']]=r
    paired=[dict(case_id=k[0],model=k[1],seed=k[2],style=k[3],
                  fused_correct=v['fused']['correct'],split_correct=v['split']['correct'],
                  fused_latency=v['fused']['latency'],split_latency=v['split']['latency'],
                  fused_tokens=v['fused']['tokens'],split_tokens=v['split']['tokens'])
            for k,v in pairs.items() if 'fused' in v and 'split' in v]
    cross=[]
    for k,v in models.items():
        if len(v)<2:continue
        items=list(v.values());cross.append(dict(case_id=k[0],method=k[1],seed=k[2],style=k[3],
            correct=[r['correct'] for r in items],answers=[r['answer'] for r in items],
            both_wrong=all(not r['correct'] for r in items),both_correct=all(r['correct'] for r in items)))
    contingency=Counter(tuple(x['correct']) for x in cross)
    phi=None
    if len(cross)>=8:
        a=contingency[(False,False)];b=contingency[(False,True)]
        c=contingency[(True,False)];d=contingency[(True,True)]
        denom=((a+b)*(c+d)*(a+c)*(b+d))**0.5
        if denom:
            phi=round((a*d-b*c)/denom,5)
    return {'trial_count':len(records),'scored':len(successful),'status_counts':dict(Counter(x['status'] for x in records)),
        'by_condition':[{'model':k[0],'method':k[1],'style':k[2],'n':len(v),'correct':sum(r['correct'] for r in v),
                         'mean_latency':round(statistics.mean(r['latency'] for r in v),4),
                         'mean_tokens':round(statistics.mean(r['tokens'] for r in v),2)} for k,v in sorted(bycond.items())],
        'paired_fused_split':paired,'cross_model_exact_pairs':cross,
        'joint_wrong_count':sum(x['both_wrong'] for x in cross),
        'cross_model_error_contingency':{str(k):v for k,v in contingency.items()},
        'cross_model_error_phi_if_estimated':phi,
        'all_records':records}


def main():
    p=argparse.ArgumentParser();p.add_argument('journal',type=Path);p.add_argument('--fixtures',type=Path,default=Path(__file__).parent/'formal_holdout_v2.json');a=p.parse_args()
    journal=[json.loads(x) for x in a.journal.read_text().splitlines() if x.strip()]
    print(json.dumps(score(journal,load(a.fixtures)),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
