#!/usr/bin/env python3
"""Descriptive comparisons against PRE-FROZEN researcher labels, not human truth."""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

def summarize(fixture,journal):
    cases={x['case_id']:x for x in fixture['cases']}
    requests={};halt=[];response=[]
    for entry in journal:
        t=entry['trial_id'];event=entry['event']
        if event=='request_submitted':
            if t in requests:raise ValueError('repeated_request')
            c=cases[entry['case_id']]
            for k in ('gold','role','goal_id','source_sha256','source_char_start','source_char_end'):
                expected={'gold':c['gold_score'],'role':c['role'],'goal_id':c['goal_id'],
                          'source_sha256':c['source_sha256'],'source_char_start':c['source_char_start'],'source_char_end':c['source_char_end']}[k]
                if entry[k]!=expected:raise ValueError('fixture_tampering_'+k)
            requests[t]=entry
        elif event=='halt':halt.append(entry)
        elif event=='response':
            if t not in requests or any(r['trial_id']==t for r in response):raise ValueError('unmatched_or_duplicate_response')
            response.append(entry)
        else:raise ValueError('unknown_event')
    complete=[]
    for item in response:
        q=requests[item['trial_id']];gold=q['gold'];score=item.get('score')
        complete.append({'trial':item['trial_id'],'model':q['model'],'encoding':q['encoding'],'goal':q['goal_id'],
                         'role':q['role'],'case_id':q['case_id'],'gold':gold,'score':score,
                         'correct_extreme':score==gold,'binary_agreement':(score>=6)==(gold==8) if type(score) is int else None,
                         'absolute_error':abs(score-gold) if type(score) is int else None,
                         'elapsed_seconds':item['elapsed_seconds'],
                         'total_tokens':(item.get('usage') or {}).get('total_tokens'),
                         'cache_tokens':((item.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')})
    paired=defaultdict(dict)
    for x in complete:paired[(x['model'],x['case_id'])][x['encoding']]=x
    comparisons=[]
    for (model,case),v in paired.items():
        if 'number' in v and 'category' in v:
            a=v['number'];b=v['category']
            comparisons.append({'model':model,'case':case,'gold':a['gold'],
                                'numeric':a['score'],'category':b['score'],
                                'numeric_absolute_error':a['absolute_error'],'category_absolute_error':b['absolute_error'],
                                'numeric_time':a['elapsed_seconds'],'category_time':b['elapsed_seconds']})
    counts=defaultdict(list)
    for x in complete:counts[(x['model'],x['encoding'])].append(x)
    return {'completed':len(complete),'halted':len(halt),'incomplete_calls':len(requests)-len(complete),
            'matched_encoding_pairs':comparisons,
            'groups':[{'model':k[0],'encoding':k[1],'n':len(v),'binary_anchor_agreement':sum(x['binary_agreement'] is True for x in v),
                       'exact_extreme_agreement':sum(x['correct_extreme'] for x in v),
                       'mean_absolute_error':sum(x['absolute_error'] for x in v if x['absolute_error'] is not None)/len(v),
                       'mean_full_wall_seconds':sum(x['elapsed_seconds'] for x in v)/len(v)} for k,v in sorted(counts.items())],
            'completed_details':complete,'halts':[{'trial':x['trial_id'],'reason':x['reason']} for x in halt]}

def main():
    p=Path(__file__).parent
    fixture=json.loads((p/'authentic_relevance_v1.json').read_text())
    observations=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    print(json.dumps(summarize(fixture,observations),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
