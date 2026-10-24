#!/usr/bin/env python3
"""Source-anchored outcome and real total-cost comparison; no LLM judge.

Importantly, this scorer never marks capacity refusal as an incorrect answer.
"""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,json,statistics

def score(dataset,journal):
    cases={x['case_id']:x for x in dataset['cases']}
    grouped=defaultdict(list)
    for event in journal:grouped[event['trial_id']].append(event)
    results=[]
    for trial,events in grouped.items():
        begins=[e for e in events if e['event']=='begin']
        terminals=[e for e in events if e['event'] in ('completed','stopped')]
        if len(begins)>1 or len(terminals)>1:raise ValueError('repeat_or_multiple_terminal')
        if not begins:
            results.append({'trial':trial,'status':'admission_refused','reason':terminals[-1].get('reason') if terminals else 'unknown'});continue
        b=begins[0];case=b['case'];gold=cases.get(case['case_id'])
        if not gold or gold!=case or b.get('ground_truth')!=gold['gold_answer']:
            raise ValueError('source_or_expected_answer_changed')
        if not terminals:status='unknown_request_outcome';t={}
        else:
            t=terminals[0];status=t.get('status') if t['event']=='completed' else 'stopped'
        answers=[e for e in events if e['event']=='response' and e['stage']=='answer']
        selections=[e for e in events if e['event']=='response' and e['stage']=='selector']
        if status=='scored':
            if len(answers)!=1:raise ValueError('missing_answer_evidence')
            if t['answer']!=answers[0]['decoded'].get('answer'):raise ValueError('mismatched_answer')
            if t['correct']!=(t['answer']==gold['gold_answer']):raise ValueError('oracle_mismatch')
        results.append({'trial':trial,'case_id':case['case_id'],'variant':b['variant'],'model':b['model'],
            'gold':gold['gold_answer'],'status':status,'answer':t.get('answer'),'correct':t.get('correct') if status=='scored' else None,
            'selected_ids':t.get('selected_ids'),'support_recall':t.get('support_recall'),
            'wall_seconds':t.get('total_wall_seconds'), 'tokens':t.get('total_tokens'),
            'actual_inference_calls':len(answers)+len(selections),
            'prompt_tokens':sum((e.get('usage') or {}).get('prompt_tokens',0) for e in events if e['event']=='response'),
            'completion_tokens':sum((e.get('usage') or {}).get('completion_tokens',0) for e in events if e['event']=='response')})
    completed=[r for r in results if r['status']=='scored']
    bycond=defaultdict(list)
    bycase=defaultdict(dict)
    for r in completed:
        bycond[(r['model'],r['variant'])].append(r)
        bycase[(r['case_id'],r['model'])][r['variant']]=r
    matched=[]
    for (case,model),v in bycase.items():
        if len(v)>1:
            matched.append({'case':case,'model':model,'conditions':{cond:{'correct':row['correct'],
                'tokens':row['tokens'],'wall_seconds':row['wall_seconds']} for cond,row in sorted(v.items())}})
    return {'trial_count':len(results),'valid_final_answers':len(completed),
        'statuses':dict(Counter(r['status'] for r in results)),
        'conditions':[{'model':k[0],'variant':k[1],'n':len(v),'correct':sum(x['correct'] for x in v),
          'mean_tokens':round(statistics.mean(x['tokens'] for x in v),2),
          'mean_wall_seconds':round(statistics.mean(x['wall_seconds'] for x in v),3)} for k,v in sorted(bycond.items())],
        'matched':matched,'results':results}

def main():
    p=argparse.ArgumentParser();p.add_argument('journal',type=Path);a=p.parse_args();root=Path(__file__).parent
    dataset=json.loads((root/'context_grounding_v1.json').read_text())
    journal=[json.loads(x) for x in a.journal.read_text().splitlines() if x.strip()]
    print(json.dumps(score(dataset,journal),indent=2))
if __name__=='__main__':main()
