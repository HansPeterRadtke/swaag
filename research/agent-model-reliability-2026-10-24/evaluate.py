#!/usr/bin/env python3
"""Offline independent-oracle scorer. Never calls a model or uses output majority as truth."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

LABELS = ('irrelevant','very_low','low','somewhat_low','medium','somewhat_high','high','very_high','critical')

def score(rows):
    results = []
    for case in rows:
        gold = case['gold']
        judgments = case.get('judgments', [])
        matches = []
        for j in judgments:
            answer = j['answer']
            if case['kind'] == 'relevance':
                if j['encoding'] == 'ordinal_enum':
                    if answer not in LABELS:
                        matches.append(None); continue
                    value = LABELS.index(answer)
                elif j['encoding'] == 'integer_0_8' and type(answer) is int and 0 <= answer <= 8:
                    value = answer
                else:
                    matches.append(None); continue
                matches.append(abs(value-gold) <= case.get('tolerance', 0))
            else:
                matches.append(answer == gold)
        valid = [x for x in matches if x is not None]
        wrong_consensus = len(valid) > 1 and len(set(json.dumps(j['answer'], sort_keys=True) for j,x in zip(judgments,matches) if x is not None)) == 1 and not all(valid)
        results.append({'id':case['id'],'kind':case['kind'],'n':len(judgments),'valid':len(valid),'correct':sum(valid),'invalid':len(matches)-len(valid),'unanimous_wrong':wrong_consensus})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);a=p.parse_args()
    rows=json.loads(a.fixture.read_text())
    print(json.dumps(score(rows),indent=2))
if __name__=='__main__': main()
