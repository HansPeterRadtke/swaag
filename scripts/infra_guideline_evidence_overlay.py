"""Apply manually reviewed, exact-phrase acceptance evidence to source trace.

Unverified units remain unverified. A nearby passing test does not prove all
requirements in a paragraph, whole guide, or product. Reuses untouched source
hashes and rejects unsupported/ambiguous source matches.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

VERDICTS = frozenset({"MECHANISM_TESTED_SCOPE_LIMITED", "TESTED_BUT_FAILING",
                      "PARTIALLY_TESTED", "NOT_ACCEPTED", "SOURCE_PRESERVED_ONLY"})


def apply(trace:Path, checks:Path, output:Path) -> dict:
    items=[json.loads(line) for line in trace.read_text().splitlines()]
    evidence=json.loads(checks.read_text())
    marked=[]
    for row in evidence:
        assert isinstance(row,dict) and row['verdict'] in VERDICTS
        for key in ('id','source_file','source_phrase','owner','implementation','tests','observation','remaining'):
            if not row.get(key):raise ValueError(f'missing or empty {key} in {row.get("id")}')
        candidates=[item for item in items if item['source_file']==row['source_file']
                    and row['source_phrase'].casefold() in item['source_text'].casefold()]
        if len(candidates)!=1:
            raise ValueError(f"ambiguous/nonexistent source phrase {row['id']}: {len(candidates)}")
        source=candidates[0]
        entry={k:row[k] for k in ('id','verdict','owner','implementation','tests','observation','remaining')}
        source['proof'].append(entry)
        # The original paragraph may include several independent obligations.
        # Acceptance remains partial until every one has a verified effect.
        source['acceptance']='AT_LEAST_ONE_SUBCLAUSE_REVIEWED_NOT_WHOLE_BLOCK_CERTIFIED'
        marked.append({'id':row['id'],'file':source['source_file'],
                       'line':source['source_start_line'],'source_hash':source['source_block_sha256'],
                       'verdict':row['verdict']})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'requirement_with_evidence.jsonl').open('w') as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n')
    (output/'reviewed_gates.json').write_text(json.dumps(marked,indent=2)+'\n')
    counts=Counter(row['verdict'] for row in evidence)
    return {'source_blocks':len(items),'reviewed_evidence_links':len(marked),
            'distinct_source_blocks_with_evidence':sum(bool(x['proof']) for x in items),
            'unverified_source_blocks':sum(not x['proof'] for x in items),
            'evidence_verdicts':dict(counts)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(apply(a.trace,a.evidence,a.output),indent=2))

if __name__=='__main__':main()
