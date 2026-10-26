#!/usr/bin/env python3
"""Create a label-neutral V2 **input**, and a separate researcher-gold file.

Keeps original source spans and all 8 historical questions unchanged. This is
NOT a new independent task sample or a new Qwen result. Never overwrite v1.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from audit_qwen_source_leakage_20261026 import dataset, HERE

VERSION='swaag-context-same-source-opaque-ids-v2-20261026'


def create():
    cases,source_hashes=dataset()
    inputs=[]; labels=[]
    for c in cases:
        source_ids={}
        candidates=[]
        for p in c['passages']:
            salt=f"{VERSION}|{c['case_id']}|{p['file_sha256']}|{p['start']}|{p['end']}"
            opaque_id='s_'+hashlib.sha256(salt.encode()).hexdigest()[:16]
            if opaque_id in source_ids.values():raise ValueError('duplicate opaque source identifier')
            source_ids[p['pid']]=opaque_id
            candidates.append({'source_id':opaque_id,'file':p['file'],'file_sha256':p['file_sha256'],
                               'start':p['start'],'end':p['end'],'text':p['text']})
        if not all(x['source_id'].startswith('s_') for x in candidates):raise ValueError('nonopaque id')
        inputs.append({'case_id':c['case_id'],'question':c['question'],
                       'answer_options':c['answer_options'],'passages':candidates})
        labels.append({'case_id':c['case_id'],'expected_answer':c['gold_answer'],
                       'authored_expected_evidence_ids':[source_ids[pid] for pid in c['support_ids']],
                       'label_origin':'same_original_researcher_labels_NOT_independent_human_gold',
                       'original_source_label_mapping_for_audit_only':source_ids})
    return {'version':VERSION,'input_cases':inputs,'model_input_disclosure':
                'No support IDs, gold answer, relevance label, source-role-prefixed r/d IDs or annotations included'},\
           {'version':VERSION,'labels':labels,'source_hashes':source_hashes,
            'label_integrity':'Original researcher-authored support spans retained; unvalidated for human/effect gold'}


def save():
    public, labels=create()
    # JSON is deterministic: exported files can be independently hashed.
    root=HERE/'evidence/source-leakage'
    root.mkdir(exist_ok=True,parents=True)
    for obj,name in [(public,'source_opaque_model_input_v2.json'),(labels,'source_opaque_researcher_labels_v2.json')]:
        path=root/name
        if path.exists():
            existing=json.loads(path.read_text())
            if existing!=obj:raise ValueError('v2 is frozen; refuse overwrite after change: '+str(path))
            continue
        path.write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
        print('wrote',path)
    for name in ['source_opaque_model_input_v2.json','source_opaque_researcher_labels_v2.json']:
        path=root/name;print(name,hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=='__main__':save()
