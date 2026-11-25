#!/usr/bin/env python3
"""Independent offline adjudication of the original public natural-audio run.

Checks every public row identity, labeled pause, actual causal step boundary,
score range, timing count and aggregate result. Never loads ONNX or repeats
inference. This is a bounded shadow screen, not a release authorization.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

CORPUS='f4e217a4c14287c3b1d4eaa8c872a2273d5a37ff7da38c9d114d639ed804892e'
MODEL='e747b182d685c7db489a3ea2e12bdcd65bac506fcddec00ed3cd0bf9f209fd11'
SCRIPT='f62de1542984dd6930a1cdcbbb79beaa5318471195847ad4122122cc82a4307d'
THRESHOLDS=(0.06,0.16,0.36,0.5,0.8,0.95)
STEP=2560
RATE=16000


def check(ok,why):
    if not ok:raise ValueError(why)


def read(root,name,limit=30_000_000):
    p=root/name
    check(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'unsafe evidence')
    try:p.resolve().relative_to(root.resolve())
    except ValueError as e:raise ValueError('external evidence path') from e
    return p.read_bytes()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_row(source,observed):
    for key in ('id','language','wav_sha256','frames'):
        check(source[key]==observed[key],'wrong audio/row identity')
    check(observed['step_count']==source['frames']//STEP==len(observed['step_seconds']),
          'incomplete stream')
    check(all(type(t) in (float,int) and math.isfinite(t) and t>=0
              for t in observed['step_seconds']),'invalid step timing')
    check(len(source['spans'])==len(observed['spans']),'missing pause')
    for gold,actual in zip(source['spans'],observed['spans']):
        check({k:v for k,v in actual.items() if k!='points'}==gold,'changed annotated pause')
        # Derive independently by enumerating only completed native audio
        # chunks. A score from a chunk ending after the pause is not causal
        # at the earlier decision time and cannot be counted.
        valid=[n*STEP for n in range(1,observed['step_count']+1)
               if gold['start_sample']+3200<=n*STEP<=gold['end_sample']]
        points=actual['points']
        check([x[0] for x in points]==valid,'missing, duplicated or noncausal score')
        check(all(type(x[1]) in (int,float) and math.isfinite(x[1])
              and 0<=x[1]<=1 for x in points),'invalid score')


def independent_metrics(rows):
    output={}
    for lang in ('en','de'):
        group=[r for r in rows if r['language']==lang]
        flat=[(r['id'],s) for r in group for s in r['spans']]
        hold=[(i,s) for i,s in flat if s['label']=='hold']
        eot=[s for _,s in flat if s['label']=='eot']
        eligible_hold=len([s for _,s in hold if s['points']])
        eligible_eot=len([s for s in eot if s['points']])
        for threshold in THRESHOLDS:
            wrong=[(i,s) for i,s in hold if any(v>=threshold for _,v in s['points'])]
            hits=[s for s in eot if any(v>=threshold for _,v in s['points'])]
            delays=[(min(n for n,v in s['points'] if v>=threshold)-s['start_sample'])/RATE
                    for s in hits]
            output.setdefault(lang,{})[str(threshold)]={
                'hold_spans_total':len(hold),'hold_spans_eligible':eligible_hold,
                'false_hold_spans':len(wrong),'turns_with_false_hold':len({i for i,_ in wrong}),
                'eot_spans_total':len(eot),'eot_spans_eligible':eligible_eot,
                'eot_detected':len(hits),'eot_recall_all':len(hits)/len(eot),
                'hold_false_positive_rate_eligible':len(wrong)/max(1,eligible_hold),
                'detected_eot_latency_median_s':statistics.median(delays) if delays else None,
            }
    return output


def verify(root,run,script):
    manifest_raw=read(root,'corpus/manifest.json')
    check(digest(manifest_raw)==CORPUS,'wrong source corpus')
    manifest=json.loads(manifest_raw)
    check(manifest['revision']=='ae834af9e9bddf73d1f550c190aaeb99bf31f523'
          and manifest['license']=='CC-BY-4.0','wrong public corpus provenance')
    check(script.is_file() and not script.is_symlink() and digest(script.read_bytes())==SCRIPT,
          'wrong as-executed script')
    check(digest(read(root,'p99-model.onnx'))==MODEL,'wrong public model weights')
    a=json.loads(read(run,'attempt.json'));r=json.loads(read(run,'result.json'))
    check(a['script_sha256']==SCRIPT and a['model_sha']==MODEL and
          a['corpus_manifest_sha']==CORPUS and tuple(a['thresholds'])==THRESHOLDS
          and a['primary']==0.36 and a['minimum_pause_samples']==3200
          and a['step_samples']==STEP and a['no_model_service_agent_phone_or_tools'] is True,
          'wrong original frozen policy')
    check(a['research_gate']=={'zero_false_hold':True,'min_eot_recall_all':0.95,'max_step_p95_s':0.025},
          'changed acceptance gate')
    raw=read(run,'rows.jsonl')
    check(digest(raw)==r['rows_sha256'],'changed raw results')
    rows=[json.loads(line) for line in raw.splitlines()]
    check(len(rows)==len(manifest['rows'])==r['rows']==800,'wrong complete row count')
    check(collections.Counter(x['language'] for x in rows)=={'en':400,'de':400},'wrong strata')
    check(len({x['id'] for x in rows})==800,'duplicate turn')
    for source,observed in zip(manifest['rows'],rows):validate_row(source,observed)
    metrics=independent_metrics(rows)
    check(metrics==r['metrics'],'aggregate mismatch')
    times=sorted(t for row in rows for t in row['step_seconds'])
    check(len(times)==r['steps'] and statistics.median(times)==r['step_p50_s'] and
          times[math.ceil(len(times)*0.95)-1]==r['step_p95_s'] and times[-1]==r['step_max_s'],
          'timing mismatch')
    gate=(r['step_p95_s']<=0.025 and all(
         metrics[l]['0.36']['false_hold_spans']==0 and
         metrics[l]['0.36']['eot_recall_all']>=0.95 for l in ('en','de')))
    check(r['passes_predeclared_primary_screen'] is gate and
          r['status']=='completed_narrow_research_screen' and
          r['provider']==['CPUExecutionProvider'],'wrong research verdict')
    for k in ('release_authorized','automatic_command_authorized','speaker_or_interrupt_tested',
              'german_officially_supported','official_eot_benchmark_score'):
        check(r[k] is False,'unsupported authority or benchmark claim')
    check(r['model_or_agent_remote_calls']==0,'unexpected external action')
    return {'status':'independently_verified_original_natural_pause_screen',
            'rows':len(rows),'annotated_spans':sum(len(row['spans']) for row in rows),
            'native_cpu_steps':len(times),'primary_gate_passed':gate,
            'primary_english':metrics['en']['0.36'],'primary_german':metrics['de']['0.36'],
            'step_p95_s':r['step_p95_s'],'release_authorized':False,
            'original_result_sha256':digest(read(run,'result.json')),
            'original_rows_sha256':digest(raw)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-evidence',type=Path,required=True)
    p.add_argument('--as-run-script',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.root,a.run_evidence,a.as_run_script),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
