#!/usr/bin/env python3
"""Offline independent comparison of acoustic and ideal-transcript signals.

Both original runs are immutable. No model inference, action, or tuning.
This checks whether a fixed conjunction can meet a narrow research screen.
It is not a measured ASR pipeline, speaker authorization, or release gate.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics

from verify_p99_natural_turn_evidence import (
    check, read, digest, verify as verify_audio, CORPUS, RATE,
)

TEXT_SHA='c25e8871ab61930edbad6a385b2cef21096077f994e56088604d7d4e38ca5169'
TEXT_SCRIPT='8fc2e503d5252d5a75cab4c46f0cb565703fcf92a9b3163fceacb054609f56fb'
MODEL_REV='59245aac1f0a0f170277ebecc32ddaea418083b4'
MODEL_SHA='67be31dac7c49abd43a277a105222c9031a0e22ac37f0e22dab395f68b318fc9'
FUSION=((0.36,0.5),(0.36,0.8),(0.8,0.8))
POLICIES=(('audio_0.36',0.36,None),('audio_0.8',0.8,None),
          ('text_0.5',None,0.5),('text_0.8',None,0.8),
          ('and_0.36_0.5',0.36,0.5),('and_0.36_0.8',0.36,0.8),
          ('and_0.8_0.8',0.8,0.8))


def validate_text_case(gold,observed):
    check({k:v for k,v in gold.items() if k!='text'}==
          {k:observed[k] for k in gold if k!='text'},'changed aligned text case')
    check(digest(gold['text'].encode())==gold['text_sha256'] and
          gold['has_prefix']==bool(gold['text']),'wrong prefix identity')
    if not gold['has_prefix']:
        check(observed['status']=='no_aligned_prefix','empty input scored')
    elif observed['status']=='valid':
        p,q=observed['p_finished'],observed['p_incomplete']
        check(type(p) in (int,float) and type(q) in (int,float)
              and math.isfinite(p) and math.isfinite(q)
              and 0<=p<=1 and 0<=q<=1 and abs(p+q-1)<1e-6,'invalid probability')
        check(type(observed['tokens']) is int and 1<=observed['tokens']<=512,'invalid tokens')
        t=observed['inference_seconds']
        check(type(t) in (int,float) and math.isfinite(t) and t>=0,'invalid model time')
    else:
        check(observed['status']=='oversized_prefix','unknown unscored case')
    if observed['status']!='valid':
        check(not any(k in observed for k in ('p_finished','p_incomplete','inference_seconds')),
              'invalid text case has decision')


def verify_text(root,run,script):
    raw=read(root,'causal-text-manifest.json')
    check(digest(raw)==TEXT_SHA,'changed text manifest')
    d=json.loads(raw)
    check(d['source_manifest_sha256']==CORPUS and d['no_current_model_or_phone'] is True,'wrong source')
    a=json.loads(read(run,'attempt.json'));r=json.loads(read(run,'result.json'))
    check(script.is_file() and not script.is_symlink() and digest(script.read_bytes())==TEXT_SCRIPT
          and a['script_sha']==r['source_script_sha']==TEXT_SCRIPT,'wrong executed text source')
    check(a['model_revision']==r['model_revision']==MODEL_REV
          and a['model_sha']==r['model_sha']==MODEL_SHA
          and a['manifest_sha']==r['manifest_sha']==TEXT_SHA,'source/model drift')
    check(tuple(a['thresholds'])==(0.5,0.8) and tuple(tuple(x) for x in a['fusion'])==FUSION
          and a['ideal_word_alignments_not_asr'] is True
          and a['one_use_no_remote_model_or_agent'] is True
          and a['missing_prefix_never_authorizes'] is True,'changed declared policy')
    source=d['cases'];actual_raw=read(run,'cases.jsonl');rows=[json.loads(l) for l in actual_raw.splitlines()]
    check(len(rows)==len(source)==r['rows']==1730 and digest(actual_raw)==r['cases_sha256'],'incomplete/altered rows')
    check(len({(x['id'],x['span_index']) for x in rows})==1730,'duplicate text span')
    for g,o in zip(source,rows):validate_text_case(g,o)
    valid=[x for x in rows if x['status']=='valid']
    times=sorted(x['inference_seconds'] for x in valid)
    check(len(valid)==r['valid'] and r['missing_or_oversized']==len(rows)-len(valid)
          and r['inference_p50_s']==statistics.median(times)
          and r['inference_p95_s']==times[math.ceil(len(times)*.95)-1]
          and r['inference_max_s']==times[-1],'timing or coverage mismatch')
    check(r['status']=='completed_actual_cpu_aligned_prefix_screen'
          and r['provider']==['CPUExecutionProvider']
          and r['remote_model_calls']==r['agent_actions']==0
          and r['release_authorized'] is r['automatic_command_authorized'] is r['live_asr_tested'] is False,
          'unexpected action or claim')
    return rows,r


def compare(audio,text):
    lookup={(x['id'],x['span_index']):x for x in text}
    output={}
    for language in ('en','de'):
        rows=[x for x in audio if x['language']==language]
        joined=[]
        for row in rows:
            for s in row['spans']:
                t=lookup.pop((row['id'],s['index'])) if s['points'] else None
                if t:
                    check(t['language']==language and t['label']==s['label']
                          and t['cutoff_sample']==s['start_sample'],'misaligned annotation')
                joined.append((row['id'],s,t))
        table={}
        for label,ath,tth in POLICIES:
            eot=hold=eligible_hold=eligible_eot=hit=wrong=0
            wrong_turns=set();delays=[]
            for ident,s,t in joined:
                is_eot=s['label']=='eot'
                eot+=is_eot;hold+=not is_eot
                if not s['points']:continue
                eligible_eot+=is_eot;eligible_hold+=not is_eot
                if tth is not None and (t is None or t['status']!='valid' or t['p_finished']<tth):
                    continue
                eligible=[at for at,score in s['points'] if ath is None or score>=ath]
                if not eligible:continue
                if is_eot:
                    hit+=1
                    delays.append((min(eligible)-s['start_sample'])/RATE)
                else:
                    wrong+=1;wrong_turns.add(ident)
            table[label]={'all_eot':eot,'all_hold':hold,'eligible_eot':eligible_eot,'eligible_hold':eligible_hold,
                'eot_detected':hit,'eot_recall_all':hit/eot,'false_hold_spans':wrong,
                'false_hold_rate_eligible':wrong/max(1,eligible_hold),'turns_with_false_hold':len(wrong_turns),
                'ideal_detected_delay_median_s':statistics.median(delays) if delays else None,
                'passes_zero_false_hold_and_95pct_eot':wrong==0 and hit/eot>=.95}
        output[language]=table
    check(not lookup,'text case without eligible audio pause')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','audio-run','text-run','audio-script','text-script'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    av=verify_audio(a.root,a.audio_run,a.audio_script)
    rows,r=verify_text(a.root,a.text_run,a.text_script)
    audio=[json.loads(l) for l in read(a.audio_run,'rows.jsonl').splitlines()]
    table=compare(audio,rows)
    verdict={'status':'independent_public_natural_pause_fusion_adjudication',
        'audio_result_sha256':av['original_result_sha256'],
        'text_result_sha256':digest(read(a.text_run,'result.json')),
        'text_valid':r['valid'],'text_unscored':r['missing_or_oversized'],
        'text_p95_s':r['inference_p95_s'],'audio_p95_s':av['step_p95_s'],
        'policies':table,
        'any_fixed_policy_passes_both_languages':any(
            table['en'][p]['passes_zero_false_hold_and_95pct_eot']
            and table['de'][p]['passes_zero_false_hold_and_95pct_eot'] for p,_,_ in POLICIES),
        'ideal_transcript_not_real_asr':True,'official_benchmark_score':False,
        'speaker_and_interruption_unverified':True,'release_authorized':False,
        'automatic_command_authorized':False}
    print(json.dumps(verdict,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
