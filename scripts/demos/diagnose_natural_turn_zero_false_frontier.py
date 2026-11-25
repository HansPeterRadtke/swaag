#!/usr/bin/env python3
"""Descriptive optimistic threshold frontier on already-scored natural pauses.

No inference, tuning for release, or new observations. This asks whether a
static threshold or conjunction could even separate the *observed* labeled
pauses. The threshold is selected on the very same cases being measured,
so the resulting recall is an optimistic in-sample upper bound, not a
generalization claim, model calibration, or a deployable decision policy.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from verify_p99_natural_turn_evidence import read, verify as verify_audio
from verify_natural_turn_fusion_evidence import verify_text, check


def pause_scores(audio, text):
    lookup = {(r['id'], r['span_index']): r for r in text}
    out = []
    for row in audio:
        for span in row['spans']:
            pts = span['points']
            t = lookup.pop((row['id'], span['index'])) if pts else None
            if t:
                check(t['language'] == row['language'] and t['label'] == span['label']
                      and t['cutoff_sample'] == span['start_sample'], 'mismatched label')
            out.append({'language': row['language'], 'label': span['label'],
                        'audio': max((v for _, v in pts), default=None),
                        'text': t['p_finished'] if t and t['status'] == 'valid' else None})
    check(not lookup, 'orphan aligned scores')
    return out


def above(x, threshold):
    return x is not None and x >= threshold


def no_hold_threshold(scores):
    # A greater-or-equal comparison needs the next representable double
    # after the largest unsafe score; exactly equal is not safe.
    return math.nextafter(max(scores), math.inf) if scores else 0.0


def best_single(rows, field):
    false_scores = [r[field] for r in rows if r['label']=='hold' and r[field] is not None]
    threshold = no_hold_threshold(false_scores)
    positives = [r for r in rows if r['label']=='eot']
    hits = sum(above(r[field],threshold) for r in positives)
    return {'threshold_in_sample_only': threshold, 'eot_detected': hits,
            'eot_total':len(positives), 'recall':hits/len(positives),
            'false_holds':sum(above(r[field],threshold) for r in rows if r['label']=='hold')}


def best_and(rows):
    positives=[r for r in rows if r['label']=='eot']
    holds=[r for r in rows if r['label']=='hold'
           and r['audio'] is not None and r['text'] is not None]
    # For any audio threshold, the minimum safe text threshold is just
    # above the greatest text score among the holds that audio admits.
    # Only hold-score breakpoints can improve this bound. Threshold zero
    # is also considered. Missing signals cannot authorize a positive.
    candidates=[0.0]+sorted({math.nextafter(r['audio'],math.inf) for r in holds})
    best=None
    for a in candidates:
        t=no_hold_threshold([r['text'] for r in holds if r['audio']>=a])
        hits=sum(above(r['audio'],a) and above(r['text'],t) for r in positives)
        # Deterministic conservative tie break. This remains in-sample.
        record=(hits,a,t)
        if best is None or record>best:best=record
    hits,a,t=best
    check(not any(above(r['audio'],a) and above(r['text'],t) for r in holds),
          'frontier computation failed zero-false condition')
    return {'audio_threshold_in_sample_only':a,'text_threshold_in_sample_only':t,
            'eot_detected':hits,'eot_total':len(positives),'recall':hits/len(positives),
            'false_holds':0}


def frontier(rows):
    return {lang:{'audio':best_single(group,'audio'),
                  'text':best_single(group,'text'),
                  'conjunction':best_and(group)}
            for lang,group in [('all',rows)]+[(l,[r for r in rows if r['language']==l])
                                             for l in ('en','de')]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','audio-run','text-run','audio-script','text-script'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    verify_audio(a.root,a.audio_run,a.audio_script)
    text,_=verify_text(a.root,a.text_run,a.text_script)
    audio=[json.loads(l) for l in read(a.audio_run,'rows.jsonl').splitlines()]
    result={'status':'descriptive_in_sample_zero_false_hold_upper_bound',
            'thresholds_fit_to_evaluation_labels':True,
            'not_independent_acceptance':True,'release_authorized':False,
            'automatic_command_authorized':False,
            'frontier':frontier(pause_scores(audio,text))}
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
