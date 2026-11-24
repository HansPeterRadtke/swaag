#!/usr/bin/env python3
"""Independent no-inference verifier of first real 51-case Namo CPU benchmark.

Authenticates immutable source/model inputs, original benchmark labels and
assistant-authored bilingual holdout, and all recorded real ONNX probabilities,
timings, decisions, and aggregate counts. Explicitly marks interruption as
NOT TESTED. Never imports ONNX, generates speech, repeats an inference, calls
SWAAG or modifies any user device. A model's fast benchmark does not grant
speaker or tool-action authorization.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,statistics
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

from run_isolated_namo_multilingual_cpu_once import MODEL_FILES,ORACLE_SHA,HOLDOUT_SHA,REPOSITORY,REVISION,checked_sources,p95,stats,sha,regular

ORIGINAL_RESULT_SHA='ddbd7a4650fe8b0ac385be9bcbdb9f074c2b44697214433cc63479d3c01dab9f'
ORIGINAL_RUNNER_SHA='111efb61c51aa718e2a333db424bdf0a4bbc8a4a663ed87345a3a32194905b47'


def check_candidate(model:Path,oracle:Path,holdout:Path,result:Path,source:Path)->dict[str,Any]:
    regular(source)
    if sha(source)!=ORIGINAL_RUNNER_SHA:
        raise ValueError('Actual model runner changed after inference')
    cases=checked_sources(model,oracle,holdout)
    regular(result)
    if sha(result)!=ORIGINAL_RESULT_SHA:raise ValueError('Immutable original ONNX inference result SHA changed')
    data=json.loads(result.read_text())
    if (data.get('status')!='completed_actual_CPU_ONNX_inference'
        or data.get('repository')!=REPOSITORY or data.get('source_revision')!=REVISION
        or data.get('source_oracle_sha')!=ORACLE_SHA or data.get('holdout_sha')!=HOLDOUT_SHA
        or data.get('onnx_model_sha')!=MODEL_FILES['model_quant.onnx'][0]
        or data.get('cpu_only') is not True
        or data.get('model_provider_requests')!=0 or data.get('swaag_tool_or_phone_calls')!=0
        or data.get('original_interruption_cases_not_supported_or_inferred')!=18
        or data.get('one_use_selected_cases')!=51
        or data.get('single_primary_threshold')!=.5
        or data.get('secondary_predeclared_conservative_threshold')!=.8):
        raise ValueError('Result mislabeled or contains real provider/phone/tool effects')
    if data.get('onnx_inputs')!=['input_ids','attention_mask']:
        raise ValueError('ONNX model schema changed')
    actual=data.get('cases')
    if not isinstance(actual,list) or len(actual)!=len(cases):
        raise ValueError('Incomplete source-backed case coverage')
    checked=[]
    for want,got in zip(cases,actual):
        for k in ('case_id','source','language','expected','text'):
            if got.get(k)!=want.get(k):
                raise ValueError('Original held-out text/label/source tampered '+want['case_id'])
        if got.get('invalid') is not False or got.get('error_type'):
            raise ValueError('Actual ONNX returned error or invalid inference not a valid no')
        p1=got.get('p_finished');p0=got.get('p_incomplete')
        if (type(p1) not in (float,int) or type(p0) not in (float,int)
            or not math.isfinite(p1) or not math.isfinite(p0)
            or not 0<=p1<=1 or not 0<=p0<=1 or abs(p1+p0-1)>1e-6):
            raise ValueError('Non-finite/non-normalized class probability')
        if (got.get('predicted_complete_0_5') is not (p1>=.5)
            or got.get('predicted_complete_0_8') is not (p1>=.8)
            or got.get('pass_0_5') is not ((p1>=.5) is want['expected'])):
            raise ValueError('Reported decision or correctness forged')
        if (type(got.get('tokens')) is not int or not 1<=got['tokens']<=512
            or type(got.get('inference_seconds')) not in (int,float)
            or not 0<got['inference_seconds']<8):
            raise ValueError('Missing source-backed inference timing or tokenizer count')
        checked.append(got)
    original=[x for x in checked if x['source']=='original_SWAAG_oracle']
    hold=[x for x in checked if x['source']=='assistant_authored_bilingual_holdout']
    groups={'original_turn_oracle':original,'holdout_all':hold,
        'holdout_english':[x for x in hold if x['language']=='en'],
        'holdout_german':[x for x in hold if x['language']=='de'],'all_51':checked}
    for k,items in groups.items():
        if stats(items)!=data.get(k):
            raise ValueError('Statistics in original model result are inaccurate '+k)
    if (len(original)!=17 or len(hold)!=34
        or data.get('quality')!='rejected_for_automatic_voice_completion_on_fixed_oracle'
        or data['all_51']['false_finished_incomplete_0_5']!=14
        or data['all_51']['false_finished_incomplete_0_8']!=2):
        raise ValueError('Benchmark was incorrectly accepted or false-positive rate hidden')
    details={}
    for name,items in groups.items():
        incomplete=[c for c in items if c['expected'] is False]
        complete=[c for c in items if c['expected'] is True]
        details[name]={**stats(items),
            'completed_correct_0_5':sum(c['predicted_complete_0_5'] is True for c in complete),
            'incomplete_correct_0_5':sum(c['predicted_complete_0_5'] is False for c in incomplete),
            'incomplete_false_finished_0_8':sum(c['predicted_complete_0_8'] is True for c in incomplete),
            'true_finished_detected_0_8':sum(c['predicted_complete_0_8'] is True for c in complete),
            'positive_cases':len(complete),'negative_cases':len(incomplete)}
    return {'status':'independently_verified_real_51_case_CPU_ONNX_model_REJECTED_for_auto_voice',
        'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'repository':REPOSITORY,'revision':REVISION,
        'actual_one_use_source_model_result_SHA256':ORIGINAL_RESULT_SHA,
        'exact_as_executed_model_test_script_SHA256':ORIGINAL_RUNNER_SHA,
        'actual_public_CPU_only_model_weights_SHA256':MODEL_FILES['model_quant.onnx'][0],
        'model_license':'Apache-2.0 original public source verified',
        'fixed_source_oracle_SHA256':ORACLE_SHA,'predeclared_bilingual_holdout_SHA256':HOLDOUT_SHA,
        'actual_total_inference_cases':51,'actual_user_provider_tool_audio_effects':0,
        'model_accuracy_verdict':'REJECTED: high false-completion counts even though fast',
        'observed_metrics':details,
        'interruption_oracle_cases_not_tested_or_authorized':18,
        'owner_speaker_authentication_or_model_completion_authorized':False,
        'real_user_audio_or_physical_Android_test':False,
        'original_private_model_file_path':str(model),
        'note':'Only assistant-authored/fixed source text. No natural pause, recorded owner speech, model-based semantic interruptions or end-to-end voice.'}


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir','oracle','holdout','original-result','runner-source','private-output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(argv)
    result=check_candidate(a.model_dir,a.oracle,a.holdout,a.original_result,a.runner_source)
    dest=a.private_output
    if dest.exists() or dest.is_symlink() or not dest.is_absolute() or not str(dest).startswith('/data/var/swaag-benchmarks/'):
        raise FileExistsError('Independent immutable verdict may only be created once in private project storage')
    fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':result['status'],'selected':result['actual_total_inference_cases'],
       'original_pass':result['observed_metrics']['original_turn_oracle']['passed_vendor_threshold_0_5'],
       'holdout_pass':result['observed_metrics']['holdout_all']['passed_vendor_threshold_0_5'],
       'total_false_early_completion':result['observed_metrics']['all_51']['false_finished_incomplete_0_5'],
       'p95_seconds':result['observed_metrics']['all_51']['inference_p95_seconds']}))
    return 0

if __name__=='__main__':raise SystemExit(main())
