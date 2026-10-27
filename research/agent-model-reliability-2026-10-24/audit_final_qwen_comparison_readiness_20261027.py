#!/usr/bin/env python3
"""Read-only full-cell Qwen result eligibility and real Nitro runtime audit.

No model call/retry, GPU allocation, SQLite writes, production configuration,
or original experiment mutation. Final judgments distinguish original
single-researcher anchors and actual independent externally graded task gold.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'evidence/final-qwen-readiness'
ARCHIVE=HERE/'evidence/source-leakage/archived_qwen_source_and_cost_v1.json'
CONTEXT=HERE/'context_grounding_v1.json'
SOURCE_SHA='68ccccd63ac080acb13d6caaf4ae090494229b7b58d89d4d46b560fbc72c6e27'
CONTEXT_SHA='74e9e1bcd1e4381971374d11f73bffb0145e7af35294134b85b5c50fc0ceb321'
RUNTIME=Path('/data/var/swaag')
VERSION='swaag_final_qwen_end_to_end_evidence_eligibility_v1_20261027'
MODEL_FAMILIES=('qwen3-coder','qwen38')
STRATEGIES=('full','gold_select','model_select')
PORTS_CHECK=(8080,8081,18081,14833,14835,19333)


def hash_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read_source():
    if hash_file(ARCHIVE)!=SOURCE_SHA or hash_file(CONTEXT)!=CONTEXT_SHA:
        raise ValueError('Original Qwen archived source or dataset version mismatch')
    a=json.loads(ARCHIVE.read_text());ctx=json.loads(CONTEXT.read_text())
    if len(ctx['cases'])!=8:
        raise ValueError('Original SWAAG source-question corpus changed')
    for case in ctx['cases']:
        if case['support_ids']!=['r1','r2']:
            raise ValueError('Original source role-label leakage check no longer valid')
    return a,ctx


def cell_index(a,ctx):
    all_ids={c['case_id'] for c in ctx['cases']}
    states={}
    for case in all_ids:
        for family in MODEL_FAMILIES:
            for strategy in STRATEGIES:
                states[(case,family,strategy)]={
                   'case_id':case,'model_family':family,'strategy':strategy,
                   'status':'no_recorded_attempt','source_question_is_original_swaag_document':True,
                   'expected_answer_origin':'researcher_authored_not_blinded_independent_gold',
                   'independent_final_task_outcome_available':False,
                   'finished_actual_model_selected_two_stage_answer':False,
                   'model_qwen_used_in_previous_archive':None,
                   'observed_total_tokens':None,'observed_wall_seconds':None,
                }
    def field(trial_id):
        vals=trial_id.split(':')
        if len(vals)!=5 or vals[0]!='ground-v1' or vals[4]!='s17':
            raise ValueError('Invalid original experiment trial_id')
        case=vals[1];port=vals[2];strategy=vals[3]
        if case not in all_ids or strategy not in STRATEGIES:
            raise ValueError('Unknown original task/variant')
        family={'14833':'qwen3-coder','14835':'qwen38'}.get(port)
        if family not in MODEL_FAMILIES:raise ValueError('Source model route unknown')
        return (case,family,strategy)
    for item in a['direct_final_records']+a['privileged_gold_source_final_records']:
        variant=item['variant']
        if variant not in ('full','gold_select'):raise ValueError('Original answer variant mislabeled')
        key=(item['case_id'],item['model_family'],variant)
        dest=states[key]
        if dest['status']!='no_recorded_attempt':raise ValueError('Duplicate original final response')
        dest.update(status='completed_final_answer',model_qwen_used_in_previous_archive=item['model_id'],
                    observed_total_tokens=int(item['tokens']),
                    observed_wall_seconds=float(item['wall_seconds']),
                    matches_original_researcher_authored_answer_key=bool(item['matches_researcher_answer']),
                    researcher_privileged_selected_sources_only=(variant=='gold_select'))
    for item in a['source_selector_records']:
        k=field(item['trial_id'])
        if k[2]!='model_select':raise ValueError('Selector is not original model_select source condition')
        dest=states[k]
        if dest['status']!='no_recorded_attempt':raise ValueError('Duplicate original selector stage')
        dest.update(status='first_stage_selector_only_no_final_answer',
                    model_qwen_used_in_previous_archive=item['model_id'],
                    selected_source_ids=item['selected_ids'],
                    observed_total_tokens=int(item['tokens']),
                    observed_wall_seconds=float(item['wall_seconds']),
                    stage2_final_correctness='unknown_stage2_not_executed',
                    completed_model_calls=1)
    seen_failed=set()
    for item in a['preflight_refusal_records']:
        k=field(item['trial_id'])
        if k in seen_failed:raise ValueError('Duplicate preflight refusal')
        seen_failed.add(k)
        dest=states[k]
        if dest['status']!='no_recorded_attempt':raise ValueError('Preflight refused after output')
        dest.update(status='admission_refused_before_inference',
                    source_preflight_category=item['reason_category'],
                    completed_model_calls=0)
    for item in a['unfinished_selector_records']:
        k=field(item['trial_id'])
        if states[k]['status']!='first_stage_selector_only_no_final_answer':
            raise ValueError('Selection interruption has no actual selected stage')
    counts=Counter(v['status'] for v in states.values())
    needed={
      'completed_final_answer':13,
      'first_stage_selector_only_no_final_answer':8,
      'admission_refused_before_inference':10,
      'no_recorded_attempt':17,
    }
    if counts!=needed or len(states)!=48:raise ValueError('Original Qwen source-cell attrition drift')
    if sum(v['status']=='completed_final_answer' and v['strategy']=='full' for v in states.values())!=9:
        raise ValueError('Source complete direct response count drift')
    if sum(v['status']=='completed_final_answer' and v['strategy']=='gold_select' for v in states.values())!=4:
        raise ValueError('Source gold_oracle answer count drift')
    return states


def matched_costs(a,states):
    bygroup=defaultdict(dict)
    for row in states.values():bygroup[(row['case_id'],row['model_family'])][row['strategy']]=row
    matched=[];direct_oracle=0;selector_oracle=0;exact_context=0;triples=0
    for (case,model),forms in sorted(bygroup.items()):
        direct=forms['full'];select=forms['model_select'];oracle=forms['gold_select']
        full=direct['status']=='completed_final_answer'
        stage1=select['status']=='first_stage_selector_only_no_final_answer'
        oracle_output=oracle['status']=='completed_final_answer'
        direct_oracle+=int(full and oracle_output)
        selector_oracle+=int(stage1 and oracle_output)
        triples+=int(full and stage1 and oracle_output)
        if stage1 and oracle_output:
            gold_context=[p for p in select.get('selected_source_ids',[]) if p in {'r1','r2'}]
            exact_context+=int(set(select['selected_source_ids'])=={'r1','r2'} and len(select['selected_source_ids'])==2)
        if full and stage1:
            matched.append({'case_id':case,'model_family':model,
                'full_direct_tokens':direct['observed_total_tokens'],
                'full_direct_wall_seconds':direct['observed_wall_seconds'],
                'selector_stage1_tokens':select['observed_total_tokens'],
                'selector_stage1_wall_seconds':select['observed_wall_seconds'],
                'researcher_answer_key_full_direct_correct':direct['matches_original_researcher_authored_answer_key'],
                'independent_action_effect_gold_not_available':True,
                'selector_stage2_is_not_completed':True})
    if len(matched)!=6:raise ValueError('Reproducible source-case model-cost matches changed')
    total=lambda key:sum(row[key] for row in matched)
    direct_tokens=total('full_direct_tokens');stage_tokens=total('selector_stage1_tokens')
    direct_s=total('full_direct_wall_seconds');stage_s=total('selector_stage1_wall_seconds')
    if (direct_tokens,stage_tokens)!=(5593,5600):raise ValueError('Original observed token records changed')
    if abs(direct_s-20.874356)>0.0001 or abs(stage_s-21.826964)>0.0001:
        raise ValueError('Original observed wall seconds changed')
    if not (direct_oracle==2 and selector_oracle==2 and exact_context==1 and triples==0):
        raise ValueError('Alternative stage-pair and triple-completion attribution changed')
    return {
       'paired_original_model_task_units':len(matched),
       'full_direct_correct_against_researcher_key':sum(r['researcher_answer_key_full_direct_correct'] for r in matched),
       'full_direct_incorrect_against_researcher_key':sum(not r['researcher_answer_key_full_direct_correct'] for r in matched),
       'completed_direct_tokens':direct_tokens,'selector_stage1_alone_tokens':stage_tokens,
       'completed_direct_wall_seconds':round(direct_s,6),'selector_stage1_alone_wall_seconds':round(stage_s,6),
       'selector_stage1_excess_tokens_before_any_answer':stage_tokens-direct_tokens,
       'selector_stage1_excess_seconds_before_any_answer':round(stage_s-direct_s,6),
       'selector_stage1_already_slower_paired_tasks':sum(r['selector_stage1_wall_seconds']>r['full_direct_wall_seconds'] for r in matched),
       'separately_completed_direct_plus_researcher_privileged_answer_pairs':direct_oracle,
       'selector_stage1_plus_separately_completed_oracle_answer_pairs':selector_oracle,
       'selector_plus_oracle_answer_pairs_with_exact_same_original_selected_source_IDS':exact_context,
       'full_direct_plus_selector_stage1_plus_oracle_answer_triples':triples,
       'genuinely_completed_model_selected_stage2_answers':0,
       'staged_total_cost_available':False,
       'staged_correctness_available':False,
       'max_possible_improvement_against_researcher_key_on_six_paired_direct_cases':sum(not r['researcher_answer_key_full_direct_correct'] for r in matched),
       'critical_contraint':'Even an apparently matching oracle answer is a different trial, not the actual model-selected next-stage execution. Never sum distinct trials as if one completed chain.',
       'matched_pair_details':matched,
    }


def runtime_metadata():
    timestamp=datetime.now(timezone.utc).isoformat()
    sql=RUNTIME/'sessions'
    counts={}
    sources={'history.sqlite3':('events','indexed_sessions','control_messages'),
             'inference_requests.sqlite3':('inference_requests',),
             'delegated_tools.sqlite3':('delegated_tool_calls',),
             'model_preemption.sqlite3':('active_calls',)}
    for name,tables in sources.items():
        dbpath=sql/name
        if not dbpath.is_file():
            counts[name]={'available':False}
            continue
        db=sqlite3.connect(f'file:{dbpath.as_posix()}?mode=ro',uri=True,timeout=1)
        try:
            counts[name]={'available':True,
                  'counts':{table:int(db.execute('SELECT count(*) FROM "'+table+'"').fetchone()[0]) for table in tables}}
        finally:db.close()
    logpath=RUNTIME/'logs/operations.jsonl'
    classes=Counter()
    if logpath.exists():
        for line in logpath.open(encoding='utf8'):
            try:rec=json.loads(line)
            except json.JSONDecodeError:classes['invalid_json_lines']+=1;continue
            name=rec.get('event')
            if not isinstance(name,str):name='missing_or_nontext_event_type'
            classes[name]+=1
    command=subprocess.run(['ss','-H','-lnt'],capture_output=True,text=True,timeout=7,check=False)
    listening={str(port):False for port in PORTS_CHECK}
    if command.returncode==0:
        # Read ONLY passive socket metadata, not model endpoint execution.
        for line in command.stdout.splitlines():
            addresses=line.split()
            if len(addresses)<4:continue
            addr=addresses[3]
            for port in PORTS_CHECK:
                if addr.endswith(':'+str(port)):listening[str(port)]=True
    else:listening={str(port):None for port in PORTS_CHECK}
    process=subprocess.run(['ps','-eo','args='],capture_output=True,text=True,timeout=8,check=False)
    matching=[]
    if process.returncode==0:
        for line in process.stdout.splitlines():
            if 'llama-server ' in line and '--port 19333' in line:
                model_match=re.search(r'--model\s+(\S+)',line)
                if model_match:
                    matching.append(Path(model_match.group(1)).name)
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=False)
        vals=[int(v.strip()) for v in gpu.stdout.splitlines()[0].split(',')]
        used,total,util=vals if gpu.returncode==0 else (None,None,None)
    except Exception:used=total=util=None
    standalone=RUNTIME/'legacy-benchmark-artifacts/openai__Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf.jetson-sweagent-qwen32k-harvested-matrix-on-nitro.json'
    separate=None
    if standalone.is_file():
        saved=json.loads(standalone.read_text())
        separate={'available':True,'file_sha256':hash_file(standalone),
          'total_original_task_instances':saved.get('total_instances'),
          'completed_instances':saved.get('completed_instances'),
          'resolved_instances':saved.get('resolved_instances'),
          'has_matched_direct_and_reviewer_arms':False,
          'is_swe_agent_single_legacy_job_not_swaag_matched_qwen':True}
    return {'sample_timestamp_UTC':timestamp,
       'source_host':'Nitro only — does not establish current Thor or Xavier availability',
       'sqlite_open_mode':'file:...mode=ro',
       'current_nitro_original_SWAAG_SQLite_counts_by_table':counts,
       'current_nitro_operations_log_event_kind_counts_only':dict(classes),
       'current_local_readonly_listening_port_snapshot':listening,
       'confirmed_non_qwen_listener_port_19333_model_basenames':sorted(set(matching)),
       'gpu_memory_used_MiB_passive_read':used,
       'gpu_memory_total_MiB_passive_read':total,
       'gpu_utilization_pct_passive_read':util,
       'GPU_queue_wait_for_archived_Qwen_calls':'not recorded',
       'per_call_GPU_VRAM_for_archived_Qwen_calls':'not recorded',
       'legacy_source_separate_single_swe_agent_outcome':separate,
       'no_inference_or_production_effect_from_audit':True,
       'live_values_are_time_bounded_not_persistent_assertions':True}


def evaluate():
    started=time.perf_counter()
    a,ctx=read_source()
    cells=cell_index(a,ctx)
    cost=matched_costs(a,cells)
    status=runtime_metadata()
    meta={'version':VERSION,'recorded_original_Qwen_evidence_sha256':SOURCE_SHA,
       'original_context_manifest_sha256':CONTEXT_SHA,
       'original_journal_lineage_source':'Previous SHA-pinned audit_qwen_source_leakage_20261026.py, not new inference',
       'source_authentic_case_count':len(ctx['cases']),'source_passage_spans_verified_in_previous_archive':40,
       'source_role_encoded_expected_selection_all_cases':True,
       'full_original_model_task_condition_cells':len(cells),
       'source_cell_status_counts':dict(Counter(c['status'] for c in cells.values())),
       'completed_full_direct_answers':sum(c['strategy']=='full' and c['status']=='completed_final_answer' for c in cells.values()),
       'completed_privileged_researcher_gold_selected_answers':sum(c['strategy']=='gold_select' and c['status']=='completed_final_answer' for c in cells.values()),
       'completed_model_selected_stage_one_only':sum(c['strategy']=='model_select' and c['status']=='first_stage_selector_only_no_final_answer' for c in cells.values()),
       'completed_model_selected_end_to_end_chains':0,
       'original_gold_independent_for_end_to_end_task':False,
       'Qwen_reviewer_correction_harm_confusion_available':False,
       'numeric_vs_word_rating_independent_human_gold_available':False,
       'current_Qwen_training_syntax_aux_matched_FLOP_available':False,
       'source_task_final_actual_effect_receipts_available':False,
       'per_original_model_call_input_and_output_token_coverage':'archived selected observations only; no full multi-stage comparison',
       'GPU_queue_VRAM_cost_record_coverage':'none for these original Qwen task calls',
       'alternative_architecture_cost':cost,'runtime_passive_snapshot':status,
       'source_cell_matrix':[cells[key] for key in sorted(cells)],
       'original_question_acceptance':{
           'role_conditioned_relevance_direction_supported':True,
           'automatic_correct_role_selector_verified':False,
           'numeric_better_than_text_or_vice_versa_proven':False,
           'one_user_initial_call_is_current_operational_default':True,
           'decomposed_model_stage2_full_final_accuracy_proven':False,
           'Qwen_peer_review_reliability_proven':False,
           'Qwen_syntax_aux_training_gain_proven':False,
       },
       'comparison_readiness_gate':{
         'original_case_source_validity':'source_authentic_but_expected_relevance_ID_leaks',
         'same_task_full_model_selected_stage2_completed':False,
         'independently_adjudicated_final_task_correctness_with_negatives':False,
         'complete_same_task_all_stage_input_output_tokens':False,
         'complete_same_task_GPU_queue_and_VRAM_costs':False,
         'Qwen_reviewer_right_to_wrong_and_wrong_to_right_golds':False,
         'safe_to_claim_winning_live_Qwen_architecture':False,
       },
       'new_model_inference_requests':0,
       'original_model_requests_repeated':False,
       'previously_denied_operations_retried':False,
       'actual_production_SQLite_mutations':0,
       'no_model_service_unloaded_or_restarted':True,
       'observational_audit_wall_seconds':round(time.perf_counter()-started,5),
       'limitations':[
           'Source case and answer anchors are author-selected and exposed role-coded relevant IDs; cannot infer independent real-task correctness.',
           'A cached completed researcher-oracle source-selected answer is a distinct trial, not a finished stage two after model source selection.',
           'Current Nitro listening ports and SQLite counts are not evidence that Thor has no active Qwen model or source data.',
           'Runtime GPU occupancy is a passive snapshot and cannot be retrospectively allocated to prior model calls or establish queue latency.',
           'A single resolved legacy SWE-bench instance is not a matched architecture comparison and has no independent reviewer confusion.',
           'No real model or original user task effect was rerun in this audit; earlier denied Thor operations were left denied.',
       ]}
    return meta


def main():
    out=evaluate()
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'original_qwen_source_and_live_readonly_eligibility_v1.json'
    if path.exists():raise FileExistsError('Existing versioned evidence; refuse to overwrite')
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('=== COVERAGE ===',json.dumps({k:out[k] for k in (
        'source_cell_status_counts','completed_full_direct_answers',
        'completed_privileged_researcher_gold_selected_answers',
        'completed_model_selected_stage_one_only','completed_model_selected_end_to_end_chains')},indent=2))
    print('=== MATCHED LOWER BOUND ===',json.dumps({k:v for k,v in out['alternative_architecture_cost'].items() if k!='matched_pair_details'},indent=2))
    print('=== TIME-BOUND RUNTIME ===',json.dumps(out['runtime_passive_snapshot'],indent=2))
    print('=== READY ===',out['comparison_readiness_gate']['safe_to_claim_winning_live_Qwen_architecture'])
    print('OUTPUT',path)

if __name__=='__main__':main()
