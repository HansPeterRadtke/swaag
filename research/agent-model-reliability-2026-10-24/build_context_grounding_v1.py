#!/usr/bin/env python3
"""Freeze source-exact authentic SWAAG investigation contexts for task-outcome trials.

The historical technical facts are exact paragraphs from pre-existing published
project reports. Questions/option sets are study-authored and anchored to those
facts; this is not an independently human-rater adjudicated relevance dataset.
"""
import hashlib, json, random
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FILES={'worker':'docs/fast-worker-world-context-2026-10-20.md',
       'history':'docs/fast-long-history-candidates-2026-10-21.md',
       'token':'docs/tokenizer-admission-correction-2026-10-16.md',
       'world':'docs/foreground-world-context-2026-10-19.md'}
TASKS=[
 ('fresh_worker_status','worker',1,4,'When SWAAG has no fresh or revision-consistent cached worker snapshot, which behavior is permitted for a status answer?',
  ['do_not_present_stale_state','trust_stale_state','fabricate_completion'],'do_not_present_stale_state'),
 ('worker_completion_percentage','worker',2,3,'Which source justifies a worker-level completion percentage rather than a plan-node percentage?',
  ['validated_worker_step_weights','elapsed_time_ratio','finite_plan_node_count'],'validated_worker_step_weights'),
 ('older_history_lookup','history',0,2,'What actually locates older user utterances cheaply before a brief first response, without scanning the full tool/event database?',
  ['scoped_user_message_index','full_global_tool_log_scan','only_six_most_recent_turns'],'scoped_user_message_index'),
 ('oversized_message','history',2,4,'When a historical user message is too large for the fast candidate character budget, what is preserved in its place?',
  ['exact_event_pointer','arbitrarily_truncated_quote','deleted_original_message'],'exact_event_pointer'),
 ('old_message_authority','history',2,3,'What authority does a matching old user statement have if a later current request or correction conflicts with it?',
  ['current_request_and_supersession_win','old_quote_overrides_current','matching_text_grants_tool_permission'],'current_request_and_supersession_win'),
 ('tokenizer_zero_defect','token',1,2,'What caused the earlier oversized completion-review request to be admitted incorrectly?',
  ['empty_token_count_accepted_for_nonempty_prompt','new_large_context_model_was_loaded','text_was_under_context_limit'],'empty_token_count_accepted_for_nonempty_prompt'),
 ('tokenizer_correction','token',3,5,'After the verified counting correction, what happened to the same oversized original request?',
  ['rejected_as_over_context','silently_truncated_to_fit','accepted_with_zero_token_count'],'rejected_as_over_context'),
 ('compact_world_projection','world',1,2,'How are full active worker and plan facts prepared for the foreground without dumping all terminal worker histories?',
  ['background_prepared_world_view','requery_each_worker_with_llm_per_turn','keep_only_four_opaque_ids'],'background_prepared_world_view'),
]
DECOYS={
 'worker': [('history',5),('token',4),('world',4)],
 'history':[('worker',5),('token',4),('world',7)],
 'token':[('worker',5),('history',6),('world',6)],
 'world':[('history',5),('token',4),('worker',7)],
}

def paragraph(source,index):
    f=PROJECT/FILES[source]
    content=f.read_text('utf-8')
    matches=[];cursor=0
    for block in content.split('\n\n'):
        start=content.find(block,cursor)
        if start<0:raise ValueError('paragraph_offset')
        cursor=start+len(block)
        trimmed=block.strip();left=len(block)-len(block.lstrip())
        if trimmed and not trimmed.startswith('#'):
            matches.append((trimmed,start+left,start+left+len(trimmed)))
    s,start,end=matches[index]
    assert content[start:end]==s
    return {'file':FILES[source], 'file_sha256':hashlib.sha256(content.encode()).hexdigest(),
            'start':start,'end':end,'text':s}

def build():
    cases=[]
    for name,src,i,j,question,choices,gold in TASKS:
        assert len(set(choices))==3 and gold in choices
        # Required support set. It is investigator-selected—not proof that both
        # are logically indispensable on every language interpretation.
        passes=[dict(pid='r1',**paragraph(src,i)),dict(pid='r2',**paragraph(src,j))]
        passes += [dict(pid=f'd{k}',**paragraph(s,p)) for k,(s,p) in enumerate(DECOYS[src],1)]
        rng=random.Random('20261024-'+name);rng.shuffle(passes)
        cases.append({'case_id':name,'question':question,'answer_options':choices,'gold_answer':gold,
                      'source_basis':'independent explicit observed descriptions from unchanged SWAAG engineering reports; authored question and expected option, not blinded human rating',
                      'support_ids':['r1','r2'],'passages':passes})
    return {'version':'2026-10-24-context-grounding-v1','cases':cases}

if __name__=='__main__':
    out=HERE/'context_grounding_v1.json'
    if out.exists():raise SystemExit('frozen dataset already present; no overwrite')
    result=build();out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('cases',len(result['cases']),'source_files',len(FILES),'max_prompt_chars',max(sum(len(p['text']) for p in c['passages']) for c in result['cases']))
