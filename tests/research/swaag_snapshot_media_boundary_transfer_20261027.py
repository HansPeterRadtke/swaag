#!/usr/bin/env python3
"""Independent live-runtime replica on heldout actual Android screenshot files.

Actual original publisher PNG captures + author mutations, same-app different
Activity negatives and intentional invalid-PNG end-boundary controls.
The selection is frozen before any tool invocation; source gold is publisher
version/source IDs, not independently human-rated visual severity.

Never touches production history/media/services. All effects occur inside a
NEW isolated /data/var root and cannot overwrite previous research receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
import struct
import time

from swaag.config import load_config
from swaag.perception.acceptance_policy import RenderSourceBinding
from swaag.runtime import AgentRuntime
from swaag.tools.verified_snapshots import HostAttestedSnapshot

BASE = Path('/data/var/perceptual-independent-20261024')
SRC = BASE/'rotbench-unused-source-split-20261026'
PREV = BASE/'rotbench-android-heldout-20261026'
WRONG = BASE/'rotbench-same-app-clean-state-negative-20261026'
ROOT = BASE/'snapshot_media_source_replication-20261027'
RELEASE_SHA='0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7'


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for fragment in iter(lambda:handle.read(1048576),b''):
            h.update(fragment)
    return h.hexdigest()


def save(path,obj):
    if path.exists():raise RuntimeError('Frozen research receipt exists: '+str(path))
    partial=path.with_suffix('.tmp')
    partial.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
    partial.rename(path)


def load(path):return json.loads(path.read_text())


def original_png(rel):
    for p in (SRC/'media'/rel,PREV/'media'/rel):
        if p.is_file():return p
    raise FileNotFoundError('No new source retrieval: '+rel)


def source_state(ref):
    a=ref.split('/')
    if len(a)!=4 or a[0]!='image' or a[1]!='right':raise ValueError(ref)
    filename=a[-1]
    m=re.match(r'^(.+)_(portrait|landscape)\.png$',filename)
    if not m:raise ValueError(ref)
    return dict(source_kind='author-approved Android source screen',source_revision=RELEASE_SHA,
                subject_id=a[2],state_id=m.group(1),orientation=m.group(2))


def width_height(path):
    with path.open('rb') as handle:hdr=handle.read(24)
    if hdr[:16]!=b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR':raise ValueError('bad original PNG')
    return struct.unpack('>II',hdr[16:24])


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    if any((ROOT/x).exists() for x in ('frozen.json','runtime_results.json')):
        raise RuntimeError('Earlier experiment already frozen')
    old_release=PREV/'source_test.json';assert file_sha(old_release)==RELEASE_SHA
    selection=load(SRC/'selection.json')['unused_samples']
    prev=load(SRC/'predictions.json')['cases'];previous_by_id={row['id']:row for row in prev}
    assert len(selection)==len(prev)==240
    # Publisher-authored changed screenshots. Hash ranks are decided without
    # consulting detector outcomes or final pixels.
    candidates=[x for x in selection if any('/defective/' in p for p in x['native_screenshots'])]
    candidates=sorted(candidates,key=lambda x:hashlib.sha256(f"v11-positive|{x['id']}".encode()).hexdigest())
    modified=[];clean=[]
    for case in candidates:
        if len(modified)>=8:break
        i=next((i for i,(a,b) in enumerate(zip(case['native_screenshots'],case['same_orientation_source_refs'])) if a!=b),None)
        if i is None:continue
        ref=case['same_orientation_source_refs'][i];final=case['native_screenshots'][i]
        try: pa,pb=original_png(ref),original_png(final)
        except FileNotFoundError:continue  # Do not retry two known unavailable author references.
        predicted=previous_by_id[case['id']]
        assert file_sha(pa)==predicted['refs_hash'][i]
        assert file_sha(pb)==predicted['imgs_hash'][i]
        if file_sha(pa)==file_sha(pb):continue
        astate=source_state(ref);a_dims=width_height(pa);b_dims=width_height(pb)
        if a_dims!=b_dims:continue
        state={**astate,'width_px':a_dims[0],'height_px':a_dims[1]}
        item=dict(id=case['id'],kind='changed',expected='fail',approved_ref=ref,
                  actual_ref=final,approved_sha=file_sha(pa),actual_sha=file_sha(pb),
                  approved_state=state,actual_state=state)
        modified.append(item)
        clean.append({**item,'kind':'clean','expected':'pass','actual_ref':ref,'actual_sha':item['approved_sha']})
    assert len(modified)==len(clean)==8
    # Publisher-author approved different Activity, same app/orientation.
    arr=load(WRONG/'frozen_pairs.json')['comparisons']
    arr=sorted(arr,key=lambda x:hashlib.sha256(f"v11-wrong|{x['case_id']}|{x['other_case_id']}".encode()).hexdigest())
    wrong=[];apps=set()
    for q in arr:
        if len(wrong)>=8:break
        if q['app'] in apps:continue
        i=0
        ap=q['other_clean_screenshots'][i];actual=q['screenshots'][i]
        pa=original_png(ap);pb=original_png(actual)
        assert file_sha(pa)==q['other_ref_sha'][i] and file_sha(pb)==q['actual_sha'][i]
        w,h=width_height(pa)
        assert (w,h)==width_height(pb)
        approved_state={**source_state(ap),'width_px':w,'height_px':h}
        actual_state={**source_state(actual),'width_px':w,'height_px':h}
        assert approved_state['state_id']!=actual_state['state_id']
        wrong.append(dict(id=q['case_id'],kind='wrong_activity',expected='review',approved_ref=ap,
                          actual_ref=actual,approved_sha=q['other_ref_sha'][i],actual_sha=q['actual_sha'][i],
                          approved_state=approved_state,actual_state=actual_state))
        apps.add(q['app'])
    assert len(wrong)==8
    # Synthetic corruption of exact previously approved public capture, PNG
    # header (and dimensions) preserved, last IEND removed. This is labelled
    # by the script, NOT an independent human perception defect.
    invalid=[]
    for item in clean:
        p=original_png(item['approved_ref']);b=p.read_bytes()
        assert b.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82')
        invalid.append({**item,'kind':'truncated_final_png','expected':'review','bad_bytes_sha':hashlib.sha256(b[:-12]).hexdigest()})
    manifest={'source_release_sha':RELEASE_SHA,'source_selection_sha':file_sha(SRC/'selection.json'),
              'prior_prediction_sha':file_sha(SRC/'predictions.json'),
              'wrong_activity_source_sha':file_sha(WRONG/'frozen_pairs.json'),
              'frozen_before_evaluation':True,'strict_property':'verified exact source and final PNG bytes under immutable-only host contract',
              'cases':clean+modified+wrong+invalid,'n_by_class':{'clean':8,'changed':8,'wrong_activity':8,'truncated_final_png':8},
              'honest_gold':'Publisher authored source right/mutated state plus script-controlled PNG boundary truncations; not independently human-rated visual defects.',
              'host_trust':'Host pins both ORIGINAL source and actual final capture digests before tool sees attachment IDs, not re-derived from model input.',
              'previous_source_references_unchanged':True,'runtime':'temporary isolated SWAAG/SQLite >=3.51.3; pure tool, GPU not used'}
    save(ROOT/'frozen.json',manifest)
    print('INDEPENDENT_REAL_SCREENS_FROZEN',file_sha(ROOT/'frozen.json'),manifest['n_by_class'],'apps_wrong',len(apps),flush=True)


def runtime():
    assert tuple(map(int,sqlite3.sqlite_version.split('.'))) >= (3,51,3)
    root=ROOT/'sessions'
    env={'SWAAG__SESSIONS__ROOT':str(root),'SWAAG__AGENT_DATA__ROOT':str(ROOT/'agent_data'),
         'SWAAG__MODEL__BASE_URL':'http://127.0.0.1:9999','SWAAG__MODEL__CACHE_ENABLED':'false',
         'SWAAG__RETRIEVAL__BACKEND':'unavailable',
         'SWAAG__TOOLS__READ_ROOTS':json.dumps([str(ROOT)]),
         'SWAAG__TOOLS__STAGED_DISCOVERY':'false',
         'SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED':'false'}
    config=load_config(env=env)
    config.tools.enabled=['verify_immutable_snapshot','list_attachments']
    config.perception.enabled=False
    config.embedding_index.enabled=False
    return AgentRuntime(config,model_client=object())


def run():
    if (ROOT/'runtime_results.json').exists():raise RuntimeError('Prior run must not be repeated')
    manifest=load(ROOT/'frozen.json')
    assert manifest['source_selection_sha']==file_sha(SRC/'selection.json')
    assert manifest['source_release_sha']==file_sha(PREV/'source_test.json')
    assert manifest['prior_prediction_sha']==file_sha(SRC/'predictions.json')
    assert manifest['wrong_activity_source_sha']==file_sha(WRONG/'frozen_pairs.json')
    start=time.monotonic();rt=runtime();session=rt.create_or_load_session()
    rows=[]
    for case in manifest['cases']:
        ref_bytes=original_png(case['approved_ref']).read_bytes()
        if case['kind']=='truncated_final_png':
            final_bytes=ref_bytes[:-12]
            assert hashlib.sha256(final_bytes).hexdigest()==case['bad_bytes_sha']
        else:final_bytes=original_png(case['actual_ref']).read_bytes()
        assert hashlib.sha256(ref_bytes).hexdigest()==case['approved_sha']
        if case['kind']!='truncated_final_png':assert hashlib.sha256(final_bytes).hexdigest()==case['actual_sha']
        ref=rt.add_attachment(ref_bytes, original_name='reference.png',media_type='image/png',source='pinned_publisher_original',session_id=session.session_id)
        actual=rt.add_attachment(final_bytes,original_name='final.png',media_type='image/png',source='pinned_author_or_scripted_capture',session_id=session.session_id)
        source_bind=RenderSourceBinding(**case['approved_state']);actual_bind=RenderSourceBinding(**case['actual_state'])
        # The two trusted independent expected source/capture digests were
        # saved in frozen research manifest, NOT copied from the attachment
        # metadata during this tool invocation.
        def attest(approved,observed,record=case):
            return HostAttestedSnapshot(approved_attachment_id=approved.attachment_id,
                actual_attachment_id=observed.attachment_id,
                approved_identity=RenderSourceBinding(**record['approved_state']),
                actual_identity=RenderSourceBinding(**record['actual_state']),
                manifest_sha256=file_sha(ROOT/'frozen.json'),
                approved_source_sha256=record['approved_sha'],
                actual_capture_sha256=record.get('bad_bytes_sha',record['actual_sha']),
                reference_manifest_attested=True,rendered_state_attested=True,immutable_snapshot_contract=True)
        rt.bind_tool_runtime_capability(session.session_id,'immutable_snapshot_attestor',attest)
        call_start=time.monotonic()
        answer=rt.execute_tool_once('verify_immutable_snapshot',
          {'approved_attachment_id':ref.attachment_id,'actual_attachment_id':actual.attachment_id},
          session_id=session.session_id)
        latency=round(time.monotonic()-call_start,4)
        if answer.error or not answer.tool_result:raise RuntimeError('SWAAG tool failed '+str(answer.error))
        actual_verdict=answer.tool_result.output['verdict']
        assert actual_verdict==case['expected'],(case['kind'],case['id'],actual_verdict,answer.tool_result.output)
        events=[event for event in rt.history.read_history(session.session_id) if event.event_type=='tool_result' and event.payload.get('tool_name')=='verify_immutable_snapshot']
        assert events and events[-1].payload['output']['verdict']==actual_verdict
        rows.append({'source_case_id':case['id'],'kind':case['kind'],'expected':case['expected'],
            'tool_verdict':actual_verdict,'tool_reason':answer.tool_result.output['reason'],
            'png_header_intact_on_corrupted_final':case['kind']=='truncated_final_png',
            'source_sha_match':ref.sha256==case['approved_sha'],
            'capture_sha_match':actual.sha256==case.get('bad_bytes_sha',case['actual_sha']),
            'tool_wall_s':latency,'event_hash':events[-1].hash})
        print('INDEPENDENT_REPLICATION',len(rows),'/',len(manifest['cases']),case['kind'],actual_verdict,'s',latency,flush=True)
    assert len(rows)==32
    counts={x:{'n':sum(r['kind']==x for r in rows),'correct':sum(r['kind']==x and r['tool_verdict']==r['expected'] for r in rows)} for x in manifest['n_by_class']}
    assert all(v['n']==v['correct']==8 for v in counts.values())
    elapsed=round(time.monotonic()-start,3)
    result={'source_manifest_sha':file_sha(ROOT/'frozen.json'), 'status':'PROPERTY_RULES_PASS_32_OF_32_SOURCE_CONTROLS',
        'results':rows,'totals':counts,'wall_s_including_attachment_and_history':elapsed,
        'CPU_GPU':'CPU only; no model servers or production sessions',
        'interpretation':'32 source-labeled image examples across 8 per class, correlated and not human ratings of visual quality. No general population accuracy claim.',
        'real_runtime_and_durable_history':True,'live_external_user_or_live_Android_attestor_tested':False}
    save(ROOT/'runtime_results.json',result)
    print('INDEPENDENT_SOURCE_PNG_END_BOUNDARY_REPLICATION_FROZEN',file_sha(ROOT/'runtime_results.json'),counts,'wall_s',elapsed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run'])
    globals()[parser.parse_args().stage]()
