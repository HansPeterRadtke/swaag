#!/usr/bin/env python3
"""Actual SWAAG tool dispatcher, raw Android PNGs, durable saved-state reentry.

Uses ONLY an isolated test SQLite 3.54.1 runtime, no live production session,
no GPU/model call. The host attestor is reconstructed from SHA-verified
original publisher manifests; it is NOT live Android device attestation.

Stages freeze -> execute -> recover are different process invocations.
All stages refuse to overwrite receipts, original source pixels never edited.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

from swaag.config import load_config
from swaag.perception.acceptance_policy import RenderSourceBinding
from swaag.runtime import AgentRuntime
from swaag.tools.verified_snapshots import HostAttestedSnapshot

BASE=Path('/data/var/perceptual-independent-20261024')
OLD=BASE/'rotbench-android-heldout-20261026'
CLEAN=BASE/'rotbench-same-app-clean-state-negative-20261026'
OUT=BASE/'swaag_real_android_tool_roundtrip_20261027'
SOURCE_RELEASE_SHA='0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7'
ACTUAL_BUG_CASE_ID=247
SQLITE_REQUIRE=(3,51,3)


def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def read(path):return json.loads(path.read_text())


def write_once(path,obj):
 if path.exists():raise RuntimeError('FROZEN result already exists: '+str(path))
 temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
 temp.rename(path)


def original_path(relative):
 candidates=(OLD/'media'/relative,BASE/'rotbench-unused-source-split-20261026/media'/relative)
 for candidate in candidates:
  if candidate.is_file():return candidate
 raise FileNotFoundError('Original source not locally available, no download: '+relative)


def binding_dict(relative, width,height):
 parts=relative.split('/')
 assert parts[0]=='image' and parts[2] and parts[-1].endswith('.png')
 name=parts[-1]
 orientation=next((x for x in ('portrait','landscape') if name.endswith('_'+x+'.png')),None)
 if orientation is None:raise ValueError('Source output must identify exact orientation')
 activity=name[:-(len(orientation)+5)]
 return dict(source_kind='author-approved-Android-screen-pixels',source_revision=SOURCE_RELEASE_SHA,
             subject_id=parts[2],state_id=activity,orientation=orientation,width_px=width,height_px=height)


def native_png_size(path):
 with path.open('rb') as f:header=f.read(24)
 assert header[:16]==b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR'
 return struct.unpack('>II',header[16:24])


def runtime():
 import sqlite3
 assert tuple(int(x) for x in sqlite3.sqlite_version.split('.'))>=SQLITE_REQUIRE,sqlite3.sqlite_version
 env={'SWAAG__SESSIONS__ROOT':str(OUT/'sessions'),
      'SWAAG__AGENT_DATA__ROOT':str(OUT/'agent-data'),
      'SWAAG__TOOLS__READ_ROOTS':json.dumps([str(OUT)]),
      'SWAAG__TOOLS__STAGED_DISCOVERY':'false',
      'SWAAG__RUNTIME__COMPLETION_EVALUATION_ENABLED':'false',
      'SWAAG__RETRIEVAL__BACKEND':'unavailable',
      'SWAAG__MODEL__BASE_URL':'http://127.0.0.1:9999',
      'SWAAG__MODEL__CACHE_ENABLED':'false'}
 config=load_config(env=env)
 config.tools.enabled=['list_attachments','verify_immutable_snapshot']
 config.perception.enabled=False
 config.embedding_index.enabled=False
 return AgentRuntime(config,model_client=object())


def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 if any((OUT/name).exists() for name in ('frozen_manifest.json','execution.json','recovery.json')):raise RuntimeError('Historical run exists')
 original=OLD/'source_test.json';assert file_sha(original)==SOURCE_RELEASE_SHA
 selected=read(OLD/'selection.json');prediction=read(OLD/'predictions.json');gold=read(OLD/'scored.json')
 assert len(selected['selection'])==100 and len(prediction['cases'])==100
 case=next(x for x in selected['selection'] if x['record_index']==ACTUAL_BUG_CASE_ID)
 model=next(x for x in prediction['cases'] if x['record_index']==ACTUAL_BUG_CASE_ID)
 goldrow=next(x for x in gold['selected_cases'] if x['record_index']==ACTUAL_BUG_CASE_ID)
 assert goldrow['gold_bug'] and not goldrow['predictions']['changed_area_0p05pct']
 index=1
 approved=case['references'][index];altered=case['images'][index]
 assert approved.startswith('image/right/')
 changes={}
 for key,rel,sha_rec in (('reference',approved,model['reference_sha256'][index]),('changed',altered,model['images_sha256'][index])):
  path=original_path(rel)
  assert file_sha(path)==sha_rec
  width,height=native_png_size(path)
  assert (height,width)==tuple(model['orientation_pixel_stats'][index]['size_hw'])
  changes[key]={'relative_path':rel,'sha256':sha_rec,'original_dimensions':[width,height],'state':binding_dict(approved,width,height),'source_test_record':ACTUAL_BUG_CASE_ID}
 assert changes['reference']['sha256']!=changes['changed']['sha256']
 negatives=read(CLEAN/'frozen_pairs.json')['comparisons'];q=negatives[0]
 wrong={}
 for typ,path,expected,state in (('reference',q['other_clean_screenshots'][0],q['other_ref_sha'][0],q['other_activity']),('actual',q['screenshots'][0],q['actual_sha'][0],q['activity'])):
  originalfile=original_path(path)
  assert file_sha(originalfile)==expected
  dimensions=native_png_size(originalfile)
  ident=binding_dict(path,*dimensions)
  assert ident['state_id']==state
  wrong[typ]={'relative_path':path,'sha256':expected,'original_dimensions':list(dimensions),'state':ident}
 assert wrong['reference']['state']['subject_id']==wrong['actual']['state']['subject_id']
 assert wrong['reference']['state']['state_id']!=wrong['actual']['state']['state_id']
 manifest={'created':'2026-10-27','input_source_sha256':SOURCE_RELEASE_SHA,'old_100_selection_sha256':file_sha(OLD/'selection.json'),'old_100_prediction_sha256':file_sha(OLD/'predictions.json'),'old_100_labels_sha256':file_sha(OLD/'scored.json'),'wrong_activity_352_original_manifest_sha256':file_sha(CLEAN/'frozen_pairs.json'),'derived_sources':{'clean':{'approved':changes['reference'],'actual':changes['reference']},'changed':{'approved':changes['reference'],'actual':changes['changed']},'wrong_activity':{'approved':wrong['reference'],'actual':wrong['actual']}},'expected':{'clean':'pass','changed':'fail','wrong_activity':'review'},'target':'actual SWAAG AgentRuntime attachment store->tool invocation->journal->process restart->trusted rebind->durable replay','source_attestation_limit':'Trusted RESEARCH harness only reproduces author source file lineage, not a live Android device Activity/viewport attestation; NOT production enabled','verified_final_png_decoder':'ffmpeg will decode each source PNG to a real raster; no OCR/model prediction used','min_sqlite_version':'3.51.3','control_no_web_or_gpu_or_existing_app_changes':True}
 write_once(OUT/'frozen_manifest.json',manifest)
 print('REAL_ANDROID_SOURCE_TO_SWAAG_FROZEN',file_sha(OUT/'frozen_manifest.json'),'cases',[(k,v['expected'] if 'expected' in v else manifest['expected'][k]) for k,v in manifest['derived_sources'].items()])


def capability_from_saved_host_manifest(case,manifest):
 ref=case['approved'];actual=case['actual']
 def provide(reference,observed):
  if reference.sha256!=ref['sha256'] or observed.sha256!=actual['sha256']:
   return None
  return HostAttestedSnapshot(
   approved_attachment_id=reference.attachment_id,
   actual_attachment_id=observed.attachment_id,
   approved_identity=RenderSourceBinding(**ref['state']),
   actual_identity=RenderSourceBinding(**actual['state']),
   manifest_sha256=file_sha(OUT/'frozen_manifest.json'),
   reference_manifest_attested=True,rendered_state_attested=True,
   immutable_snapshot_contract=True)
 return provide


def actual_png_ffmpeg(p):
 t=time.monotonic()
 run=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-frames:v','1','-f','null','-'],capture_output=True,text=True,timeout=18)
 if run.returncode or run.stderr.strip():raise RuntimeError('Actually rendered PNG does not decode: '+run.stderr[:300])
 return round(time.monotonic()-t,3)


def execute():
 if (OUT/'execution.json').exists():raise RuntimeError('Prior executed source state')
 manifest=read(OUT/'frozen_manifest.json')
 assert manifest['input_source_sha256']==file_sha(OLD/'source_test.json')
 start=time.monotonic();a=runtime();out={}
 for name,case in manifest['derived_sources'].items():
  session=a.create_or_load_session()
  actuals=[]
  for side in ('approved','actual'):
   source=case[side]
   path=original_path(source['relative_path'])
   assert file_sha(path)==source['sha256']
   assert list(native_png_size(path))==source['original_dimensions']
   actuals.append(actual_png_ffmpeg(path))
  approved=a.add_attachment(original_path(case['approved']['relative_path']).read_bytes(), original_name=name+'_reference.png',media_type='image/png',source='verified_research_original_publisher',session_id=session.session_id)
  actual=a.add_attachment(original_path(case['actual']['relative_path']).read_bytes(), original_name=name+'_final.png',media_type='image/png',source='verified_research_final_pixels',session_id=session.session_id)
  a.bind_tool_runtime_capability(session.session_id,'immutable_snapshot_attestor',capability_from_saved_host_manifest(case,manifest))
  before=time.monotonic()
  result=a.execute_tool_once('verify_immutable_snapshot',{'approved_attachment_id':approved.attachment_id,'actual_attachment_id':actual.attachment_id},session_id=session.session_id)
  runtime_s=round(time.monotonic()-before,4)
  if result.error or not result.tool_result:raise RuntimeError(f'User-facing SWAAG dispatch failure {name}: {result.error}')
  content=result.tool_result.output
  assert content['verdict']==manifest['expected'][name],(name,content)
  assert content['approved_sha256']==approved.sha256==case['approved']['sha256']
  assert content['actual_sha256']==actual.sha256==case['actual']['sha256']
  assert content['universal_media_release_approved'] is False
  assert len(content['source_event_references'])==2
  history=a.history.read_history(session.session_id)
  events=[e for e in history if e.event_type=='tool_result' and e.payload.get('tool_name')=='verify_immutable_snapshot']
  assert len(events)==1
  out[name]={'session_id':session.session_id,'approved_attachment_id':approved.attachment_id,'actual_attachment_id':actual.attachment_id,'source_approved_sha':approved.sha256,'actual_sha':actual.sha256,'tool_output':content,'tool_result_event_sequence':events[0].sequence,'tool_result_event_hash':events[0].hash,'raw_source_PNGs_decoded_as_real_images_s':actuals,'duration_tool_once_s':runtime_s,'original_context_sessions_are_isolated_from_production':True}
  print('REAL_SWAAG_PNG_ATTACHMENT_TOOL',name,content['verdict'],'result_hash',events[0].hash,'tool_s',runtime_s,'PNG_decode_s',actuals,flush=True)
 write_once(OUT/'execution.json',{'frozen_manifest_sha':file_sha(OUT/'frozen_manifest.json'),'sqlite_version':__import__('sqlite3').sqlite_version,'research_runtime_tool_register_enabled_in_temporary_config':True,'process_start_epoch':start,'total_wall_s':round(time.monotonic()-start,3),'cases':out})
 print('REAL_SWAAG_TOOL_OUTPUT_FROZEN',file_sha(OUT/'execution.json'))


def recover():
 if (OUT/'recovery.json').exists():raise RuntimeError('Already recovered persisted states')
 manifest=read(OUT/'frozen_manifest.json');previous=read(OUT/'execution.json')
 assert previous['frozen_manifest_sha']==file_sha(OUT/'frozen_manifest.json')
 a=runtime();out={}
 start=time.monotonic()
 for name,info in previous['cases'].items():
  case=manifest['derived_sources'][name]
  sid=info['session_id']
  reconstructed=a.history.rebuild_from_history(sid,write_projections=False)
  assert {x.attachment_id for x in reconstructed.attachments}=={info['approved_attachment_id'],info['actual_attachment_id']}
  source_events=[x for x in a.history.read_history(sid) if x.event_type=='tool_result' and x.payload.get('tool_name')=='verify_immutable_snapshot']
  assert len(source_events)==1 and source_events[0].hash==info['tool_result_event_hash']
  assert source_events[0].payload['output']['verdict']==manifest['expected'][name]
  # Critical: runtime capability is IN-MEMORY and untrusted sources MUST NOT
  # obtain a PASS/FAIL merely by replaying the old session journal.
  assert not a.tool_runtime_capabilities(sid)
  no_trust=a.execute_tool_once('verify_immutable_snapshot',{'approved_attachment_id':info['approved_attachment_id'],'actual_attachment_id':info['actual_attachment_id']},session_id=sid)
  assert no_trust.error and no_trust.tool_result is None
  assert 'requires immutable_snapshot_attestor' in no_trust.error['error']
  # A fresh process obtains same SHA-verified manifest from independently
  # preserved original publisher source, then explicitly REBINDS the trusted
  # host capability, rather than inferring it from a previous successful call.
  for side in ('approved','actual'):
   ref=case[side];p=original_path(ref['relative_path'])
   assert file_sha(p)==ref['sha256']
  a.bind_tool_runtime_capability(sid,'immutable_snapshot_attestor',capability_from_saved_host_manifest(case,manifest))
  after=a.execute_tool_once('verify_immutable_snapshot',{'approved_attachment_id':info['approved_attachment_id'],'actual_attachment_id':info['actual_attachment_id']},session_id=sid)
  assert after.error is None and after.tool_result is not None
  assert after.tool_result.output['verdict']==manifest['expected'][name]
  assert after.tool_result.output['actual_sha256']==info['actual_sha']
  newhistory=a.history.read_history(sid)
  result_events=[e for e in newhistory if e.event_type=='tool_result' and e.payload.get('tool_name')=='verify_immutable_snapshot']
  assert len(result_events)==2
  assert result_events[0].hash==info['tool_result_event_hash']
  assert all(e.hash for e in result_events)
  assert all(x.metadata.get('source_event_hash') for x in a.history.rebuild_from_history(sid,write_projections=False).attachments)
  # A real ARCHIVE/re-entry check verifies the saved original tool verdict
  # survives when active projections are removed. Never reuse archived input
  # for a second action.
  archived=a.history.archive_session(sid,remove_active=True)
  reentry=a.history.rebuild_from_history(sid,write_projections=False)
  assert any(m.role=='tool' and m.name=='verify_immutable_snapshot' for m in reentry.messages)
  out[name]={'session_id':sid,'initial_result_sha256':info['tool_result_event_hash'],'rebound_tool_verdict':after.tool_result.output['verdict'],'rebind_result_event_hash':result_events[1].hash,'result_events_persisted':len(result_events),'archive_event_count':archived['event_count'],'reentry_still_has_source_attachments':len(reentry.attachments)==2,'no_capability_failure_is_explicit':True,'archived_safely':True}
  print('REAL_SWAAG_SESSION_RELOADED_AND_ARCHIVED',name,'verdict',out[name]['rebound_tool_verdict'],'journal_results',len(result_events),'archive_events',archived['event_count'],flush=True)
 write_once(OUT/'recovery.json',{'frozen_manifest_SHA':file_sha(OUT/'frozen_manifest.json'),'execution_SHA':file_sha(OUT/'execution.json'),'all_trusted_metadata_reconstructed_from_saved_independent_host_source':True,'trusted_host_capability_was_volatile_not_in_journal':True,'CPU_wall_s':round(time.monotonic()-start,3),'all_archived_session_journals_had_exact_lineage':True,'cases':out})
 print('COMPLETE_ORIGINAL_ANDROID_TO_RUNTIME_ATTACHMENT_TOOL_RELOAD_AND_ARCHIVE',file_sha(OUT/'recovery.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','execute','recover'));globals()[p.parse_args().stage]()
