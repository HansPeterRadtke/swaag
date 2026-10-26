#!/usr/bin/env python3
"""Evidence-driven source-identity repair: replay original Android controls.

The old frozen 352 same-APP/different-ACTIVITY clean comparisons suffered
352/352 image-difference false alarms when a wrong approved reference was
used. A verified-reference flag alone cannot protect a high-level caller from
misbinding sources. This replay checks original source/version/state and
both actual file SHAs before any EXACT IMMUTABLE BYTE contract may PASS/FAIL.

Previously selected author-mutated cases and missing public references provide
controls. This IS NOT a human-rated GUI clipping/contrast/occlusion evaluation;
new labels are not invented, original data and past outcomes are never edited.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

BASE=Path('/data/var/perceptual-independent-20261024')
OLD=BASE/'rotbench-android-heldout-20261026'
NEW=BASE/'rotbench-unused-source-split-20261026'
CLEAN=BASE/'rotbench-same-app-clean-state-negative-20261026'
OUT=BASE/'rotbench_source_binding_replay_20261026'
REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'src/swaag/perception/acceptance_policy.py'
PIN='0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7'


def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as source:
  for chunk in iter(lambda:source.read(1024*1024),b''):
   h.update(chunk)
 return h.hexdigest()


def read(path):return json.loads(path.read_text())

def save(path,value):
 if path.exists():raise RuntimeError('Frozen historical study already exists '+str(path))
 tmp=path.with_suffix('.tmpnew')
 tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
 tmp.replace(path)


def original_png(src_path):
 for folder in (OLD/'media',NEW/'media'):
  f=folder/src_path
  if f.exists():return f
 # Explicitly do not try unavailable original public paths again.
 return OLD/'media'/src_path


def policy():
 spec=importlib.util.spec_from_file_location('source_bound_policy_replay_standalone',POLICY)
 module=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=module
 spec.loader.exec_module(module)
 return module


def make_binding(P,app,state,orientation,dims):
 return P.RenderSourceBinding(
  source_kind='original-android-screen',source_revision=PIN,
  subject_id=app,state_id=state,orientation=orientation,
  width_px=int(dims[0]),height_px=int(dims[1]))


def compare(P,approved,actual,reference_file,actual_file,reference_sha,actual_sha,**override):
 opt=dict(approved_identity=approved,actual_identity=actual,
          approved_file=reference_file,actual_file=actual_file,
          expected_approved_sha256=reference_sha,expected_actual_sha256=actual_sha,
          reference_manifest_attested=True,rendered_state_attested=True,
          immutable_snapshot_contract=True)
 opt.update(override)
 return P.decide_bound_immutable_snapshot(**opt).value


def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 if any((OUT/n).exists() for n in ('frozen_protocol.json','source_binding_replay.json')):raise RuntimeError('Already has prior receipts')
 paths={
  'original_340_cases':OLD/'source_test.json',
  'first_100_selection':OLD/'selection.json',
  'first_100_frozen_predictions':OLD/'predictions.json',
  'first_100_author_gold':OLD/'scored.json',
  'remaining_240_selection':NEW/'selection.json',
  'remaining_240_frozen_predictions':NEW/'predictions.json',
  'remaining_240_author_gold':NEW/'evaluated.json',
  '352_author_clean_wrong_activity_manifest':CLEAN/'frozen_pairs.json',
  '352_author_clean_wrong_activity_results':CLEAN/'scored.json',
 }
 assert digest_file(paths['original_340_cases'])==PIN
 assert len(read(paths['first_100_selection'])['selection'])==100
 assert len(read(paths['remaining_240_selection'])['unused_samples'])==240
 assert len(read(paths['352_author_clean_wrong_activity_manifest'])['comparisons'])==352
 assert len(read(paths['352_author_clean_wrong_activity_results'])['cases'])==352
 frozen={k:digest_file(p) for k,p in paths.items()}
 spec={'date':'2026-10-26','source':'RotBench author-pinned original real Android screen captures and source authored mutations, NOT a new human-certified GUI corpus',
       'root_cause':'The existing low-level trusted Observation accepts caller verified flags; it cannot check actual source/Activity/viewport bindings. Earlier wrong-Activity comparator wrongly flagged 352/352 author-clean screen pairs. Add stricter source binding and immutable byte contract helper as a separate opt-in boundary.',
       'inputs_from_preregistered_prior_work':frozen,'source_revision_sha256':PIN,
       'test_groups':['all 352 real author-clean same-app same-size different-activity source comparisons, two orientations each','same 352 real right screen originals against themselves','all previously frozen 100 RotBench author-mutated/normal test cases as exact snapshot property','two previously unavailable source reference cases, without network retry','source version, attestation, output hash and immutable-contract negative controls'],
       'expected_verdict_wrong_activity':'REVIEW without attempting to imply source-authorized exact pixel defect',
       'expected_verdict_author_mutation':'FAIL only for exact immutable snapshot byte equivalence and correct activity+reference; NOT a human-visible clipping or occlusion outcome',
       'expected_verdict_self':'PASS only exact byte equivalence; NO general GUI release approval',
       'statistical_scope':'Wrong-Activity pairs from 28 correlated Android apps, original mutation cases source-author generated; no independent original human semantic labels',
       'no_gpu_or_external_requests':True,'no_old_denied_media_transfer_retry':True,
       'safety':'Source metadata attestation supplied by prior pinned author release and verified local prior research manifests, not a live authenticated Android Activity capture. This helper must not be advertised as production verified identity.'}
 save(OUT/'frozen_protocol.json',spec)
 print('SOURCE_BINDING_REPAIR_FROZEN',digest_file(OUT/'frozen_protocol.json'),'source receipts',len(frozen))


def evaluate():
 if (OUT/'source_binding_replay.json').exists():raise RuntimeError('Prior replay outcome preserved, not rerunning')
 protocol=read(OUT/'frozen_protocol.json')
 source_paths={'original_340_cases':OLD/'source_test.json','first_100_selection':OLD/'selection.json','first_100_frozen_predictions':OLD/'predictions.json','first_100_author_gold':OLD/'scored.json','remaining_240_selection':NEW/'selection.json','remaining_240_frozen_predictions':NEW/'predictions.json','remaining_240_author_gold':NEW/'evaluated.json','352_author_clean_wrong_activity_manifest':CLEAN/'frozen_pairs.json','352_author_clean_wrong_activity_results':CLEAN/'scored.json'}
 for label,expected in protocol['inputs_from_preregistered_prior_work'].items():
  if digest_file(source_paths[label])!=expected:raise RuntimeError('Frozen earlier source changed '+label)
 P=policy();start=time.monotonic()
 same_manifest=read(CLEAN/'frozen_pairs.json');same_scored=read(CLEAN/'scored.json')
 assert len(same_manifest['comparisons'])==len(same_scored['cases'])==352
 assert same_manifest['pinned_source_sha']==PIN
 assert all(x['localized_bad_flag'] for x in same_scored['cases'])
 checksums={}
 wrong=[];selfchecks=[];invalid=[]
 for q in same_manifest['comparisons']:
  for i,orientation in enumerate(('portrait','landscape')):
   actualpath=original_png(q['screenshots'][i]);otherpath=original_png(q['other_clean_screenshots'][i])
   own_sha=q['actual_sha'][i];other_sha=q['other_ref_sha'][i]
   for p,expected in ((actualpath,own_sha),(otherpath,other_sha)):
    if p not in checksums:
     checksums[p]=digest_file(p) if p.is_file() else None
    if checksums[p]!=expected:raise RuntimeError('Original PNG mismatch '+str(p))
   dims=q['native_sizes'][i]
   own=make_binding(P,q['app'],q['activity'],orientation,dims)
   wrong_binding=make_binding(P,q['app'],q['other_activity'],orientation,dims)
   verdict=compare(P,wrong_binding,own,otherpath,actualpath,other_sha,own_sha)
   checkself=compare(P,own,own,actualpath,actualpath,own_sha,own_sha)
   wrong.append({'case_id':q['case_id'],'orientation':orientation,'actual_sha':own_sha,'wrong_ref_sha':other_sha,'naive_byte_mismatch':own_sha!=other_sha,'wrong_identity_outcome':verdict,'same_author_source_against_itself':checkself})
 assert len(wrong)==704
 assert all(x['wrong_identity_outcome']=='review' and x['same_author_source_against_itself']=='pass' for x in wrong)
 assert len({x['case_id'] for x in wrong})==137  # 352 correlated comparisons, not 352 independent cases
 matched_wrong_count=len({q['case_id'] for q in same_manifest['comparisons'] if q['case_id']})
 assert matched_wrong_count==137
 # Test the previous author-injected change / right originals with independent
 # frozen image SHA receipts, without recomputing a visual classifier.
 initial=read(OLD/'selection.json')['selection']
 predictions=read(OLD/'predictions.json');pred={x['record_index']:x for x in predictions['cases']}
 gold={x['record_index']:x for x in read(OLD/'scored.json')['selected_cases']}
 synthetic=[]
 for q in initial:
  sid=q['record_index'];entry=pred[sid];g=gold[sid]
  assert len(q['images'])==len(q['references'])==2
  trials=[]
  for i,orientation in enumerate(('portrait','landscape')):
   actual_file=original_png(q['images'][i]);ref_file=original_png(q['references'][i]);a_sha=entry['images_sha256'][i];r_sha=entry['reference_sha256'][i]
   for path,expected in ((actual_file,a_sha),(ref_file,r_sha)):
    if path not in checksums:checksums[path]=digest_file(path) if path.is_file() else None
    if checksums[path]!=expected:raise RuntimeError('Previously frozen first100 PNG has changed')
   # The prior manifest identifies each mutation as a variant of its original
   # activity and viewport. Exact state is from the recorded ORIGINAL source,
   # not an image model's description.
   path=q['references'][i]
   if not path.startswith('image/right/') or not path.endswith('_'+orientation+'.png'):raise ValueError('unexpected source ref format')
   parts=path.split('/');app=parts[2];state=parts[3][:-(len(orientation)+5)]
   # Documented width/height from previous actual native image decode.
   W,H=entry['orientation_pixel_stats'][i]['size_hw'][::-1]
   ref=make_binding(P,app,state,orientation,(W,H))
   decision=compare(P,ref,ref,ref_file,actual_file,r_sha,a_sha)
   trials.append({'orientation':orientation,'different_approved_bytes':a_sha!=r_sha,'immutable_bytes_verdict':decision})
  agg='review' if any(x['immutable_bytes_verdict']=='review' for x in trials) else ('fail' if any(x['immutable_bytes_verdict']=='fail' for x in trials) else 'pass')
  synthetic.append({'source_test_id':sid,'source_author_injected_change':g['gold_bug'],'earlier_visual_detector_flag':g['predictions']['changed_area_0p05pct'],'byte_snapshot_group_outcome':agg,'orientation_outcomes':trials})
 assert len(synthetic)==100
 # Strict checker must not promote unverified metadata or files to FAIL/PASS.
 q=same_manifest['comparisons'][0]
 p=original_png(q['screenshots'][0]);digest=q['actual_sha'][0];app=q['app'];state=q['activity'];dims=q['native_sizes'][0]
 ref=make_binding(P,app,state,'portrait',dims)
 valid=compare(P,ref,ref,p,p,digest,digest)
 changed_version=replace(ref,source_revision='0000000000000000000000000000000000000000')
 negatives={
  'true_self_snapshot':valid,
  'wrong_data_revision':compare(P,ref,changed_version,p,p,digest,digest),
  'bad_expected_artifact_sha':compare(P,ref,ref,p,p,'0'*64,digest),
  'bad_actual_artifact_sha':compare(P,ref,ref,p,p,digest,'0'*64),
  'manifest_not_attested':compare(P,ref,ref,p,p,digest,digest,reference_manifest_attested=False),
  'render_state_not_attested':compare(P,ref,ref,p,p,digest,digest,rendered_state_attested=False),
  'not_an_immutable_snapshot_contract':compare(P,ref,ref,p,p,digest,digest,immutable_snapshot_contract=False),
  'unknown_reference_file':compare(P,ref,ref,p.with_name('author_source_missing_do_not_fetch.png'),p,digest,digest),
 }
 assert negatives['true_self_snapshot']=='pass' and all(v=='review' for k,v in negatives.items() if k!='true_self_snapshot')
 # The two missing references from last round were HTTP 404 at the pinned
 # release; do NOT retry requests or re-label them as safe/false.
 unselected=read(NEW/'selection.json')['unused_samples'];missing={z['id'] for z in read(NEW/'evaluated.json')['table'] if z['predictions']['aligned']['localized'] is None}
 assert missing=={122,275}
 missing_results=[]
 newpred={x['id']:x for x in read(NEW/'predictions.json')['cases']}
 for q in unselected:
  if q['id'] not in missing:continue
  i=0;orientation='portrait';path=q['same_orientation_source_refs'][i]
  parts=path.split('/');app=parts[2];state=parts[3][:-(len(orientation)+5)]
  actual=original_png(q['native_screenshots'][i]);refpath=original_png(path)
  actualsha=newpred[q['id']]['imgs_hash'][i];refsha=newpred[q['id']]['refs_hash'][i]
  assert refsha is None and not refpath.exists()
  w,h=(720,1600)
  ident=make_binding(P,app,state,orientation,(w,h))
  decision=compare(P,ident,ident,refpath,actual,refsha,actualsha)
  missing_results.append({'id':q['id'],'source_ref_not_available':True,'outcome':decision})
 assert len(missing_results)==2 and all(x['outcome']=='review' for x in missing_results)
 elapsed=round(time.monotonic()-start,3)
 wrong_counts=dict(Counter(x['wrong_identity_outcome'] for x in wrong));same_counts=dict(Counter(x['same_author_source_against_itself'] for x in wrong));snap_counts=dict(Counter(x['byte_snapshot_group_outcome'] for x in synthetic))
 prior_clipped=sum(x['source_author_injected_change'] and x['earlier_visual_detector_flag'] for x in synthetic)
 auth_mutants=sum(x['source_author_injected_change'] for x in synthetic)
 mutated_byte_fails=sum(x['source_author_injected_change'] and x['byte_snapshot_group_outcome']=='fail' for x in synthetic)
 normal_snapshot_passes=sum((not x['source_author_injected_change']) and x['byte_snapshot_group_outcome']=='pass' for x in synthetic)
 report={'date':'2026-10-26','status':'PARTIAL','source_pinned_release_SHA256':PIN,'frozen_protocol_SHA256':digest_file(OUT/'frozen_protocol.json'),'policy_code_SHA256':digest_file(POLICY),'prior_wrong_ref_source_manifest_SHA256':digest_file(CLEAN/'frozen_pairs.json'),'source_file_hashes_recomputed':len(checksums),'source_file_hashes_all_match_original_receipts':True,'measured_CPU_wall_s':elapsed,'peak_RSS_KB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_used':False,'real_author_clean_wrong_activity_pairs':352,'real_author_clean_wrong_activity_orientation_trials':len(wrong),'old_naive_automated_alerts_at_pair_level':352,'actual_correct_source_binding_decisions_on_mismatched_orientation_trials':wrong_counts,'same_right_original_exact_byte_snapshots':same_counts,'source_generated_paired_real_screen_shots_100':{'cases':len(synthetic),'original_author_injected_positive':auth_mutants,'correct_source_exact_immutable_snapshot_fail_on_author_edits':mutated_byte_fails,'correct_source_exact_immutable_snapshot_pass_on_original_right':normal_snapshot_passes,'previous_localized_pixel_CV_detected_author_mutations':prior_clipped,'note':'Exact-byte contract does not prove any semantic UI clipping defect. It flags ANY authorized change, so use ONLY for truly immutable approved image contract.'},'bad_metadata_and_missing_file_fail_closed':negatives,'previously_unavailable_original_right_refs_without_retry':missing_results,'comparison_per_case':wrong,'source_mutation_per_case':synthetic,'limitations':['Author-rendered Android screenshots with author-injected mutations; not independent human rated real app UI bugs','352 wrong-state pairs are 704 orientation trials but only 137 distinct original clean right screen sources from 28 correlated apps','Source version/activity were attributed by an original author release, not authenticated from a LIVE operating Android Activity, so this remains an offline demonstration','Caller source_manifest_attested and rendered_state_attested must themselves come from a trusted system; untrusted client flags do not establish truth','Exact SHA equality is a useful only for explicit immutable byte-for-byte snapshot contract; varying anti-aliasing, output format or legitimate changes must instead REVIEW','This path does not improve real defect appearance recognition, audio synchronization, diagram arrows or 3D semantic correctness','Nitro CPU load and GPU service contention are reported separately; no heavy GPU models were loaded','Old explicitly denied storyboard data transfers and unavailable HTTP404 source-references not retried','SWAAG acceptance integration into live worker interface is NOT proven']}
 save(OUT/'source_binding_replay.json',report)
 print('SOURCE_BINDING_NEW_FAULT_PROOF',json.dumps({'wrong_source_reviews':wrong_counts,'self_snapshot_passes':same_counts,'actual_author_changes':snap_counts,'missing_refs':missing_results,'controls':negatives,'time_s':elapsed,'source_image_unique_SHA_verifications':len(checksums)},sort_keys=True)[:3600],'SHA',digest_file(OUT/'source_binding_replay.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','evaluate'));globals()[p.parse_args().stage]()
