#!/usr/bin/env python3
"""Publish verified source-backed warehouse mechanical contact gate research.

Read only preexisting frozen CPU Blender ray/vertex results and unchanged source
receipts. Never export/ship copyrighted model assets, images, videos or raw 3D
mesh values to Git. Preserve negative v44 intermediate result: vertical only
appeared good (72 pallet-wood contacts) but actual rack both-end contact fails
(0/72), whereas corrected v47 72/72 both ends. Site-specific gate, no generic
physical capacity proof, no independent IID human error rate.
"""
from pathlib import Path
import json,hashlib,statistics,datetime
RAW=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics')
DOCS=Path(__file__).resolve().parents[2]/'docs/research'
OUT=DOCS/'perceptual-owner-warehouse-physical-contact-chain-2026-10-25.json'
SOURCEQA=Path('/data/var/warehouse-film-v47/independent_reloaded_camera_geometry_v47_qa.json')
V45=Path('/data/var/warehouse-film-v45/fix_actual_rack_orientation_and_contacts_v45.py')

def get(name):
 p=RAW/name;b=p.read_bytes();return json.loads(b),hashlib.sha256(b).hexdigest()

def main():
 if OUT.exists():print('PREVIOUS_VERIFIED_CONTACT_AGGREGATE_PRESERVED_NOT_REGENERATED');return
 repair_wood,hwood=get('real_v47_pallet_to_new_support_mesh_contact_20261025.json')
 repair_rack,hrack=get('real_v47_support_to_rack_mesh_contact_20261025.json')
 before,hbefore=get('real_pre_repair_warehouse_original_pallet_to_rack_gap_20261025.json')
 middle,hmiddle=get('real_v44_intermediate_contact_chain_20261025.json')
 fps,hfps=get('v47_support_mesh_5_camera_frame_persistence_20261025.json')
 previous,hpre=get('before_repair_v30_source_mesh_identity_20261025.json')
 intermediate,hmid=get('intermediate_v44_source_mesh_identity_20261025.json')
 corrected,hcor=get('after_repair_v47_source_mesh_identity_20261025.json')
 original_camera=json.loads(SOURCEQA.read_text())
 assert original_camera['source_sha256']==repair_wood['actual_source_scene_sha256']
 assert repair_wood['source_counts']=={'buyer_pallet_roots':24,'bearing_support_meshes':72,'original_rack_roots':12}
 assert repair_rack['summary']['n_bars']==72
 assert repair_rack['summary']['left_end_geometric_contacts']==repair_rack['summary']['right_end_geometric_contacts']==72
 assert repair_rack['summary']['both_ends_geometrically_contact_actual_rack_mesh']==72
 assert repair_wood['source_independent_summary']['support_bars_with_contact_within_2mm']==72
 assert all(x['actual_mesh_bottom_hit_above_support_top_m'] is not None and abs(x['actual_mesh_bottom_hit_above_support_top_m'])<=.002 for x in repair_wood['support_geometry_cases'])
 assert middle['summary']['wood_to_steel_contact_within2mm_bars']==72
 assert middle['summary']['support_to_rack_left_mesh_intersections']==0
 assert middle['summary']['support_to_rack_right_mesh_intersections']==0
 assert middle['summary']['complete_pallet_bearing_rack_chain_on_both_ends']==0
 assert middle['summary']['rack_probe_count']==repair_rack['summary']['rays_cast']==432
 assert before['original_pallets']==24 and before['extra_correction_supports']==0
 assert before['summary']['probe_ray_hits']==0 and before['summary']['probe_ray_attempts']==144
 for s in (previous,intermediate,corrected):
  assert len(s['actual_3d_mesh_fingerprint_instances'])==36
 def entries(s):return {v['root_sha256']:v for v in s['actual_3d_mesh_fingerprint_instances']}
 A=entries(previous);B=entries(intermediate);C=entries(corrected)
 assert set(A)==set(B)==set(C)
 matched={}
 for typ in ('pallet','rack'):
  ids=[k for k in A if A[k]['kind']==typ]
  matched[typ]={'count':len(ids),'pre_equals_intermediate':sum(A[k]['actual_mesh_world_vertex_digest_1e5m']==B[k]['actual_mesh_world_vertex_digest_1e5m'] for k in ids),'intermediate_equals_corrected':sum(B[k]['actual_mesh_world_vertex_digest_1e5m']==C[k]['actual_mesh_world_vertex_digest_1e5m'] for k in ids)}
 assert matched['pallet']=={'count':24,'pre_equals_intermediate':24,'intermediate_equals_corrected':24}
 assert matched['rack']=={'count':12,'pre_equals_intermediate':12,'intermediate_equals_corrected':0}
 support44={r['source_support_id_sha256']:r['source_support_world_AABB_m'] for r in middle['source_cases']}
 support47={r['support_source_name_sha256']:r['support_bounds_world_metres'] for r in repair_wood['support_geometry_cases']}
 assert set(support44)==set(support47) and len(support44)==72
 same_supports=sum(all(abs(support44[k][i][j]-support47[k][i][j])<1e-5 for i in (0,1) for j in (0,1,2)) for k in support44)
 assert same_supports==72
 assert fps['mesh_world_geometry_stable_across_sampled_frames'] and fps['n_source_objects_with_animation_data']==0
 assert fps['sampled_source_frame_indices']==[1,91,210,330,420]
 assert fps['n_source_pallet_meshes']==24 and fps['n_source_steel_bars']==72 and fps['n_source_rack_meshes']==12
 assert all(n==0 for n in fps['mesh_instances_changed_at_each_sampled_frame'].values())
 script=V45.read_text()
 assert 'math.pi/2' in script and 'ob.matrix_world=M@old' in script
 d={
  'date':'2026-10-25','status':'VERIFIED narrow source-engineering contact chain, NOT physical load capacity or vision photorealism nor industrial benchmark accuracy',
  'project':'SWAAG perceptual analyzer research, owner warehouse production scene v30 (pre), v44 (intermediate), v47 (corrected)',
  'source_assets_STAY_OUTSIDE_GIT':True,
  'source_blender_version':repair_wood['resource']['blender_version'],
  'unchanged_production_scene_SHA256':repair_wood['actual_source_scene_sha256'],
  'source_provenance':{'v47_matches_preexisting_independent_camera_source_receipt':True,'source_pre_repair_wood_and_rack_identity':hpre,'source_intermediate_wood_and_rack_identity':hmid,'source_corrected_wood_and_rack_identity':hcor,'physical_rack_reorientation_source_script_sha256':hashlib.sha256(V45.read_bytes()).hexdigest(),'rack_repair_confirmed_from_source':'90 degree rotation about local center in existing real v45 build script, not inferred from model prose'},
  'raw_unmodified_results_SHA256':{'original_before_repair_rack_local_probe':hbefore,'intermediate_v44_both_contact_stages':hmiddle,'corrected_v47_wood_to_bearing':hwood,'corrected_v47_bearing_to_rack_both_ends':hrack,'corrected_v47_five_frame_mesh_stability':hfps},
  'actual_source_scene_counts':{'pallets':24,'steel_bearing_bars':72,'rack_bays':12},
  'individual_frame_sample_indices':fps['sampled_source_frame_indices'],
  'original_v30_pre_repair':{'real_steel_bearing_bars':0,'rack_end_location_wood_to_rack_ray_hits':0,'predefined_endpoint_source_rack_rays':144,'limitation':'No hit at selected endpoint ray XY positions does not establish missing every possible rack contact; source racks misoriented and this before/after is not a single-variable pair.'},
  'real_intermediate_v44':{'wood_to_steel_contact_bars_within_2mm':72,'source_steel_bars_with_rack_contact_at_LEFT_end':0,'source_steel_bars_with_rack_contact_at_RIGHT_end':0,'source_steel_bars_with_full_wood_bearing_rack_contact_chain':0,'failed_component':'Horizontal original rack axes misoriented. A simple wood/support vertical gap check falsely implies the steel bars are themselves supported.'},
  'real_corrected_v47':{'wood_to_steel_contact_bars_within_2mm':72,'wood_to_steel_sampled_min_vertical_gap_m':repair_wood['source_independent_summary']['min_support_wood_bottom_gap_m'],'wood_to_steel_sampled_max_vertical_gap_m':repair_wood['source_independent_summary']['max_support_wood_bottom_gap_m'],'original_rack_triangle_geometric_hits_at_both_ends':72,'original_rack_left_end_hits':72,'original_rack_right_end_hits':72,'sampled_rack_endpoint_rays':432,'dynamic_sampled_source_meshes_unchanged_per_frame':108,'sampled_scene_frame_count':5,'no_animation_data_on_these_108_mesh_objects':True},
  'version_controlled_geometric_identity':{'pallet_world_triangle_meshes_identical_pre_v30_through_v47':24,'original_racks_same_v30_v44':12,'original_racks_CHANGED_world_geometry_v44_to_v47':12,'steel_bearing_bars_same_v44_v47_world_AABB':same_supports,'matches':matched,'causal_limit':'The intermediate-to-corrected comparison holds all pallet world-space triangle meshes and steel bar world-space bounding geometry fixed, while racks are explicitly reoriented. Therefore this specific spatial rack orientation repair explains why the tested full bearing contact chain changes. It does not prove real engineered load strength.'},
  'what_this_proves':'Source-aware geometry ray casting along the full declared load path can detect a real incomplete manufactured assembly-like visual product state, while earlier vertical-only measurement returned a misleading apparent pass; original author contact labels were not used to calculate raw geometry.',
  'what_is_NOT_proved':['It is one internally correlated warehouse with one before/after correction, not 72 independent industrial defect+clean experiments. No independent TP/FP estimate for other machinery.','Ray hits on selected contact points prove triangle geometry proximity/intersection, not steel weld integrity, friction, allowable load, true structural simulation, mechanical engineering certification or intended whole-scene appearance.','No actual rendered image/video frame was judged for plausibility or photographic realism.','Five sample animation frames with no object animation show stable observed contact geometry, but not an independent test of every aspect of all 420 rendered frames.','CAD/load-bearing design oracle could itself be wrong; stage source-level assumptions and scope must be recorded.'],
  'narrow_tool_recommendation':'For source-available industrial 3D work: source-identity checksum and object role tags → actual world triangle contact between required components → vertical AND horizontal support path with intended endpoints and no unsupported components → joints/collision/animation static-or-dynamic checks → separate actual rendered image human/visual-model review and independent mechanical design comparison. No universal acceptance from generic VLM.',
  'no_render_no_shared_GPU_no_model_called':True,
 }
 OUT.write_text(json.dumps(d,indent=2)+'\n')
 print('VERIFIED_REAL_OWNER_PHYSICAL_SOURCE_CONTACT_REPORT_SAVED',OUT,'BYTES',OUT.stat().st_size,'V44_FALSE_PASS_V47_FULL_CHAIN',(d['real_intermediate_v44']['source_steel_bars_with_full_wood_bearing_rack_contact_chain'],d['real_corrected_v47']['original_rack_triangle_geometric_hits_at_both_ends']),'CONTROLLED_SOURCE_IDENTITIES',matched)
if __name__=='__main__':main()
