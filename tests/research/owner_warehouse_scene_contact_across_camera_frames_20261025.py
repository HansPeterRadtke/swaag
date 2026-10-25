"""Independent static 3D contact-chain persistence over real 420-frame film.

Source Blender v47 film; sample frames 1,91,210,330,420 (door crossing and
final intentional camera hold), compute 1e-5m quantized world triangle vertex
SHA for all 24 pallet, 72 steel support and 12 rack meshes. Test that physical
contact proof at initial scene pose remains applicable to sampled film frames.
No GPU, render, product edit, or user-video source-framehash workaround.
Between-frame proof limited without checking every frame; log animation_data.
"""
import bpy,hashlib,json,struct,time,collections
from pathlib import Path
SRC=Path('/data/var/warehouse-film-v47/warehouse_THROUGH_OPENING_DOOR_to_REAR_CORNER_three_second_HOLD_v47.blend')
OUT=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics/v47_support_mesh_5_camera_frame_persistence_20261025.json')
FRAMES=[1,91,210,330,420]

def world_digest(obj):
 d=hashlib.sha256()
 for vertex in obj.data.vertices:
  pos=obj.matrix_world@vertex.co
  for v in pos:d.update(struct.pack('<q',round(float(v)*100000)))
 return d.hexdigest()

def main():
 if OUT.exists():print('PREVIOUS_MOTION_PERSISTENCE_RESULT_PRESERVED');return
 assert Path(bpy.data.filepath).resolve()==SRC.resolve()
 scene=bpy.context.scene;assert scene.frame_end>=420 and scene.render.fps==30
 pallets=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('instance_of_verified_buyer_prefab')=='pallet_euro_1200x800']
 rackroot=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400']
 palletmesh=[a for r in pallets for a in r.children_recursive if a.type=='MESH' and not a.hide_render]
 rackmesh=[a for r in rackroot for a in r.children_recursive if a.type=='MESH' and not a.hide_render]
 steel=[o for o in bpy.data.objects if o.type=='MESH' and o.name.startswith('RENDER_RACK_STEEL_PALLET_SUPPORT_')]
 assert (len(palletmesh),len(rackmesh),len(steel))==(24,12,72)
 objects=palletmesh+rackmesh+steel
 t=time.monotonic();rows={};original_frame=scene.frame_current
 try:
  for frame in FRAMES:
   scene.frame_set(frame);bpy.context.view_layer.update()
   rows[str(frame)]={hashlib.sha256(o.name.encode()).hexdigest():world_digest(o) for o in objects}
   print('PHYSICAL_SCENE_WORLD_MESH_FRAME',frame,'OBJECTS',len(rows[str(frame)]),'SOURCE_ONLY',True,flush=True)
 finally:scene.frame_set(original_frame)
 baseline=rows[str(FRAMES[0])]
 changed={str(f):[k for k in baseline if baseline[k]!=rows[str(f)][k]] for f in FRAMES[1:]}
 allstable=all(not v for v in changed.values())
 with_actions={hashlib.sha256(o.name.encode()).hexdigest():bool(o.animation_data or o.data.shape_keys and o.data.shape_keys.animation_data) for o in objects}
 out={'date':'2026-10-25','source':'same original warehouse video v47 420 scene frames, CPU only','sampled_source_frame_indices':FRAMES,'n_source_pallet_meshes':len(palletmesh),'n_source_steel_bars':len(steel),'n_source_rack_meshes':len(rackmesh),'blender_version':bpy.app.version_string,'mesh_world_geometry_stable_across_sampled_frames':allstable,'mesh_instances_changed_at_each_sampled_frame':{k:len(v) for k,v in changed.items()},'n_source_objects_with_animation_data':sum(with_actions.values()),'source_world_geometry_hashes_by_frame':rows,'seconds':round(time.monotonic()-t,2),'limitation':'Five selected checkpoints and no mesh animation data provide strong evidence of static contact positions in this saved scene. Not a rendered-frame visual acceptance and not proof of all 420 intermediate frame transforms in any other scene version. No video decoding, frame hashes, GPU or product changes.'}
 OUT.parent.mkdir(exist_ok=True,parents=True);OUT.write_text(json.dumps(out,indent=2)+'\n')
 print('SOURCE_PHYSICAL_CONTACT_CHAIN_STATIC_ACROSS_SAMPLE_FRAMES',allstable,'CHANGED_COUNTS',out['mesh_instances_changed_at_each_sampled_frame'],'ACTION_DATA_N',out['n_source_objects_with_animation_data'],'WALL_S',out['seconds'],flush=True)
if __name__=='__main__':main()
