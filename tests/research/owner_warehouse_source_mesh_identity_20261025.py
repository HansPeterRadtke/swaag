"""Independently compare actual original pallet/rack meshes before/after repair.

Run on each original Blender scene CPU-only via --disable-autoexec, no rendering.
Hash real world-space vertex coordinates (precision 1e-5m) for 24 pallet and 12
rack meshes, to test whether support geometry was truly the only physical
change, before combining other ray-cast outcomes. Does not modify scenes or
trust third-party purported identical-object labels as itself sufficient.
"""
import bpy,hashlib,struct,json,time
from pathlib import Path
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics')
NAME=Path(bpy.data.filepath).name
MODE='before_repair_v30' if NAME=='warehouse_photo_pbr_lodclean_v30.blend' else 'after_repair_v47' if NAME=='warehouse_THROUGH_OPENING_DOOR_to_REAR_CORNER_three_second_HOLD_v47.blend' else 'intermediate_v44' if NAME=='warehouse_pallet_contacts_animated_camera_v44.blend' else None
assert MODE
OUT=ROOT/(MODE+'_source_mesh_identity_20261025.json')

def digest(o):
 d=hashlib.sha256()
 pts=[o.matrix_world@v.co for v in o.data.vertices]
 for p in pts:
  for v in p:d.update(struct.pack('<q',round(float(v)*100000)))
 return d.hexdigest(),len(pts),[[round(float(min(q[i] for q in pts)),5) for i in range(3)],[round(float(max(q[i] for q in pts)),5) for i in range(3)]]

def main():
 if OUT.exists():print('MESH_IDENTITY_PREVIOUSLY_CHECKED_SOURCE_PRESERVED');return
 t=time.monotonic();rows=[]
 pallets=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('instance_of_verified_buyer_prefab')=='pallet_euro_1200x800']
 racks=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400']
 assert len(pallets)==24 and len(racks)==12
 for kind,roots in [('pallet',pallets),('rack',racks)]:
  for root in roots:
   obj=[o for o in root.children_recursive if o.type=='MESH' and not o.hide_render]
   assert len(obj)==1
   sha,n,bounds=digest(obj[0])
   rows.append({'root_sha256':hashlib.sha256(root.name.encode()).hexdigest(),'kind':kind,'actual_mesh_world_vertex_digest_1e5m':sha,'vertex_count':n,'world_axis_aligned_bounds_m':bounds})
 OUT.write_text(json.dumps({'mode':MODE,'source_file_size':Path(bpy.data.filepath).stat().st_size,'actual_3d_mesh_fingerprint_instances':rows,'quantization_grid_metres':.00001,'runtime_s':round(time.monotonic()-t,2),'scope':'Actual original scene mesh, not rendered pixels; identity hash is geometry comparison not physical performance testing.'},indent=2)+'\n')
 print('BLENDER_REAL_SOURCE_GEOMETRY_IDENTITY',MODE,'PALLETS',len(pallets),'RACKS',len(racks),'S',round(time.monotonic()-t,2),flush=True)
if __name__=='__main__':main()
