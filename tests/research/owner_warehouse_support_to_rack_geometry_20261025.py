"""Independent CPU-only ray contact test of actual steel bearing bars to original rack.

Reload owner v47 warehouse source.blend without autoexec or rendering. For each
of 72 pallet support bars, probe x offsets at both left and right beam ends
for real triangle intersections with the linked rack mesh. This tests geometry
contact at both ends, not physics load capacity, welding, camera visibility or
human-perceived photorealism. No production source file writes.
"""
import bpy,json,hashlib,datetime,math,time
from pathlib import Path
from mathutils import Vector
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics')
OUT=ROOT/'real_v47_support_to_rack_mesh_contact_20261025.json'
SRC=Path('/data/var/warehouse-film-v47/warehouse_THROUGH_OPENING_DOOR_to_REAR_CORNER_three_second_HOLD_v47.blend')
OFF=(-.565,-.54,-.50,.50,.54,.565)

def box_world(obj):
 ps=[obj.matrix_world@p.co for p in obj.data.vertices]
 return [[min(q[i] for q in ps) for i in range(3)],[max(q[i] for q in ps) for i in range(3)]]

def ray(obj,x,y,z):
 inv=obj.matrix_world.inverted_safe()
 origin=inv@Vector((x,y,z));direction=inv.to_3x3()@Vector((0,0,-1));direction.normalize()
 hit,loc,normal,face=obj.ray_cast(origin,direction,distance=.4)
 if not hit:return None
 return [round(float(v),6) for v in obj.matrix_world@loc]

def main():
 if OUT.exists():print('PRIOR_RACK_RAY_TEST_PRESERVED_NO_REPLAY');return
 assert Path(bpy.data.filepath).resolve()==SRC.resolve()
 start=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True)
 racks={o.name:o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400'}
 bars=[o for o in bpy.data.objects if o.type=='MESH' and o.name.startswith('RENDER_RACK_STEEL_PALLET_SUPPORT_')]
 assert len(racks)==12 and len(bars)==72
 cases=[]
 for bar in sorted(bars,key=lambda obj:obj.name):
  rack=racks[bar['support_rack_name']]
  meshes=[o for o in rack.children_recursive if o.type=='MESH' and not o.hide_render]
  assert len(meshes)==1
  obj=meshes[0]
  low,hi=box_world(bar);center=bar.matrix_world.translation
  assert abs((hi[0]-low[0])-1.16)<.002 and abs((hi[1]-low[1])-.06)<.002
  observations=[]
  for side in [-1,1]:
   probes=[]
   for offset in OFF:
    if math.copysign(1,offset)!=side:continue
    pt=ray(obj,float(center.x+offset),float(center.y),float(hi[2]+.05))
    contact=bool(pt is not None and low[2]-.002<=pt[2]<=hi[2]+.002)
    probes.append({'X_offset_m':offset,'rack_triangle_ray_hit_xyz_m':pt,'inside_support_steel_z_slab':contact})
   observations.append({'side':side,'rack_triangle_contact_any':any(p['inside_support_steel_z_slab'] for p in probes),'fixed_probes':probes})
  cases.append({'bearing_bar_id_SHA256':hashlib.sha256(bar.name.encode()).hexdigest(),'rack_id_SHA256':hashlib.sha256(rack.name.encode()).hexdigest(),'support_z_slab_m':[round(low[2],6),round(hi[2],6)],'source_rack_contact_at_both_ends':all(x['rack_triangle_contact_any'] for x in observations),'end_probes':observations})
 result={'date':'2026-10-25','method':'actual original warehouse rack mesh downward ray casting from both ends of 72 pallet support bars; source CPU only','source_file_size_bytes':SRC.stat().st_size,'source_scene_blender_version':bpy.app.version_string,'probe_end_X_offsets_metres':OFF,'n_source_racks':len(racks),'n_source_support_bars':len(bars),'results':cases,'summary':{'both_ends_geometrically_contact_actual_rack_mesh':sum(c['source_rack_contact_at_both_ends'] for c in cases),'left_end_geometric_contacts':sum(c['end_probes'][0]['rack_triangle_contact_any'] for c in cases),'right_end_geometric_contacts':sum(c['end_probes'][1]['rack_triangle_contact_any'] for c in cases),'n_bars':len(cases),'rays_cast':len(cases)*len(OFF)},'wall_seconds':round(time.monotonic()-start,2),'scope_limits':'Mesh ray intersection in support Z slab demonstrates geometry on tested lines only. It is not proof of beam strength, friction, welding or visually plausible appearance. Negative ray may reflect sampling, not actual non-contact. Original production video/render preserved.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('ACTUAL_OWNER_RACK_AND_STEEL_TRIANGLE_CONTACT',result['summary'],'RUN_S',result['wall_seconds'],'SOURCE_UNCHANGED',True,flush=True)
if __name__=='__main__':main()
