"""Detect missing structural contact overlooked by a vertical-only repair gate.

The actual owner warehouse intermediate v44 stage includes 24 real pallet meshes
and 72 extra steel bearing meshes, but actual pallet racks have not yet been
reoriented (documented next-stage v45 real fix). Measure on exact Blender mesh
triangles: pallet wood-to-bearing near-contact, and each bearing's both-end
rack triangle intersection on 6 declared X-offset probes. No GPU, no renders,
no product writes, no author QA flags as an oracle.

This is a genuine source-stage comparison to the v47 real corrected scene, not
an IID 72-sample independent human defect benchmark. Intended load capacity
and pixel appearance remain separate and unverified.
"""
import bpy,json,hashlib,math,time,statistics
from pathlib import Path
from mathutils import Vector
SRC=Path('/data/var/warehouse-film-v44/warehouse_pallet_contacts_animated_camera_v44.blend')
OUT=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics/real_v44_intermediate_contact_chain_20261025.json')
WOOD_OFF=[-.35,0,.35]
RACK_OFF=[-.565,-.54,-.50,.50,.54,.565]

def pts(o):return [o.matrix_world@v.co for v in o.data.vertices]
def bounds(o):
 v=pts(o)
 return [[min(z[i] for z in v) for i in range(3)],[max(z[i] for z in v) for i in range(3)]]
def ray(o,x,y,z,up):
 inv=o.matrix_world.inverted_safe();ori=inv@Vector((x,y,z))
 d=inv.to_3x3()@Vector((0,0,1 if up else -1));d.normalize()
 hit,pos,normal,face=o.ray_cast(ori,d,distance=.7 if up else .4)
 if not hit:return None
 return float((o.matrix_world@pos).z)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(4*1024*1024):h.update(chunk)
 return h.hexdigest()

def main():
 if OUT.exists():print('V44_INTERMEDIATE_SOURCE_EVIDENCE_PRESERVED_NO_REPLAY');return
 assert Path(bpy.data.filepath).resolve()==SRC.resolve()
 t=time.monotonic();pallet={o.name:o for o in bpy.data.objects if o.type=='EMPTY' and o.get('instance_of_verified_buyer_prefab')=='pallet_euro_1200x800'}
 rack={o.name:o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400'}
 bearing=[o for o in bpy.data.objects if o.type=='MESH' and o.name.startswith('RENDER_RACK_STEEL_PALLET_SUPPORT_')]
 assert (len(pallet),len(rack),len(bearing))==(24,12,72)
 rows=[]
 for support in sorted(bearing,key=lambda o:o.name):
  pname=support.get('supported_pallet_root');rname=support.get('support_rack_name')
  assert pname in pallet and rname in rack
  wood=[o for o in pallet[pname].children_recursive if o.type=='MESH' and not o.hide_render]
  rackmesh=[o for o in rack[rname].children_recursive if o.type=='MESH' and not o.hide_render]
  assert len(wood)==len(rackmesh)==1
  lo,hi=bounds(support);center=support.matrix_world.translation
  wood_hits=[]
  for xoff in WOOD_OFF:
   z=ray(wood[0],float(center.x+xoff),float(center.y),float(hi[2]-.08),True)
   wood_hits.append(round(z-hi[2],6) if z is not None else None)
  wood_near=any(gap is not None and abs(gap)<=.002 for gap in wood_hits)
  rack_end=[]
  for sign in (-1,1):
   test=[]
   for off in RACK_OFF:
    if math.copysign(1,off)!=sign:continue
    z=ray(rackmesh[0],float(center.x+off),float(center.y),float(hi[2]+.05),False)
    test.append({'Xoffset_m':off,'rack_face_z_m':round(z,6) if z is not None else None,'intersects_support_z_slab':bool(z is not None and lo[2]-.002<=z<=hi[2]+.002)})
   rack_end.append({'sign':sign,'contact':any(r['intersects_support_z_slab'] for r in test),'fixed_probes':test})
  rows.append({'source_support_id_sha256':hashlib.sha256(support.name.encode()).hexdigest(),'source_pallet_id_sha256':hashlib.sha256(pname.encode()).hexdigest(),'source_rack_id_sha256':hashlib.sha256(rname.encode()).hexdigest(),'source_support_world_AABB_m':[[round(x,6) for x in a] for a in (lo,hi)],'wood_support_upward_ray_gap_m':wood_hits,'wood_bar_contact_within_2mm':wood_near,'steel_support_attached_to_rack_left_end':rack_end[0]['contact'],'steel_support_attached_to_rack_right_end':rack_end[1]['contact'],'rack_ray_probe_details':rack_end})
 summary={'real_source_pallets':len(pallet),'real_source_racks':len(rack),'real_steel_bearing_bars':len(rows),'wood_to_steel_contact_within2mm_bars':sum(x['wood_bar_contact_within_2mm'] for x in rows),'support_to_rack_left_mesh_intersections':sum(x['steel_support_attached_to_rack_left_end'] for x in rows),'support_to_rack_right_mesh_intersections':sum(x['steel_support_attached_to_rack_right_end'] for x in rows),'complete_pallet_bearing_rack_chain_on_both_ends':sum(x['wood_bar_contact_within_2mm'] and x['steel_support_attached_to_rack_left_end'] and x['steel_support_attached_to_rack_right_end'] for x in rows),'rack_probe_count':len(rows)*len(RACK_OFF)}
 out={'date':'2026-10-25','source':'actual owner 3D warehouse v44 intermediate stage with supports but not yet reoriented original racks','source_file_sha256':sha(SRC),'blender_version':bpy.app.version_string,'fixed_wood_probe_offsets_X_m':WOOD_OFF,'fixed_rack_probe_offsets_X_m':RACK_OFF,'source_cases':rows,'summary':summary,'runtime_s':round(time.monotonic()-t,2),'limitations':'Original v44 stage not a broad independent human rated benchmark. One correlated 3D world, 72 within-scene bars. A missing sampled steel/rack ray does not prove absolutely no contact elsewhere; real room support and load engineering require more checks. Nonnegative geometry still may look visually wrong, and no image rendering or physics engine simulation run. Comparability to corrected v47 source must separately validate source support and pallet geometry identities.'}
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps(out,indent=2)+'\n')
 print('REAL_V44_INTERMEDIATE_TRIANGLE_CONTACT_CHAIN',summary,'S',out['runtime_s'],flush=True)
if __name__=='__main__':main()
