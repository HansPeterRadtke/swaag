"""Read-only full 3D geometry contact audit of one REAL owner warehouse source.

Run with Blender 4.0.2 -b source.blend --disable-autoexec -t 1 -P thisfile,
not in the Blender text editor. No Cycles render, CUDA, material change, save,
model service or frame sequence. Source is the already completed warehouse v47
film scene, actual same GLB-derived timber pallets and welded rack 3D geometry.

Independent oracle: world-space bounding and mesh ray casting to identify actual
pallet wood underside relative to each of its 72 material support bars. This
proves the chosen spatial contact *if ray hits near top plane*, NOT structural
load capacity, human-visible plausibility, texture/shadow or rack loading.
Compare source-provided previous QA only after this raw geometry output froze.
No files from owner 3D production are modified, no render or GPU work.
"""
import bpy,math,json,hashlib,datetime,statistics,os,time
from pathlib import Path
from mathutils import Vector
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics')
RAW=ROOT/'real_v47_pallet_to_new_support_mesh_contact_20261025.json'
SOURCE=Path('/data/var/warehouse-film-v47/warehouse_THROUGH_OPENING_DOOR_to_REAR_CORNER_three_second_HOLD_v47.blend')
CONFIG={'source_blender_file':str(SOURCE),'blender_version':bpy.app.version_string,'method':'actual mesh world vertex bounding and mesh forward vertical upward ray casts from near support steel upper face to pallet wood, no render','semantic_contact_tolerance_metres':.002,'ray_start_below_steel_top_metres':.08,'raycast_scan_distance_local':.7,'probe_offsets_local_x_metres':[-.35,0,.35],'all_operations_read_only':True,'physical_scope':'check position/contact of pallet wood underside with independent support geometry, does NOT prove primary rack beam contact/load/strength or final visual quality'}

def sha(path):
 digest=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(4*1024*1024):digest.update(chunk)
 return digest.hexdigest()

def points_world(obj):
 mat=obj.matrix_world
 return [mat@v.co for v in obj.data.vertices]

def bounds(ps):
 return [[float(min(p[i] for p in ps)) for i in range(3)],[float(max(p[i] for p in ps)) for i in range(3)]]

def ray_at(obj,x,y,z):
 inv=obj.matrix_world.inverted_safe()
 start=inv@Vector((x,y,z-.08));direction=inv.to_3x3()@Vector((0,0,1))
 if direction.length<1e-6:return None
 direction.normalize()
 hit,where,normal,face=obj.ray_cast(start,direction,distance=.7)
 if not hit:return None
 world=obj.matrix_world@where
 return {'hit_world':[round(float(v),6) for v in world],'gap_hit_above_support_top_metres':round(float(world.z-z),6),'face_index':int(face)}

def main():
 if RAW.exists():
  d=json.loads(RAW.read_text());print('PREVIOUS_CONTACT_RAW_PRESERVED_NOT_REPEATED',d['status']);return
 assert Path(bpy.data.filepath).resolve()==SOURCE.resolve()
 ROOT.mkdir(parents=True,exist_ok=True)
 original_stat=SOURCE.stat();t=time.monotonic()
 pallet_roots={o.name:o for o in bpy.data.objects if o.type=='EMPTY' and o.get('instance_of_verified_buyer_prefab')=='pallet_euro_1200x800'}
 racks={o.name:o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400'}
 supports=[o for o in bpy.data.objects if o.type=='MESH' and o.name.startswith('RENDER_RACK_STEEL_PALLET_SUPPORT_')]
 assert len(pallet_roots)==24 and len(supports)==72 and len(racks)==12
 sresult=[]
 for sup in sorted(supports,key=lambda o:o.name):
  pname=sup.get('supported_pallet_root');rname=sup.get('support_rack_name')
  assert pname in pallet_roots and rname in racks
  pallet=pallet_roots[pname]
  meshes=[x for x in pallet.children_recursive if x.type=='MESH' and not x.hide_render]
  assert len(meshes)==1,(pname,len(meshes))
  wood=meshes[0]
  bwood=bounds(points_world(wood));bsteel=bounds(points_world(sup));top=bsteel[1][2]
  wood_min_z=bwood[0][2]
  s_center=sup.matrix_world.translation
  hits=[]
  for offset in CONFIG['probe_offsets_local_x_metres']:
   x=float(s_center.x+offset);y=float(s_center.y)
   hits.append({'x_offset_metres':offset,'hit':ray_at(wood,x,y,top)})
  exact=[r['hit']['gap_hit_above_support_top_metres'] for r in hits if r['hit'] is not None]
  nearest=min(exact,key=abs) if exact else None
  overlaps_x=max(0,min(bwood[1][0],bsteel[1][0])-max(bwood[0][0],bsteel[0][0]))
  overlaps_y=max(0,min(bwood[1][1],bsteel[1][1])-max(bwood[0][1],bsteel[0][1]))
  sresult.append({'support_source_name_sha256':hashlib.sha256(sup.name.encode()).hexdigest(),'pallet_source_name_sha256':hashlib.sha256(pname.encode()).hexdigest(),'rack_source_name_sha256':hashlib.sha256(rname.encode()).hexdigest(),'support_bounds_world_metres':[[round(z,6) for z in a] for a in bsteel],'pallet_bounds_world_metres':[[round(z,6) for z in a] for a in bwood],'source_support_top_world_z_m':round(top,6),'wood_bottom_world_z_m':round(wood_min_z,6),'global_wood_bottom_minus_steel_top_m':round(wood_min_z-top,6),'AABB_xy_overlap_metres':[round(overlaps_x,6),round(overlaps_y,6)],'actual_mesh_bottom_hit_above_support_top_m':nearest,'probe_vertical_mesh_ray_hits':hits})
 byp={}
 for row in sresult:byp.setdefault(row['pallet_source_name_sha256'],[]).append(row)
 assert len(byp)==24 and all(len(rows)==3 for rows in byp.values())
 allnearest=[x['actual_mesh_bottom_hit_above_support_top_m'] for x in sresult if x['actual_mesh_bottom_hit_above_support_top_m'] is not None]
 near=[x for x in sresult if x['actual_mesh_bottom_hit_above_support_top_m'] is not None and abs(x['actual_mesh_bottom_hit_above_support_top_m'])<=CONFIG['semantic_contact_tolerance_metres']]
 btm=[x['global_wood_bottom_minus_steel_top_m'] for x in sresult]
 original_post=SOURCE.stat();assert (original_stat.st_size,original_stat.st_mtime_ns)==(original_post.st_size,original_post.st_mtime_ns)
 report={'date':'2026-10-25','status':'complete_source_mesh_world_geometry_CPU_only','actual_source_scene_sha256':sha(SOURCE),'original_file_bytes':original_stat.st_size,'original_mtime_ns':original_stat.st_mtime_ns,'config':CONFIG,'source_counts':{'buyer_pallet_roots':len(pallet_roots),'bearing_support_meshes':len(supports),'original_rack_roots':len(racks)},'support_geometry_cases':sresult,'source_independent_summary':{'all_24_pallets_have_three_referenced_support_meshes':all(len(rows)==3 for rows in byp.values()),'support_ray_probes_total':len(sresult)*len(CONFIG['probe_offsets_local_x_metres']),'support_bars_with_at_least_one_upward_ray_hit':len(allnearest),'support_bars_with_contact_within_2mm':len(near),'contact_fraction_detected':len(near)/len(sresult),'min_support_wood_bottom_gap_m':min(btm),'max_support_wood_bottom_gap_m':max(btm),'nearest_actual_ray_hit_abs_gap_m_min':min(map(abs,allnearest)) if allnearest else None,'nearest_actual_ray_hit_abs_gap_m_max':max(map(abs,allnearest)) if allnearest else None,'mean_ray_gap_m':statistics.mean(allnearest) if allnearest else None,'bars_with_nonzero_world_xy_bbox_overlap':sum(all(y>0 for y in x['AABB_xy_overlap_metres']) for x in sresult)},'resource':{'elapsed_wall_s':round(time.monotonic()-t,2),'blender_version':bpy.app.version_string},'validity_limits':['Vertical upward ray-cast hit is a geometric contact property, not proof of engineered load-bearing design or intact construction','Steel bars may intersect pallet wood slightly (~0.3mm), a designed tolerance rather than simulated contact force','No full physics simulation; no time-varying pallet position tested, no rendered pixels analyzed','Racking beams supporting steel bars have not been independently triangle-ray-validated with load constraints','This actual owner video stage scene remains separate from external 3D-DefectBench human benchmark and is not independent human-rated defect vs clean dataset; do not report diagnostic true positive/false positive rates on this one warehouse scene']}
 RAW.write_text(json.dumps(report,indent=2)+'\n')
 print('REAL_OWNER_3D_CONTACT_GEOMETRY',report['source_independent_summary'],'RUNTIME',report['resource'],'SAVED',RAW,flush=True)
if __name__=='__main__':main()
