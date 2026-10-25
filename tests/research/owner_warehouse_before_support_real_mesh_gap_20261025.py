"""READ-ONLY real before/after contact evaluation on the owner's warehouse.

Load original pre-repair source warehouse_photo_pbr_lodclean_v30.blend, previously
reported visible-hovering pallet problem. Use actual rack triangles and pallet
bottom world mesh vertices at the same physically defined footprint locations
where later v44 made 72 bearing supports. No GPU, no Blender renders, no edits.

Source author geometry includes intermittent steel guides. This test only
measures local top-down contact at the original missing front/rear beam flange
areas. A physically credible engineered support must be explicit, not inferred
solely from global AABB or one camera image. A failed hit does not prove all
racking supports absent elsewhere. Dataset one correlated product design.
"""
import bpy,json,hashlib,math,time,statistics
from pathlib import Path
from mathutils import Vector
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_warehouse_physics')
OUT=ROOT/'real_pre_repair_warehouse_original_pallet_to_rack_gap_20261025.json'
SRC=Path('/data/var/warehouse-film-v29/warehouse_photo_pbr_lodclean_v30.blend')
XY_X_OFFSETS=[-.54,.54]
XY_Y_OFFSETS=[-.325,0,.325]

def raydown(mesh,x,y,z):
 inv=mesh.matrix_world.inverted_safe()
 start=inv@Vector((x,y,z));direction=inv.to_3x3()@Vector((0,0,-1));direction.normalize()
 hit,pos,norm,face=mesh.ray_cast(start,direction,distance=.28)
 if not hit:return None
 w=mesh.matrix_world@pos
 return float(w.z)

def main():
 if OUT.exists():print('SOURCE_PRE_REPAIR_ORIGINAL_GAP_EVIDENCE_ALREADY_SAVED');return
 assert Path(bpy.data.filepath).resolve()==SRC.resolve()
 t=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True)
 pallets=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('instance_of_verified_buyer_prefab')=='pallet_euro_1200x800']
 racks=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400']
 additions=[o for o in bpy.data.objects if o.name.startswith('RENDER_RACK_STEEL_PALLET_SUPPORT_')]
 assert len(pallets)==24 and len(racks)==12 and not additions
 results=[]
 for pallet in sorted(pallets,key=lambda o:o.name):
  timber=[o for o in pallet.children_recursive if o.type=='MESH' and not o.hide_render]
  assert len(timber)==1
  wood=timber[0];bottom=min(float((wood.matrix_world@v.co).z) for v in wood.data.vertices)
  center=pallet.matrix_world.translation
  rack=min(racks,key=lambda o:(o.matrix_world.translation.xy-center.xy).length)
  assert (rack.matrix_world.translation.xy-center.xy).length<2.1
  mesh=[o for o in rack.children_recursive if o.type=='MESH' and not o.hide_render]
  assert len(mesh)==1
  probes=[]
  for oy in XY_Y_OFFSETS:
   for ox in XY_X_OFFSETS:
    z=raydown(mesh[0],float(center.x+ox),float(center.y+oy),bottom+.05)
    probes.append({'source_steel_flange_region_offset_xy_m':[ox,oy],'rack_surface_below_bottom_z_m':round(z,6) if z is not None else None,'wood_to_first_rack_face_vertical_gap_m':round(bottom-z,6) if z is not None else None})
  measured=[p['wood_to_first_rack_face_vertical_gap_m'] for p in probes if p['wood_to_first_rack_face_vertical_gap_m'] is not None]
  results.append({'pallet_root_hash':hashlib.sha256(pallet.name.encode()).hexdigest(),'original_rack_root_hash':hashlib.sha256(rack.name.encode()).hexdigest(),'source_wood_bottom_z_m':round(bottom,6),'actual_rack_mesh_probes':probes,'positive_gap_probe_count_more_than_2mm':sum(z>.002 for z in measured),'rack_contact_probe_count_within_2mm':sum(abs(z)<=.002 for z in measured),'rack_ray_hits':len(measured),'minimum_measured_gap_m':min(measured) if measured else None,'maximum_measured_gap_m':max(measured) if measured else None})
 gaps=[p['wood_to_first_rack_face_vertical_gap_m'] for r in results for p in r['actual_rack_mesh_probes'] if p['wood_to_first_rack_face_vertical_gap_m'] is not None]
 record={'date':'2026-10-25','source':'Original owner warehouse source scene BEFORE extra bearing bars; CPU-only Blender 4.0.2 geometry raycast','pre_repair_file_size':SRC.stat().st_size,'one_correlated_product_source_not_independent_examples':True,'original_pallets':len(pallets),'original_racks':len(racks),'extra_correction_supports':len(additions),'geometry_probe_offsets_X_m':XY_X_OFFSETS,'geometry_probe_offsets_Y_m':XY_Y_OFFSETS,'per_pallet':results,'summary':{'probe_ray_attempts':len(pallets)*len(XY_X_OFFSETS)*len(XY_Y_OFFSETS),'probe_ray_hits':len(gaps),'measured_gap_median_m':statistics.median(gaps) if gaps else None,'measured_gap_min_m':min(gaps) if gaps else None,'measured_gap_max_m':max(gaps) if gaps else None,'contact_under_2mm_probe_count':sum(abs(x)<=.002 for x in gaps),'positive_gap_more_than_2mm_probe_count':sum(x>.002 for x in gaps),'pallets_all_six_probes_show_positive_gaps':sum(r['positive_gap_probe_count_more_than_2mm']==6 for r in results),'pallets_at_least_one_true_rack_triangle_contact_in_probe':sum(r['rack_contact_probe_count_within_2mm']>=1 for r in results)},'elapsed_s':round(time.monotonic()-t,2),'limits':'Probes sample specified front/back support endpoints that motivated engineering repair; other real steel runners may contact elsewhere. This single source scene is not a multi-objectively human-labelled industrial benchmark; cannot prove true load failure, load capacity or visual fidelity. Before/after product versions carry correlated geometry; not separate independent samples.'}
 OUT.write_text(json.dumps(record,indent=2)+'\n')
 print('REAL_PRE_REPAIR_UNSUPPORTED_SHELF_ENDPOINT_GEOMETRY',record['summary'],'SCENE',SRC.name,'S',record['elapsed_s'],flush=True)
if __name__=='__main__':main()
