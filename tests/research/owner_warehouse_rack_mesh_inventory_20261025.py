"""Read-only inspect rack mesh hierarchy, world AABBs and support attachment candidates.

Only for designing a physically truthful downstream ray contact test; no new
model inference, no mesh mutation, render, save, or production system changes.
Print aggregate counts, source-free dimensions only."
"""
import bpy,collections,statistics,hashlib
from mathutils import Vector
pallets=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('instance_of_verified_buyer_prefab')=='pallet_euro_1200x800']
racks=[o for o in bpy.data.objects if o.type=='EMPTY' and o.get('prefab_id')=='pallet_rack_double_5400']
supports=[o for o in bpy.data.objects if o.name.startswith('RENDER_RACK_STEEL_PALLET_SUPPORT_') and o.type=='MESH']
print('ROOTS',len(pallets),len(racks),len(supports),flush=True)
print('RACK_CHILD_HIERARCHY',[(len(r.children_recursive),len([c for c in r.children_recursive if c.type=='MESH'])) for r in racks],flush=True)
rack=racks[0];mesh=[o for o in rack.children_recursive if o.type=='MESH']
for o in mesh[:18]:
 ps=[o.matrix_world@v.co for v in o.data.vertices]
 print('RACK_SAMPLE_MESH',o.name[:90],'VERTICES',len(ps),'X',(round(min(p.x for p in ps),2),round(max(p.x for p in ps),2)),'Y',(round(min(p.y for p in ps),2),round(max(p.y for p in ps),2)),'Z',(round(min(p.z for p in ps),3),round(max(p.z for p in ps),3)),flush=True)
palsupport=[s for s in supports if s.get('support_rack_name')==rack.name]
print('SUPPORTS_THIS_RACK',len(palsupport))
for sup in palsupport[:3]:
 pts=[sup.matrix_world@v.co for v in sup.data.vertices]
 print('SUPPORT_BOX',sup.name[:60],'X',(round(min(p.x for p in pts),3),round(max(p.x for p in pts),3)),'Y',(round(min(p.y for p in pts),3),round(max(p.y for p in pts),3)),'Z',(round(min(p.z for p in pts),3),round(max(p.z for p in pts),3)),flush=True)
