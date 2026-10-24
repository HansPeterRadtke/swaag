#!/usr/bin/env python3
"""Independent human-labelled, real GLB geometry sanity checks, no GPU/renders.

Predeclared invariants detect corrupt geometry, NOT missing semantic parts.
GLBs are original 3D-DefectBench golden human-review assets under /data/var.
Compares fixed geometry flags to ORIGINAL human missing-parts label to expose
semantic-vs-structural mismatch, not to pick a learned threshold/physics winner.
No archived media, mesh or benchmark labels are modified or committed.
"""
from __future__ import annotations
from pathlib import Path
import json,hashlib,time,resource,collections
import numpy as np
import trimesh
ROOT=Path('/data/var/perceptual-independent-20261024/3d-defectbench')
MANIFEST=ROOT/'mechanical_golden_manifest.json'
OUT=ROOT/'mesh_structure_vs_human_missing_parts_20261024.json'
TOL=1e-12
# Geometry flags predetermined before model results/mesh measurements, not tuned.
FLAGS=('nonfinite_vertex_coordinates','any_open_or_nonmanifold_submesh','degenerate_triangle_faces','coarse_aabb_overlap')

def overlap(a,b):
 a0,a1=a;b0,b1=b
 intersect=np.maximum(0.,np.minimum(a1,b1)-np.maximum(a0,b0));inter=np.prod(intersect)
 volumea=np.prod(np.maximum(0,a1-a0));volumeb=np.prod(np.maximum(0,b1-b0))
 denom=min(volumea,volumeb)
 return float(inter/denom) if denom>0 else 0.

def inspect(path):
 start=time.monotonic()
 scene=trimesh.load(path,force='scene',process=False)
 meshes=[g for g in scene.dump(concatenate=False) if isinstance(g,trimesh.Trimesh)]
 if not meshes:raise ValueError('no actual triangle geometry')
 extent=np.vstack([m.vertices for m in meshes]);size=np.ptp(extent,axis=0)
 scale=float(np.linalg.norm(size));scale=max(scale,1e-12)
 bad_coords=any(not np.isfinite(m.vertices).all() for m in meshes)
 faces=sum(len(m.faces) for m in meshes)
 vertices=sum(len(m.vertices) for m in meshes)
 zero=0
 any_open=False
 boxes=[]
 for m in meshes:
  if len(m.faces):
   areas=m.area_faces
   zero+=int((areas<=TOL*scale*scale).sum())
   if not m.is_watertight:any_open=True
  if len(m.vertices)>0 and np.isfinite(m.vertices).all():boxes.append(m.bounds)
 coarse=False;max_fraction=0.
 for i,a in enumerate(boxes):
  for b in boxes[i+1:]:
   fraction=overlap(a,b);max_fraction=max(max_fraction,fraction)
   if fraction>.02:coarse=True
 flags={'nonfinite_vertex_coordinates':bool(bad_coords),'any_open_or_nonmanifold_submesh':bool(any_open),'degenerate_triangle_faces':bool(zero>0),'coarse_aabb_overlap':bool(coarse)}
 return {'source_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mesh_nodes':len(meshes),'triangle_faces':faces,'vertices':vertices,'world_scene_extents':size.tolist(),'open_submeshes':sum(not m.is_watertight for m in meshes),'degenerate_triangles':zero,'max_aabb_overlap_fraction_of_smaller_part':round(max_fraction,6),'detector_flags':flags,'seconds':round(time.monotonic()-start,3)}

def stats(rows,key,gold):
 tp=sum(x['metrics']['detector_flags'][key] and x[gold] for x in rows)
 fp=sum(x['metrics']['detector_flags'][key] and not x[gold] for x in rows)
 fn=sum(not x['metrics']['detector_flags'][key] and x[gold] for x in rows)
 tn=sum(not x['metrics']['detector_flags'][key] and not x[gold] for x in rows)
 return {'n':len(rows),'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}

def main():
 if OUT.exists():print('PREVIOUS_RESULTS_NOT_REPLAYED',OUT);return
 record=json.loads(MANIFEST.read_text())
 assert len(record['cases'])==10 and sum(x['gold_missing_parts'] for x in record['cases'])==4
 samples=[]
 for idx,r in enumerate(record['cases']):
  path=ROOT/'golden-glbs'/(r['object_id']+'.glb')
  assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==r['file']['sha256']
  outcome=inspect(path)
  samples.append({'id':r['object_id'],'prompt':r['prompt'],'independent_human_raters':r['num_independent_human_votes'],'gold_missing_parts':bool(r['gold_missing_parts']),'gold_fused_incomplete':bool(r['gold_fused_incomplete']),'gold_form_surface':bool(r['gold_geometry_form']),'metrics':outcome})
  print('SOURCE',idx+1,'/',len(record['cases']),'ID',r['object_id'],'GOLD_MISSING',r['gold_missing_parts'],'N_SUBMESHES',outcome['mesh_nodes'],'FACE',outcome['triangle_faces'],'FLAGS',outcome['detector_flags'],flush=True)
 results={gold:{flag:stats(samples,flag,gold) for flag in FLAGS} for gold in ['gold_missing_parts','gold_fused_incomplete','gold_form_surface']}
 report={'date':'2026-10-24','method':'real glTF/GLB numeric topology checks on 10 externally human-labelled generated mechanic-objects (trimesh), NOT rendered image perception','sources':'3D-DefectBench golden human rater majority; CC BY-NC 4.0, non-commercial research','input_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'thresholds':{'relative_degenerate_area':TOL,'aabb_overlap_fraction':.02},'cases':samples,'comparison_to_human_labels':results,'memory_peak_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':'Only 10 preselected machine-ish assets and only 2 raters each, not random; glTF unit scale varies, watertight meshes are not required for game/rendering assets and naive world AABB overlaps do NOT prove interpenetration. Unrepresented or absent intended parts cannot be discovered through valid-triangle checks. Numerical flag TP/FP against missing-parts is a deliberately mismatched construct; not a trustworthy estimate of mechanical QA sensitivity. Source meshes are actual generated assets, no new simulation or render performed.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULTS_MISSING',results['gold_missing_parts'],'TOTAL_SECONDS',round(sum(x['metrics']['seconds'] for x in samples),2),'SAVED',OUT)
if __name__=='__main__':main()
