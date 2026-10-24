#!/usr/bin/env python3
"""Read-only bounded actual 3D glTF metadata and previously completed video frames.

Finds signal/structural anomalies, never declares aesthetic or motion correctness.
"""
import json, struct, math, time, statistics, platform
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]/'docs'/'research'/'perceptual-real-assets-20261024'
GLB=Path('/data/src/3d-commercial-asset-publish/outputs/forklift_publish_candidate/forklift_publish_candidate.glb')
VIDEO=Path('/data/var/3d-video-investigation-20261014/v4/forklift_steering_and_continuous_motion_v4.mp4')

def inspect_glb():
 t=time.perf_counter()
 with GLB.open('rb') as f:
  h=f.read(20);magic,version,declared_size,chunk_size,chunk_type=struct.unpack('<4sIIII',h)
  if magic!=b'glTF' or chunk_type!=0x4e4f534a:raise ValueError('unexpected glTF header')
  document=json.loads(f.read(chunk_size))
 pos=[];invalid=[];flat=[]
 for idx,acc in enumerate(document.get('accessors',[])):
  if acc.get('type')!='VEC3' or not ('min' in acc and 'max' in acc):continue
  lo,hi=acc['min'],acc['max'];pos.append(idx)
  if any(not math.isfinite(v) for v in lo+hi) or any(a>b for a,b in zip(lo,hi)):invalid.append(idx)
  elif any(abs(a-b)<1e-8 for a,b in zip(lo,hi)):flat.append(idx)
 return {'path':str(GLB),'bytes':GLB.stat().st_size,'glb_version':version,'declared_bytes':declared_size,'nodes':len(document.get('nodes',[])),'meshes':len(document.get('meshes',[])),'checked_position_accessors':len(pos),'invalid_bbox_count':len(invalid),'flat_axis_bbox_count':len(flat),'example_named_nodes':[x.get('name') for x in document.get('nodes',[]) if x.get('name')][:8],'seconds':round(time.perf_counter()-t,3),'limits':'valid accessor bounds do not establish correct machinery proportions, contacts, topology, motion or photorealism'}

def inspect_video():
 t=time.perf_counter();cap=cv2.VideoCapture(str(VIDEO))
 if not cap.isOpened():raise RuntimeError('video cannot be opened read-only')
 fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));wh=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
 indices=[i for i in range(min(count,90))]
 if count>200:indices+=list(range(count//2,min(count//2+90,count)))
 # Seek only once to the middle; no new render/encode, downscale to 192x108.
 groups=[indices[:min(count,90)]]
 if count>200:groups.append(indices[min(count,90):])
 diffs=[];exact=[];flow_median=[];previews=[];decoded=0
 for grp in groups:
  cap.set(cv2.CAP_PROP_POS_FRAMES,grp[0]);previous=None;prevgray=None
  for i in grp:
   ok,frame=cap.read()
   if not ok:break
   decoded+=1;im=cv2.resize(frame,(192,108),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
   if i in (0,30,grp[0],grp[min(30,len(grp)-1)]):previews.append(im)
   if previous is not None:
    mae=float(cv2.absdiff(im,previous).mean());diffs.append((i,mae));exact.append(i) if np.array_equal(im,previous) else None
    pts=cv2.goodFeaturesToTrack(prevgray,maxCorners=75,qualityLevel=.05,minDistance=5)
    if pts is not None and len(pts)>=4:
     nxt,status,_=cv2.calcOpticalFlowPyrLK(prevgray,gray,pts,None,winSize=(13,13),maxLevel=1)
     if nxt is not None and status is not None:
      motion=np.linalg.norm((nxt-pts).reshape((-1,2))[status.flatten()>0],axis=1)
      if len(motion):flow_median.append((i,float(np.median(motion))))
   previous=im;prevgray=gray
 cap.release()
 vals=np.array([x[1] for x in diffs]);q=np.percentile(vals,[25,50,75,95,99]) if len(vals) else np.zeros(5)
 iqr=q[2]-q[0];boundary=float(q[2]+3*iqr);outliers=[{'frame':i,'seconds':round(i/fps,3),'difference':round(v,3)} for i,v in diffs if v>boundary]
 if previews:
  cv2.imwrite(str(ROOT/'real_video_contact_sheet.jpg'),np.concatenate(previews[:4],axis=1),[cv2.IMWRITE_JPEG_QUALITY,75])
 return {'path':str(VIDEO),'bytes':VIDEO.stat().st_size,'fps':fps,'frame_count':count,'size':wh,'sampled_frames':decoded,'sampled_adjacent_pairs':len(diffs),'mean_absolute_pixel_difference_quantiles':[round(float(v),3) for v in q],'upper_outlier_rule_q75_plus_3iqr':round(boundary,3),'candidate_change_outliers':outliers[:30],'exact_duplicate_frame_indices':exact[:30],'flow_median_px_per_frame_quantiles':[round(float(x),3) for x in np.percentile([v for _,v in flow_median],[25,50,75,95])] if flow_median else [],'seconds':round(time.perf_counter()-t,3),'limits':'no timestamp ground truth; cuts, camera motion, holds and transitions may cause outliers; cannot assert stutter or quality from threshold alone'}

def main():
 ROOT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 d={'environment':{'python':platform.python_version(),'opencv':cv2.__version__},'glb':inspect_glb(),'video':inspect_video()}
 (ROOT/'results.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
