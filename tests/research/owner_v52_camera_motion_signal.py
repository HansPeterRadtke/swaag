#!/usr/bin/env python3
"""Read-only camera/motion continuity DIAGNOSTIC on TWO finished owner Cycles 3D MP4s.

520? No: source must contain exactly 420 frames at 30 FPS as independently
verified by ffprobe. Use 256x144 grayscale original frames with timestamp
cadence unchanged; sparse Lucas-Kanade + RANSAC affine partial transform.

A high/low motion outlier is not necessarily perceptual stutter; camera cuts,
foreground machinery, occlusion, video blur can change the fit. This script
reports reproducible motion statistics and abstains from QA pass/fail when no
independent camera transform reference or human label exists. CPU-only.

No user video is written, edited, rendered, published, or modified.
"""
from pathlib import Path
import json,hashlib,subprocess,time,statistics,resource
import numpy as np
import cv2
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_product_video')
OUT=ROOT/'owner_prod_v52_per_frame_motion_signal_20261025.json'
INPUTS={
 'warehouse_v52':Path('/data/var/3d-smooth-hd-20261024/warehouse/animated_warehouse_true_Cycles_1920x1080_native_30fps_motionblur_v52.mp4'),
 'dock_v52':Path('/data/var/3d-smooth-hd-20261024/dock/animated_dock_true_Cycles_1920x1080_native_30fps_motionblur_v52.mp4')}
CONFIG={'frame_size':[256,144],'expected_fps':30,'expected_frames':420,'goodFeaturesToTrack':{'maxCorners':160,'qualityLevel':0.012,'minDistance':6,'blockSize':5},'calcOpticalFlowPyrLK':{'winSize':[17,17],'maxLevel':2,'criteria':['EPS',18,0.02]},'estimateAffinePartial2D':{'method':'RANSAC','ransacReprojThreshold':2.0,'maxIters':500,'confidence':0.97},'motion_signal_units':'median 2D feature displacement pixels/frame at 256x144 resolution, camera+moving objects entangled','robust_jump_baseline':'nonstationary median local 30-frame scale, no acceptance threshold tuned or human jerk labels'}

def sample(path):
 before=path.stat()
 args=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-i',str(path),'-map','0:v:0','-an','-vf','scale=256:144:flags=area','-pix_fmt','gray','-f','rawvideo','pipe:1']
 p=subprocess.run(args,capture_output=True,timeout=90,check=True)
 n=len(p.stdout)//(256*144)
 if len(p.stdout)%(256*144)!=0 or n!=CONFIG['expected_frames'] or p.stdout[:20]==bytes(20):raise ValueError('Unexpected raw decoded frame count or format '+path.name)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during read')
 data=np.frombuffer(p.stdout,dtype=np.uint8).reshape((n,144,256))
 return data,{'file':str(path),'source_bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'source_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoded_frame_count':n}

def calculate(frames):
 cv2.setNumThreads(1)
 rows=[];w=256;h=144
 for i,(pre,post) in enumerate(zip(frames[:-1],frames[1:])):
  t=i+1
  if np.array_equal(pre,post):
   rows.append({'frame_number':t,'identical_frame':True,'mean_abs_frame_change':0,'source_points':0,'tracked_points':0,'affine_fit':'identical','median_displacement_px':0,'p95_displacement_px':0,'affine_translation_px':0,'affine_rotation_deg':0,'affine_scale':1,'affine_inlier_fraction':None});continue
  x=cv2.goodFeaturesToTrack(pre,mask=None,**CONFIG['goodFeaturesToTrack'])
  info={'frame_number':t,'identical_frame':False,'mean_abs_frame_change':round(float(cv2.absdiff(pre,post).mean()),5),'source_points':len(x) if x is not None else 0,'tracked_points':0,'affine_fit':'not_enough_features'}
  if x is None or len(x)<8:
   rows.append(info);continue
  y,ok,_=cv2.calcOpticalFlowPyrLK(pre,post,x,None,winSize=(17,17),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,18,0.02))
  if y is None or ok is None:
   rows.append(info);continue
  v=ok.reshape((-1,))!=0;points=x.reshape((-1,2))[v];targets=y.reshape((-1,2))[v]
  info['tracked_points']=len(points)
  if len(points)<6:
   rows.append(info);continue
  distances=np.linalg.norm(targets-points,axis=1)
  info['median_displacement_px']=round(float(np.median(distances)),6)
  info['p95_displacement_px']=round(float(np.quantile(distances,.95)),6)
  mat,inliers=cv2.estimateAffinePartial2D(points,targets,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=500,confidence=.97)
  if mat is None:
   info['affine_fit']='failed';rows.append(info);continue
  info['affine_fit']='ok';info['affine_translation_px']=round(float(np.hypot(mat[0,2],mat[1,2])),6)
  info['affine_rotation_deg']=round(float(np.degrees(np.arctan2(mat[1,0],mat[0,0]))),6)
  info['affine_scale']=round(float(np.hypot(mat[0,0],mat[1,0])),6)
  info['affine_inlier_fraction']=round(float(np.mean(inliers)),5) if inliers is not None else None
  rows.append(info)
 return rows

def summarize(rows):
 valid=[r for r in rows if r.get('affine_fit')=='ok']
 if not valid:raise ValueError('No tractable camera flow pairs')
 disps=np.asarray([r['median_displacement_px'] for r in valid]);trans=np.asarray([r['affine_translation_px'] for r in valid]);changes=np.asarray([r['mean_abs_frame_change'] for r in rows]);
 all_signal=np.full((len(rows),),np.nan)
 for row in valid:all_signal[row['frame_number']-1]=row['median_displacement_px']
 # Flag candidates ONLY by pure preset local heuristic, not physics truth.
 candidates=[]
 for i,x in enumerate(all_signal):
  if not np.isfinite(x):continue
  lo=max(0,i-15);hi=min(len(all_signal),i+16);neighbor=all_signal[lo:hi]
  neighbor=neighbor[np.isfinite(neighbor)]
  if len(neighbor)<8:continue
  center=float(np.median(neighbor));mad=float(np.median(np.abs(neighbor-center)))
  if x>center+8*max(0.05,1.4826*mad):candidates.append({'frame':i+1,'seconds':round((i+1)/30,4),'feature_motion':round(float(x),5),'local_median':round(center,5),'robust_spike':True})
 return {'adjacent_frame_pairs':len(rows),'affine_success':len(valid),'unfit_pairs':len(rows)-len(valid),'exact_pixel_identical_adjacent_pairs':sum(r['identical_frame'] for r in rows),'median_flow_px_q':[round(float(np.quantile(disps,q)),5) for q in (0,.05,.25,.5,.75,.95,1)],'camera_affine_translation_px_q':[round(float(np.quantile(trans,q)),5) for q in (0,.05,.25,.5,.75,.95,1)],'frame_change_grayscale_q':[round(float(np.quantile(changes,q)),5) for q in (.05,.5,.95,.99)],'large_feature_displacement_robust_local_outliers':candidates,'max_affine_rotation_deg':max(abs(r['affine_rotation_deg']) for r in valid),'median_affine_inlier_fraction':statistics.median(r['affine_inlier_fraction'] for r in valid if r['affine_inlier_fraction'] is not None)}

def main():
 if OUT.exists():
  print('PREVIOUS_COMPLETE_OWNER_SIGNAL_PRESERVED');return
 ROOT.mkdir(exist_ok=True)
 details={};t0=time.monotonic()
 for name,path in INPUTS.items():
  t=time.monotonic();frame,source=sample(path);signals=calculate(frame);summary=summarize(signals)
  details[name]={'source':source,'summary':summary,'signal_per_pair':signals,'wall_seconds':round(time.monotonic()-t,2)}
  print(name,'SOURCE_FRAMES',source['decoded_frame_count'],'AFFINE_FIT',summary['affine_success'],'EXACT_REPEATS',summary['exact_pixel_identical_adjacent_pairs'],'MOTION_OUTLIER_CANDIDATES',len(summary['large_feature_displacement_robust_local_outliers']),'MEDIAN_FLOW_PX',summary['median_flow_px_q'][3],'WALL_S',details[name]['wall_seconds'],flush=True)
 result={'date':'2026-10-25','scope':'actual completed 3D Cycles product MP4 motion signal, source-only static 2D pixel tracking, not a calibrated temporal video judger','analysis_configuration':CONFIG,'videos':details,'total_cpu_wall_s':round(time.monotonic()-t0,2),'limitations':'2D feature motion combines camera pan, parallax, moving objects, texture and occlusions; 8x local motion-jump heuristic is neither expert-rated stutter threshold nor a guarantee that a product is smooth. Smooth timestamp cadence/zero duplicate frames do not prove good visuals. Exact camera transforms/Blender output, full human judgement and reference ratings missing. No product touched.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('SOURCE_SIGNAL_SAVED',OUT,'RESULTS_TWO_VIDEOS',len(details))
if __name__=='__main__':main()
