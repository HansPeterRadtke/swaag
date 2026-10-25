#!/usr/bin/env python3
"""SOURCE-BLIND video-level detectors on 24 ORIGINAL Godot gameplay MP4s.

Novel independent source video test vs previous original author Videosham edit and
TempGlitch full source model reanalysis: 12 actual source-paired original
Platformer videos with character ANIMATION FROZEN (engine fault) versus 12
source normal Platformer gameplay clips. Original source exact LFS SHA verified.
Author source injected labels NOT read by frame predictor, only 24 image pixels.

Test 4 PREDECLARED SAME native source frame-video segment signals:
 A. Official FFmpeg entire-frame freezedetect n=-50dB duration >=0.6sec.
 B. raw local ROI still while outside scene moving, center x.35-.65/y.25-.85;
    define center pixel MAE<=2.0 and outer MAE>=4.0 at downsampled 5fps,
    at least 3 consecutive 5fps frame transitions (~0.6s).
 C. phase-registered local ROI still, same region and outside motion >4.0;
    cv2.phaseCorrelate on centered grayscale and shift compensation, local
    MAE<=3.0, consecutive >=3; cap image shift +/-16 working pixels.
 D. Farneback optical-flow mismatch center 0.75 median pixel-frame
    displacement versus outside >=2.0, at least 3 consecutive samples.
 E. trivial background-motion-independent whole-screen near-static frame
    MAE<=1.0 for 3 transitions; false-positive control.

Frozen/normal source author engine labels are not subjective independent human
qualitative ratings; exact bug event intervals absent, only video-level
source author fault category ground truth. Very narrow sample chosen by
encoded pair sizes (<=70MB) before looking at pixels, all selected source
scenes same Godot Platformer; no camera/domain independence.

CPU nice19, OpenCV one thread and native FFmpeg6.1 thread1. Pixel decoder
sends `fps=5,scale=192:108:flags=area,format=gray` video only; audio not tested.
Predictions frozen before separate label scoring; abstain on decoder/size/time
limits (never classify undetected as clean). Preserve original videos, prior
negative external findings, no licensed media/model weights in project Git.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,subprocess,re,time,collections,resource,math
import cv2,numpy as np
ROOT=Path('/data/var/perceptual-independent-20261024/tempglitch-pose-frozen-external-20261025')
PLAN=ROOT/'source_original_12_platformer_pairs_fixed_selection_20261025.json'
RECEIPT=ROOT/'source_24_original_MP4_verified_SHA256_receipts_20261025.json'
OUT=ROOT/'24_actual_original_video_fullframe_vs_character_appearance_features_FILTER_FIXED_20261025.json'
W,H=192,108;FPS=5
CONFIG={'date':'2026-10-25','input':'original external Godot author-injected FROZEN CHARACTER ANIMATION versus normal original rendered Platformer clips, same source family/index, exact source SHA verified; not natural frozen encoded frames','video_frames_per_second_analysis':5,'scaled_pixel_resolution':[W,H],'image_sampling':'FFmpeg fps5 scale192:108:flags=area grayscale, -threads1 -filter_threads1, decoder returns all sampled frames max600','ffmpeg_full_frame_freezedetect':'freezedetect=n=-50dB:d=0.6','source_blind_character_region':'x fractions 0.35..0.65, y fractions 0.25..0.85, no learned segmentation; may not contain avatar, no source bounding boxes','local_raw_still_when_background_moves':'ROI MAE<=2.0 AND outside MAE>=4.0, at least3 adjacent samples at5fps','local_phase_aligned_still_when_background_moves':'cv2.phaseCorrelate ROI registered +/-16px full plane, grayscale aligned residual MAE<=3.0 AND outside raw MAE>=4.0, at least3 samples','local_opticalflow_still_when_outer_moves':'OpenCV Farneback defaults pyrScale0.5 levels3 winsize15 iterations3 poly_n5 poly_sigma1.2 flags0, median magnitude ROI<=0.75 and outer>=2.0 on adjacent sampled frames, at least3','whole_frame_MAE_stall':'entire picture MAE<=1.0 for at least3 transitions','local_robustness':'freeze anomaly requires 3 consecutive transitions while backdrop moving to avoid certifying intentional whole-scene hold','no_gold_label_or_selected_source_family_used_inside_feature_logic':True,'invalid_previously_attempted_instrumentation_run_preserved_outside_Git':'INVALID_FILTER_STRING_partial_source_analysis_preserved_20261025.json, partial six cases not evaluated', 'no_gpu_use':True,'time_limit_per_original_video_s':45,'input_source_rights':'TempGlitch original dataset MIT research release, underlying game content may have separate rights; original MP4 files stay /data/var, not Git'}

def sha(p):
 h=hashlib.sha256();size=0
 with p.open('rb') as f:
  while b:=f.read(2097152):size+=len(b);h.update(b)
 return size,h.hexdigest()

def check_video(path):
 c=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,nb_frames,width,height,duration','-of','json',str(path)]
 a=json.loads(subprocess.run(c,text=True,capture_output=True,check=True,timeout=10).stdout)['streams'][0]
 num,den=[int(x) for x in a['avg_frame_rate'].split('/')]
 return {'native_fps':num/den,'native_fps_token':a['avg_frame_rate'],'native_width':int(a['width']),'native_height':int(a['height']),'native_frame_count':int(a['nb_frames']),'duration_seconds':float(a['duration'])}

def decode(path):
 c=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-i',str(path),'-an','-vf',f'fps={FPS},scale={W}:{H}:flags=area,format=gray','-f','rawvideo','-pix_fmt','gray','pipe:1']
 p=subprocess.run(c,capture_output=True,timeout=CONFIG['time_limit_per_original_video_s'])
 if p.returncode:raise RuntimeError('ffmpeg decode error '+p.stderr.decode('utf8','replace')[-160:])
 if len(p.stdout)%(W*H):raise RuntimeError('Incomplete video raw frame bytes')
 n=len(p.stdout)//(W*H)
 if not 15<=n<=600:raise RuntimeError('SAMPLE_COUNT_OUTSIDE_DECLARED_LIMIT '+str(n))
 return np.frombuffer(p.stdout,dtype=np.uint8).reshape((n,H,W))

def consecutive(mask):
 best=0;cur=0
 for t in mask:
  if t:cur+=1;best=max(best,cur)
  else:cur=0
 return int(best)

def native_ffmpeg_freeze(path):
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-threads','1','-filter_threads','1','-i',str(path),'-an','-vf',CONFIG['ffmpeg_full_frame_freezedetect'],'-f','null','-']
 p=subprocess.run(cmd,capture_output=True,timeout=CONFIG['time_limit_per_original_video_s'])
 if p.returncode:raise RuntimeError('original FFmpeg full freeze error '+p.stderr.decode('utf8','replace')[-160:])
 text=p.stderr.decode('utf8','replace')
 starts=[float(x) for x in re.findall(r'freeze_start:\s*([-+]?\d+(?:\.\d+)?)',text)]
 ends=[float(x) for x in re.findall(r'freeze_end:\s*([-+]?\d+(?:\.\d+)?)',text)]
 return {'original_ffmpeg_full_frame_frozen_event_count':len(starts),'original_ffmpeg_full_frame_freeze_bool':bool(starts),'source_freeze_start_s':starts,'source_freeze_end_s':ends,'original_ffmpeg_event_unclosed':len(starts)>len(ends)}

def features(frames):
 cv2.setNumThreads(1)
 n=len(frames)
 x0,x1=int(W*.35),int(W*.65);y0,y1=int(H*.25),int(H*.85)
 mask=np.ones((H,W),dtype=bool);mask[y0:y1,x0:x1]=False
 arr=frames.astype(np.float32);d=np.abs(arr[1:]-arr[:-1])
 global_changes=d.mean(axis=(1,2));inside=d[:,y0:y1,x0:x1].mean(axis=(1,2));outside=d[:,mask].mean(axis=1)
 local_raw=(inside<=2.) & (outside>=4.)
 whole=global_changes<=1.
 phase_changes=[];phase_shifts=[];flow_inside=[];flow_outside=[]
 roi0=arr[:,:][0,y0:y1,x0:x1]
 for i in range(1,n):
  prev=arr[i-1];next=arr[i]
  old=prev[y0:y1,x0:x1];new=next[y0:y1,x0:x1]
  shift,response=cv2.phaseCorrelate(old.copy(),new.copy())
  dx=float(np.clip(shift[0],-16,16));dy=float(np.clip(shift[1],-16,16))
  M=np.float32([[1,0,-dx],[0,1,-dy]])
  M2=np.float32([[1,0,dx],[0,1,dy]])
  a=cv2.warpAffine(new,M,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  b=cv2.warpAffine(new,M2,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  center=(slice(6,-6),slice(6,-6))
  mismatch=min(float(np.abs(old[center]-a[center]).mean()),float(np.abs(old[center]-b[center]).mean()))
  phase_changes.append(mismatch);phase_shifts.append([round(dx,3),round(dy,3)])
  fl=cv2.calcOpticalFlowFarneback(prev.astype(np.uint8),next.astype(np.uint8),None,.5,3,15,3,5,1.2,0)
  mag=np.linalg.norm(fl,axis=2)
  flow_inside.append(float(np.median(mag[y0:y1,x0:x1])))
  flow_outside.append(float(np.median(mag[mask])))
 phase=np.asarray(phase_changes);flow_inside=np.asarray(flow_inside);flow_outside=np.asarray(flow_outside)
 phase_mask=(phase<=3.) & (outside>=4.)
 flow_mask=(flow_inside<=.75) & (flow_outside>=2.)
 result={'source_video_frame_count_at_5fps':n,'original_global_gray_frame_MAE_median':float(np.median(global_changes)),'original_center_character_ROI_MAE_median':float(np.median(inside)),'original_outside_character_ROI_MAE_median':float(np.median(outside)),'original_phase_registered_center_MAE_median':float(np.median(phase)),'original_Farneback_center_flow_median':float(np.median(flow_inside)),'original_Farneback_outside_flow_median':float(np.median(flow_outside)),
 'max_adjacent_run_raw_center_freeze_while_outer_motion':consecutive(local_raw),'max_adjacent_run_phase_registered_center_freeze_while_outer_motion':consecutive(phase_mask),'max_adjacent_run_Farneback_center_frozen_while_outer_motion':consecutive(flow_mask),'max_adjacent_run_full_frame_still':consecutive(whole),
 'raw_center_3transition_frozen_bool':consecutive(local_raw)>=3,'phase_registered_center_3transition_frozen_bool':consecutive(phase_mask)>=3,'Farneback_center_3transition_frozen_bool':consecutive(flow_mask)>=3,'full_frame_3transition_still_bool':consecutive(whole)>=3,'consecutive_sample_seconds':.2,'source_avg_fraction_center_low_motion_conditional_on_outer_active':float(np.mean(local_raw)),'source_avg_fraction_phase_stable_conditional_on_outer_active':float(np.mean(phase_mask)),'source_avg_fraction_Farneback_stable_conditional_on_outer_active':float(np.mean(flow_mask))}
 return result

def main():
 plan=json.loads(PLAN.read_text());rec=json.loads(RECEIPT.read_text())
 assert len(plan['selected'])==12 and len(rec['completed'])==24
 if OUT.exists():
  state=json.loads(OUT.read_text())
  if state['status']=='complete':print('NEW_TEMPGLITCH_REAL_VIDEOS_FEATURES_ALREADY_FROZEN_NO_REPLAY');return
 else:state={'created':'2026-10-25','status':'in_progress','config':CONFIG,'source_original_selection_SHA256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'original_verified_24_MP4_SHA_receipts_at_start':hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),'source_label_blind_results':{}}
 assert state['config']==CONFIG and state['source_original_selection_SHA256']==hashlib.sha256(PLAN.read_bytes()).hexdigest()
 t0=time.monotonic()
 for row in plan['selected']:
  for klass in ['buggy','normal']:
   key=str(row['source_original_pair_index'])+'|'+klass
   if key in state['source_label_blind_results']:continue
   x=rec['completed'][key];p=ROOT/x['source_relpath'];size,h=sha(p)
   assert h==x['source_original_SHA256'] and size==x['source_bytes']
   state['source_label_blind_results'][key]={'status':'pending_or_uncertain'}
   OUT.write_text(json.dumps(state,indent=2)+'\n')
   t=time.monotonic()
   try:
    props=check_video(p);frames=decode(p);sig=features(frames);del frames
    native=native_ffmpeg_freeze(p)
    record={'status':'complete','source_MP4_SHA256':h,'native_source_video_props':props,'source_pixel_signals':sig,'source_fullframe_freeze_detector':native,'CPU_video_analysis_seconds':round(time.monotonic()-t,2)}
   except Exception as e:record={'status':'abstain_due_source_or_compute_no_retry','source_MP4_SHA256':h,'error':type(e).__name__+': '+str(e)[:180]}
   state['source_label_blind_results'][key]=record
   OUT.write_text(json.dumps(state,indent=2)+'\n')
   print('ACTUAL_ORIGINAL_GODOT_VIDEO_FROZEN_IMAGE_SIGNAL',len(state['source_label_blind_results']),'OF24','KEY',key,'STATUS',record['status'],'5FPS_FRAMES',record.get('source_pixel_signals',{}).get('source_video_frame_count_at_5fps'),'SECONDS',record.get('CPU_video_analysis_seconds'),flush=True)
 assert len(state['source_label_blind_results'])==24
 state['status']='complete';state['run_wall_seconds']=round(time.monotonic()-t0,2);state['run_peak_RSS_KB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 OUT.write_text(json.dumps(state,indent=2)+'\n')
 print('FROZEN_NEW_ORIGINAL_REAL_ENGINE_VIDEO_FRAME_ANALYSIS',OUT,'RESULTS',dict(collections.Counter(x['status'] for x in state['source_label_blind_results'].values())),'SECONDS',state['run_wall_seconds'],'GPU_USED',False)
if __name__=='__main__':main()
