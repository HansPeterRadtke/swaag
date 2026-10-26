#!/usr/bin/env python3
"""Read-only, real FINAL encoded owner video vs source Cycles PNG frames.

stage freeze: source sha256 check and protocol lock, NO video decoded.
stage decode: FFmpeg native full frame decode + PTS + same-source and two true
              negative controls; generate contact sheet only under /data/var.
source/approved .blend, .mp4 and PNG files are READ ONLY. No GPU.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

PROJECT=Path('/data/var/3d-publication-20261024')
SOURCE=PROJECT/'warehouse_v47_native'
FINAL=PROJECT/'approved_warehouse_door_rear_corner_v47.mp4'
DOCK=PROJECT/'approved_dock_cinematic_preview_v46.mp4'
SCENE=Path('/data/var/warehouse-film-v47/warehouse_THROUGH_OPENING_DOOR_to_REAR_CORNER_three_second_HOLD_v47.blend')
RELOADED=Path('/data/var/warehouse-film-v47/independent_reloaded_camera_geometry_v47_qa.json')
ISOLATED=Path('/data/var/perceptual-independent-20261024/final_encoded_owner_mp4_20261026')

EXPECTED_SOURCE_SCENE_SHA='6609f09ac0637dd232539f7d5de49ce3e59dc477a5bf25beb125ddcbf73849b6'
EXPECTED_FINAL_SHA='e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb'
EXPECTED_DOCK_SHA='75ef01e7316d099759f4e1dccec304f4c3ea8a995949deebe7f6b41962c6ef93'
KEYFRAMES=(1,40,50,75,90,120,145,165,166,180,200,210)
DESIGN={
 'date':'2026-10-26','goal':'Independently decode actual final owner warehouse MP4; verify approved specific source PNG frame sequence, PTS and intended camera-hold *source contract*, instead of trusting report metadata alone',
 'scope':'source-to-decoded-encoded-frame fidelity and browser playback (separate stage); does not judge owner photorealism, machine stress or audible event synchronization',
 'positive_video':str(FINAL),'positive_original_sha_expected':EXPECTED_FINAL_SHA,'negative_dock_video':str(DOCK),'negative_dock_sha_expected':EXPECTED_DOCK_SHA,
 'source_scene':str(SCENE),'scene_SHA_expected':EXPECTED_SOURCE_SCENE_SHA,'source_render_receipt':str(SOURCE/'status.json'),'source_geometry_verification':str(RELOADED),
 'source_frame_count':210,'native_frame_fps':15,'rendered_frame_source_numbers':'(i-1)*2+1, i in [1,210], source scene 30fps; exact original PNG hashes in status.json','resolution':[960,540],
 'final_expected_duration_seconds':14,'expected_hold_source_camera_3_seconds':'last 45 encoded frames are sourced from camera-static scene frames; particles/denoising/other objects may still move',
 'expected_camera_door_crossing_scene_frames':[90,91],
 'reference_comparison':'native RGB decoded pixels vs original source RGB PNG at identical frame index, full 960x540; H264 is lossy',
 'quality_proxy_declared_before_decode':'min aligned per-frame PSNR >=25 dB AND median >=30 dB and median aligned SSIM >=0.8; this only checks encoded image resemblance, not aesthetics',
 'controls':'unmodified approved dock movie (same 960x540 @15fps/210) against warehouse source; SAME warehouse encoded movie against original PNG 15 frames later (circular). Both are wrong-reference/time controls, NOT bad videos.',
 'negative_expected':'median aligned PSNR > median each wrong-control PSNR by >=5dB (for shifted source may fail in static tail; not an auto rejection, report descriptive)',
 'encoded_timebase':'ffprobe best-effort PTS: exact 210 frames, 15/1 fps, 14 seconds, first PTS=0, intervals within 0.00001 of 1/15, no missing/duplicate timestamps',
 'browser_playback':'independent headless Firefox and normal HTML <video> element should play and reach ended over 14 s; any browser decoder failure makes REVIEW not inferred PASS',
 'sound':'expected silent original warehouse and dock MP4, no audio stream. Audio synchronization with actual event CANNOT be judged and is always REVIEW.',
 'final_view':'twelve extracted *DECODED* MP4 frames in isolated contact sheet; source identity/hashes and scene image evidence retained. Only summary and reproducible code in Git.',
 'do_not_touch':'prior explicitly denied older high-resolution owner v52 tail-framehash retrieval; no production renderer, upload, YouTube, gallery or asset editing',
 'limits':'One related final warehouse video and different approved dock film; author original source receipts are internally audited but not independent human quality labels, so no general detector FP/FN rates.',
}

def hash_file(p):
 with p.open('rb') as f:
  digest=sha256()
  for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()

def write_once(path,obj):
 if path.exists():raise RuntimeError('Frozen research evidence exists, refuse overwrite: '+str(path))
 b=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode()
 tmp=path.with_suffix('.newtmp');tmp.write_bytes(b);os.replace(tmp,path)

def probe_video(path):
 command=['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]
 r=subprocess.run(command,capture_output=True,text=True,timeout=18,check=True)
 d=json.loads(r.stdout);v=[x for x in d['streams'] if x.get('codec_type')=='video']
 if len(v)!=1:raise RuntimeError('Must have exactly one video stream')
 return {'format_duration':float(d['format']['duration']),'streams':[{k:s.get(k) for k in ('index','codec_type','codec_name','width','height','r_frame_rate','avg_frame_rate','nb_frames','pix_fmt','start_time')} for s in d['streams']]}

def freeze():
 ISOLATED.mkdir(parents=True,exist_ok=True)
 if any((ISOLATED/p).exists() for p in ('protocol.json','source_manifest.json','decoded_frame_evidence.json')):raise RuntimeError('Existing evaluation, refuse new freeze')
 assert hash_file(SCENE)==EXPECTED_SOURCE_SCENE_SHA
 assert hash_file(FINAL)==EXPECTED_FINAL_SHA
 assert hash_file(DOCK)==EXPECTED_DOCK_SHA
 source=json.loads((SOURCE/'status.json').read_text())
 reloaded=json.loads(RELOADED.read_text())
 assert source['scene_sha256']==EXPECTED_SOURCE_SCENE_SHA and reloaded['source_sha256']==EXPECTED_SOURCE_SCENE_SHA
 assert reloaded['status']=='INDEPENDENT_RELOADED_REAL_DOOR_CROSSING_REAR_CORNER_WIDE_VIEW_AND_EXACT_THREE_SECOND_STILL_HOLD_PASS'
 assert source['status']=='ALL_210_WAREHOUSE_V47_ACCEPTED_CAMERA_SOURCE_FRAMES_COMPLETE'
 assert source['total_frames']==len(source['completed'])==210
 assert source['resolution']==[960,540] and source['preview_output_fps']==15
 assert reloaded['door_open_crossing_frames']==[90,91] and reloaded['final_stationary_frames']==90 and reloaded['final_stationary_seconds']==3.0
 original=[]
 for i in range(1,211):
  p=SOURCE/f'frame_{i:04d}.png'; rec=source['completed'][str(i)]
  assert rec['source_frame']==(i-1)*2+1
  assert p.stat().st_size==rec['bytes']
  h=hash_file(p)
  assert h==rec['sha256'],('source PNG changed',i)
  original.append({'native_render_frame':i,'source_scene_frame':rec['source_frame'],'sha256':h,'bytes':rec['bytes']})
 ffprobe={name:probe_video(path) for name,path in (('warehouse',FINAL),('dock',DOCK))}
 for x in ffprobe.values():
  v=[s for s in x['streams'] if s['codec_type']=='video'][0]
  assert (v['width'],v['height'])==(960,540) and v['avg_frame_rate']=='15/1' and v['nb_frames']=='210'
  assert abs(x['format_duration']-14.)<1e-3
 write_once(ISOLATED/'protocol.json',DESIGN)
 write_once(ISOLATED/'source_manifest.json',{'protocol_sha256':hash_file(ISOLATED/'protocol.json'),'source_scene_sha256':EXPECTED_SOURCE_SCENE_SHA,'render_status_sha256':hash_file(SOURCE/'status.json'),'reloaded_original_scene_QA_sha256':hash_file(RELOADED),'original_source_PNGs':original,'original_final_MP4_sha256':EXPECTED_FINAL_SHA,'unrelated_real_dock_MP4_sha256':EXPECTED_DOCK_SHA,'pre_decode_probe':ffprobe})
 print('SOURCE_FROZEN',len(original),'original Cycles PNGs',hash_file(ISOLATED/'source_manifest.json'),'video',EXPECTED_FINAL_SHA,'dock',EXPECTED_DOCK_SHA)

def video_pts(path):
 cmd=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time,key_frame','-of','json',str(path)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=90,check=True)
 frames=json.loads(p.stdout)['frames'];times=[float(x['best_effort_timestamp_time']) for x in frames]
 return {'count':len(frames),'first_sec':times[0],'last_sec':times[-1],'min_interval_s':min(b-a for a,b in zip(times,times[1:])),'max_interval_s':max(b-a for a,b in zip(times,times[1:])),'nonmonotonic_count':sum(b<=a for a,b in zip(times,times[1:])),'frame_timebase_max_error_seconds':max(abs(v-i/15) for i,v in enumerate(times)),'key_frames':sum(x.get('key_frame')==1 for x in frames)}


def frame_reader(path,w=960,h=540):
 cmd=['ffmpeg','-nostdin','-hide_banner','-v','error','-threads','2','-i',str(path),'-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
 n=w*h*3;count=0
 try:
  while True:
   buf=bytearray()
   while len(buf)<n:
    piece=p.stdout.read(n-len(buf))
    if not piece:break
    buf.extend(piece)
   if not buf:break
   if len(buf)!=n:raise RuntimeError('truncated rawvideo frame '+str(count))
   count+=1
   yield np.frombuffer(buf,dtype=np.uint8).reshape((h,w,3)).copy()
  out=p.wait(timeout=20)
  stderr=p.stderr.read().decode(errors='replace')
  if out!=0 or stderr.strip(): raise RuntimeError(f'ffmpeg decoding issue rc={out}: {stderr[:900]}')
 finally:
  if p.poll() is None:p.terminate();p.wait(timeout=5)


def metrics(a,b):
 if a.shape!=b.shape:raise ValueError('wrong source resolution')
 d=cv2.absdiff(a,b)
 mae=float(cv2.mean(d)[:3][0]+cv2.mean(d)[:3][1]+cv2.mean(d)[:3][2])/3
 psnr=float(cv2.PSNR(a,b))
 grayA=cv2.cvtColor(cv2.resize(a,(240,135),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
 grayB=cv2.cvtColor(cv2.resize(b,(240,135),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
 ssim=float(structural_similarity(grayA,grayB,data_range=255))
 return {'PSNR_dB':round(psnr,4),'MAE_rgb':round(mae,4),'SSIM_gray240':round(ssim,5)}

def do_decode():
 if (ISOLATED/'decoded_frame_evidence.json').exists():raise RuntimeError('Do not rerun already-frozen video result')
 protocol=ISOLATED/'protocol.json'; plan=json.loads((ISOLATED/'source_manifest.json').read_text())
 assert plan['protocol_sha256']==hash_file(protocol)
 assert hash_file(FINAL)==EXPECTED_FINAL_SHA and hash_file(DOCK)==EXPECTED_DOCK_SHA
 assert hash_file(SOURCE/'status.json')==plan['render_status_sha256']
 t=time.perf_counter();times={}
 pts={name:video_pts(path) for name,path in (('warehouse',FINAL),('dock',DOCK))}
 times['ffprobe_pts_seconds']=round(time.perf_counter()-t,3)
 source=[]
 for rec in plan['original_source_PNGs']:
  p=SOURCE/f'frame_{rec["native_render_frame"]:04d}.png'
  if hash_file(p)!=rec['sha256']:raise RuntimeError('SOURCE VERSION CHANGED')
  orig=cv2.imread(str(p),cv2.IMREAD_COLOR)
  if orig is None or orig.shape!=(540,960,3):raise RuntimeError('PNG decode failure')
  source.append(cv2.cvtColor(orig,cv2.COLOR_BGR2RGB))
 times['read_verified_original_PNGs_seconds']=round(time.perf_counter()-t-times['ffprobe_pts_seconds'],3)
 scores={};motion={};storyboard={}
 for name,path in (('warehouse',FINAL),('dock',DOCK)):
  begin=time.perf_counter();arr=[];motion_gray=[];prev=None
  for i,frame in enumerate(frame_reader(path),start=1):
   if i>210:raise RuntimeError('unexpected extra video frame')
   record={'index':i,'time_s':round((i-1)/15.,6),'aligned_to_warehouse_source':metrics(frame,source[i-1]), 'warehouse_source_15frame_shift':metrics(frame,source[(i-1+15)%210])}
   small=cv2.cvtColor(cv2.resize(frame,(240,135),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
   record['native_render_motion_gray_absmean_240']=None if prev is None else round(float(cv2.mean(cv2.absdiff(small,prev))[0]),5)
   prev=small
   arr.append(record)
   if name=='warehouse' and i in KEYFRAMES:storyboard[i]=frame.copy()
  if len(arr)!=210:raise RuntimeError('decode count != 210: '+name+':'+str(len(arr)))
  scores[name]=arr;times['decode_native_RGB_and_compare_'+name+'_seconds']=round(time.perf_counter()-begin,3)
 if len(storyboard)!=len(KEYFRAMES):raise RuntimeError('storyboard missing samples')
 W=960;H=4*206;sheet=Image.new('RGB',(W,H),(25,25,27));draw=ImageDraw.Draw(sheet)
 for num,i in enumerate(KEYFRAMES):
  x=(num%3)*320;y=(num//3)*206
  shot=Image.fromarray(storyboard[i]).resize((320,180),Image.Resampling.LANCZOS)
  sheet.paste(shot,(x,y));draw.text((x+8,y+184),f'Actual decoded MP4 frame {i} — {round((i-1)/15,2)} s',fill='white')
 contact=ISOLATED/'actual_final_warehouse_mp4_decoded_contact_sheet.jpg'
 if contact.exists():raise RuntimeError('Prior contact sheet exists')
 sheet.save(contact,quality=85)
 def summary(rows):
  ps=[z['aligned_to_warehouse_source']['PSNR_dB'] for z in rows]
  ss=[z['aligned_to_warehouse_source']['SSIM_gray240'] for z in rows]
  sh=[z['warehouse_source_15frame_shift']['PSNR_dB'] for z in rows]
  moves=[z['native_render_motion_gray_absmean_240'] for z in rows if z['native_render_motion_gray_absmean_240'] is not None]
  return {'source_alignment_PSN_dB_min':min(ps),'source_alignment_PSN_dB_median':round(statistics.median(ps),3),'source_alignment_PSN_dB_max':max(ps),'SSIM_gray240_median':round(statistics.median(ss),5),'PSNR_dB_less_than_25_count':sum(x<25 for x in ps),'shifted_source_15frame_PSN_dB_median':round(statistics.median(sh),3),'correct_minus_shifted_median_dB':round(statistics.median(ps)-statistics.median(sh),3),'native_pixel_motion_median':round(statistics.median(moves),5),'original_frames_count':len(rows)}
 sums={key:summary(val) for key,val in scores.items()}
 wh=scores['warehouse'];early=[x['native_render_motion_gray_absmean_240'] for x in wh[1:165]];hold=[x['native_render_motion_gray_absmean_240'] for x in wh[165:]]
 tail={'frames_after_source_camera_stop':45,'tail_last_45_decoded_adjacent_frame_gray_motion_median':round(statistics.median(hold),5),'early_first_165_decoded_adjacent_frame_gray_motion_median':round(statistics.median(early),5),'tail_motion_nonzero_frames':sum(x>0 for x in hold),'measures_are_pixels_not_camera_keyframes':True}
 video_pass=pts['warehouse']['count']==210 and pts['warehouse']['nonmonotonic_count']==0 and pts['warehouse']['frame_timebase_max_error_seconds']<1e-5 and sums['warehouse']['source_alignment_PSN_dB_min']>=25 and sums['warehouse']['source_alignment_PSN_dB_median']>=30 and sums['warehouse']['SSIM_gray240_median']>=.8
 score={'protocol_sha256':hash_file(protocol),'source_manifest_sha256':hash_file(ISOLATED/'source_manifest.json'),'video_sha256':EXPECTED_FINAL_SHA,'dock_negative_sha256':EXPECTED_DOCK_SHA,'PTS':pts,'summary':sums,'tail':tail,'source_matched_frame_sequence_property':'PASS' if video_pass else 'FAIL','silent_video_has_NO_AUDIO_SYNC_GOLD':'REVIEW','wrong_other_real_dock_video_is_a_valid_video_not_an_anomaly':True,'source_verification_complete':True,'contact_sheet_sha256':hash_file(contact),'contact_sheet_path':str(contact),'latency':times,'GPU_used':False,'source_keyframes_raw_SHA_receipts':[rec['sha256'] for rec in plan['original_source_PNGs'] if rec['native_render_frame'] in KEYFRAMES],'all_encoded_native_frames':scores,'limits':['Final encoded MP4 compared to its own author-rendered PNGs, so perfect provenance is not independent subjective visual quality gold','Wrong original DOCK movie and 15-frame shifted same movie are invalid references, not bad videos','Camera hold is source QA reference; PNG-frame motion cannot certify camera transforms or scene physical plausibility','Silent movie has no audio events and therefore cannot validate audiovisual synchronization','No independently rated natural defective owner films, no end-to-end viewer playback proven by this decode stage']}
 write_once(ISOLATED/'decoded_frame_evidence.json',score)
 print('FINAL_VIDEO_DECODER_COMPLETE',json.dumps({'source_property':score['source_matched_frame_sequence_property'],'summaries':sums,'PTS':pts,'tail':tail,'latency':times,'contact_sheet_SHA':score['contact_sheet_sha256']},sort_keys=True)[:5700],flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('freeze','decode'));globals()['do_decode' if ap.parse_args().stage=='decode' else 'freeze']()
