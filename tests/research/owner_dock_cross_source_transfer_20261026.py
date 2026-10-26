#!/usr/bin/env python3
"""Preregistered warehouse -> different owner DOCK video, no rule retuning.

Four separate stages: freeze, extract, encode, evaluate. Exact clean and
faulted actual final H.264 films all isolated in /data/var. No production
GPU/renders, no model inference, no calendar or gallery modifications.

Gold is SCRIPT AUTHORED from frozen source indices, NOT independent human
ratings of natural video faults. Interventions are on authentic final media.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

BASE=Path('/data/var/perceptual-independent-20261024')
ROOT=BASE/'owner_dock_cross_source_transfer_20261026'
SCENE=Path('/data/var/dock-film-v46/dock_real_OUTWARD_cabinet_and_CONTINUOUS_camera_v46.blend')
PNGROOT=Path('/data/var/dock-film-v46/continuous_real_camera_960x540_32samples')
QA=Path('/data/var/dock-film-v46/owner_review/3D-Dock-OUTWARD-Cabinet-Continuous-Real-Camera-v46-14s.owner_review_qa.json')
MOVIE=Path('/data/var/3d-publication-20261024/approved_dock_cinematic_preview_v46.mp4')
WAREHOUSE=Path('/data/var/3d-publication-20261024/approved_warehouse_door_rear_corner_v47.mp4')
SCENE_SHA='7c1bda0b4c78df455b609c1022b79e17c4e3bce48dbd6675c8d014425f97dbc4'
MOVIE_SHA='75ef01e7316d099759f4e1dccec304f4c3ea8a995949deebe7f6b41962c6ef93'
WAREHOUSE_SHA='e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb'
HISTORIC=BASE/'owner_real_encoded_fault_injection_20261026/frozen_design.json'
SOURCE_THRESHOLD=28.0
NOREF_THRESHOLD=1.0
NFRAMES=210
W=960;H=540;FPS=15


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def write_once(path,obj):
 if path.exists():raise RuntimeError('Previously frozen artifact, refuse overwrite: '+str(path))
 tmp=path.with_suffix('.incomplete')
 tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n');os.replace(tmp,path)

def run(args,timeout):
 t=time.monotonic();p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(f'Failed rc={p.returncode} for {args[:4]}: {p.stderr[-650:]}')
 return round(time.monotonic()-t,3),p.stdout,p.stderr

def probe(path):
 _,out,_=run(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,codec_type,avg_frame_rate,nb_frames,codec_name','-of','json',str(path)],25)
 x=json.loads(out);v=[s for s in x['streams'] if s['codec_type']=='video'];assert len(v)==1
 o=v[0];assert (o['width'],o['height'],o['avg_frame_rate'],int(o['nb_frames']))==(W,H,'15/1',210)
 assert abs(float(x['format']['duration'])-14)<.001
 return {'frames':int(o['nb_frames']),'width':o['width'],'height':o['height'],'fps':o['avg_frame_rate'],'duration_s':float(x['format']['duration']),'codec':o['codec_name']}

def choose_sites():
 # Source+protocol bound, ALL names + windows frozen before reading video pixels.
 # Distinct from warehouse's original 40,91,145 holds / 55-56,105-106,150-151 reversals.
 bands={'hold_a':(25,42),'reverse_a':(48,69),'hold_b':(77,97),'reverse_b':(106,125),'hold_c':(131,150),'reverse_c':(159,182)}
 pos={}
 for name,(lo,hi) in bands.items():
  pos[name]=min(range(lo,hi+1),key=lambda i:hashlib.sha256((SCENE_SHA+'|TRANSFER_DOCK|'+name+'|'+str(i)).encode()).hexdigest())
 holds={pos[k]:pos[k]-1 for k in ('hold_a','hold_b','hold_c')}
 swaps={}
 for k in ('reverse_a','reverse_b','reverse_c'):
  i=pos[k];swaps[i]=i+1;swaps[i+1]=i
 assert len(set(holds)|set(swaps))==9
 assert max(set(holds)|set(swaps))<=185
 return pos,{'clean_crf17':{},'clean_crf23':{},'holds_crf17':holds,'reversals_crf17':swaps}

def freeze():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'frozen_protocol.json').exists() or (ROOT/'source_manifest.json').exists():raise RuntimeError('freeze already in place')
 assert sha(MOVIE)==MOVIE_SHA and sha(SCENE)==SCENE_SHA and sha(WAREHOUSE)==WAREHOUSE_SHA
 status=json.loads((PNGROOT/'status.json').read_text());qa=json.loads(QA.read_text());prior=json.loads(HISTORIC.read_text())
 assert status['status']=='ALL_210_ACTUAL_MOVING_CAMERA_FRAMES_WITH_OUTWARD_DOORS_COMPLETE'
 assert status['scene_sha256']==SCENE_SHA and status['total_output_frames']==len(status['completed'])==210
 assert status['native_output_fps']==15 and status['camera_source_frames_per_output_frame']==2 and status['first_source_frame']==1
 assert status['resolution']==[960,540] and qa['video_sha256']==MOVIE_SHA and qa['native_fps']==15
 assert qa['status']=='OWNER_NONSTOP_TRUE_3D_CAMERA_PREVIEW_FULL_DECODE_AND_EVERY_FRAME_SHA_QA_PASS'
 assert prior['detector_A_source_bound'].startswith('native framewise RGB PSNR') and prior['detector_B_no_ref_low_motion'].startswith('gray 240x135')
 assert prior['detector_C_ffmpeg_freezedetect'].startswith('FFmpeg freezedetect')
 source=[]
 for i in range(1,211):
  row=status['completed'][str(i)];p=PNGROOT/f'frame_{i:04d}.png'
  assert p.is_file() and sha(p)==row['sha256'] and row['source_frame']==(i-1)*2+1
  source.append({'frame':i,'source_blend_frame':row['source_frame'],'sha256':row['sha256'],'bytes':row['bytes']})
 sites,variants=choose_sites()
 protocol={'date':'2026-10-26','role':'Frozen transfer from owner warehouse video to DIFFERENT actual DOCK handling-kit Blender scene and finished 210-frame film, source identities distinct','source_blend_SHA':SCENE_SHA,'original_approved_dock_final_MP4_SHA':MOVIE_SHA,'negative_other_product_WAREHOUSE_SHA':WAREHOUSE_SHA,'original_rack_warehouse_threshold_source_file_SHA':sha(HISTORIC),'frame_count':NFRAMES,'resolution':[W,H],'fps':FPS,'distinct_source_camera':'dock author QA 420 source frames continually moving unlike warehouse approved 3 sec hold','frozen_sites':sites,'variants_with_original_1based_source_frame_mappings':variants,'gold_type':'scripted media index manipulations frozen before decoding, not naturally occurring errors or independent human labels','detector_source_bound':'RGB PSNR to exact correctly aligned original decoded dock frame, candidate if <28dB','detector_no_reference':'same frozen warehouse grayscale mean abs adjacent 240x135 <1.0, prehold frames 2..165; dock camera never stops but transfer keeps original warehouse time eligibility, so last 45 deliberately exempted','detector_ffmpeg':'original FFmpeg freezedetect noise=-50dB minimum duration=.12s','primary_acceptance':{'clean_crf17': 'no source sequence candidate; clean correctly version-bound native H264 transfer', 'clean_crf23':'no source sequence candidate; transcode-only negative control different allowed compression level, note not production-approved codec contract','fault_recall_at_least':8,'scripted_fault_frames':9,'source_sequence_false_alarms_max_on_all_unedited_positions':6,'all_video_PTS':'210 frames, 15fps, monotonic; this is necessary but not sufficient','do_not_retune_thresholds':True},'encoder_CRFS':{'clean_crf17':17,'clean_crf23':23,'holds_crf17':17,'reversals_crf17':17},'encoder_other':['libx264','preset fast','threads 2','pix_fmt yuv420p','-an','-movflags +faststart'],'negative_controls':['dock original final vs original 210 Cycles PNGs','new H264 CRF17 reencode','new H264 CRF23 legitimate ordered-sequence encode','all unmodified frames in positive copies','wrong original warehouse film same format but different source must REVIEW, not auto-fail','dock true nonstop camera (no expected 3s hold)'],'uncertainty':'Clopper-Pearson 95% frame-level descriptive CIs plus group-level case counts; correlated frames from ONE new dock movie, no independent subject-level error-rate claim','no_GPU':True,'do_not_touch':'Production source Blender, original PNGs, original MP4, gallery, services, old denied video source transfer'}
 write_once(ROOT/'frozen_protocol.json',protocol)
 write_once(ROOT/'source_manifest.json',{'protocol_sha256':sha(ROOT/'frozen_protocol.json'),'dock_MP4_SHA':MOVIE_SHA,'dock_scene_SHA':SCENE_SHA,'dock_render_status_SHA':sha(PNGROOT/'status.json'),'dock_author_QA_SHA':sha(QA),'previous_warehouse_protocol_SHA':sha(HISTORIC),'original_210_source_pngs':source,'original_dock_probe':probe(MOVIE),'negative_other_warehouse_probe':probe(WAREHOUSE)})
 print('DOCK_CROSS_SOURCE_FROZEN',sha(ROOT/'frozen_protocol.json'),'frames',len(source),'sites',sites,'variants',{k:sorted(v) for k,v in variants.items()})

def extract():
 if (ROOT/'canonical_source_decoded.json').exists():raise RuntimeError('source extraction already completed')
 f=json.loads((ROOT/'source_manifest.json').read_text());assert f['dock_MP4_SHA']==sha(MOVIE)==MOVIE_SHA
 frames=ROOT/'canonical_dock_final_decode'
 if frames.exists():raise RuntimeError('Prior frame extraction directory exists; do not overwrite')
 frames.mkdir()
 cmd=['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(MOVIE),'-vsync','0','-start_number','1','-frames:v','210',str(frames/'frame_%04d.png')]
 dt,_,_=run(cmd,150)
 paths=sorted(frames.glob('frame_*.png'))
 assert len(paths)==210
 result={'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'source_MP4_sha256':MOVIE_SHA,'runtime_extract_sec':dt,'decoded_frame_sha256':{str(i):sha(p) for i,p in enumerate(paths,1)},'decoder_frame_count':len(paths)}
 write_once(ROOT/'canonical_source_decoded.json',result)
 print('DOCK_FINAL_OWNER_MOVIE_DECODED',dt,'sec','frames',len(paths),'sha',sha(ROOT/'canonical_source_decoded.json'))

def encode():
 if (ROOT/'encodes.json').exists():raise RuntimeError('encodes already completed')
 frozen=json.loads((ROOT/'frozen_protocol.json').read_text());src=json.loads((ROOT/'canonical_source_decoded.json').read_text())
 assert sha(MOVIE)==MOVIE_SHA
 frames=ROOT/'canonical_dock_final_decode'
 out=ROOT/'encoded_variants'
 if out.exists():raise RuntimeError('encoding partially started, inspect before repeating')
 out.mkdir()
 rec={}
 for name,maps in frozen['variants_with_original_1based_source_frame_mappings'].items():
  links=ROOT/('links_'+name)
  if links.exists():raise RuntimeError('links already present; no replay')
  links.mkdir()
  for i in range(1,211):
   s=maps.get(str(i),i) if isinstance(next(iter(maps),None),str) else maps.get(i,i)
   target=frames/f'frame_{s:04d}.png';assert target.is_file()
   assert sha(target)==src['decoded_frame_sha256'][str(s)]
   (links/f'frame_{i:04d}.png').symlink_to(target)
  dest=out/(name+'.mp4');crf=int(frozen['encoder_CRFS'][name])
  dt,_,_=run(['ffmpeg','-nostdin','-hide_banner','-v','error','-framerate','15','-start_number','1','-i',str(links/'frame_%04d.png'),'-frames:v','210','-an','-c:v','libx264','-preset','fast','-threads','2','-crf',str(crf),'-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],220)
  rec[name]={'path':str(dest),'sha256':sha(dest),'original_source_frame_override_positions':[int(x) for x in maps],'H264_CRF':crf,'encode_wall_s':dt,'probe':probe(dest)}
  print('DOCK_FINISHED_MP4',name,'encoded_sec',dt,'sha',rec[name]['sha256'],'source_edits',rec[name]['original_source_frame_override_positions'],flush=True)
 assert len(rec)==4
 write_once(ROOT/'encodes.json',{'frozen_protocol_sha256':sha(ROOT/'frozen_protocol.json'),'canonical_source_decoded_SHA256':sha(ROOT/'canonical_source_decoded.json'),'original_dock_MOVIE_sha256':MOVIE_SHA,'variants':rec})
 print('DOCK_ENCODES_LOCKED',sha(ROOT/'encodes.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','extract','encode'));globals()[p.parse_args().stage]()
