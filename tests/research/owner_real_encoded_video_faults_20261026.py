#!/usr/bin/env python3
"""Frozen real-owner-video fault injection and final encoded-output comparison.

Synthetic frame-order faults are independently fixed BEFORE image decoding.
These are NOT naturally arising glitches or human subjectively adjudicated
visual errors. Original owner MP4/source .blend/PNGs are READ ONLY.
All derived video bytes stay outside project Git under /data/var.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import cv2
import numpy as np

ORIG=Path('/data/var/3d-publication-20261024/approved_warehouse_door_rear_corner_v47.mp4')
ORIG_SHA='e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb'
SCENE=Path('/data/var/warehouse-film-v47/warehouse_THROUGH_OPENING_DOOR_to_REAR_CORNER_three_second_HOLD_v47.blend')
SCENE_SHA='6609f09ac0637dd232539f7d5de49ce3e59dc477a5bf25beb125ddcbf73849b6'
PREVIOUS=Path('/data/var/perceptual-independent-20261024/final_encoded_owner_mp4_20261026')
ROOT=Path('/data/var/perceptual-independent-20261024/owner_real_encoded_fault_injection_20261026')
FRAMECOUNT=210
INJECT={
 'clean_reencoded':{},
 'single_frame_holds':{40:39,91:90,145:144},
 'short_reversals':{55:56,56:55,105:106,106:105,150:151,151:150},
}
DESIGN={
 'date':'2026-10-26','dataset':'owner ACTUAL approved final warehouse Cycles flythrough encoded MP4 SHA locked',
 'source':'Exact approved production-adjacent v47 final 14.0s 960x540 H264 @15fps 210 frames, with independently verified 210 source Cycles PNG hashes',
 'source_MP4_sha256':ORIG_SHA,'source_scene_sha256':SCENE_SHA,'source_receipt_sha256_requirement':'prior final_encoded_owner_mp4_20261026 source_manifest and decoded output verify 210 frames and final browser ended',
 'ground_truth':'Independently fixed proposed frame source-index edits in this PRE-DECODE manifest; script-authored synthetic interventions, not independent human-rated real natural faults',
 'indices_1_based':True,'planned_mutations':INJECT,
 'positive_video_counts':{'single_frame_holds':3,'short_reversals':6},
 'true_negative_video':'clean_reencoded is newly reencoded from the exact same decoded original MP4, not unmodified production MP4',
 'true_negative_time':'approved last three seconds (frames 166-210) have a deliberately stationary CAMERA, even if pixels/animation continue moving slightly',
 'encoder_identical_all_cases':'ffmpeg H264 libx264 preset fast CRF17 threads2 pix_fmt yuv420p 15fps 210 frames movflags faststart, -an',
 'detector_A_source_bound':'native framewise RGB PSNR versus same-index canonical decoded approved video PNG; auto-fail source-sequence mismatch only for verified exact source/video identity, threshold <28dB',
 'detector_B_no_ref_low_motion':'gray 240x135 mean adjacent abs pixel difference <1.0 within frames 2..165, flag temporal suspect frame only; outside this motion window proposal/review, not fail',
 'detector_C_ffmpeg_freezedetect':'FFmpeg freezedetect noise -50dB minimum 0.12 sec, full actual encoded output; count events and false detections on clean and permitted static-camera tail',
 'frozen_thresholds_before_new_video':'prior frozen approved source-aligned PSNR minimum 33.28dB, chosen 28 dB margin for second encode; no-ref motion 1.0 relative to prior moving median 13.12, tail median .00071',
 'reference_precondition':'original approved final MP4 SHA, original 210 frame PNG SHA and scene SHA must match; otherwise abstain rather than auto-fail',
 'decisive_test':'real encoded playable output must fail only exact source contract on injected edits while clean second-generation encoding and legitimate camera hold remain negatives',
 'limitations':'Temporal faults are scripted changes to REAL finished owner film, no natural unforeseen recording glitch or human subjective visual-quality gold; 3+6 frames highly correlated within one scene, not independent deployment estimates',
 'do_not_modify':'original MP4, source Blender, native Cycles PNGs, production gallery, active service, old v52 explicitly denied retrieval',
}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def save(path,obj):
 if path.exists():raise RuntimeError('FROZEN_ALREADY: '+str(path))
 tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
 os.replace(tmp,path)

def run(command,timeout):
 t=time.monotonic();p=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(f'cmd failed rc={p.returncode}, err={p.stderr[-450:]}')
 return round(time.monotonic()-t,3),p.stdout

def probe(path):
 _,out=run(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames,avg_frame_rate,codec_type:format=duration,size','-of','json',str(path)],20)
 info=json.loads(out);v=[s for s in info['streams'] if s.get('codec_type')=='video']
 assert len(v)==1
 s=v[0]
 assert (s['width'],s['height'],s['avg_frame_rate'],int(s['nb_frames']))==(960,540,'15/1',210)
 assert abs(float(info['format']['duration'])-14)<.0001
 return {'duration_sec':float(info['format']['duration']),'frame_count':int(s['nb_frames']),'fps':s['avg_frame_rate'],'codec':s['codec_name'],'bytes':int(info['format']['size'])}

def freeze():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'frozen_design.json').exists() or (ROOT/'source_receipts.json').exists():raise RuntimeError('original freeze exists')
 assert sha(ORIG)==ORIG_SHA and sha(SCENE)==SCENE_SHA
 prev=json.loads((PREVIOUS/'decoded_frame_evidence.json').read_text())
 assert prev['video_sha256']==ORIG_SHA and prev['source_matched_frame_sequence_property']=='PASS'
 assert prev['PTS']['warehouse']['count']==210
 native=json.loads((PREVIOUS/'source_manifest.json').read_text())
 assert native['source_scene_sha256']==SCENE_SHA and len(native['original_source_PNGs'])==210
 for name,mods in INJECT.items():
  assert all(1<=i<=FRAMECOUNT and 1<=src<=FRAMECOUNT for i,src in mods.items())
  assert all(i!=src for i,src in mods.items())
  assert all(i<=165 for i in mods)
 save(ROOT/'frozen_design.json',DESIGN)
 save(ROOT/'source_receipts.json',{'frozen_design_sha256':sha(ROOT/'frozen_design.json'),'prior_source_manifest_sha256':sha(PREVIOUS/'source_manifest.json'),'prior_final_video_pixel_evidence_sha256':sha(PREVIOUS/'decoded_frame_evidence.json'),'video_sha256':ORIG_SHA,'source_blend_sha256':SCENE_SHA,'source_clean_probe':probe(ORIG),'gold_edit_masks':{k:sorted(v) for k,v in INJECT.items()},'author_intent_hold_indices_1based':[166,210]})
 print('REAL_OWNER_MEDIA_INJECTION_FROZEN',sha(ROOT/'source_receipts.json'),'gold masks', {k:sorted(v) for k,v in INJECT.items()})

def extract():
 if (ROOT/'source_frames.json').exists():raise RuntimeError('Source decoder already executed, no retry')
 receipts=json.loads((ROOT/'source_receipts.json').read_text())
 assert receipts['frozen_design_sha256']==sha(ROOT/'frozen_design.json')
 assert sha(ORIG)==ORIG_SHA
 frames=ROOT/'canonical_decoded_png'
 if frames.exists():raise RuntimeError('Source frame directory already exists; inspect uncertain effects')
 frames.mkdir()
 cmd=['ffmpeg','-nostdin','-hide_banner','-v','error','-threads','2','-i',str(ORIG),'-vsync','0','-start_number','1','-frames:v','210',str(frames/'frame_%04d.png')]
 dt,_=run(cmd,150)
 fs=sorted(frames.glob('frame_*.png'))
 if len(fs)!=210:raise RuntimeError('Source decode yielded '+str(len(fs)))
 digest={str(i):sha(p) for i,p in enumerate(fs,start=1)}
 from PIL import Image
 for i in (1,40,91,145,166,210):
  with Image.open(frames/f'frame_{i:04d}.png') as im:assert im.size==(960,540)
 save(ROOT/'source_frames.json',{'frozen_receipts_sha256':sha(ROOT/'source_receipts.json'),'source_video_sha256':ORIG_SHA,'decode_wall_s':dt,'exact_png_hashes':digest,'all_210_frames_complete':True,'original_prod_video_modified':False})
 print('DECODED_REAL_OWNER_VIDEO_TO_ISOLATED_SOURCE',len(fs),'sec',dt,'sha',sha(ROOT/'source_frames.json'))

def encode():
 if (ROOT/'encodes.json').exists():raise RuntimeError('encodes already finalized, not repeating')
 first=json.loads((ROOT/'source_frames.json').read_text());assert first['source_video_sha256']==ORIG_SHA
 videos={};orig=ROOT/'canonical_decoded_png';encdir=ROOT/'final_encoded_variants'
 if encdir.exists():raise RuntimeError('prior output directory: inspect partial outputs first')
 encdir.mkdir()
 for name,mapping in INJECT.items():
  links=ROOT/('input_'+name)
  if links.exists():raise RuntimeError('prior input links exist')
  links.mkdir()
  for i in range(1,211):
   from_frame=mapping.get(i,i)
   reference=orig/f'frame_{from_frame:04d}.png';p=links/f'frame_{i:04d}.png'
   assert sha(reference)==first['exact_png_hashes'][str(from_frame)]
   p.symlink_to(reference)
  path=encdir/(name+'.mp4')
  dt,_=run(['ffmpeg','-nostdin','-hide_banner','-v','error','-framerate','15','-start_number','1','-i',str(links/'frame_%04d.png'),'-frames:v','210','-an','-c:v','libx264','-preset','fast','-threads','2','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],210)
  videos[name]={'path':str(path),'sha256':sha(path),'video':probe(path),'encode_wall_s':dt,'edit_positions':sorted(mapping)}
  print('REAL_FINAL_OWNER_CONTROLLED_VARIANT',name,'sec',dt,'bytes',path.stat().st_size,'sha',videos[name]['sha256'],flush=True)
 save(ROOT/'encodes.json',{'frozen_receipt_sha256':sha(ROOT/'source_receipts.json'),'source_frame_receipt_sha256':sha(ROOT/'source_frames.json'),'variants':videos,'all_variants_same_encoder_settings':True,'original_video_remains_unmodified':sha(ORIG)==ORIG_SHA})
 print('ENCODES_FROZEN',sha(ROOT/'encodes.json'))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','extract','encode'));globals()[parser.parse_args().stage]()
