#!/usr/bin/env python3
"""Publish verified provenance for two READ-ONLY owner v52 video signals.

Uses ONLY earlier completed and verified read-only reports. Does not revisit
uncertain/denied source-resolution tail-framehash check. No video decoder,
model/GPU, product edit or verification relabel. This is numerical signal
evidence, NOT a pass/fail video quality label.
"""
from pathlib import Path
import json,hashlib
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_product_video')
OUT=Path(__file__).resolve().parents[2]/'docs/research/perceptual-owner-video-verified-signal-2026-10-25.json'

def load(file):
 b=(ROOT/file).read_bytes();return json.loads(b),hashlib.sha256(b).hexdigest()

def main():
 if OUT.is_file():print('SAVED_AGGREGATE_PRESERVED');return
 pts,hpts=load('owner_prod_v52_presentation_timestamps_20261025.json')
 moves,hmove=load('owner_prod_v52_per_frame_motion_signal_20261025.json')
 names=['warehouse_v52','dock_v52'];rows={}
 for name in names:
  a=pts['results'][name];b=moves['videos'][name]
  assert a['decoded_timestamp_frames']==420 and int(a['declared_stream']['nb_frames'])==420
  assert a['declared_stream']['avg_frame_rate']=='30/1'
  assert len(a['missing_time_intervals_larger_than_1p5_frame'])==0
  assert len(a['duplicate_pts_indices'])==0
  assert b['source']['source_bytes']==a['bytes'] and b['source']['mtime_ns']==a['mtime_ns']
  s=b['summary'];assert s['adjacent_frame_pairs']==419
  coarse_dups=[x['frame_number'] for x in b['signal_per_pair'] if x['identical_frame']]
  assert len(coarse_dups)==s['exact_pixel_identical_adjacent_pairs']
  rows[name]={'source_mp4_sha256':b['source']['source_file_sha256'],'source_mp4_bytes':b['source']['source_bytes'],'nominal_fps':30,'encoded_video_frames':420,'encoded_pts_min_delta_s':a['minimum_dt_s'],'encoded_pts_max_delta_s':a['maximum_dt_s'],'encoded_missing_pts_gaps':0,'encoded_duplicate_pts':0,'coarse_gray_resolution_wh':[256,144],'coarse_near_identical_adjacent_pairs':len(coarse_dups),'coarse_gray_repeat_first_to_last_frame_indexes':[min(coarse_dups),max(coarse_dups)] if coarse_dups else [],'flow_computable_pairs':s['affine_success'],'flow_median_px_at_coarse_scale':s['median_flow_px_q'][3],'large_local_motion_outlier_candidates':len(s['large_feature_displacement_robust_local_outliers']),'source_resolution_exact_frame_repeat_count':'NOT_VERIFIED; a separate requested tail-framehash run result retrieval was explicitly denied, not bypassed; do not interpret grayscale matches as original 1080p duplicates.'}
 assert rows['warehouse_v52']['coarse_near_identical_adjacent_pairs']==20
 assert rows['dock_v52']['coarse_near_identical_adjacent_pairs']==0
 assert rows['warehouse_v52']['coarse_gray_repeat_first_to_last_frame_indexes']==[396,419]
 result={'date':'2026-10-25','scope':'actual completed warehouse and dock Cycles motion-blur 1080p 30fps v52; source files left untouched','source_report_sha256':{'video_pts':hpts,'low_resolution_pixel_flow':hmove},'results':rows,'interpretation':'Both actual encoded outputs have regular 30 fps presentation timing and zero PTS gaps, ruling out irregular PTS as immediate cause of visible stuttering in these two files. Warehouse closing frames are nearly static only AFTER downscale to grayscale, with 20 near-identical frame pairs close to the end; dock has zero such coarse matches. This might be an intentional outro, static camera/object, tiny original image variations or render cadence; source-resolution exact duplication and subjective perceptual smoothness were NOT verified. No author-rated good/bad reference for these two clips. A separate tail-source-resolution run output was denied; no retry or workaround was attempted.','production_write_effects':'none, read-only videos'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('VERIFIED_OWNED_SOURCE_VIDEO_AGGREGATE',OUT,'bytes',OUT.stat().st_size)
 for k,v in rows.items():print(k,'PTS_GAPS',v['encoded_missing_pts_gaps'],'coarse_duplicates',v['coarse_near_identical_adjacent_pairs'],'median_motion_px',v['flow_median_px_at_coarse_scale'])
if __name__=='__main__':main()
