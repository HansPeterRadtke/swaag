#!/usr/bin/env python3
"""Untrained temporal anomaly signals on ORIGINAL third-party edited footage.

Source Adobe VideoSham release v1.0 38 fixed selected original Vimeo scenes:
19 authentic-video temporal frame/speed/reversal edits versus 19 matched
other edits (audio, text, object, background) with same source control. Every
original and edited MP4 was extracted by public HTTP ZIP range and CRC/SHA.
No source author Task/manipulation labels reach pixel processing functions.

Freeze ONLY original video pixels and derived video features BEFORE source gold
join, no search/tuning. Two complementary modes:
 no-reference: ≥3 adjacent near-static grayscale transitions bracketed by real
 motion and local high frame-difference spikes;
 reference-aware: compare 64-bit DCT perceptual frame hashes of the edited
 video against its original matching source clip and estimate departure from
 1x normal wall-clock source-frame progression at the video fps. Content
 unmatched (<25% nearest pHash distance<=12 bits) is ABSTAIN, never 'good'.
 Source and edited video FPS and duration also saved as potentially confounding
 metadata baselines; no clip quality judgement from duration alone.

Hardware: Nitro CPU nice level nineteen, FFMPEG thread1, cv2 one thread, no
GPU/model or original source media modifications. This is an author-labelled
forgery category benchmark, NOT human-objectionable or natural playback freeze.
Video frames are actual Vimeo footage/edits, not unit-test synthetic copies.
No media/raw hashes/third-party labels distributed inside Git.
"""
from pathlib import Path
import json,hashlib,subprocess,time,datetime,resource,math,statistics,collections,os
import numpy as np,cv2
SRC=Path('/data/var/perceptual-independent-20261024/videosham-20261025')
MAN=SRC/'source_selection_original_38_video_pairs_20261025.json'
RECEIPT=SRC/'selected_video_member_receipts_20261025.json'
RAW=SRC/'38_real_clip_source_temporal_pixels_frozen_20261025.json'
W,H=128,72
CFG={'source':'Adobe VideoSham release v1.0 38 source-independent original/edited Vimeo pairs','license':'Adobe Research License v1.1a noncommercial', 'source_original_and_altered_human_temporal_labels_NOT_read_by_inference':True,'video_frames':'Decode each native FPS original and processed sequence, grayscale 128x72 with ffmpeg6 scale area, no rate conversion, max 2100 original frames/clip, ffmpeg -vsync 0','phash':'64-bit 32x32 DCT pHash 8x8 threshold of AC coefficients, no fit; Hamming distance to every original source-frame','no_ref_freeze':'at least 3 consecutive adjacent scaled grayscale frame MAE<0.8, with preceding and following five frame context moving >3.0 grayscale units (median); read max across intervals','no_ref_cut_jerk':'frame difference outlier > max(8, median(diff)+6*1.4826*MAD), fraction of all valid transitions','ref_match_distance_threshold_bits':12,'ref_min_valid_frame_fraction':0.25,'ref_expected_timing_source_fps_divided_by_processed_fps':'unit real physical clock progression plus arbitrary source segment start offset','ref_bad_frame_alignment_threshold_original_frames':4.,'ref_min_abnormal_fraction_for_flag':0.20,'ref_source_motion_quality_requirement':'at least 0.15 fraction of source pHash adjacent frames differ by two or more bits','reference_selection':'known exact associated original source ID, no author task labels used','duration_ratio_baseline':'absolute log edit/original duration > log(1.3) flagged; separate confound not accepted video perception','cv2_version':cv2.__version__,'numpy':np.__version__}


def probe(path):
 command=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(path)]
 doc=json.loads(subprocess.run(command,capture_output=True,text=True,check=True,timeout=13).stdout)
 s=doc['streams'][0]
 fps=s['avg_frame_rate'];a,b=map(float,fps.split('/'))
 return {'frame_rate':a/b,'frame_rate_text':fps,'width':int(s['width']),'height':int(s['height']),'reported_nb_frames':int(s.get('nb_frames') or 0),'duration_seconds':float(s['duration'])}

def read_frames(path,info):
 args=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-i',str(path),'-an','-vf','scale=128:72:flags=area','-vsync','0','-pix_fmt','gray','-f','rawvideo','pipe:1']
 p=subprocess.run(args,capture_output=True,timeout=55)
 if p.returncode:raise RuntimeError('ffmpeg decoder failed '+p.stderr.decode(errors='replace')[-140:])
 size=W*H
 if len(p.stdout)%size:raise RuntimeError('partial raw video frame in output')
 n=len(p.stdout)//size
 if not 6<=n<=2100:raise RuntimeError(f'frame count out of bounds {n}')
 if info['reported_nb_frames'] and abs(n-info['reported_nb_frames'])>2:raise RuntimeError(f'coded/decoded count mismatch {n}/{info["reported_nb_frames"]}')
 return np.frombuffer(p.stdout,dtype=np.uint8).reshape(n,H,W)

def phash(frame):
 gray=cv2.resize(frame,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
 dct=cv2.dct(gray)[:8,:8];threshold=float(np.median(dct.ravel()[1:]))
 flag=(dct>threshold).reshape(64);flag[0]=False
 return int.from_bytes(np.packbits(flag).tobytes(),'big')

def features(frames,fps):
 t=time.monotonic();arr=frames.astype(np.int16)
 diff=np.abs(arr[1:]-arr[:-1]).mean(axis=(1,2))
 hashes=np.array([phash(f) for f in frames],dtype=np.uint64)
 median=float(np.median(diff));mad=float(np.median(np.abs(diff-median)))
 cutoff=max(8.,median+6*1.4826*mad)
 spike=(diff>cutoff).astype(np.uint8)
 low=(diff<.8)
 maxrun=0;runs=[];i=0
 while i<len(low):
  if not low[i]:i+=1;continue
  start=i
  while i<len(low) and low[i]:i+=1
  end=i
  context_left=diff[max(0,start-5):start];context_right=diff[end:min(len(diff),end+5)]
  dynamic=bool(len(context_left)>0 and len(context_right)>0 and np.median(context_left)>3. and np.median(context_right)>3.)
  runs.append({'index_after_frame':start+1,'length_adjacent_deltas':end-start,'seconds_approx':round((end-start)/fps,4),'bracketed_by_motion':dynamic})
  if dynamic:maxrun=max(maxrun,end-start)
 hashdiff=np.bitwise_count(hashes[1:]^hashes[:-1])
 return {'frame_count':len(frames),'fps':fps,'pHash64_hex':[f'{int(x):016x}' for x in hashes],'source_same_grayscale_threshold_0p8_count':int(np.sum(low)),'true_motion_bracketed_run_max':maxrun,'no_ref_hold_flag':maxrun>=3,'no_ref_jerk_fraction':float(np.mean(spike)),'no_ref_jerk_flag':float(np.mean(spike))>=.08,'frame_diff_median':median,'frame_diff_mad':mad,'frame_diff_max':float(np.max(diff)),'frame_diff_jump_cutoff':cutoff,'adjacent_dhash_motion_at_least_2_bits_fraction':float(np.mean(hashdiff>=2)),'selected_bracketed_freeze_runs':[r for r in runs if r['bracketed_by_motion']][:25],'elapsed_s':round(time.monotonic()-t,2)}

def reference_features(ref,edit,original,altered):
 # Compute all pairwise source-frame matches from actual image content;
 # labels aren't consulted. Use np.bitwise_count on broadcast uint64 bitmaps.
 a=np.array([int(x,16) for x in ref['pHash64_hex']],dtype=np.uint64)
 b=np.array([int(x,16) for x in edit['pHash64_hex']],dtype=np.uint64)
 dist=np.bitwise_count(b[:,None]^a[None,:]);j=np.argmin(dist,axis=1)
 mindist=np.min(dist,axis=1)
 matched=mindist<=CFG['ref_match_distance_threshold_bits']
 validfrac=float(np.mean(matched));matched_j=j[matched];source_motion=ref['adjacent_dhash_motion_at_least_2_bits_fraction']
 abstain_reason=None
 if validfrac<CFG['ref_min_valid_frame_fraction']:abstain_reason='too_few_actual_content_matches'
 elif source_motion<.15:abstain_reason='authentic_source_content_nearly_static'
 if abstain_reason:
  return {'status':'abstain','reason':abstain_reason,'valid_pHash_matches_n':int(np.sum(matched)),'valid_matches_fraction':validfrac,'min_hamming_p50':float(np.median(mindist)),'source_content_motion_fraction':source_motion,'ref_original_duration_sec':original['duration_seconds'],'edited_duration_sec':altered['duration_seconds'],'fps_original':original['frame_rate'],'fps_edited':altered['frame_rate']}
 steps=float(original['frame_rate']/altered['frame_rate'])
 idx=np.arange(len(b))[matched]
 offset=float(np.median(matched_j-idx*steps))
 resid=np.abs(matched_j-idx*steps-offset)
 # The fixed 4 source frames residual threshold is an absolute timeline
 # alignment diagnostic. A trim aligned by one unknown constant source offset
 # remains no-error, while inserted/freezed/reversed sections deviate.
 abnormal=float(np.mean(resid>CFG['ref_bad_frame_alignment_threshold_original_frames']))
 local_jump_ratio=None
 if len(matched_j)>4:
  jj=np.asarray(matched_j,dtype=np.float64)
  if len(jj)>4:
   d=np.diff(jj);idxdiff=np.diff(idx)
   rate=d/np.maximum(1,idxdiff)
   local_jump_ratio=float(np.mean((rate<0.)|(rate>steps*1.5)|(rate<steps*.5)))
 return {'status':'scorable','valid_pHash_matches_n':int(np.sum(matched)),'valid_matches_fraction':validfrac,'min_hamming_p50':float(np.median(mindist)),'min_hamming_p90':float(np.quantile(mindist,.9)),'source_content_motion_fraction':source_motion,'aligned_unknown_reference_start_offset_frames':offset,'ref_abnormal_deviation_gt_4_frames_fraction':abnormal,'ref_abnormal_flag':abnormal>.20,'ref_abs_alignment_error_median_in_source_frames':float(np.median(resid)),'ref_local_jump_anomaly_fraction':local_jump_ratio,'ref_original_duration_sec':original['duration_seconds'],'edited_duration_sec':altered['duration_seconds'],'fps_original':original['frame_rate'],'fps_edited':altered['frame_rate']}


def gather():
 if not SRC.exists():raise RuntimeError('source dataset not downloaded')
 manifest=json.loads(MAN.read_text());receipts=json.loads(RECEIPT.read_text())
 if RAW.exists():
  state=json.loads(RAW.read_text())
  if state['status']=='complete':print('PREVIOUS_FULL_VIDEO_CONTENT_FEATURES_FROZEN');return
 else:state={'created':'2026-10-25','status':'in_progress','config':CFG,'source_selection_SHA256':hashlib.sha256(MAN.read_bytes()).hexdigest(),'source_download_receipts_SHA256_at_start':hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),'cases':{}}
 assert state['config']==CFG and state['source_selection_SHA256']==hashlib.sha256(MAN.read_bytes()).hexdigest()
 t0=time.monotonic();cv2.setNumThreads(1)
 for entry in manifest['items']:
  ident=str(entry['source_video_id'])
  if ident in state['cases']:continue
  # Only source IDs, original/processed paths & true image bytes used in
  # pixel features. Never read entry['gold_temporal_intent'] or Task here.
  if not all(k+':'+ident in receipts['completed'] for k in ['original','processed']):continue
  row={'status':'pending_or_uncertain'};state['cases'][ident]=row;RAW.write_text(json.dumps(state,indent=2)+'\n')
  try:
   data={}
   for side in ['original','processed']:
    path=SRC/receipts['completed'][side+':'+ident]['relative_path']
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=receipts['completed'][side+':'+ident]['SHA256']:raise RuntimeError('source media sha mismatch')
    info=probe(path);frames=read_frames(path,info)
    data[side]={'video_info':info,'signals':features(frames,info['frame_rate']),'source_mp4_sha256':sha}
    del frames
   compare=reference_features(data['original']['signals'],data['processed']['signals'],data['original']['video_info'],data['processed']['video_info'])
   if 'pHash64_hex' in data['original']['signals']:pass # keep raw image fingerprint source out of project Git
   ratio=data['processed']['video_info']['duration_seconds']/data['original']['video_info']['duration_seconds']
   row.update({'status':'complete','original':data['original'],'processed':data['processed'],'reference':compare,'duration_ratio':ratio,'duration_confounded_flag':abs(math.log(ratio))>math.log(1.3),'no_ref_combined_flag':data['processed']['signals']['no_ref_hold_flag'] or data['processed']['signals']['no_ref_jerk_flag']})
  except Exception as e:row.update(status='error_or_ambiguous_no_retry',error=type(e).__name__+': '+str(e)[:220])
  RAW.write_text(json.dumps(state,indent=2)+'\n')
  print('FROZEN_ACTUAL_ORIGINAL_AND_EDITED_CLIP_FEATURES',len(state['cases']),'/',len(manifest['items']),ident,'STATUS',row['status'],'ORIGINAL_FRAMES',row.get('original',{}).get('signals',{}).get('frame_count'),'EDIT_FRAMES',row.get('processed',{}).get('signals',{}).get('frame_count'),'REFERENCE_STATUS',row.get('reference',{}).get('status'),'SECONDS_ELAPSED',round(time.monotonic()-t0,2),flush=True)
 if len(state['cases'])==manifest['source_pairs_expected']:
  state['status']='complete'
  state['processing_elapsed_seconds']=round(time.monotonic()-t0,2)
  state['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  RAW.write_text(json.dumps(state,indent=2)+'\n')
  print('FULL_VIDEO_ORIGINAL_AND_EDITED_PIXEL_EVIDENCE_FROZEN',len(state['cases']),'SECONDS',state['processing_elapsed_seconds'],'STATUS',dict(collections.Counter(v['status'] for v in state['cases'].values())),flush=True)
 else:
  print('PARTIAL_VIDEO_CONTENT_FEATURES_SAVED',len(state['cases']),'/38','PENDING_INPUT_MEDIA',len(manifest['items'])-len(state['cases']))

if __name__=='__main__':gather()
