#!/usr/bin/env python3
"""Predeclared OFFICIAL FFmpeg freeze detector on original third-party MP4s.

Adobe VideoSham release first 179 scenes, 38 fixed author temporally edited vs
other edited source videos. This tool sees only selected 38 processed video
PIXELS with `ffmpeg -vf freezedetect=n=-50dB:d=0.12` at native frame FPS,
no original control/reference and no source manipulation labels.

FFmpeg 6.1.1 built-in filter, not a learned vision model. Full frame decode
CPU one filter thread, no CUDA, nice19. Extract detected start/end seconds,
count clips with any event >=0.12s and freeze duration. Freeze can be natural
stasis or approved edit and not necessarily an unplanned visual defect.
Time truth unknown, event F1 cannot be scored. All labels joined later by a
separate source-human edit-type scorer; detection raw frozen first.
Original licensed media and full ffmpeg output stay outside project Git.
"""
from pathlib import Path
import subprocess,re,json,hashlib,time,datetime,resource,collections
ROOT=Path('/data/var/perceptual-independent-20261024/videosham-20261025')
SELECT=ROOT/'source_selection_original_38_video_pairs_20261025.json'
RECEIPT=ROOT/'selected_video_member_receipts_20261025.json'
OUT=ROOT/'videosham_38_external_real_ffmpeg_freezedetect_20261025.json'
CONFIG={'source':'Adobe VideoSham original public edited videos 38 author label stratified selected video family IDs','ffmpeg_version':'FFmpeg 6.1.1-3ubuntu5','filter':'freezedetect=n=-50dB:d=0.12','freezedetect_minimum_duration_s':.12,'pixel_noise_tolerance_dB':-50.,'positive_if_at_least_one_event':True,'timeout_seconds_per_video':40,'decode':'CPU threads1 filter_threads1 entire native input no audio and no frame rate conversion','model_input':'processed authored video ONLY no reference and no editorial intent task label','run_kind':'predeclared source-blind frozen predictions, no threshold search'}

def detect(path):
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-threads','1','-filter_threads','1','-i',str(path),'-an','-vf',CONFIG['filter'],'-f','null','-']
 t=time.monotonic()
 p=subprocess.run(cmd,capture_output=True,timeout=CONFIG['timeout_seconds_per_video'])
 if p.returncode:raise RuntimeError('ffmpeg failed exit '+str(p.returncode))
 txt=p.stderr.decode('utf8','replace')
 begins=[float(x) for x in re.findall(r'freeze_start:\s*([-+]?(?:\d*\.)?\d+)',txt)]
 ends=[float(x) for x in re.findall(r'freeze_end:\s*([-+]?(?:\d*\.)?\d+)',txt)]
 lengths=[float(x) for x in re.findall(r'freeze_duration:\s*([-+]?(?:\d*\.)?\d+)',txt)]
 # Some events extend to file end, no freeze_end. Keep explicit unknown end.
 return {'status':'complete','freeze_start_seconds':begins,'freeze_end_seconds':ends,'freeze_duration_seconds':lengths,'event_count':len(begins),'event_positive_flag':len(begins)>0,'known_freeze_duration_sum_seconds':sum(lengths),'max_recorded_complete_freeze_seconds':max(lengths,default=0),'had_unclosed_freeze_event':len(begins)>len(ends),'elapsed_seconds':round(time.monotonic()-t,2)}

def main():
 if OUT.exists():
  state=json.loads(OUT.read_text())
  if state['status']=='complete':print('PRIOR_FFMPEG_SOURCE_RESULT_COMPLETED_NOT_REPLAYED');return
 else:state={'status':'in_progress','config':CONFIG,'source_selection_sha256':hashlib.sha256(SELECT.read_bytes()).hexdigest(),'results':{},'created':'2026-10-25'}
 assert state['config']==CONFIG and state['source_selection_sha256']==hashlib.sha256(SELECT.read_bytes()).hexdigest()
 manifest=json.loads(SELECT.read_text());receipts=json.loads(RECEIPT.read_text())
 t0=time.monotonic()
 for case in manifest['items']:
  ident=str(case['source_video_id'])
  if ident in state['results']:continue
  x=receipts['completed'].get('processed:'+ident)
  if x is None:continue
  p=ROOT/x['relative_path'];sha=hashlib.sha256(p.read_bytes()).hexdigest()
  if sha!=x['SHA256']:raise RuntimeError('source MP4 SHA mismatch')
  row={'status':'pending_or_uncertain'};state['results'][ident]=row;OUT.write_text(json.dumps(state,indent=2)+'\n')
  try:row.update(detect(p))
  except Exception as e:row.update(status='error_or_timeout_no_retry',exception=type(e).__name__+': '+str(e)[:200])
  row['source_MP4_sha256']=sha
  OUT.write_text(json.dumps(state,indent=2)+'\n')
  print('FFMPEG_NATIVE_EDITED_VIDEO_FREEZE',len(state['results']),'/',manifest['source_pairs_expected'],ident,'RESULT',row['status'],'STILL_EVENTS',row.get('event_count'),'SECONDS',row.get('elapsed_seconds'),flush=True)
 if len(state['results'])==manifest['source_pairs_expected']:
  state['status']='complete';state['total_elapsed_s_this_iteration']=round(time.monotonic()-t0,2);state['max_RSS_KB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  OUT.write_text(json.dumps(state,indent=2)+'\n')
  print('OFFICIAL_FFMPEG_SOURCE_VIDEO_FROZEN_ALL',len(state['results']),'STATUSES',dict(collections.Counter(x['status'] for x in state['results'].values())),'T',state['total_elapsed_s_this_iteration'])
 else:print('PARTIAL_FFMPEG_SOURCE_RESULTS',len(state['results']),'FULL',manifest['source_pairs_expected'])
if __name__=='__main__':main()
