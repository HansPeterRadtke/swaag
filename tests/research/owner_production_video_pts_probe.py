#!/usr/bin/env python3
"""Read-only exact presentation timestamp checks on completed owner 3D videos.

This is a targeted production media diagnostic, NOT a subjective stutter judge.
Never modify, re-encode, move or publish products; no model or GPU work.
Clip file sizes and mtimes must remain stable during read. Source PTS can prove
missing/duplicate timestamps but cannot prove camera path smoothness/perception.
"""
from pathlib import Path
import json,subprocess,statistics,hashlib,time
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_product_video')
OUT=ROOT/'owner_prod_v52_presentation_timestamps_20261025.json'
FILES={
 'warehouse_v52':Path('/data/var/3d-smooth-hd-20261024/warehouse/animated_warehouse_true_Cycles_1920x1080_native_30fps_motionblur_v52.mp4'),
 'dock_v52':Path('/data/var/3d-smooth-hd-20261024/dock/animated_dock_true_Cycles_1920x1080_native_30fps_motionblur_v52.mp4'),
}

def analyse(name,path):
 before=path.stat();start=time.monotonic()
 if not path.is_file() or before.st_size<1000000:raise RuntimeError('original file not complete')
 args=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height:frame=best_effort_timestamp_time,pkt_duration_time,pict_type','-of','json',str(path)]
 p=subprocess.run(args,capture_output=True,text=True,timeout=32,check=True)
 v=json.loads(p.stdout);frames=v['frames'];streams=v['streams'];assert len(streams)==1
 stamps=[float(f['best_effort_timestamp_time']) for f in frames];diff=[round(b-a,7) for a,b in zip(stamps,stamps[1:])]
 assert stamps==sorted(stamps)
 expected=1/30
 gaps=[{'frame_index':i+1,'dt_s':x} for i,x in enumerate(diff) if x>1.5*expected]
 duplicates=[i+1 for i,x in enumerate(diff) if x<=1e-7]
 stat={'name':name,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'file_sha256':None,'declared_stream':streams[0],'decoded_timestamp_frames':len(frames),'frame_interval_count':len(diff),'first_timestamp':stamps[0],'last_timestamp':stamps[-1],'median_dt_s':statistics.median(diff),'minimum_dt_s':min(diff),'maximum_dt_s':max(diff),'missing_time_intervals_larger_than_1p5_frame':gaps[:50],'duplicate_pts_indices':duplicates[:50],'distinct_delta_values':{str(k):diff.count(k) for k in sorted(set(diff))[:12]},'run_seconds':round(time.monotonic()-start,2)}
 after=path.stat();assert after.st_size==before.st_size and after.st_mtime_ns==before.st_mtime_ns,'original was modified concurrently'
 # no need to hash entire multi-MB mp4 if machine under heavy load; metadata and stream plus exact names recorded.
 return stat

def main():
 if OUT.exists():print('PRESERVE_OWNED_PTS_REPORT_NOT_REPLAYED');return
 ROOT.mkdir(exist_ok=True)
 results={key:analyse(key,path) for key,path in FILES.items()}
 state={'source':'owner existing finished Nitro production warehouse/dock MP4 v52 on 2026-10-24','date':'2026-10-25','results':results,'scope':'read-only exact encoded video PTS/frame count, NOT viewing smoothness or renderer acceptance, intentionally limited to completed encoded files','caveats':'Correct timestamp cadence cannot detect repeated but non-identical frames, camera jerk, motion blur or high frequency visual artefacts. Incorrect timestamp/gaps can reveal a different class of playback fault. No independent humans labelled these exact final v52 clips smooth or bad. Original 3D rendering untouched.'}
 OUT.write_text(json.dumps(state,indent=2)+'\n')
 for key,v in results.items():print('OWNER_VIDEO',key,'frames',v['decoded_timestamp_frames'],'fps',v['declared_stream']['avg_frame_rate'],'median_dt',v['median_dt_s'],'max_dt',v['maximum_dt_s'],'large_gaps',len(v['missing_time_intervals_larger_than_1p5_frame']),'duplicate_pts',len(v['duplicate_pts_indices']),'sec',v['run_seconds'])
if __name__=='__main__':main()
