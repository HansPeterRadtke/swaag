#!/usr/bin/env python3
"""Read-only source-resolution decoded frame hash audit for v52 video last35.

Important distinction: 256x144 grayscale resized frame duplicates can be
visually identical at coarse scale while original 1920x1080 decoded frames
remain distinct. This is not a human-validated stutter threshold.
"""
from pathlib import Path
import json,subprocess,hashlib,time
ROOT=Path('/data/var/perceptual-independent-20261024/owner_actual_product_video')
OUT=ROOT/'owner_prod_v52_source_resolution_tail_hashes_20261025.json'
VIDEOS={
 'warehouse_v52':Path('/data/var/3d-smooth-hd-20261024/warehouse/animated_warehouse_true_Cycles_1920x1080_native_30fps_motionblur_v52.mp4'),
 'dock_v52':Path('/data/var/3d-smooth-hd-20261024/dock/animated_dock_true_Cycles_1920x1080_native_30fps_motionblur_v52.mp4')}

def check(path):
 before=path.stat();start=time.monotonic()
 cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-i',str(path),'-an','-vf','select=gte(n\\,385)','-fps_mode','vfr','-pix_fmt','yuv420p','-f','framemd5','pipe:1']
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 if p.returncode:raise ValueError('ffmpeg framemd5 failed: '+p.stderr[-250:])
 lines=[x.strip() for x in p.stdout.splitlines() if x.strip() and not x.startswith('#')]
 rows=[]
 for i,line in enumerate(lines):
  fields=[y.strip() for y in line.split(',')]
  if len(fields)!=6:raise ValueError('unexpected decoded frame-md5 record '+str(fields))
  rows.append({'absolute_frame_index':385+i,'source_pts':int(fields[2]),'hash':fields[5]})
 if len(rows)!=35:raise ValueError('expected 35 last frames, got '+str(len(rows)))
 adjacent=[r['absolute_frame_index'] for old,r in zip(rows[:-1],rows[1:]) if old['hash']==r['hash']]
 hashes=[r['hash'] for r in rows]
 repeat_across_nonadjacent=sum(1 for i in range(len(hashes)) if hashes[i] in hashes[:i] and hashes[i]!=hashes[i-1] if i>0)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('source changed during read')
 return {'source_video_path':str(path),'size_bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'frame_indexes_checked':[385,419],'decoded_frames':len(rows),'adjacent_source_resolution_identical_frame_indices':adjacent,'unique_hashes':len(set(hashes)),'nonadjacent_repeated_hash_count':repeat_across_nonadjacent,'decoded_framehashes':[{'frame_index':x['absolute_frame_index'],'md5':x['hash']} for x in rows],'wall_s':round(time.monotonic()-start,2)}

def main():
 if OUT.exists():print('SOURCE_FRAME_RESULT_ALREADY_SAVED');return
 ROOT.mkdir(exist_ok=True)
 data={name:check(path) for name,path in VIDEOS.items()}
 report={'date':'2026-10-25','operation':'read-only decoded native-resolution tail 35-frame MD5 audit; original yuv420p at 1920x1080, 30fps','results':data,'interpretation':'Full decoded YUV hash comparison distinguishes low-resolution near-identical output from exact decoded frame repetition; even consecutive exact repeats can be intentional fade/cut, and distinct frames can still have perceptual judder. No human acceptance claim.','source_change':'none'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 for k,v in data.items():print('TAIL',k,'LAST35_FULL_RESOLUTION_FRAMES','adjacent_exact_repeat',len(v['adjacent_source_resolution_identical_frame_indices']),'unique',v['unique_hashes'],'nonadjacent_repeated',v['nonadjacent_repeated_hash_count'],'seconds',v['wall_s'])
if __name__=='__main__':main()
