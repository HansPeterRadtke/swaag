#!/usr/bin/env python3
"""CPU-only read-only real-render video with externally specified frame edit manifests.

Labels are editorial policies / exact known inserted frame edits, NOT human quality labels.
The two different machine scenes are encoded at two resolutions each; correlated.
Never touches active render assets or video service. No raw output is committed.
"""
from __future__ import annotations
import cv2,hashlib,json,time,collections,statistics
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher

ROOT=Path('/data/var/perceptual-independent-20261024/real_render_video_continuity')
SOURCES={
 'forklift_fullhd':Path('/data/var/3d-video-investigation-20261014/v4/forklift_steering_and_continuous_motion_v4.mp4'),
 'industrial_pack_fullhd':Path('/data/var/3d-video-investigation-20261014/v4/industrial_machinery_pack_motion_corrected_v4.mp4'),
 'forklift_published_720':Path('/data/var/3d-video-investigation-20261014/public_youtube/forklift_public_720.mp4'),
 'industrial_pack_published_720':Path('/data/var/3d-video-investigation-20261014/public_youtube/pack_public_720.mp4'),
}
OFFSETS=[30,160]
COUNT=64
# Predeclaration of timeline semantics. Deliberate held identical pixels are
# observationally indistinguishable from unintended content freeze.
VARIANTS={
 'unchanged':'clean_reference_control',
 'freeze_three':'bad_inserted_freeze',
 'drop_three':'bad_dropped_frames',
 'reverse_three':'bad_temporal_reversal',
 'intentional_hold':'allowed_director_hold',
 'intentional_cut':'allowed_editorial_cut',
}

def decode(path,starts):
 cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
 results=[]
 for start in starts:
  cap.set(cv2.CAP_PROP_POS_FRAMES,start)
  seq=[]
  for i in range(COUNT):
   ok,f=cap.read()
   if not ok:raise ValueError(f'incomplete video {path.name} at {start}+{i}')
   image=cv2.resize(f,(192,108),interpolation=cv2.INTER_AREA)
   seq.append(image)
  results.append(seq)
 cap.release()
 return results

def make(base,alternate,variant):
 arr=list(base)
 if variant in ('freeze_three','intentional_hold'):
  for i in range(28,31):arr[i]=arr[27]
 elif variant=='drop_three':
  # In a 64-frame segment drop three true source frames; preserve original source order.
  arr=arr[:28]+arr[31:]
 elif variant=='reverse_three':
  arr[28:31]=list(reversed(arr[28:31]))
 elif variant=='intentional_cut':
  # Explicit approved shot cut to an alternate position in the SAME original video.
  arr=arr[:28]+list(alternate[28:])
 return arr

def frame_sig(frame):return hashlib.blake2b(frame.tobytes(),digest_size=12).hexdigest()

def metrics(seq,orig):
 sigs=[frame_sig(x) for x in seq]
 ref=[frame_sig(x) for x in orig]
 dup=[i for i in range(1,len(sigs)) if sigs[i]==sigs[i-1]]
 runs=[];start=None
 for i in range(1,len(sigs)):
  if sigs[i]==sigs[i-1] and start is None:start=i-1
  if (sigs[i]!=sigs[i-1] or i==len(sigs)-1) and start is not None:
   end=i if i==len(sigs)-1 and sigs[i]==sigs[i-1] else i-1
   runs.append(end-start+1);start=None
 max_repeat=max(runs,default=1)
 diffs=[float(cv2.absdiff(a,b).mean()) for a,b in zip(seq[:-1],seq[1:])]
 med=float(np.median(diffs));mad=float(np.median(np.abs(np.asarray(diffs)-med)))
 q95=float(np.percentile(diffs,95));maxchange=max(diffs)
 # No re-tuning on cases: simple diagnostic threshold from local median / MAD.
 spike=bool(any(v>med+6*max(mad,.005) for v in diffs))
 refedit=SequenceMatcher(None,ref,sigs,autojunk=False).ratio()
 return {'frame_count':len(seq),'has_three_or_more_identical_frames':max_repeat>=3,'max_identical_run':max_repeat,'has_diff_spike_six_mad':spike,'reference_changed':refedit<.999999,'reference_similarity_ratio':round(refedit,6),'num_duplicate_pair':len(dup),'median_frame_mae':round(med,4),'max_frame_mae':round(maxchange,4),'frame_mae_p95':round(q95,4)}

def summary(rows,key,mask):
 tp=fp=fn=tn=0
 selected=[r for r in rows if mask(r)]
 for row in selected:
  violation=row['variant'] in ('freeze_three','drop_three','reverse_three')
  predicted=row[key]
  if predicted and violation:tp+=1
  elif predicted and not violation:fp+=1
  elif not predicted and violation:fn+=1
  else:tn+=1
 return {'n_cases':len(selected),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}

def main():
 cv2.setNumThreads(1)
 ROOT.mkdir(parents=True,exist_ok=True)
 out=ROOT/'20261024_real_video_results.json'
 if out.exists():
  print('PREVIOUS_COMPLETE_RESULT_PRESERVED_NOT_REPEATED',out)
  return
 rows=[];sources=[];beg=time.monotonic()
 for name,path in SOURCES.items():
  before=path.stat();t0=time.monotonic()
  seqs=decode(path,OFFSETS)
  after=path.stat()
  assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed concurrently'
  sig=hashlib.sha256(path.read_bytes()).hexdigest()
  sources.append({'name':name,'path':str(path),'sha256':sig,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'decoded_window_offsets':OFFSETS,'decoded_size':[192,108],'decode_seconds':round(time.monotonic()-t0,3)})
  for j,base in enumerate(seqs):
   alt=seqs[1-j]
   variant_hashes={}
   for variant in VARIANTS:
    edit=make(base,alt,variant)
    t1=time.monotonic();meter=metrics(edit,base)
    hashes=[frame_sig(x) for x in edit]
    variant_hashes[variant]=hashlib.sha256('|'.join(hashes).encode()).hexdigest()
    rows.append({'source':name,'window_index':j,'source_frame_offset':OFFSETS[j],'variant':variant,'gold_edit_semantics':VARIANTS[variant],'pixel_hash_stream_sha256':variant_hashes[variant],'metric_seconds':round(time.monotonic()-t1,4),**meter})
   # Mathematically identical frames with opposite intentional/unintentional labels.
   assert variant_hashes['freeze_three']==variant_hashes['intentional_hold']
  print('SOURCE',name,'WINDOWS',len(seqs),'CASES',len(seqs)*len(VARIANTS),'DECODE_S',round(time.monotonic()-t0,2),flush=True)
 for method in ['has_three_or_more_identical_frames','has_diff_spike_six_mad','reference_changed']:
  print('METHOD',method,'all',summary(rows,method,lambda x:True),'observable_only',summary(rows,method,lambda x:x['variant'] not in ('intentional_cut','intentional_hold')),flush=True)
 report={'date':'2026-10-24','purpose':'Compare three nonlearned pixel/known-good-reference continuity checks on finished real rendered videos, with controlled true frame edits and approved discontinuities','not_an_external_human_label_dataset':True,'labels':'Applied frame-index edits and approved counterfactual editorial intent are established from script, NOT inferred from video; unchanged originals are only no-added-defect controls, NOT certified smooth videos','inputs':sources,'parameters':{'offsets':OFFSETS,'frames_per_window':COUNT,'pixel_resolution':[192,108],'independent_media_families':2,'deliveries':4,'variants':VARIANTS,'duplicate_min_run_frames':3,'mae_outlier':'median+6*MAD with MAD floor .005'},'cases':rows,'results':{method:{'all':summary(rows,method,lambda x:True),'observable_only_excluding_approved_cuts_and_holds':summary(rows,method,lambda x:x['variant'] not in ('intentional_cut','intentional_hold'))} for method in ['has_three_or_more_identical_frames','has_diff_spike_six_mad','reference_changed']},'wall_seconds':round(time.monotonic()-beg,2),'limits':'No human-rated real-world stutter or 3D physics labels; edited freeze and director-approved hold have IDENTICAL pixel streams and opposite intent labels; no screenshot-only analyzer can distinguish them. Matched exact reference only applies if a trusted approved final edit/reference exists. Two scenes rendered into four correlated delivery variants; CPU only.'}
 out.write_text(json.dumps(report,indent=2)+'\n')
 print('WRITTEN',out,'N',len(rows),'SECONDS',report['wall_seconds'],flush=True)
if __name__=='__main__':main()
