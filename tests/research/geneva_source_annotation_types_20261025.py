#!/usr/bin/env python3
"""Read original GeneVA public artifact annotations for source-research feasibility.

Public S3 GeneVA/WACV2026 16k AI-generated clips, per-frame artifact annotations
and quality. Stream ~113MB JSON directly from source to memory; never store raw
userID, personal annotation identifiers, timestamp or descriptions on disk.
Aggregate category strings into a source-count table and choose candidate
source VIDEO IDs (not annotator IDs) only for externally labeled video defects.
No model inference or videos downloaded here; source metadata original rights
and endpoints remain in isolated private /data/var, no full raw identities in
Git. This is a source-data inspection and does not validate video judge yet.
"""
from pathlib import Path
import urllib.request,json,collections,hashlib,time,datetime,statistics,math,os
ROOT=Path('/data/var/perceptual-independent-20261024/geneva-human')
OUT=ROOT/'geneva_source_taxonomy_and_selection_20261025.json'
URL='https://geneva-annotations.s3.us-east-1.amazonaws.com/AnnotatedVideos.json'
BYTES=112664805

def main():
 if OUT.exists():print('GENEVA_SOURCE_ALREADY_SELECTED_DO_NOT_REPEAT');return
 ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
 t=time.monotonic()
 req=urllib.request.Request(URL,headers={'User-Agent':'visual-perceptual-research/1.0'})
 with urllib.request.urlopen(req,timeout=80) as r:
  raw=r.read(BYTES+1)
 assert len(raw)==BYTES
 parsed=json.loads(raw)
 assert isinstance(parsed,list) and len(parsed)>10000
 print('SOURCE_BYTES_VERIFIED',len(raw),'RECORDS',len(parsed),'SHA256',hashlib.sha256(raw).hexdigest(),'TIME_S',round(time.monotonic()-t,2),flush=True)
 # Raw annotation records include annotator identities; do NOT save them.
 videos=collections.defaultdict(list);cat=collections.Counter();labels=collections.Counter();generators=collections.Counter();bad=collections.Counter()
 for row in parsed:
  if not row.get('isDoneAnnotated'):continue
  rawid=str(row.get('videoID','')).strip().strip('"')
  if not rawid.endswith('.mp4') or '/' not in rawid:bad['bad_video_id']+=1;continue
  try:ann=json.loads(row['annotation'])
  except (ValueError,TypeError):bad['bad_annotation_json']+=1;continue
  if not isinstance(ann,list):bad['bad_annotation_type']+=1;continue
  simplified=[]
  for a in ann:
   label=str(a.get('label',''))
   category=label.split(' - ')[0].strip().casefold()
   # Sanitized category names only; full descriptive label text not retained.
   cat[category]+=1
   frame=a.get('frame');stamp=a.get('time')
   if not isinstance(frame,int) or not isinstance(stamp,(int,float)):bad['bad_frame_marker']+=1;continue
   simplified.append({'category':category,'frame':frame,'time':stamp,'bbox_coordinates_present':all(k in a for k in ('x','y','w','h'))})
  generator=rawid.split('/')[0]
  generators[generator]+=1
  videos[rawid].append({'n':len(simplified),'events':simplified,'quality':float(row['videoRating']) if str(row.get('videoRating','')).replace('.','',1).isdigit() else None,'width':int(row['videoWidth']) if str(row.get('videoWidth','')).isdigit() else None,'height':int(row['videoHeight']) if str(row.get('videoHeight','')).isdigit() else None})
 # Only ORIGINAL public clip IDs and coarse human category/frame annotations.
 # Avoid storing broad original annotation texts or any user identifier.
 patterns=collections.Counter()
 for vid,ratings in videos.items():
  positive_categories={a['category'] for r in ratings for a in r['events']}
  for c in positive_categories:patterns[c]+=1
 print('HUMAN_ANNO_EVENTS',sum(cat.values()),'UNIQUE_VIDEOS',len(videos),'CAT',cat.most_common(35),'CLIP_CATEGORIES',patterns.most_common(25),'GENERATORS',generators,'BAD',bad,flush=True)
 # Choose minimally ambiguous temporal issues in OTHER than 'physics':
 # human labels marking in-video flicker/stutter/morphing/interpolation.
 keys=['flick','stutt','froz','freez','jitter','teleport','warping','morph','temporal','frame','inconsistent','disappear','motion']
 matchcats=sorted(c for c in cat if any(k in c for k in keys))
 print('POSSIBLE_TEMPORAL_CATEGORIES',[(c,cat[c],patterns[c]) for c in matchcats],flush=True)
 # Source JSON may contain many multi-rater duplicates; record agreement.
 multi=collections.Counter(len(v) for v in videos.values())
 result={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'GeneVA human per-frame AI-generated video artifacts, WACV 2026, https://www.immersivecomputinglab.org/publication/geneva/','source_annotation_url':URL,'source_raw_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'row_count':len(parsed),'unique_videos':len(videos),'generation_group_counts':dict(generators),'annotation_category_counts':dict(cat),'clip_positive_category_counts':dict(patterns),'video_number_raters':dict(multi),'source_errors':dict(bad),'relevant_temporal_category_prefixes':matchcats,'license_note':'Public source availability and published paper do not imply permissive redistribution; no raw human identifiers, original texts or media placed in project Git. UserID and annotation timestamps intentionally discarded.','selected_video_ids':[]}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 os.chmod(OUT,0o600)
 print('AGGREGATE_SOURCE_TAXONOMY_STORED',OUT,'N_BYTES',OUT.stat().st_size,flush=True)
if __name__=='__main__':main()
