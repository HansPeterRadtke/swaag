#!/usr/bin/env python3
"""New ORIGINAL frozen-pose VIDEO dataset test, not synthetic edited frames.

TEMPGLITCH author-engine-injected Godot Frozen Animation bug original 150 buggy
and 150 normal video sources; some first source IDs use nonmatching file
naming and must not be assumed equivalent. Select ONLY 12 Platformer pairs
with matching original `..._Platformer_(Buggy|Normal)_ID.mp4` scene-ID and
total pair bytes <=70MB. Selection SHA256 sorted based ONLY on official source
file-size metadata and original IDs, before any video pixels/decoder/model.
This size restriction is NOT random over all video duration/source families:
source-bias must be disclosed. Original source metadata SHA pinned from prior
complete 1500 actual source listing, NO prior ambiguous replay.

Original entire release 60GB, selected 12 source-scene 24 original MP4 716MB,
retrieval from public official pinned HuggingFace dataset revision, verify
source exact SHA256 from original HF LFS metadata and original byte size;
source original moving video files are NOT added to SWAAG Git.

External creator-injected bug labels are independently defined in original
Godot game engine, NOT fresh blind human ratings, exact fault frame times
unavailable. Test semantic class `character-animation-poses-frozen`, which
is explicitly different from frozen **entire video frame**. Two same-budget
pixel methods and official FFmpeg freezedetect are declared separately.
No production system, proprietary owner video or GPUs touched.
"""
from pathlib import Path
import argparse,json,hashlib,os,time,urllib.request,urllib.parse
BASE=Path('/data/var/perceptual-independent-20261024')
SOURCE=BASE/'tempglitch-control-20261025/full_1500_original_metadata_and_pair_integrity_20261025.json'
ROOT=BASE/'tempglitch-pose-frozen-external-20261025'
PLAN=ROOT/'source_original_12_platformer_pairs_fixed_selection_20261025.json'
REC=ROOT/'source_24_original_MP4_verified_SHA256_receipts_20261025.json'
SEED='SWAAG-TEMPGLITCH-platformer-20261025-gold-blind'
CAP=70_000_000
REPO='asgaardlab/TempGlitch'

def hash_file(p):
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  while b:=f.read(2*1024*1024):n+=len(b);h.update(b)
 return n,h.hexdigest()

def plan():
 if PLAN.exists():print('TEMPGLITCH_NEW_ORIGINAL_VIDEO_SELECTION_EXISTS_NO_RESAMPLE');return
 original=json.loads(SOURCE.read_text())
 rows=original['original_source_file_listing']
 fault={r['index']:r for r in rows if r['category']=='Frozen Animation' and r['label_buggy'] and '_Platformer_' in r['file_path']}
 normal={r['index']:r for r in rows if r['category']=='Frozen Animation' and not r['label_buggy'] and '_Platformer_' in r['file_path']}
 eligible=[]
 for i in fault:
  if i not in normal:continue
  a,b=fault[i],normal[i]
  if a['bytes']+b['bytes']>CAP:continue
  basename_a=a['file_path'].rsplit('/',1)[-1]
  basename_b=b['file_path'].rsplit('/',1)[-1]
  assert basename_a.replace('_Buggy_','_ORIGINAL_')==basename_b.replace('_Normal_','_ORIGINAL_'),str(i)
  assert len(a['source_lfs_sha256'])==len(b['source_lfs_sha256'])==64
  eligible.append({'source_scene_family':'Godot Platformer','source_original_pair_index':i,'source_buggy_original_mp4':a,'source_normal_original_mp4':b,'expected_source_pair_total_bytes':a['bytes']+b['bytes']})
 selected=sorted(eligible,key=lambda r:hashlib.sha256((SEED+'|'+str(r['source_original_pair_index'])).encode()).hexdigest())[:12]
 assert len(eligible)>=12 and len(selected)==12 and len({x['source_original_pair_index'] for x in selected})==12
 ROOT.mkdir(parents=True,mode=0o700,exist_ok=True)
 meta={'date':'2026-10-25','original_release':'TempGlitch (2026) authored source original Godot animation frozen vs normal, MIT dataset card, underlying game footage rights require their own review','source_dataset_revision':original['revision'],'original_full_dataset_source_receipt_SHA256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'original_source_repo':REPO,'positive_type':'original engine intentionally injected frozen character animation pose; NOT whole-video frame freeze, human subjective abnormality rating or exact event interval','negative_type':'original same-scene-index Normal authored gameplay; not necessarily time-wise registered or same action path','input_selection':'ONLY original metadata `Frozen Animation` Platformer with matching original filename source identity, same numeric ID and total bytes <=70e6, then SHA256 ranking predeclared seed; no image decoded or labels consulted for method tuning. Biased toward short/compressible Platformer clips.','seed':SEED,'max_original_source_pair_bytes':CAP,'eligible_source_pairs_under_cap':len(eligible),'selected_independent_source_scene_pairs':len(selected),'total_source_original_MP4_files':24,'total_download_MB':round(sum(x['expected_source_pair_total_bytes'] for x in selected)/1e6,2),'selected':selected,'source_safety':'No uncertain Calendar effect touched. Source originals and videos under isolated research /data/var only, no Git redistribution.'}
 PLAN.write_text(json.dumps(meta,indent=2)+'\n')
 print('NEW_TEMPGLITCH_REAL_VIDEO_SOURCE_PAIRS_FROZEN',len(selected),'FROM_ELIGIBLE',len(eligible),'TOTAL_MB',meta['total_download_MB'],'SCENES',[i['source_original_pair_index'] for i in selected],flush=True)

def fetch(start,count):
 metadata=json.loads(PLAN.read_text());assert metadata['selected_independent_source_scene_pairs']==12
 if REC.exists():
  rec=json.loads(REC.read_text());assert rec['selection_SHA256']==hashlib.sha256(PLAN.read_bytes()).hexdigest()
 else:rec={'selection_SHA256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'completed':{}}
 for item in metadata['selected'][start:start+count]:
  for klass in ['normal','buggy']:
   key=str(item['source_original_pair_index'])+'|'+klass
   if key in rec['completed']:continue
   info=item['source_'+klass+'_original_mp4']
   target=ROOT/'media'/klass/(str(item['source_original_pair_index'])+'.mp4')
   if target.exists():
    size,sha=hash_file(target)
    if size!=info['bytes'] or sha!=info['source_lfs_sha256']:raise RuntimeError('Existing source mismatch or ambiguous: refuse overwrite/retry')
    stat='already_present_exact_official_source_SHA_verified'
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.mp4.part')
    if tmp.exists():raise RuntimeError('UNCERTAIN incomplete source .part; do not retry')
    url='https://huggingface.co/datasets/'+REPO+'/resolve/'+metadata['source_dataset_revision']+'/'+urllib.parse.quote(info['file_path'],safe='/')+'?download=true'
    h=hashlib.sha256();n=0;t=time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SWAAG-source-perception/1.0'}),timeout=60) as f,tmp.open('xb') as out:
     while b:=f.read(2*1024*1024):
      n+=len(b)
      if n>info['bytes']+1024:raise RuntimeError('source response larger than original metadata manifest')
      h.update(b);out.write(b)
     out.flush();os.fsync(out.fileno())
    if n!=info['bytes'] or h.hexdigest()!=info['source_lfs_sha256']:raise RuntimeError('official source LFS SHA+size verification failed')
    tmp.rename(target);size=n;sha=h.hexdigest();stat='new_source_MP4_verified_official_SHA+size'
    print('ORIGINAL_GODOT_SOURCE_MP4_VERIFIED',key,'BYTES',n,'WALL_SEC',round(time.monotonic()-t,1),flush=True)
   rec['completed'][key]={'source_relpath':str(target.relative_to(ROOT)),'source_original_SHA256':sha,'source_bytes':size,'status':stat}
   REC.write_text(json.dumps(rec,indent=2)+'\n')
 print('GODOT_SOURCE_MEDIA_CHECKPOINT',len(rec['completed']),'OF',metadata['total_source_original_MP4_files'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','fetch']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=4);a=ap.parse_args()
 if a.mode=='plan':plan()
 else:fetch(a.start,a.count)
