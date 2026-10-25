#!/usr/bin/env python3
"""Read-only release-asset ranged extraction of real VideoSham reference pairs.

Adobe VideoSham original Vimeo video snippets and genuinely edited clips, release
v1.0: first 179 source group only, original and processed 1.37/1.46GB zip.
Author labels 19 temporal editing cases and other manipulation categories.
Freezes all 19 temporal and 19 class-stratified non-temporal processed controls
BEFORE observing their video images; each source has matched original footage.

Do not download full 2.8GB archives: GitHub public release supports HTTP range
reads and ZipInfo central directory. Download only selected members with their
exact ZIP CRC32/size, SHA256 provenance. Licensed Adobe Research License 1.1a,
noncommercial academic evaluation only; original MP4s under /data/var, not Git.
No GPUs, video models, original production operations, or publication actions.

Mode plan: immutable label + archive manifest, no videos downloaded.
Mode fetch --start/--count: one bounded group of source pairs, never repeat
completed/uncertain downloads. Resume only from independently CRC-verified
existing selected file, not from ambiguous unfinished .part; no force replay.
"""
from pathlib import Path
from urllib.request import urlopen,Request
import argparse,io,json,zipfile,hashlib,csv,collections,os,time,datetime,re,zlib
ROOT=Path('/data/var/perceptual-independent-20261024/videosham-20261025')
MAN=ROOT/'source_selection_original_38_video_pairs_20261025.json'
RECEIPT=ROOT/'selected_video_member_receipts_20261025.json'
REPO='adobe-research/VideoSham-dataset';RELEASE='v1.0';TAG_API='https://api.github.com/repos/'+REPO+'/releases/tags/'+RELEASE
TYPES={
 'Removing an entity/Removing a subject':4,
 'Adding an entity/ Adding a subject':4,
 'Background/Color Change':4,
 'Audio replaced/Audio added':4,
 'Text Replaced/Text Added':3,
}
SOURCE_META='VideoSourceandEdits.csv'


def request_json(url):
 with urlopen(Request(url,headers={'User-Agent':'SWAAG-independent-visual-research/1.0','Accept':'application/vnd.github+json'}),timeout=15) as f:return json.load(f)

class RangedZIP(io.RawIOBase):
 def __init__(self,uri,length,block=4*1024*1024):
  self.uri=uri;self.total=length;self.pos=0;self.block_bytes=block;self.start=-1;self.data=b'';self.requests=0;self.bytes_read=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  p=offset if whence==0 else self.pos+offset if whence==1 else self.total+offset
  if not 0<=p<=self.total:raise ValueError('ZIP seek outside bounds')
  self.pos=p;return self.pos
 def read(self,n=-1):
  if n is None or n<0:n=self.total-self.pos
  out=bytearray()
  while n>0 and self.pos<self.total:
   if not self.start<=self.pos<self.start+len(self.data):
    self.start=(self.pos//self.block_bytes)*self.block_bytes
    end=min(self.total-1,self.start+self.block_bytes-1)
    req=Request(self.uri,headers={'Range':f'bytes={self.start}-{end}','User-Agent':'Mozilla/5.0'})
    with urlopen(req,timeout=35) as r:
     if r.status!=206:raise RuntimeError('HTTP range not honored')
     data=r.read(self.block_bytes+1)
    if len(data)!=end-self.start+1:raise RuntimeError('Incomplete selected ZIP HTTP block')
    self.data=data;self.requests+=1;self.bytes_read+=len(data)
   take=min(n,len(self.data)-(self.pos-self.start))
   if take<=0:raise RuntimeError('empty cache segment')
   out+=self.data[self.pos-self.start:self.pos-self.start+take];self.pos+=take;n-=take
  return bytes(out)


def assets():
 data=request_json(TAG_API)
 assert data['tag_name']==RELEASE
 values={a['name']:a for a in data['assets']}
 assert all(k in values for k in ['Processed-Part1.zip','Unedited-Part1.zip'])
 return values,data

def headurl(url):
 with urlopen(Request(url,method='HEAD',headers={'User-Agent':'Mozilla/5.0'}),timeout=20) as f:
  if f.status!=200 or f.headers.get('Accept-Ranges')!='bytes':raise RuntimeError('HEAD/range unavailable')
  return f.geturl()

def remote_indexes(assets_map):
 maps={};total_mb=0
 for assetname in ['Unedited-Part1.zip','Processed-Part1.zip']:
  a=assets_map[assetname]
  reader=RangedZIP(headurl(a['browser_download_url']),a['size'])
  with zipfile.ZipFile(reader) as z:
   files={}
   for member in z.infolist():
    name=os.path.basename(member.filename)
    if name.startswith('._') or not name.endswith('.mp4'):continue
    m=re.fullmatch(r'(\d{3,4})_(processed|unedited)\.mp4',name)
    if not m:
     # The release also contains a few unrelated .mp4 aux entries; exclude
     # them rather than guessing sample IDs or letting labels shift.
     continue
    vid=int(m.group(1));assert vid not in files
    files[vid]={'archive_member_name':member.filename,'file_size':member.file_size,'compressed_size':member.compress_size,'CRC32':member.CRC,'compress_type':member.compress_type}
   maps[assetname]=files
  print('VIDEO_SHAM_SOURCE_ORIGINAL_ZIP_INDEX',assetname,'N',len(files),'RANGE_FETCHED_MB',round(reader.bytes_read/1e6,2),flush=True)
 return maps

def public_labels(git_sha):
 url='https://raw.githubusercontent.com/'+REPO+'/'+git_sha+'/'+SOURCE_META
 with urlopen(Request(url,headers={'User-Agent':'SWAAG-research/1.0'}),timeout=18) as f:blob=f.read(300000)
 rows=list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig','replace'))))
 assert len(rows)==352
 return rows,hashlib.sha256(blob).hexdigest()

def plan():
 if MAN.exists():print('PRIOR_FROZEN_SELECTION_PRESERVED',MAN);return
 ROOT.mkdir(parents=True,exist_ok=True)
 sha=request_json('https://api.github.com/repos/'+REPO+'/commits/main')['sha']
 rows,hlabels=public_labels(sha);A,rel=assets();indexes=remote_indexes(A)
 eligible=[]
 for r in rows:
  if int(r['S.No.'])>179:continue
  i=int(r['Video-Name'][:-4]);
  if not all(i in ix for ix in indexes.values()):continue
  eligible.append({'source_video_id':i,'author_manipulation_group':r['Manipulation'],'author_instruction':r['Task'],'original':indexes['Unedited-Part1.zip'][i],'processed':indexes['Processed-Part1.zip'][i]})
 positives=sorted([x for x in eligible if x['author_manipulation_group'].startswith('Frame duplication/')],key=lambda z:hashlib.sha256(str(z['source_video_id']).encode()).hexdigest())
 assert len(positives)==19
 negatives=[]
 for group,n in TYPES.items():
  candidates=sorted([x for x in eligible if x['author_manipulation_group']==group],key=lambda z:hashlib.sha256((group+'|'+str(z['source_video_id'])).encode()).hexdigest())
  # Bound selected download size without inspecting source frames. A maximum
  # 40 MB original-plus-edited filesize avoids monopolizing production WAN.
  picked=[x for x in candidates if x['original']['file_size']+x['processed']['file_size']<40_000_000][:n]
  assert len(picked)==n,(group,len(picked))
  negatives.extend(picked)
 assert len(negatives)==19
 selected=positives+negatives
 assert len({x['source_video_id'] for x in selected})==38
 for item in selected:
  item['gold_temporal_intent']=item['author_manipulation_group'].startswith('Frame duplication/')
  item['source_set']='videoSham_part1'
 chosen=sorted(selected,key=lambda z:hashlib.sha256(('videosham-20261025|'+str(z['source_video_id'])).encode()).hexdigest())
 desc={'date':'2026-10-25','source':'Adobe VideoSham dataset original authentic Vimeo references and separately author edited clips','source_repo':REPO,'pinned_git_commit':sha,'github_release_tag':RELEASE,'release_id':rel['id'],'original_author_labels_sha256':hlabels,'license':'Adobe Research License Version 1.1a, noncommercial research; original media stays outside Git','criteria':'All 19 author temporal-manipulation cases in original Part1; 19 other editing-type negative controls matched to separately edited original source using predeclared quotas 4 removal/4 adding/4 background/4 audio/3 text, controls total size under40MB, hash-based per category fixed ranking; no model features or video previews used for sample selection. Exact same authentic Vimeo source used for each original and altered pair. Temporal positives intentionally edited, not naturally occurring stutter or human-rated perceptual failure.','no_ground_truth_timings':'Author Task contains human editor instructions but not frame-accurate independently adjudicated anomaly intervals. Evaluate only video-level known temporal vs other edits, no temporal localization F1.','model_input_policy':'Detector receives decoded edited video ONLY for no-reference, may receive associated original source for reference-aware; NEVER task labels until all features frozen','release_zip':{k:{'name':k,'bytes':A[k]['size'],'id':A[k]['id']} for k in indexes},'n_temporal_authored':len(positives),'n_other_authored':len(negatives),'n_distinct_source_families':38,'source_pairs_expected':38,'approx_total_paired_download_mb':round(sum(x['original']['file_size']+x['processed']['file_size'] for x in chosen)/1e6,2),'items':chosen}
 MAN.write_text(json.dumps(desc,indent=2)+'\n')
 os.chmod(MAN,0o600)
 print('FROZEN_VIDEO_SHAM_REAL_MEDIA_SELECTION',MAN,'POS',len(positives),'NON_TEMPORAL_PROCESSED_NEG',len(negatives),'TOTAL_EXPECTED_MB',desc['approx_total_paired_download_mb'],'LABEL_DISTRIBUTION',dict(collections.Counter(x['author_manipulation_group'] for x in chosen)),flush=True)


def file_crc(path):
 dig=hashlib.sha256();crc=0;sz=0
 with path.open('rb') as f:
  while block:=f.read(4*1024*1024):
   dig.update(block);crc=zlib.crc32(block,crc);sz+=len(block)
 return sz,crc&0xffffffff,dig.hexdigest()

def acquire(reader,z,info,target):
 expected=info['file_size'];crc=info['CRC32']
 if target.exists():
  if target.with_suffix('.mp4.part').exists():raise RuntimeError('source ambiguous pending existing file')
  size,check,sha=file_crc(target)
  if size==expected and check==crc:return {'status':'preexisting_source_crc_and_size_verified_no_redownload','bytes':size,'CRC32':check,'SHA256':sha}
  raise RuntimeError('existing media changed, do not overwrite')
 target.parent.mkdir(parents=True,exist_ok=True)
 tmp=target.with_suffix('.mp4.part')
 if tmp.exists():raise RuntimeError('incomplete source material exists; do not repeat ambiguous fetch')
 with z.open(info['archive_member_name']) as infile,tmp.open('xb') as outfile:
  digest=hashlib.sha256();calc=0;total=0
  while block:=infile.read(4*1024*1024):
   total+=len(block)
   if total>100_000_000:raise RuntimeError('Unexpected >100MB per source video')
   outfile.write(block);calc=zlib.crc32(block,calc);digest.update(block)
  outfile.flush();os.fsync(outfile.fileno())
 if total!=expected or calc&0xffffffff!=crc:raise RuntimeError('CRC or length mismatch, source partial preserved')
 tmp.rename(target)
 return {'status':'new_verified_source_member','bytes':total,'CRC32':crc,'SHA256':digest.hexdigest()}

def fetch(start,count):
 if not MAN.exists():raise RuntimeError('plan before first source fetch')
 manifest=json.loads(MAN.read_text());assert manifest['source_pairs_expected']==38
 assert 0<=start<38 and 1<=count<=10
 batch=manifest['items'][start:start+count];assert batch
 A,_=assets()
 receipts=json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {'selection_sha256':hashlib.sha256(MAN.read_bytes()).hexdigest(),'completed':{}}
 assert receipts['selection_sha256']==hashlib.sha256(MAN.read_bytes()).hexdigest()
 for side,zipname in [('original','Unedited-Part1.zip'),('processed','Processed-Part1.zip')]:
  ids=[r for r in batch if side+':'+str(r['source_video_id']) not in receipts['completed']]
  if not ids:continue
  obj=A[zipname]
  reader=RangedZIP(headurl(obj['browser_download_url']),obj['size'])
  with zipfile.ZipFile(reader) as z:
   for item in ids:
    ident=item['source_video_id'];target=ROOT/side/(f'{ident:04d}.mp4')
    t=time.monotonic()
    verified=acquire(reader,z,item[side],target)
    receipts['completed'][side+':'+str(ident)]={**verified,'relative_path':str(target.relative_to(ROOT)),'source_sample_id':ident}
    RECEIPT.write_text(json.dumps(receipts,indent=2)+'\n')
    print('ORIGINAL_SOURCE_ARCHIVE_MEMBER_VERIFIED',side,ident,'BYTES',verified['bytes'],'SECONDS',round(time.monotonic()-t,2),'FETCHED_REMOTE_MB',round(reader.bytes_read/1e6,2),flush=True)
 print('SOURCE_BATCH_CHECKPOINT','START',start,'COUNT',len(batch),'CUMULATIVE_COMPLETED_MEMBERS',len(receipts['completed']),'GOAL',manifest['source_pairs_expected']*2,'HASHES_SAVED',RECEIPT,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','fetch']);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=6)
 args=p.parse_args()
 if args.mode=='plan':plan()
 else:fetch(args.start,args.count)
