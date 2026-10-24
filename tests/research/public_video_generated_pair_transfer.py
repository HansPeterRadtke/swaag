#!/usr/bin/env python3
"""Frozen same-prompt generated-vs-generated transfer check, no metric tuning.

Uses six Rapidata Physics-IQ scenarios previously SHA-selected for real/AI test
that also have a Cosmos 2.5 vs MiniMax H3 published human-vote comparison.
Only public aggregate votes loaded, no personal voting records; copied MP4s
remain outside Git. This is exploratory, not a fresh independent prompt split.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import io,json,hashlib,urllib.parse,urllib.request,collections,time
import pyarrow.parquet as pq
ROOT=Path('/data/var/perceptual-independent-20261024/rapidata-physics-video')
OUT=ROOT/'six_generated_pair_transfer_20261024.json'
SOURCE=ROOT/'train-00000-of-00043.parquet'
CHOICES=ROOT/'frozen_human_pair_source_manifest.json'
MODEL_A='Cosmos 2.5';MODEL_B='MiniMax H3'

def download(url):
 u=urllib.parse.urlparse(url)
 if u.hostname!='assets.rapidata.ai' or u.scheme!='https' or not u.path.endswith('.mp4'):raise ValueError('unexpected source URL')
 p=ROOT/'clips'/(hashlib.sha256(url.encode()).hexdigest()[:22]+'.mp4')
 if not p.is_file():
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'perceptual-research/1.0'}),timeout=20) as f:data=f.read(5000001)
  if len(data)>5000000 or len(data)<1000 or data[4:8]!=b'ftyp':raise ValueError('unexpected MP4')
  p.write_bytes(data)
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 return str(p),sha,p.stat().st_size

def main():
 if OUT.exists():print('EXISTS_NOT_RECALCULATED');return
 selected={c['prompt'] for c in json.loads(CHOICES.read_text())['samples']}
 original=pq.read_table(SOURCE,columns=['prompt','model1','model2','video1','video2','weighted_results_video1_blind','weighted_results_video2_blind','weighted_results_video1_with_reference','weighted_results_video2_with_reference']).to_pylist()
 items=[r for r in original if r['prompt'] in selected and {r['model1'],r['model2']}=={MODEL_A,MODEL_B}]
 assert len(items)==6 and len({r['prompt'] for r in items})==6
 allurls={r[k] for r in items for k in ('video1','video2')}
 files={}
 with ThreadPoolExecutor(max_workers=3) as ex:
  futures={ex.submit(download,url):url for url in allurls}
  for f in as_completed(futures):
   u=futures[f];path,sha,size=f.result();files[u]={'file':path,'sha256':sha,'bytes':size}
   print('VIDEO',len(files),'/',len(allurls),'BYTES',size,flush=True)
 # Freeze every input and model-neutral temporal prediction BEFORE reading human labels.
 from public_video_physics_temporal_metrics import compute,CONFIG
 raw={}
 for i,url in enumerate(sorted(allurls)):
  values=compute(files[url]['file']);raw[url]=values
  print('METRICS',i+1,'/',len(allurls),'fps',values['source_fps'],'motion_jerk',values['flow_magnitude_acceleration'],'s',values['wall_seconds'],flush=True)
 raw_path=ROOT/'six_generated_pair_blind_raw_metrics_20261024.json'
 raw_data={'algorithm':CONFIG,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'participant_videos':{files[u]['sha256']:raw[u] for u in allurls},'note':'Raw metrics frozen before accessing human preference when scoring.'}
 raw_path.write_text(json.dumps(raw_data,indent=2)+'\n')
 compare=[]
 for r in items:
  akey='video1' if r['model1']==MODEL_A else 'video2';bkey='video2' if r['model1']==MODEL_A else 'video1'
  a=raw[r[akey]];b=raw[r[bkey]]
  hblind=float(r['weighted_results_video1_blind'] if akey=='video1' else r['weighted_results_video2_blind'])-float(r['weighted_results_video2_blind'] if akey=='video1' else r['weighted_results_video1_blind'])
  href=float(r['weighted_results_video1_with_reference'] if akey=='video1' else r['weighted_results_video2_with_reference'])-float(r['weighted_results_video2_with_reference'] if akey=='video1' else r['weighted_results_video1_with_reference'])
  decisions={k:{'cosmos_less_than_minimax':a[k]<b[k],'exact_tie':a[k]==b[k],'cosmos_score':a[k],'minimax_score':b[k]} for k in CONFIG['methods']}
  compare.append({'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),'vote_margin_Cosmos_minus_MiniMax_blind':hblind,'vote_margin_Cosmos_minus_MiniMax_reference':href,'human_Cosmos_wins_blind':hblind>0,'human_Cosmos_wins_reference':href>0,'temporal_methods':decisions})
 scores={}
 for key in CONFIG['methods']:
  for typ in ('blind','reference'):
   x=[s for s in compare if not s['temporal_methods'][key]['exact_tie']]
   hit=sum(s['temporal_methods'][key]['cosmos_less_than_minimax']==s['human_Cosmos_wins_'+typ] for s in x)
   scores[key+'_'+typ]={'n_decided':len(x),'agree_human':hit,'disagree_human':len(x)-hit,'abstain':len(compare)-len(x),'human_Cosmos_preferred':sum(s['human_Cosmos_wins_'+typ] for s in x)}
 result={'source':'Rapidata Physics-IQ, same twelve preselected prompts, subset six with Cosmos2.5 vs MiniMaxH3 published model-pair voting','source_dataset_revision':'0adbb8421595188087df72ab2da72129c71ea227','raw_pred_hash':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'participants':[MODEL_A,MODEL_B],'sampled_prompts':6,'pairs':compare,'metric_agreement':scores,'media_receipts':[files[u] for u in sorted(allurls)],'limitations':'Six exploratory scenarios only and selected prompt set previously used real-vs-generated; not an unseen holdout. Source videos may have different codecs/fps between generated models. Human comparisons are preferences not verified frame glitches. Original preference direction was inspected before experiment, though no metric parameters were changed. Labels not given to pixel metrics. No FP estimate from truly defect-free controls.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('TRANSFER_RESULTS',json.dumps(scores,indent=2))
if __name__=='__main__':main()
