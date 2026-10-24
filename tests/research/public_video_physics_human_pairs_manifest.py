#!/usr/bin/env python3
"""Read ONLY aggregate human physical-video votes for a reproducible small sample.

Source: Rapidata public physics benchmark, 43 tiny Parquet metadata shards.
Original vote-level demographic contents are NOT retained; download bytes only
in memory, extract aggregate model pairs, retain original remote content hashes.
Select first 12 distinct PROMPTS by SHA256, never based on human vote outcome.
Compare one original real filmed Ground Truth vs Cosmos 2.5 generated video
per prompt; the human votes serve external preference truth, not exact glitches.
Media from providers remains outside Git and their use is research-only.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import urllib.request,json,hashlib,datetime,io,collections,re
import pyarrow.parquet as pq
ROOT=Path('/data/var/perceptual-independent-20261024/rapidata-physics-video')
OUT=ROOT/'frozen_human_pair_source_manifest.json'
API='https://huggingface.co/api/datasets/Rapidata/image-to-video-physics-human-feedback-benchmark'
MODEL='Cosmos 2.5'
REFERENCE='Ground Truth'
SAMPLE_N=12

def fetch(shard,revision):
 name=shard['path'];u='https://huggingface.co/datasets/Rapidata/image-to-video-physics-human-feedback-benchmark/resolve/'+revision+'/'+name+'?download=true'
 cache=ROOT/Path(name).name
 if cache.is_file() and cache.stat().st_size==shard['size']:
  data=cache.read_bytes()
 else:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'visual-research-2026/1.0'}),timeout=17) as r:data=r.read(1000000)
 if len(data)!=shard['size'] or len(data)>1000000 or data[:4]!=b'PAR1':raise ValueError('shard integrity error '+name)
 columns=['prompt','video1','video2','model1','model2','weighted_results_video1_blind','weighted_results_video2_blind','weighted_results_video1_with_reference','weighted_results_video2_with_reference']
 table=pq.read_table(io.BytesIO(data),columns=columns)
 pairs=[]
 for row in table.to_pylist():
  if set([row['model1'],row['model2']])!={REFERENCE,MODEL}:continue
  # Only public aggregate vote sum; absolutely no per-voter details read.
  pairs.append(row)
 return name,{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'rows':table.num_rows,'matching_pairs':len(pairs)},pairs

def main():
 if OUT.exists():print('SNAPSHOT_ALREADY_EXISTS_NOT_REDOWNLOADED');return
 ROOT.mkdir(exist_ok=True)
 with urllib.request.urlopen(API,timeout=12) as r:metadata=json.load(r)
 sha=metadata['sha'];assert metadata.get('gated') is False and sha
 with urllib.request.urlopen(API+'/tree/main/data?recursive=false',timeout=12) as r:items=json.load(r)
 items=sorted([f for f in items if re.fullmatch(r'data/train-\d{5}-of-00043\.parquet',f['path'])],key=lambda x:x['path'])
 assert len(items)==43
 rec=[];filtered=[]
 with ThreadPoolExecutor(max_workers=3) as ex:
  tasks={ex.submit(fetch,x,sha):x for x in items}
  for f in as_completed(tasks):
   name,receipt,pairs=f.result();rec.append({'path':name,**receipt});filtered.extend(pairs)
   print('SHARD',name,'gt_pairs',len(pairs),'source_bytes',receipt['bytes'],flush=True)
 # Choice independent of votes (only prompt hash); source has scenario-level
 # truth, but paired entries may use different generated samples.
 grouped=collections.defaultdict(list)
 for x in filtered:grouped[x['prompt']].append(x)
 scenario_ids=sorted(grouped,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:SAMPLE_N]
 selected=[]
 for text in scenario_ids:
  matches=grouped[text]
  # deterministically choose one from pair set, not using vote labels.
  m=min(matches,key=lambda r:(r['video1'],r['video2']))
  real_side=1 if m['model1']==REFERENCE else 2
  real_url=m['video1'] if real_side==1 else m['video2']
  fake_url=m['video2'] if real_side==1 else m['video1']
  pref_real_blind=float(m['weighted_results_video1_blind'] if real_side==1 else m['weighted_results_video2_blind'])
  pref_fake_blind=float(m['weighted_results_video2_blind'] if real_side==1 else m['weighted_results_video1_blind'])
  pref_real_reference=float(m['weighted_results_video1_with_reference'] if real_side==1 else m['weighted_results_video2_with_reference'])
  pref_fake_reference=float(m['weighted_results_video2_with_reference'] if real_side==1 else m['weighted_results_video1_with_reference'])
  selected.append({'scenario_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt':text,'ground_truth_url':real_url,'generated_url':fake_url,'original_model':REFERENCE,'generated_model':MODEL,'blind_ground_truth_weighted_votes':pref_real_blind,'blind_generated_weighted_votes':pref_fake_blind,'with_reference_ground_truth_weighted_votes':pref_real_reference,'with_reference_generated_weighted_votes':pref_fake_reference,'ground_truth_human_win_blind':pref_real_blind>pref_fake_blind,'ground_truth_human_win_with_reference':pref_real_reference>pref_fake_reference})
 report={'source':'Rapidata/image-to-video-physics-human-feedback-benchmark','source_revision':sha,'shards':sorted(rec,key=lambda x:x['path']),'source_total_rows':sum(x['rows'] for x in rec),'matching_model_pairs':len(filtered),'unique_scenarios':len(grouped),'scenario_selection':'12 smallest SHA256(prompt) ordered, model pair Ground Truth vs Cosmos 2.5, URL tie-break (no human vote used)','samples':selected,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'note':'Real recorded physics vs image-to-video AI-generated continuations, human preferences from public study. These are NOT binary frame-glitch ground truth or necessarily perfect human annotations. Rights for each mp4 remain with original Physics-IQ/AI providers; no media in Git. Individual demographic records explicitly never read into Python data.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('DONE',OUT,'SCENARIOS',len(grouped),'SELECTED',len(selected),'HUMAN_REAL_WINS_BLIND',sum(x['ground_truth_human_win_blind'] for x in selected),'HUMAN_REAL_WINS_WITH_REFERENCE',sum(x['ground_truth_human_win_with_reference'] for x in selected))
if __name__=='__main__':main()
