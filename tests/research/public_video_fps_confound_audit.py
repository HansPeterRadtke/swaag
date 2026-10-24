#!/usr/bin/env python3
"""Post-hoc metadata-only confound audit of two human video-preference samples.

No video decode/model calls. Pure source FPS comparator, heldout labels already
consumed, so perfect preference agreement cannot establish physical realism.
"""
import json,hashlib,collections,pyarrow.parquet as pq
from pathlib import Path
ROOT=Path('/data/var/perceptual-independent-20261024/rapidata-physics-video')
OUT=ROOT/'fps_only_preference_confound_20261024.json'
def load(path):return json.loads((ROOT/path).read_text())

def main():
 if OUT.exists():print('EXISTING_NOT_RECOMPUTED');return
 original=load('frozen_human_pair_source_manifest.json');receipts=load('selected_human_pair_clip_receipts.json');raw=load('24_video_temporal_raw_scores_20261024.json')
 byurl={x['source_url']:x['sha256'] for x in receipts['clips']};metrics=raw['clip_metrics']
 matched=[]
 for sample in original['samples']:
  real=metrics[byurl[sample['ground_truth_url']]]['source_fps'];gen=metrics[byurl[sample['generated_url']]]['source_fps']
  expected=sample['ground_truth_human_win_blind'];pred=real>gen
  matched.append({'prompt_sha256':sample['scenario_prompt_sha256'],'human_prefers_real':expected,'real_native_fps':real,'generated_native_fps':gen,'fps_only_predicts_real':pred,'match':expected==pred})
 transfer=load('six_generated_pair_transfer_20261024.json')
 transferraw=load('six_generated_pair_blind_raw_metrics_20261024.json')['participant_videos']
 # Recover pair URLs/sha from the fixed public source shard, already cached, no refetch.
 source=pq.read_table(ROOT/'train-00000-of-00043.parquet',columns=['prompt','model1','model2','video1','video2']).to_pylist()
 chosen={x['prompt_sha256'] for x in transfer['pairs']}
 loc=collections.defaultdict(dict)
 # Clip refs keyed by sha only. There are two source model versions per prompt; line
 # pairs identify URLs; local filenames produced deterministically by SHA256(url).
 for r in source:
  if hashlib.sha256(r['prompt'].encode()).hexdigest() not in chosen:continue
  if {r['model1'],r['model2']}!={'Cosmos 2.5','MiniMax H3'}:continue
  key=hashlib.sha256(r['prompt'].encode()).hexdigest()
  for fld,model in [('video1',r['model1']),('video2',r['model2'])]:
   p=ROOT/'clips'/(hashlib.sha256(r[fld].encode()).hexdigest()[:22]+'.mp4')
   sha=hashlib.sha256(p.read_bytes()).hexdigest();loc[key][model]=transferraw[sha]['source_fps']
 generated=[]
 for p in transfer['pairs']:
  a=loc[p['prompt_sha256']];expect=p['human_Cosmos_wins_blind'];pred=a['Cosmos 2.5']>a['MiniMax H3']
  generated.append({'prompt_sha256':p['prompt_sha256'],'human_prefers_Cosmos':expect,'Cosmos_native_fps':a['Cosmos 2.5'],'MiniMax_native_fps':a['MiniMax H3'],'fps_only_predicts_Cosmos':pred,'match':pred==expect})
 assert len(matched)==12 and len(generated)==6
 report={'note':'After-the-fact video frame-rate metadata diagnostic; not a trained video judge and cannot be used as an unbiased new heldout result. All 18 source preferences favored the higher-fps participant, making prior motion-metric preference agreement impossible to interpret as recognition of physics. Different codec/resolution/generator confounds remain after resampling to common 12fps.','real_vs_Cosmos':{'n':len(matched),'metadata_fps_matches':sum(x['match'] for x in matched),'pairs':matched},'Cosmos_vs_MiniMax':{'n':len(generated),'metadata_fps_matches':sum(x['match'] for x in generated),'pairs':generated},'human_source_manifest_sha256':hashlib.sha256((ROOT/'frozen_human_pair_source_manifest.json').read_bytes()).hexdigest(),'generated_transfer_sha256':hashlib.sha256((ROOT/'six_generated_pair_transfer_20261024.json').read_bytes()).hexdigest(),'source_has_frame_exact_glitch_labels':False}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('METADATA_BASELINE REAL vs AI',report['real_vs_Cosmos']['metadata_fps_matches'],'/',len(matched),'MODEL vs MODEL',report['Cosmos_vs_MiniMax']['metadata_fps_matches'],'/',len(generated))
if __name__=='__main__':main()
