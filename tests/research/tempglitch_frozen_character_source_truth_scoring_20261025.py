#!/usr/bin/env python3
"""Independent source-authored Frozen Animation label scoring after image freeze.

Source selection 12 authentic Platformer pairs original creator-injected frozen
character pose vs source normal, all 24 original MP4 SHA verified, exact image
signals+official whole-frame FFmpeg freeze outputs frozen FIRST with NO labels
used in algorithm. Author bug vs normal source folder/class is expected truth
for this narrow Godot engine-injected fault, not independent human movie visual
subjective objectionability; no precise original temporal event labels.

Report full TP FP FN TN incl abstentions, sensitivity, specificity, AUC of
predeclared continuous run lengths, paired by *source scene* sign test, and
source-pair bootstrap of binary balanced accuracy/AUC; no hyperparameter
search or best-model source selection. Native FPS, duration and sample limits
kept as confound. License/input media outside Git, concise report only.
"""
from pathlib import Path
import json,hashlib,collections,math
import numpy as np
from scipy.stats import beta,binomtest
ROOT=Path('/data/var/perceptual-independent-20261024/tempglitch-pose-frozen-external-20261025')
PLAN=ROOT/'source_original_12_platformer_pairs_fixed_selection_20261025.json'
PIX=ROOT/'24_actual_original_video_fullframe_vs_character_appearance_features_FILTER_FIXED_20261025.json'
OUT=ROOT/'24_original_external_frozen_animation_vs_normal_method_comparisons_20261025.json'
METHODS={
 'official_FFmpeg_full_frame_freezedetect':('source_fullframe_freeze_detector','original_ffmpeg_full_frame_freeze_bool','original_ffmpeg_full_frame_frozen_event_count'),
 'whole_frame_pixel_MAE_stall':('source_pixel_signals','full_frame_3transition_still_bool','max_adjacent_run_full_frame_still'),
 'center_raw_still_while_outer_motion':('source_pixel_signals','raw_center_3transition_frozen_bool','max_adjacent_run_raw_center_freeze_while_outer_motion'),
 'phase_aligned_center_still_while_outer_motion':('source_pixel_signals','phase_registered_center_3transition_frozen_bool','max_adjacent_run_phase_registered_center_freeze_while_outer_motion'),
 'Farneback_center_still_while_background_moves':('source_pixel_signals','Farneback_center_3transition_frozen_bool','max_adjacent_run_Farneback_center_frozen_while_outer_motion'),
}

def CI(k,n):
 if not n:return None
 return [round(float(beta.ppf(.025,k,n-k+1)) if k else 0,4),round(float(beta.ppf(.975,k+1,n-k)) if k<n else 1,4)]

def roc(rows,key):
 valid=[r for r in rows if r[key] is not None]
 pos=np.asarray([r[key] for r in valid if r['original_external_engine_glitch_label']]);neg=np.asarray([r[key] for r in valid if not r['original_external_engine_glitch_label']]);
 if len(pos)*len(neg)==0:return None
 return float(np.mean(pos[:,None]>neg[None])+0.5*np.mean(pos[:,None]==neg[None]))

def metrics(rows,which):
 scored=[x for x in rows if x[which+'__binary'] is not None]
 pos=sum(x['original_external_engine_glitch_label'] for x in scored)
 neg=len(scored)-pos
 tp=sum(x['original_external_engine_glitch_label'] and x[which+'__binary'] for x in scored)
 fp=sum(not x['original_external_engine_glitch_label'] and x[which+'__binary'] for x in scored)
 fn=pos-tp;tn=neg-fp
 return {'scorable_n':len(scored),'expected_original_label_positive_n':12,'expected_original_label_negative_n':12,'abstained_original_positive_n':12-pos,'abstained_original_normal_control_n':12-neg,'TP':tp,'FP':fp,'FN':fn,'TN':tn,'conditional_recall':tp/pos if pos else None,'conditional_specificity':tn/neg if neg else None,'conditional_precision_on_balanced_sample':tp/(tp+fp) if tp+fp else None,'conditional_balanced_accuracy':(tp/pos+tn/neg)/2 if pos and neg else None,'continuous_run_score_AUROC':roc(rows,which+'__score'),'source_original_exact_recall95_CI':CI(tp,pos),'source_original_exact_false_alarm95_CI':CI(fp,neg)}

def main():
 if OUT.exists():print('PREVIOUS_ORIGINAL_GODOT_ENGINE_LABEL_COMPARISONS_ALREADY_SAVED');return
 plan=json.loads(PLAN.read_text());pix=json.loads(PIX.read_text())
 assert plan['selected_independent_source_scene_pairs']==12
 assert pix['status']=='complete' and len(pix['source_label_blind_results'])==24
 assert pix['source_original_selection_SHA256']==hashlib.sha256(PLAN.read_bytes()).hexdigest()
 rows=[];pairs=[]
 for item in plan['selected']:
  ident=item['source_original_pair_index'];pair=[]
  for classkey,gold in [('buggy',True),('normal',False)]:
   result=pix['source_label_blind_results'][str(ident)+'|'+classkey]
   r={'source_scene_id_SHA256':hashlib.sha256(str(ident).encode()).hexdigest(),'original_external_engine_glitch_label':gold,'original_source_analysis_status':result['status'],'source_duration_seconds':result.get('native_source_video_props',{}).get('duration_seconds'),'source_native_frame_rate':result.get('native_source_video_props',{}).get('native_fps'),'source_image_samples_n':result.get('source_pixel_signals',{}).get('source_video_frame_count_at_5fps'),'source_metadata_original_bytes':item['source_'+classkey+'_original_mp4']['bytes'],'source_metadata_inverse_file_bytes_score':-float(item['source_'+classkey+'_original_mp4']['bytes']),'source_metadata_inverse_duration_score':-float(result.get('native_source_video_props',{}).get('duration_seconds')) if result.get('native_source_video_props',{}).get('duration_seconds') is not None else None}
   for method,(group,binary,score) in METHODS.items():
    b=result.get(group,{}).get(binary) if result['status']=='complete' else None
    c=result.get(group,{}).get(score) if result['status']=='complete' else None
    r[method+'__binary']=bool(b) if b is not None else None
    r[method+'__score']=float(c) if c is not None else None
   rows.append(r);pair.append(r)
  pairs.append(pair)
 assert len(rows)==24 and sum(r['original_external_engine_glitch_label'] for r in rows)==12
 output={}
 for method in METHODS:
  m=metrics(rows,method)
  allpairs=[p for p in pairs if all(r[method+'__score'] is not None for r in p)]
  good=sum(p[0][method+'__score']>p[1][method+'__score'] for p in allpairs)
  bad=sum(p[0][method+'__score']<p[1][method+'__score'] for p in allpairs)
  ties=len(allpairs)-good-bad
  m['same_original_scene_bug_score_exceeds_normal']=good
  m['same_original_scene_bug_score_below_normal']=bad
  m['same_original_scene_equal_score']=ties
  m['source_match_same_scene_exact_sign_p_if_nontied']=float(binomtest(good,good+bad,p=.5).pvalue) if good+bad else None
  rng=np.random.default_rng(20261025);bag={'AUROC':[],'balanced_accuracy':[]}
  for _ in range(3000):
   sample=[r for i in rng.integers(0,len(pairs),len(pairs)) for r in pairs[i]]
   auc=roc(sample,method+'__score');mm=metrics(sample,method)
   if auc is not None:bag['AUROC'].append(auc)
   if mm['conditional_balanced_accuracy'] is not None:bag['balanced_accuracy'].append(mm['conditional_balanced_accuracy'])
  m['source_scene_matched_bootstrap_95pct_CI']={k:[round(float(np.quantile(v,q)),4) for q in (.025,.975)] if len(v)>=1500 else None for k,v in bag.items()}
  output[method]=m
  print('REAL_ORIGINAL_PLATFORMER_CHARACTER_FROZEN_VS_NORMAL',method,'TP',m['TP'],'FP',m['FP'],'FN',m['FN'],'TN',m['TN'],'ABSTAIN_BUG',m['abstained_original_positive_n'],'ABSTAIN_NORMAL',m['abstained_original_normal_control_n'],'AUROC',m['continuous_run_score_AUROC'],'PAIR_BUG_GREATER',good,'PAIR_GOOD_GREATER',bad,'EQUAL',ties,flush=True)
 confounds={'negative_original_MP4_bytes_source_AUROC_NOT_VISUAL':roc(rows,'source_metadata_inverse_file_bytes_score'),'negative_original_video_duration_source_AUROC_NOT_VISUAL':roc(rows,'source_metadata_inverse_duration_score'),'scope':'Original encoded file bytes and duration metadata ONLY, source author construction shortcut control; not a vision model and file-size selection cap biases this sample.'}
 docs={'date':'2026-10-25','status':'PARTIAL: empirical real original Godot frozen-character-pose glitch vs authored normal baseline, bounded 12 Platformer scene pairs. Source-author labels NOT human blind timing truth; no validated universally reliable VLM or visual-only motion gate','source_publication':'TempGlitch arxiv 2605.21443 asgaardlab/TempGlitch original external MIT source clips and in-engine fault labels','original_release_source_revision':plan['source_dataset_revision'],'selection_caveat':'12 matched Platformer IDs selected from27 pairs <=70MB per pair before video pixels. 24 videos / 716.73MB source, not a random full 750 source/video set, source size and one game env may confound. First21 author bug file naming mismatches normal scene family, excluded rather than cross-pairing.','source_hashes':{'full_1500_original_source_reference':plan['original_full_dataset_source_receipt_SHA256'],'frozen_original_12_pair_selection':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'all24_original_video_pixel_predictions_before_author_gold':hashlib.sha256(PIX.read_bytes()).hexdigest()},'n_source_scene_matched_pairs':12,'n_actual_original_video_files':24,'external_engine_author_labels_positives':12,'external_engine_author_labels_normal_controls':12,'full_frame_and_local_ROI_methods_frozen_before_source_test_labels':pix['config'],'actual_original_source_method_metrics':output,'nonvisual_source_dataset_construction_controls':confounds,'per_source_original_video_pair_labels_and_prediction_outcomes':rows,'resource':{'GPU_used':False,'CPU_video_feature_wall_seconds':pix.get('run_wall_seconds'),'CPU_peak_RSS_KB':pix.get('run_peak_RSS_KB'),'original_video_bytes_expected_MB':plan['total_download_MB']},'limits':['The source frozen CHARACTER ANIMATION bug can leave video frame/scene/camera movement intact; full-frame freeze detector is not aligned with source truth and is expected to miss many defects.','Original creator label injected bug vs normal was not independently reviewed by subjective viewers; no exact original temporal onset or fault interval ground truth. Event localization F1 or perceptual objectionability cannot be measured from these source clips.','Only original Platformer scene family and 12 source pairs selected with file-size constraint, no complete 1500 source or cross-engine generalization.','Center ROI is a fixed guess, not ground truth segmented game avatar; a false positive on original normal may arise from intentional static pose, low background texture or camera movement.','Unknown video source frame/decoding failures count as abstentions, not negative predictions.','FFmpeg native-frame freeze not equivalent to frozen animation; no 3D engine debug pose/rig state available as gold outside injected source label.','No source original videos/images, copyrighted game rendering pixels or label-specific raw outputs added to Git.','Previous blocked owner warehouse full-resolution framehash and Spotlight outputs were not retrieved or worked around.']}
 OUT.write_text(json.dumps(docs,indent=2)+'\n')
 print('REAL_GAMEPLAY_EXTERNAL_SOURCE_ENGINE_FAULT_TEST_COMPLETED',OUT,'BINARY_METHODS',len(output))
if __name__=='__main__':main()
