#!/usr/bin/env python3
"""Publish source-verified real-human-slide geometry/perception comparison.

Source SlideAudit UIST 2025: real/synthetically altered rendered original slide
images + author object descriptions + strongly agreed human flaw regions.
Requires ALL 343 source rectangle frozen predictions, human-binary evaluations,
previously frozen pixel-only OCR/CLIP, source/pixel OR/AND, and 104-positive
source occlusion localization-vs-uniform-location randomized control.
Also source-only GeneVA 2026 frame-artifact data availability and taxonomy;
no original annotator userIDs, videoIDs, free-text artifact descriptions or
footage ever stored in project Git. No media/model inference or GPU here.
This is research about task-specific tool support, not an end-to-end QA pass.
"""
from pathlib import Path
import hashlib,json,collections,datetime
ROOT=Path('/data/var/perceptual-independent-20261024')
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'docs/research/perceptual-real-source-layout-human-visual-2026-10-25.json'


def data(path):
 p=ROOT/path;b=p.read_bytes();return json.loads(b),hashlib.sha256(b).hexdigest()

def main():
 if OUT.exists():print('AGGREGATE_EVIDENCE_ALREADY_EXISTS_UNCHANGED');return
 selection,hs=data('slideaudit_selection_20261025.json')
 geom,hg=data('slideaudit_343_source_geometry_frozen_20261025.json')
 scores,hgs=data('slideaudit_343_source_geometry_human_defects_20261025.json')
 comp,hcomp=data('slideaudit_343_source_pixel_complementarity_20261025.json')
 loc,hloc=data('slideaudit_occlusion_source_localization_random_baseline_20261025.json')
 sourceprop,hprop=data('slideaudit_343_layout_intersection_proposals_frozen_20261025.json')
 baseline,hbase=data('slideaudit_human_visual_label_performance_20261025.json')
 cl, hcl=data('slideaudit_clip_343_human_flaw_comparison_20261025.json')
 geneva,hgeneva=data('geneva-human/geneva_source_taxonomy_and_selection_20261025.json')
 assert selection['n_selected_unique']==geom['descriptions_n']==len(geom['features_only_NO_source_elements_or_text'])==343
 assert len(sourceprop['slides'])==343 and loc['results']['n_occlusion_human_positive_slides_with_bbox']==104
 assert geom['status']==sourceprop['status']=='complete'
 assert scores['source_geometry_prediction_sha256']==hg
 assert comp['source_geometry_frozen_sha256']==hg
 assert comp['source_intersection_proposals_sha256']==hprop
 assert loc['source_generated_intersection_proposals_sha256']==hprop
 assert scores['source_human_selection_sha256']==hs
 assert comp['source_selection_sha256']==hs
 assert all(k in scores['outcomes'] for k in selection['task_labels'])
 over=scores['outcomes']['Content Overflow/Cut-off']['confusion']
 occl=scores['outcomes']['Occluded Content']['confusion']
 assert (over['TP'],over['FP'],over['FN'],over['TN'])==(9,7,73,75)
 assert (occl['TP'],occl['FP'],occl['FN'],occl['TN'])==(93,33,12,72)
 assert (loc['results']['source_localization_hits'],loc['results']['source_proposal_empty'])==(85,12)
 assert geneva['unique_videos']==16351 and geneva['row_count']==16361
 categories={}
 for key in ['shape, form, geometry','motion','physics','visual artifacts','semantic mismatch with text','other']:
  categories[key]=sum(n for mixture,n in geneva['annotation_category_counts'].items() if key in mixture.split(', ') or key=='shape, form, geometry' and key in mixture)
 # Fix domain label tokenization: shape, form, geometry is one user-visible category.
 categories['shape, form, geometry']=sum(n for mixture,n in geneva['annotation_category_counts'].items() if 'shape, form, geometry' in mixture)
 comparisons={}
 for defect in ['Content Overflow/Cut-off','Occluded Content','Insufficient Color Contrast for Readability']:
  source=scores['outcomes'][defect]
  human_n=selection['task_counts'][defect]['strong_positive_count']
  primary='max_source_text_offscreen_area_fraction' if defect=='Content Overflow/Cut-off' else 'max_source_any_pair_intersection_of_smaller'
  comparisons[defect]={
   'n_positive_human_strong_agreement':human_n,
   'n_matched_human_negative_target_category':human_n,
   'source_metadata_guide':'Original authored slide element geometry, not pixel perception or reliable modern DOM',
   'source_rectangle_fixed_rule':source['confusion'],
   'source_primary_continuous_AUROC':source['all_six_predeclared_continuous_sources'][primary]['auroc'],
   'source_primary_continuous_AUROC_matched_pair_bootstrap95':source['all_six_predeclared_continuous_sources'][primary]['paired_bootstrap95'],
   'source_rectangle_binary_balanced_accuracy_bootstrap95':source['paired_bootstrap95_balanced_accuracy'],
   'old_image_only_OCR_baseline':baseline['tasks'][defect]['confusion_on_balanced_human_source'],
   'old_real_zero_shot_CLIP':cl['tasks'][defect]['binary_prompt_rule'],
  }
  if defect in comp['tasks']:
   comparisons[defect]['geometry_OR_image_heuristic']=comp['tasks'][defect]['binary_policy']['OR']
   comparisons[defect]['geometry_AND_image_heuristic']=comp['tasks'][defect]['binary_policy']['AND']
   comparisons[defect]['source_image_primary_AUROC_delta_matched95']=comp['tasks'][defect]['paired_bootstrap_95pct_auc_difference_geom_minus_pixel']
  if defect=='Occluded Content':
   comparisons[defect]['source_occlusion_localization_and_chance']=loc['results']
   comparisons[defect]['source_total_overlap_area_AUROC_exploratory_secondary']=source['all_six_predeclared_continuous_sources']['sum_source_pair_overlap_area_in_slide_units']['auroc']
 audit={
  'date':'2026-10-25',
  'overall_status':'PARTIAL — source object rectangles can help identify visually annotated occlusion but none tested reliably certifies actual GUI clipping, occlusion or contrast',
  'human_original_media':'SlideAudit UIST 2025 original/altered presentation slides, 2,400 human reviewed designs, 3 human annotators, CC BY 4.0 GitHub original pinned 642d490b7c1d2e78a50a631bfd359433397f3ecf',
  'source_description_elements':'All 343 previously frozen human-reviewed source slide descriptions, mostly TEXT_BOX and IMAGE, original authored normalized rectangles; source content text never available to human comparison code except geometry extraction.',
  'source_sha256':{'independent_human_selection':hs,'all343_source_layout_frozen_before_human_labels':hg,'source_layout_labels_scoring':hgs,'independent_frozen_source_intersection_proposals':hprop,'same_slide_binary_OR_AND_comparison':hcomp,'human_occlusion_region_uniform_spatial_null':hloc,'previous_pixels_only_Tesseract_OpenCV_scores':hbase,'previous_actual_CLIP_model_scores':hcl,'GeneVA_video_frame_artifact_source_taxonomy_only':hgeneva},
  'n_distinct_original_slide_rasters':343,
  'n_original_source_layout_descriptions':343,
  'n_valid_source_rectangles':sum(v['source_element_geometry']['source_valid_geometry_count'] for v in geom['features_only_NO_source_elements_or_text'].values()),
  'n_desc_with_source_text_outside_by_2pct':sum(v['source_element_geometry']['source_predicted_overflow'] for v in geom['features_only_NO_source_elements_or_text'].values()),
  'n_desc_with_any_source_pair_overlap_above_10pct':sum(v['source_element_geometry']['source_predicted_any_element_overlap'] for v in geom['features_only_NO_source_elements_or_text'].values()),
  'target_human_category_results':comparisons,
  'separate_candidate_video_annotation_source':{
   'source':'GeneVA WACV 2026 University/Immersive Computing Lab, S3 public frames metadata','source_link':'https://www.immersivecomputinglab.org/publication/geneva/',
   'published_dataset_video_count':16356,'actually_completed_annotation_row_count':geneva['row_count'],'distinct_source_videoIDs_seen':geneva['unique_videos'],
   'annotated_events_per_frame_record':sum(geneva['annotation_category_counts'].values()),
   'generators_source_counts':geneva['generation_group_counts'],
   'broad_original_frame_record_category_counts_overlapping':categories,
   'number_of_ratings_per_video_source':geneva['video_number_raters'],
   'new_local_video_judge_accuracy':'NOT TESTED',
   'data_access_observation':'112.7 MB original JSON S3 annotation stream read and summarized in-memory, no raw annotator IDs/free-text descriptions stored. A sample VC/Sora video object HEAD was accessible. No new video clips obtained/scored.',
   'critical_limitation':'Broad motion/physics/artifact per-frame region labels and sometimes interpolated tracks, generally one annotator per video; not a certified natural playback stutter/freeze versus truly clean matched-FPS control benchmark. Approx event/track counts are NOT independent human confirmations.'
  },
  'scientific_cautions':[
   'This is not a clean web/android GUI human screenshot dataset; source is slides, some deliberately modified, and author-provided layout rectangles may precede final pixel alterations.',
   'The 343 human-reviewed source images were already used by prior CLIP/OCR comparison and selection; source-geometry tests are exploratory on consumed development benchmark, not pristine super-test.',
   'All label no-category slides have strong agreement on THAT category only and may contain other design defects.',
   'Intentional stacking of source shapes and background text boxes produces false occlusion alerts, and text can be clipped within entirely onscreen containers.',
   'The spatial localization null is random uniform independent locations with each case original number proposals; a low null probability demonstrates geometric alignment beyond chance under that null but not that overlap is unwanted, visually obscuring or correct in real app.',
   'The category low contrast is unmeasurable from object rectangles alone; observed correlation with text-box overlap is not proof of color readability.',
   'Simple OR improves occlusion sensitivity at an increased false-positive rate and slightly lower balanced accuracy; more detectors do not automatically yield a better judge.',
   'GeneVA single-rater spatiotemporal generated-video artifacts are not clean natural-frame discontinuity truth. Current production models not qualified.'
  ],
  'original_licensed_or_private_media_in_git':False,
 }
 OUT.write_text(json.dumps(audit,indent=2)+'\n')
 print('PUBLISHED_COMPACT_REAL_SOURCE_GEOMETRY_AND_HUMAN_EVIDENCE',OUT,'BYTES',OUT.stat().st_size,'ORIGINAL_ELEMENTS',audit['n_valid_source_rectangles'])
if __name__=='__main__':main()
