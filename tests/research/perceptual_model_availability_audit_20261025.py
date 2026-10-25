#!/usr/bin/env python3
"""Small SOURCE AVAILABILITY audit, not an image/video judge.

Verify actual installable checkpoints and benchmark downloadability using
provider APIs and local manifests. Never execute benchmark generator or remote
model repo code. Only explicit public JSON/metadata requests; no secrets.
Write source receipt and task limitations to owning project research folder.
"""
from pathlib import Path
import json,urllib.request,hashlib,datetime
ROOT=Path('/data/var/perceptual-independent-20261024')
OUT=Path(__file__).resolve().parents[2]/'docs/research/perceptual-candidate-availability-2026-10-25.json'

def get(url,max_bytes=600_000):
 req=urllib.request.Request(url,headers={'User-Agent':'perceptual-scientific-evaluation/1.0','Accept':'application/json'})
 with urllib.request.urlopen(req,timeout=13) as r: b=r.read(max_bytes+1)
 if len(b)>max_bytes:raise RuntimeError('bounded public source too large '+url)
 return json.loads(b)

def main():
 if OUT.exists():print('ALREADY_AUDITED_NOT_REPLAYED',OUT);return
 model_screen=json.loads((ROOT/'models/ScreenParser-YOLO/model_manifest.json').read_text())
 model_vlm=json.loads((ROOT/'models/ScreenVLM/research_manifest.json').read_text())
 figma_meta=get('https://api.github.com/repos/gzmelon/UI-Defect-Benchmark/git/trees/main?recursive=1')
 files=figma_meta['tree'];screenshots=[x for x in files if x['path'].lower().endswith(('.png','.jpg','.jpeg','.webp'))]
 preview=json.loads((ROOT/'figma-defect-claims/dataset_full_annotations.json').read_text())
 assert len(preview)==100 and len(screenshots)==1 and figma_meta['sha']=='b1e6a2fe8809f23fcd43d43ba1812a2e8493f71f'
 candidate_phy=get('https://pypi.org/pypi/phyprobe/json')
 physical_eval=get('https://huggingface.co/api/models/nvidia/physical-plausibility-evaluator')
 principia=get('https://api.github.com/repos/PrincipiaBench/Principia')
 assert model_screen['license']=='Apache-2.0 (weights from model card); isolated Ultralytics runtime license separately reviewed' or model_screen['license'].lower().startswith('apache')
 assert model_vlm['license']=='apache-2.0'
 result={
  'research_date':'2026-10-25',
  'objective':'which judge models/benchmarks are materially installed, available, and licensable; DOES NOT imply tested model accuracy',
  'ScreenParser':{'checkpoint_model':model_screen['model'],'pinned_revision':model_screen['revision'],'weight_bytes':model_screen['weight_bytes'],'weights_sha256':model_screen['sha256'],'model_card_weight_license':model_screen['license'],'ultralytics_8_3_221_runtime':'AGPL-3.0 / separately available Enterprise commercial license according to ultralytics.com/license; legal review needed for proprietary deployment, model card weight Apache 2.0 does not alone license inference runtime','official_source':'https://huggingface.co/docling-project/ScreenParser','local_cpu_accuracy_status':'32 real full-page screenshots model evaluation pending or available separately; do not infer from installation'},
  'ScreenVLM':{'checkpoint_model':'docling-project/ScreenVLM','pinned_revision':model_vlm['revision'],'weight_bytes':model_vlm['total_bytes'],'official_max_new_tokens_example':6192,'prior_test_local_max_new_tokens':384,'prior_local_2_of_4_screen_tags_truncated':True,'note':'prior 0 of 4 expected labels at 384 tokens on small crops is NOT a fair official full-page parsing score','official_source':'https://huggingface.co/docling-project/ScreenVLM'},
  'PhyProbe':{'pip_current_version':candidate_phy['info']['version'],'release_available':True,'package':'phyprobe','model_head_weights_size_approx_MB':4,'backbone_size_approx_GB':9,'hardware':'2B parameter PE-Core-G14-448 backbone; high GPU memory, incompatible with spare shared Nitro RTX3060 6 GB; NOT executed','source':'https://pypi.org/project/phyprobe/','scientific_results_external_only':'https://research.nvidia.com/labs/cosmos-lab/phyprobe/'},
  'nvidia_physical_plausibility_evaluator':{'hf_revision':physical_eval['sha'],'license':physical_eval.get('cardData',{}).get('license'),'source':'https://huggingface.co/nvidia/physical-plausibility-evaluator','reported_same_generator_bad_recall':0.8052,'reported_same_generator_good_video_false_rejection':0.3133,'hardware':'trained head + frozen V-JEPA2.1 ViT-L, published H100 evaluation; no Nitro execution, not tested on Blender machinery'},
  'Principia':{'default_branch':principia['default_branch'],'owner_repo':'PrincipiaBench/Principia','source':'https://github.com/PrincipiaBench/Principia','tracking_requirement':'SAM3 gated HF access plus CUDA GPU for object tracking; not runnable as documented without grant/idle GPU','value':'paired-object gravity, restitution, momentum, friction invariants independent of camera calibration; not a universal subjective video judge'},
  'Figma_Defect_3_5K_claimed':{'repo':'https://github.com/gzmelon/UI-Defect-Benchmark','tree_revision':figma_meta['sha'],'source_tree_files':len(files),'public_preview_annotation_records':len(preview),'public_image_files':len(screenshots),'full_3500_claimed_release':'README itself says full ~12.7GB image+DOM+style archive to be hosted after journal acceptance, absent from pinned repository at evaluation','validation_use':'NOT suitable for independent image-based Figma-Defect-3.5K model scoring; no 3.5K original screenshots/DOM available; do not repeat author performance claim as locally confirmed'},
  'ScreenParse_parser_alternative':{'RT_DETRv2':'Authors report object-detector 43M element model on ScreenParse dataset, but no independently verified RT-DETRv2 checkpoint installed locally','OmniParser':'Microsoft v2 icon_detect_v3 MIT + MIT Florence icon_caption option, different license from YOLOv8 AGPL icon_detect; no local inference score yet','source':['https://saidgurbuz.github.io/screenparse/','https://huggingface.co/microsoft/OmniParser-v2.0']},
  'known_unanswered':['ScreenParser real-target bbox localization result and reliability on actual defects','complete ScreenVLM on full screenshots with enough output tokens under production-safe compute','installed, domain-calibrated physical plausibility video scorer on independent human labels, without encoding/generator confounds','industrial 3D mechanically verified physical assembly faults and natural real-video stutter time truth'],
  'data_source_status':'All source evidence from official upstream model/research/dataset APIs and pinned local source hashes. Availability and software licenses distinct from accuracy. Actual model outputs in dated project reports.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('CANDIDATE_AVAILABILITY_RECORDED',OUT,'BYTES',OUT.stat().st_size,'SCREENPARSER',model_screen['weight_bytes'],'SCREENVLM_LIMIT',result['ScreenVLM']['prior_test_local_max_new_tokens'],'FIGMA_REAL_SCREENSHOTS',len(screenshots))
if __name__=='__main__':main()
