#!/usr/bin/env python3
"""Download published ScreenParser YOLO11-L UI detector into isolated var only.

Frozen checkpoint and license provenance; no GPU, no untrusted Python scripts
from hub executed. To run inference use the existing isolated Ultralytics venv.
"""
from pathlib import Path
import json,hashlib,datetime
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path('/data/var/perceptual-independent-20261024/models/ScreenParser-YOLO')
MANIFEST=ROOT/'model_manifest.json'
MODEL='docling-project/ScreenParser';SHA='f029e565f1206577402e43206454522075be3f72'
def main():
 if MANIFEST.exists():
  a=json.loads(MANIFEST.read_text());print('MODEL_ALREADY_PINNED',a['revision'],a['weight_bytes']);return
 ROOT.mkdir(parents=True,exist_ok=True)
 meta=HfApi().model_info(MODEL)
 assert meta.sha==SHA
 assert meta.card_data.license=='apache-2.0'
 path=Path(hf_hub_download(repo_id=MODEL,filename='best.pt',revision=SHA,local_dir=str(ROOT)))
 digest=hashlib.sha256();start=path.stat().st_size
 with path.open('rb') as f:
  while block:=f.read(4_194_304):digest.update(block)
 assert 140_000_000<start<170_000_000
 out={'model':MODEL,'revision':SHA,'source':'https://huggingface.co/docling-project/ScreenParser','license':'Apache-2.0 (weights from model card); isolated Ultralytics runtime license separately reviewed','checkpoint':'best.pt','weight_bytes':start,'sha256':digest.hexdigest(),'model_arch':'YOLO11-L 55 UI classes, trained on ScreenParse v2 full web screenshots, 1280 input; bounding boxes/classes, no text transcription, NOT a defect detector','downloaded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 MANIFEST.write_text(json.dumps(out,indent=2)+'\n')
 print('MODEL_PINNED',start,'BYTES','SHA256',out['sha256'],'REV',SHA)
if __name__=='__main__':main()
