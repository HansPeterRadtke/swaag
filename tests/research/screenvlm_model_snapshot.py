#!/usr/bin/env python3
"""Download pinned model weights into isolated var cache, not project Git.

Known source is docling-project/ScreenVLM, Apache-2.0. Exact commit SHA +
file sizes and checksum become manifest; never replace an existing completed
checkpoint and never use untrusted Python repo code.
"""
from pathlib import Path
import hashlib,json,datetime
from huggingface_hub import HfApi,snapshot_download
ROOT=Path('/data/var/perceptual-independent-20261024/models/ScreenVLM')
ROOT.mkdir(parents=True,exist_ok=True)
MANIFEST=ROOT/'research_manifest.json'
def main():
 if MANIFEST.exists():
  d=json.loads(MANIFEST.read_text());print('SNAPSHOT_ALREADY_RECORDED',d.get('revision'),len(d.get('files',[])));return
 info=HfApi().model_info('docling-project/ScreenVLM')
 print('MODEL_SHA',info.sha,'LICENSE',info.card_data.license if info.card_data else None,flush=True)
 dir=snapshot_download(repo_id='docling-project/ScreenVLM',revision=info.sha,local_dir=str(ROOT),max_workers=2,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','*.tiktoken'],ignore_patterns=['*.md','*.gitattributes'])
 files=[]
 for p in sorted(ROOT.iterdir()):
  if p.is_file() and not p.name.startswith('research_manifest'):
   # 746MB chunk hashing one-time, no GPU.
   h=hashlib.sha256()
   with p.open('rb') as f:
    while chunk:=f.read(4*1024*1024):h.update(chunk)
   files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 state={'model':'docling-project/ScreenVLM','revision':info.sha,'license':info.card_data.license if info.card_data else 'unverified','date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'total_bytes':sum(x['bytes'] for x in files),'source':'https://huggingface.co/docling-project/ScreenVLM','evaluation_status':'MODEL AVAILABLE, NOT AN ACCURACY RESULT'}
 MANIFEST.write_text(json.dumps(state,indent=2)+'\n');print('SAVED',len(files),'BYTES',state['total_bytes'],flush=True)
if __name__=='__main__':main()
