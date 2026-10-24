#!/usr/bin/env python3
"""Fetch all original-color, real MVTec LOCO pushpins subset images read-only.

Keep CC-BY-NC-SA 4.0 photos and labels outside Git; only a SHA manifest
and downstream aggregated results may be committed. Never execute remote code.
"""
from pathlib import Path
import json,urllib.request,hashlib
BASE='https://huggingface.co/datasets/saiphyohein/spec2inspect-images/resolve/main/'
API='https://huggingface.co/api/datasets/saiphyohein/spec2inspect-images/tree/main?recursive=false'
ROOT=Path('/data/var/perceptual-independent-20261024/mvtec-loco-subset')

def download(name):
 path=ROOT/name
 if not path.is_file():
  with urllib.request.urlopen(urllib.request.Request(BASE+name+'?download=true',headers={'User-Agent':'perceptual-research/1.0'}),timeout=15) as resp:data=resp.read(600000)
  if len(data)>=600000:raise ValueError('file over size budget '+name)
  path.write_bytes(data)
 return {'file':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with urllib.request.urlopen(API,timeout=10) as f:data=json.load(f)
 names=sorted(x['path'] for x in data if x['path'].endswith('.jpg') and len(x['path'].split('_'))==2)
 receipts=[]
 for name in names:receipts.append({**download(name),'gold':'anomaly' if name.startswith('anomaly_') else 'good'})
 for aux in ['metadata.csv','README.md']:download(aux)
 manifest={'source':'saiphyohein/spec2inspect-images, original un-recolored yellow JPEG test images attributed to MVTec LOCO AD pushpins','provider_license':'CC BY-NC-SA 4.0; noncommercial research only','note':'Public repackage of a subset of MVTec real industrial photographs. JPEG conversion; not a representative industrial/machinery dataset. Labels are externally provided, zero local gold construction.','cases':receipts}
 (ROOT/'original_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('ORIGINAL IMAGES',len(receipts),'GOOD',sum(x['gold']=='good' for x in receipts),'ANOMALIES',sum(x['gold']=='anomaly' for x in receipts))
if __name__=='__main__':main()
