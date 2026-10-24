#!/usr/bin/env python3
"""Evaluate OCR text-contract and pixel-only contrast on already captured Firefox images."""
import json,time,statistics
from pathlib import Path
from PIL import Image
import numpy as np
import pytesseract
ROOT=Path(__file__).resolve().parents[2]/'docs/research/perceptual-browser-20261024'
def main():
 source=json.loads((ROOT/'firefox_manifest.json').read_text())['cases'];rows=[]
 for record in source:
  start=time.perf_counter();case=record['case']; image=Image.open(ROOT/(case+'-firefox.png')).convert('RGB')
  words=pytesseract.image_to_string(image,config='--psm 6',timeout=8)
  intended=record['expected_text'];missed=intended.lower() not in words.lower()
  prediction='defect' if missed else 'none';gold='none' if record['gold']=='none' else 'defect'
  a=np.asarray(image);roi=a[78:98,32:209,:].astype(float)
  contrast=float(np.percentile(roi.mean(axis=2),95)-np.percentile(roi.mean(axis=2),5))
  rows.append({'case':case,'gold':record['gold'],'binary_gold':gold,'ocr_prediction':prediction,'correct':gold==prediction,'ocr':words.strip()[:350], 'region_dynamic_range':round(contrast,2),'seconds':round(time.perf_counter()-start,3)})
 out={'method':'Firefox 146.0 captured screenshots; Tesseract 5.3.4 PSM6; separate intended text contract; 8 cases; not blinded holdout','cases':rows,'accuracy':sum(r['correct'] for r in rows)/len(rows),'seconds_sum':round(sum(r['seconds'] for r in rows),3)}
 (ROOT/'firefox_ocr_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
