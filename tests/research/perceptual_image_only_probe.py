#!/usr/bin/env python3
"""Bounded CPU probes on rendered pixels, with separately constructed ground truth.

No model, GPU, browser or running production service is accessed. This deliberately
preserves false positives and false negatives, not an acceptance gate.
"""
from __future__ import annotations
import json, math, platform, statistics, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT=Path(__file__).resolve().parents[2]/'docs'/'research'/'perceptual-image-only-20261024'
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
SMALL=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)

def gui_image(case):
 dark=case=='legitimate_dark_theme'; fg=(244,244,244) if dark else (18,18,18)
 bg=(28,32,38) if dark else (245,245,245); panel=(40,44,48) if dark else (255,255,255)
 im=Image.new('RGB',(440,240),bg); d=ImageDraw.Draw(im)
 d.text((20,16),'Operations control',font=FONT,fill=fg)
 d.rectangle((20,60,220,110),fill=panel,outline=(100,100,100),width=1)
 d.rectangle((20,130,320,182),fill=panel,outline=(100,100,100),width=1)
 txt='System ready'; loc=(28,74)
 if case=='legitimate_text_revision':txt='Pump ready'
 if case=='legitimate_spacing':loc=(30,74)
 if case=='legitimate_right_align':txt='Ready';loc=(154,74)
 if case=='clipped_text':txt='System ready for maintenance'
 if case=='low_contrast':fg=(242,242,242)
 if case=='overlap':
  d.text(loc,'System ready',font=FONT,fill=fg)
  d.text((28,81),'Maintenance alert',font=FONT,fill=fg)
 else:
  if case=='clipped_text':
   # Real crop of rendered glyphs within panel; gold deliberately not inferred by detector.
   overlay=Image.new('RGB',im.size,panel); od=ImageDraw.Draw(overlay)
   od.text(loc,txt,font=FONT,fill=fg)
   im.paste(overlay.crop((28,69,218,108)),(28,69));d=ImageDraw.Draw(im)
  else:d.text(loc,txt,font=FONT,fill=fg)
 d.text((28,146),'Pressure: 240 kPa',font=FONT,fill=(244,244,244) if dark else (18,18,18))
 return im

def label_scores(gold,pred):
 labels=sorted(set(gold)|set(pred));out={}
 for k in labels:
  tp=sum(g==k and p==k for g,p in zip(gold,pred));fp=sum(g!=k and p==k for g,p in zip(gold,pred));fn=sum(g==k and p!=k for g,p in zip(gold,pred))
  out[k]={'tp':tp,'fp':fp,'fn':fn,'precision':(tp/(tp+fp) if tp+fp else None),'recall':(tp/(tp+fn) if tp+fn else None)}
 return out

def gui_probe():
 cases={'clean':'none','clipped_text':'defect','overlap':'defect','low_contrast':'defect','legitimate_dark_theme':'none','legitimate_text_revision':'none','legitimate_spacing':'none','legitimate_right_align':'none'}
 images={k:gui_image(k) for k in cases};ref=np.asarray(images['clean'],dtype=np.int16)
 samples=[]
 for name,gold in cases.items():
  start=time.perf_counter();a=np.asarray(images[name]);images[name].save(ROOT/f'gui_{name}.png')
  # Screenshot pixels only; no case-specific metadata is consulted by the detectors.
  delta=float(np.abs(a.astype(np.int16)-ref).mean());pred_change='defect' if delta>1.5 else 'none'
  roi=cv2.cvtColor(a[70:104,30:211],cv2.COLOR_RGB2GRAY)
  p10,p90=np.percentile(roi,[10,90]); contrast_low=bool((p90-p10)<40)
  samples.append(dict(case=name,gold=gold,change_detector=pred_change,low_contrast_flag=contrast_low,reference_mean_abs_difference=round(delta,3),seconds=round(time.perf_counter()-start,5)))
 out={'cases':samples,'change_detector':label_scores([x['gold'] for x in samples],[x['change_detector'] for x in samples])}
 return out

def ocr_probe():
 import pytesseract
 cases={'clean':('none','System ready'),'clipped_text':('defect','System ready for maintenance'),'overlap':('defect','System ready'),'low_contrast':('defect','System ready'),'legitimate_dark_theme':('none','System ready'),'legitimate_text_revision':('none','Pump ready'),'legitimate_spacing':('none','System ready'),'legitimate_right_align':('none','Ready')}
 rows=[]
 for name,(gold,expected) in cases.items():
  t=time.perf_counter()
  try:
   observed=pytesseract.image_to_string(gui_image(name),config='--psm 6',timeout=5).strip()
   contains=expected.casefold() in observed.casefold()
   prediction='none' if contains else 'defect'
   rows.append({'case':name,'gold':gold,'predicted':prediction,'expected_text':expected,'ocr':observed[:260],'seconds':round(time.perf_counter()-t,3)})
  except Exception as e:rows.append({'case':name,'gold':gold,'error':type(e).__name__+': '+str(e)[:100]})
 valid=[r for r in rows if 'predicted' in r]
 return {'method':'Tesseract OCR of image pixels compared to independently defined text contract; contract NOT image-only','cases':rows,'classification':label_scores([x['gold'] for x in valid],[x['predicted'] for x in valid])}

def diagram_probe():
 cases={'clean':'none','broken_connection':'broken','legitimate_recolor':'none','overprinted_connection':'overprint'};results=[]
 for name,gold in cases.items():
  start=time.perf_counter();im=Image.new('RGB',(400,170),'white');d=ImageDraw.Draw(im)
  ink=(40,90,180) if name=='legitimate_recolor' else (15,15,15)
  d.rectangle((20,48,105,105),outline=ink,width=2);d.rectangle((295,48,380,105),outline=ink,width=2)
  d.text((52,65),'A',font=FONT,fill=ink);d.text((326,65),'B',font=FONT,fill=ink)
  if name=='broken_connection':d.line((105,77,176,77),fill=ink,width=3);d.line((240,77,295,77),fill=ink,width=3)
  else:d.line((105,77,295,77),fill=ink,width=3)
  if name=='overprinted_connection':d.text((183,65),'X',font=FONT,fill=ink)
  im.save(ROOT/f'diagram_{name}.png');a=np.asarray(im)
  # Edge scanning uses only pixels in the fixed line corridor, not drawing metadata.
  stripe=a[75:80,110:290,:].astype('float64').mean(axis=2)
  density=float(np.any(stripe<225,axis=0).mean());pred='broken' if density<.75 else 'none'
  results.append(dict(case=name,gold=gold,predicted=pred,line_occupancy=round(density,3),seconds=round(time.perf_counter()-start,5)))
 return {'cases':results,'classification':label_scores([x['gold'] for x in results],[x['predicted'] for x in results])}

def machine_probe():
 # Engine/scene coordinates are independently constructed; these are NOT 2D image-only judgments.
 cases={
  'door_above_floor':('none',(0.0,0.0,0.01,1.8,0.2,2.0)),
  'door_inside_floor':('penetration',(0.0,0.0,-0.16,1.8,0.2,1.83)),
  'legitimate_recessed_track':('none',(0.0,0.0,-0.05,1.8,0.2,0.10)),
  'fork_above_floor':('none',(0.0,0.0,0.04,2.0,0.25,0.12)),
 }
 rows=[]
 for name,(gold,box) in cases.items():
  t=time.perf_counter();penetration=max(0,-box[2]);pred='penetration' if penetration>.02 else 'none'
  rows.append({'case':name,'gold':gold,'predicted':pred,'penetration_m':penetration,'seconds':round(time.perf_counter()-t,6)})
 return {'measurement':'geometry-only lower z against z=0 floor; no texture or semantic understanding','cases':rows,'classification':label_scores([x['gold'] for x in rows],[x['predicted'] for x in rows])}

def render_video(name):
 frames=[];positions=[]
 for t in range(16):
  x=12+3*t
  if name in ('jump','camera_cut') and t>=9:x+=31
  if name=='valid_hold' and t>=9:x=12+3*(t-1)
  if name=='freeze' and t==9:x=12+3*8
  if name=='stationary':x=18
  bg=(230,230,230) if name not in ('cut','camera_cut') or t<8 else (35,35,35)
  im=np.zeros((96,180,3),dtype=np.uint8);im[:]=bg
  # Red target remains detectable despite the unrelated valid hard cut.
  cv2.rectangle(im,(x,35),(x+15,52),(35,35,220),thickness=-1)
  frames.append(im);positions.append(x)
 return frames,positions

def video_probe():
 gold={'smooth':'none','jump':'jump','freeze':'freeze','cut':'none','stationary':'none','valid_hold':'none','camera_cut':'none'};results=[]
 for name,expected in gold.items():
  t0=time.perf_counter();frames,_=render_video(name)
  # Observe the object from frame pixels using color segmentation, with no gold motion metadata.
  centers=[]
  for im in frames:
   mask=(im[:,:,2]>170)&(im[:,:,0]<90)&(im[:,:,1]<90)
   yy,xx=np.nonzero(mask);centers.append(float(np.median(xx)) if len(xx)>0 else None)
  displacement=[b-a for a,b in zip(centers[:-1],centers[1:])]
  median_step=float(statistics.median(displacement))
  jump=any(abs(v-median_step)>12 for v in displacement)
  freeze=(median_step>1 and any(abs(v)<.5 for v in displacement))
  predicted='jump' if jump else 'freeze' if freeze else 'none'
  # A naive global difference fires on legitimate cuts, retained as negative control.
  diffs=[float(cv2.absdiff(a,b).mean()) for a,b in zip(frames[:-1],frames[1:])]
  naive_any_spike=any(v>30 for v in diffs)
  results.append(dict(case=name,gold=expected,predicted=predicted,naive_global_spike=naive_any_spike,mean_step=round(median_step,3),max_frame_change=round(max(diffs),3),seconds=round(time.perf_counter()-t0,5)))
  if name in ('smooth','jump','freeze','cut'):
   # Save a compact inspectable contact sheet, not a large video or decoder work.
   sheet=np.concatenate([frames[i] for i in (7,8,9,10)],axis=1)
   cv2.imwrite(str(ROOT/f'video_{name}_frames.png'),sheet)
 return {'cases':results,'classification':label_scores([x['gold'] for x in results],[x['predicted'] for x in results])}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 out={'versions':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'Pillow':__import__('PIL').__version__},'scope':'CPU-only synthetic pixel comparisons plus separate 3D scene geometry check; no VLM, no real production media evaluation', 'gui':gui_probe(),'ocr_contract':ocr_probe(),'diagram':diagram_probe(),'mechanical':machine_probe(),'video':video_probe()}
 (ROOT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 for sec in ('gui','ocr_contract','diagram','mechanical','video'):
  cases=out[sec]['cases'];print(sec, 'samples',len(cases),'matches',sum(x['gold']==x.get('predicted',x.get('change_detector')) for x in cases),'/',len(cases));print('errors',[(x['case'],x['gold'],x.get('predicted',x.get('change_detector'))) for x in cases if x['gold']!=x.get('predicted',x.get('change_detector'))])
 print('SAVED',ROOT/'results.json')
if __name__=='__main__':main()
