#!/usr/bin/env python3
"""Reproducible CPU-only geometry / contrast acceptance fixtures; no VLM claims."""
import json, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'docs'/'research'/'perceptual-fixtures-20261024'
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
CASES={
 'clean':('none',[(15,15,180,45),(15,60,220,90)],(0,0,0),(255,255,255)),
 'clipped':('clipped_text',[(15,15,360,45),(15,60,220,90)],(0,0,0),(255,255,255)),
 'overlap':('overlap',[(15,15,190,45),(95,30,260,60)],(0,0,0),(255,255,255)),
 'low_contrast_error':('low_contrast_error',[(15,15,180,45),(15,60,220,90)],(242,242,242),(255,255,255)),
 'near_touch':('none',[(15,15,190,45),(191,15,295,45)],(0,0,0),(255,255,255)),
 'dark_legible':('none',[(15,15,180,45),(15,60,220,90)],(255,255,255),(20,20,20)),
}
def ratio(a,b):
 def lin(v):
  v=v/255; return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4
 def l(rgb):return sum(w*lin(v) for w,v in zip((.2126,.7152,.0722),rgb))
 x,y=sorted((l(a),l(b)));return (y+.05)/(x+.05)
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def inspect(rects,fg,bg,canvas=(300,120)):
 if any(x<0 or y<0 or x2>canvas[0] or y2>canvas[1] for x,y,x2,y2 in rects):return 'clipped_text'
 if any(overlap(a,b)>0 for i,a in enumerate(rects) for b in rects[i+1:]):return 'overlap'
 if ratio(fg,bg)<4.5:return 'low_contrast_error'
 return 'none'
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 results=[]
 for name,(gold,rects,fg,bg) in CASES.items():
  start=time.perf_counter()
  im=Image.new('RGB',(300,120),bg);draw=ImageDraw.Draw(im)
  for i,rect in enumerate(rects):
   draw.rectangle(rect,outline=(110,110,110),width=1)
   draw.text((rect[0]+3,rect[1]+2),'Error: inspect' if name=='low_contrast_error' else f'Label {i+1}',font=FONT,fill=fg)
  im.save(ROOT/f'{name}.png')
  prediction=inspect(rects,fg,bg)
  results.append(dict(case=name,ground_truth=gold,predicted=prediction,correct=gold==prediction,seconds=round(time.perf_counter()-start,6),geometry=rects,foreground=fg,background=bg))
 labels=sorted(set(x['ground_truth'] for x in results))
 stats={label:{'tp':sum(x['ground_truth']==label and x['predicted']==label for x in results),'fp':sum(x['ground_truth']!=label and x['predicted']==label for x in results),'fn':sum(x['ground_truth']==label and x['predicted']!=label for x in results)} for label in labels}
 for v in stats.values():v.update(precision=v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else None,recall=v['tp']/(v['tp']+v['fn']) if v['tp']+v['fn'] else None)
 out={'method':'known DOM-style bounding boxes + WCAG contrast formula; not image understanding','cases':results,'scores':stats,'passed':sum(x['correct'] for x in results),'total':len(results)}
 (ROOT/'cpu_results.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'passed':out['passed'],'total':out['total'],'scores':stats,'output':str(ROOT)},indent=2))
 assert out['passed']==out['total']
if __name__=='__main__':main()
