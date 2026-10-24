#!/usr/bin/env python3
"""Repair evidence visibility, NOT benchmark gold, for independently labeled real sites.

Capture the actual target element within viewport using offline archived HTML.
Save true page pixels, target rect, source SHA, and intended text separately.
The predictor receives only the PNG or masked pixel region, never the gold.
This creates an evaluation corpus, NOT a UI defect detector.
"""
from __future__ import annotations
import hashlib,io,json,os,tempfile,time,datetime
from pathlib import Path
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

ROOT=Path('/data/var/perceptual-independent-20261024/uijudge-v0.4.0')
PAGES=ROOT/'real-html'
OUT=Path('/data/var/perceptual-independent-20261024/visible-ui-real')
MANIFEST=OUT/'anchored_manifest.json'
DRIVER='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
VIEWPORT=(1920,1080)
SCROLL_JS='''const e=arguments[0];let r=e.getBoundingClientRect();
  window.scrollTo({left:0,top:window.scrollY+r.top-290,behavior:'instant'});
  r=e.getBoundingClientRect();const s=getComputedStyle(e);
  let a=[];for(let x=e.parentElement;x&&a.length<14;x=x.parentElement){let q=getComputedStyle(x);let z=x.getBoundingClientRect();if(q.overflowX!=='visible'||q.overflowY!=='visible')a.push({overflowX:q.overflowX,overflowY:q.overflowY,bounds:[z.x,z.y,z.width,z.height]});}
  return {element_rect:[r.x,r.y,r.width,r.height],viewport:[window.innerWidth,window.innerHeight],scroll_y:window.scrollY,
  client:[e.clientWidth,e.clientHeight],scroll:[e.scrollWidth,e.scrollHeight],overflow:[s.overflowX,s.overflowY],
  text_content:e.textContent?.trim().slice(0,2500)??'',inner_text:e.innerText?.trim().slice(0,2500)??'',clipping_ancestors:a};'''

def main():
 if MANIFEST.exists():
  prior=json.loads(MANIFEST.read_text());print('RESUMING saved',len(prior.get('cases',{})),'not overwriting frames',flush=True)
 else:prior={'provenance':'UIJudgeBench v0.4.0 author-labeled real-site L1 layout pages. Offline Firefox 146.0 viewport after target anchored scrolling, no remote resources','date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'viewport_px':VIEWPORT,'cases':{}}
 items=json.loads((ROOT/'real_ui_frozen_manifest.json').read_text())['items'];items=sorted((x for x in items if x['track']=='layout' and x['task_level']=='L1'),key=lambda x:x['item_id'])
 assert len(items)==32
 OUT.mkdir(parents=True,exist_ok=True)
 o=Options();o.add_argument('-headless');o.set_preference('network.proxy.type',1)
 o.set_preference('network.proxy.http','127.0.0.1');o.set_preference('network.proxy.http_port',9)
 o.set_preference('network.proxy.ssl','127.0.0.1');o.set_preference('network.proxy.ssl_port',9)
 o.set_preference('permissions.default.image',1)
 with tempfile.TemporaryDirectory(prefix='perceptual-anchored-browser-',dir='/data/var/tmp') as tmp:
  o.profile=tmp
  driver=webdriver.Firefox(service=Service(executable_path=DRIVER,log_output=os.devnull),options=o)
  try:
   driver.set_page_load_timeout(16);driver.set_script_timeout(5)
   driver.set_window_size(*VIEWPORT)
   for idx,item in enumerate(items):
    key=item['item_id'];name=item['page_id'];p=PAGES/(name+'.html');start=time.monotonic()
    if key in prior['cases']:
     print('SKIP',idx+1,key,flush=True);continue
    entry={'item_id':key,'page_id':name,'criterion':item['criterion_code'],'split':item['split'],'gold_violation':item['ground_truth']=='no','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,'oracle_provenance':item['receipt'].get('door'),'selector':item['receipt']['selector']}
    try:
     if not p.is_file():raise RuntimeError('source missing')
     if b'<script' in p.read_bytes().lower():raise ValueError('unexpected active scripts in archived HTML')
     driver.get(p.as_uri());e=driver.find_element(By.CSS_SELECTOR,item['receipt']['selector'])
     m=driver.execute_script(SCROLL_JS,e)
     # Focus image: bounding box itself might extend beyond 1920px, crop visible
     # area around target with ~80 px margin to expose local context/borders.
     data=driver.get_screenshot_as_png();im=Image.open(io.BytesIO(data)).convert('RGB')
     x,y,w,h=m['element_rect'];left=max(0,int(x-75));top=max(0,int(y-65));right=min(im.width,int(x+min(w,1400)+75));bottom=min(im.height,int(y+min(h,210)+65))
     if right<=left or bottom<=top:raise RuntimeError('anchor outside viewport after scroll')
     # The image is true Firefox output; no drawn gold annotations or borders.
     thumb=im.crop((left,top,right,bottom))
     file=OUT/(hashlib.sha256(key.encode()).hexdigest()[:18]+'.png')
     if file.exists():raise RuntimeError('unexpected image already exists without manifest entry; do not overwrite')
     thumb.save(file)
     entry.update(status='ok',image_path=str(file),image_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),image_size=list(thumb.size),image_crop=[left,top,right,bottom],target_metadata=m,anchor_intersects_image=(x<im.width and y<im.height and x+w>0 and y+h>0))
    except Exception as e:entry.update(status='error',error=type(e).__name__+': '+str(e)[:200])
    entry['seconds']=round(time.monotonic()-start,3)
    prior['cases'][key]=entry;MANIFEST.write_text(json.dumps(prior,indent=2)+'\n')
    print('CAPTURE',idx+1,'/',len(items),name,entry['status'],'visible',entry.get('anchor_intersects_image'), 's',entry['seconds'],flush=True)
  finally:driver.quit()
 print('DONE',len(prior['cases']),'OK',sum(x['status']=='ok' for x in prior['cases'].values()),'ERROR',sum(x['status']!='ok' for x in prior['cases'].values()))
if __name__=='__main__':main()
