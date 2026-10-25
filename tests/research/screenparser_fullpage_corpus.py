#!/usr/bin/env python3
"""Real Firefox full-page viewport crops with known externally labeled target.

From 32 independent UIJudgeBench v0.4.0 authored real-site HTML cases,
scroll target into the viewport and capture the WHOLE 1920x1080 scene (not
small target-only crop). This better fits a trained web page element detector.
Gold target rect and screenshot SHA kept outside predictions, never passed to
ScreenParser. UIJudgeBench model labels remain DOM full-page semantic labels.
No modifications to prior cropped screens, originals, or active browser.
"""
from pathlib import Path
import hashlib,json,tempfile,os,time,io,datetime
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from PIL import Image
ROOT=Path('/data/var/perceptual-independent-20261024')
ORIGINAL=ROOT/'visible-ui-real/anchored_manifest.json'
ARCHIVE=ROOT/'uijudge-v0.4.0/real-html'
OUT=ROOT/'screenparser-fullpage-real'
MANIFEST=OUT/'fullpage_manifest.json'
DRIVER='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
VIEWPORT=(1920,1080)
SCROLL_JS='''let r=arguments[0].getBoundingClientRect();window.scrollTo({top:window.scrollY+r.top-290,behavior:'instant'});r=arguments[0].getBoundingClientRect();return {rect:[r.x,r.y,r.width,r.height],viewport:[window.innerWidth,window.innerHeight],scrollY:window.scrollY,tag:arguments[0].tagName,ariaRole:arguments[0].getAttribute('role')||'',text:arguments[0].innerText?.trim().slice(0,300)||''};'''

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 old=json.loads(ORIGINAL.read_text())['cases'];assert len(old)==32
 state=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'source':'UIJudgeBench v0.4.0, original archived human/browser verified real sites, Firefox 146 full 1920x1080 target-centered viewport','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_anchored_manifest_sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'cases':{}}
 o=Options();o.add_argument('-headless')
 o.set_preference('network.proxy.type',1)
 for proto in ('http','ssl'):
  o.set_preference('network.proxy.'+proto,'127.0.0.1');o.set_preference('network.proxy.'+proto+'_port',9)
 with tempfile.TemporaryDirectory(prefix='ScreenParserWholePage-',dir='/data/var/tmp') as td:
  o.profile=td;driver=webdriver.Firefox(service=Service(executable_path=DRIVER,log_output=os.devnull),options=o)
  try:
   driver.set_window_size(*VIEWPORT)
   driver.set_page_load_timeout(15)
   for i,(key,row) in enumerate(sorted(old.items())):
    if key in state['cases']:
     print('PRESERVE',key,flush=True);continue
    page=ARCHIVE/(row['page_id']+'.html');src=page.read_bytes()
    if hashlib.sha256(src).hexdigest()!=row['source_sha256']:raise ValueError('source html changed unexpectedly '+str(page))
    if b'<script' in src.lower():raise ValueError('unexpected source scripts')
    t=time.monotonic();result={'page_id':row['page_id'],'criterion':row['criterion'],'split':row['split'],'independent_gold_violation':row['gold_violation'],'selector':row['selector'],'source_file_sha256':row['source_sha256'],'status':'started'}
    try:
     driver.get(page.as_uri());element=driver.find_element(By.CSS_SELECTOR,row['selector'])
     info=driver.execute_script(SCROLL_JS,element)
     data=driver.get_screenshot_as_png();im=Image.open(io.BytesIO(data));W,H=im.size
     assert W>=1200 and H>=700
     x,y,w,h=info['rect'];x1=max(0,x);y1=max(0,y);x2=min(W,x+w);y2=min(H,y+h)
     assert x1<x2 and y1<y2
     dest=OUT/(hashlib.sha256(key.encode()).hexdigest()[:18]+'.png')
     if dest.exists():raise RuntimeError('old screenshot exists without checkpoint; do not overwrite')
     dest.write_bytes(data)
     result.update(status='ok',image_path=str(dest),image_sha256=hashlib.sha256(data).hexdigest(),image_size=[W,H],target_observed_rect=info['rect'],target_visible_rect=[x1,y1,x2,y2],target_fraction_visible=(x2-x1)*(y2-y1)/(w*h) if w*h>0 else None,target_dom_tag=info['tag'],target_expected_visible_text=info['text'],target_role=info['ariaRole'],seconds=round(time.monotonic()-t,2))
    except Exception as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc)[:170],seconds=round(time.monotonic()-t,2))
    state['cases'][key]=result;MANIFEST.write_text(json.dumps(state,indent=2)+'\n')
    print('CAPTURE',i+1,'/',len(old),result['status'],row['page_id'],'pixels',result.get('image_size'),'seconds',result['seconds'],flush=True)
  finally:driver.quit()
 print('COLLECTED',len(state['cases']),'OK',sum(x['status']=='ok' for x in state['cases'].values()))
if __name__=='__main__':main()
