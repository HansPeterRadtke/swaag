#!/usr/bin/env python3
"""Cross-browser real-site screenshot / DOM geometry check on UIJudgeBench v0.4.0.

Raw archived HTML is executed only as static local file, no external navigation.
The oracle contains Chrome/benchmark measured independent ground truth. The
local Firefox detector receives the target CSS selector and TASK criterion,
NEVER the yes/no label or reference measurement. This is an instrumented DOM
analyzer, explicitly NOT screenshot-only accuracy. Pages may be off-screen.
"""
from __future__ import annotations
import datetime,hashlib,json,os,pathlib,urllib.request,tempfile,time,sys
ROOT=pathlib.Path('/data/var/perceptual-independent-20261024/uijudge-v0.4.0')
PAGES=ROOT/'real-html';INPUT=ROOT/'real_ui_frozen_manifest.json';OUT=ROOT/'firefox_dom_independent.json'
TOOL=pathlib.Path('/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver')
BASE='https://huggingface.co/datasets/gojiberries/uijudge-bench/resolve/v0.4.0/corpus/real/'
DOM_SCRIPT='''const e=arguments[0];const s=getComputedStyle(e);return {\
  scrollWidth:e.scrollWidth,clientWidth:e.clientWidth,scrollHeight:e.scrollHeight,clientHeight:e.clientHeight,\
  computedOverflowX:s.overflowX,computedOverflowY:s.overflowY,\
  rect:{x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height},\
  rootScrollWidth:document.documentElement.scrollWidth,innerWidth:window.innerWidth,\
  pageScrollWidth:document.body.scrollWidth,documentHeight:document.documentElement.scrollHeight};'''

def fetch(page):
 dest=PAGES/(page+'.html')
 if not dest.exists():
  u=BASE+page+'/page.html?download=true'
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'perceptual-research/1.0'}),timeout=16) as f:data=f.read(1600000)
  if len(data)>1500000:raise RuntimeError('HTML > 1.5MB cap')
  if b'<script' in data.lower():raise RuntimeError('Refuse page with active scripts')
  dest.write_bytes(data)
 return dest

def metric(page,criterion,selector,driver):
 from selenium.webdriver.common.by import By
 driver.get((PAGES/(page+'.html')).as_uri())
 driver.set_window_size(1920,1080)
 root=driver.execute_script('return {sw:document.documentElement.scrollWidth,ww:window.innerWidth,bodySW:document.body.scrollWidth}')
 elem=driver.find_element(By.CSS_SELECTOR,selector)
 data=driver.execute_script(DOM_SCRIPT,elem)
 if criterion=='redecheck:viewport-protrusion':bad=(root['sw']>root['ww']+1)
 elif criterion=='redecheck:element-protrusion':
  dx=data['scrollWidth']-data['clientWidth'];dy=data['scrollHeight']-data['clientHeight']
  bad=(dx>1 and data['computedOverflowX'] in ['hidden','clip','auto','scroll']) or (dy>1 and data['computedOverflowY'] in ['hidden','clip','auto','scroll'])
 else:raise ValueError('unsupported criterion '+criterion)
 return {'predicted_violation':bool(bad),'measured':data}

def main():
 from selenium import webdriver
 from selenium.webdriver.firefox.service import Service
 from selenium.webdriver.firefox.options import Options
 m=json.loads(INPUT.read_text());items=[r for r in m['items'] if r['track']=='layout' and r['task_level']=='L1']
 assert len(items)==32,'external dataset changed'
 selected=[]
 for r in items:
  # Benign selector included with both positive/negative control labels; only location given to detector.
  selector=(r.get('receipt') or {}).get('selector')
  if not selector:raise ValueError('missing element selector for '+r['item_id'])
  selected.append((r['page_id'],r['criterion_code'],selector,r['item_id'],r['ground_truth'],r['split']))
 PAGES.mkdir(parents=True,exist_ok=True)
 for i,page in enumerate(sorted({r[0] for r in selected})):
  try:fetch(page)
  except Exception as e:print('FETCH_FAILURE',page,type(e).__name__,str(e)[:140],flush=True)
 state=json.loads(OUT.read_text()) if OUT.exists() else {'source':'UIJudgeBench v0.4.0 real-site L1 layout, author verified receipts used only for gold/selector','method':'Firefox 146.0 browser DOM scroll/overflow, element locator provided; NOT screenshot-only','cases':{},'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 opts=Options();opts.add_argument('-headless');opts.set_preference('network.proxy.type',1);opts.set_preference('network.proxy.http','127.0.0.1');opts.set_preference('network.proxy.http_port',9);opts.set_preference('network.proxy.ssl','127.0.0.1');opts.set_preference('network.proxy.ssl_port',9);opts.set_preference('permissions.default.geo',2)
 driver=None
 with tempfile.TemporaryDirectory(prefix='visual-browser-geometry-',dir='/data/var/tmp') as prof:
  opts.profile=prof
  try:
   driver=webdriver.Firefox(service=Service(executable_path=str(TOOL),log_output=os.devnull),options=opts)
   driver.set_page_load_timeout(18); driver.set_script_timeout(6)
   for i,(page,criterion,selector,item_id,gold,split) in enumerate(selected):
    if item_id in state['cases']:continue
    start=time.monotonic()
    p=PAGES/(page+'.html')
    entry={'page_id':page,'criterion':criterion,'selector':selector,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,'gold':'violation' if gold=='no' else 'clean','split':split}
    if not p.exists():entry.update(status='missing_file')
    else:
     try:entry.update(status='ok',**metric(page,criterion,selector,driver))
     except Exception as e:entry.update(status='error',error=type(e).__name__+': '+str(e)[:180])
    entry['seconds']=round(time.monotonic()-start,3)
    state['cases'][item_id]=entry;OUT.write_text(json.dumps(state,indent=2)+'\n')
    print('CASE',i+1,'/',len(selected),page,entry['status'],entry.get('predicted_violation'),entry['seconds'],flush=True)
  finally:
   if driver is not None:driver.quit()
 out={}
 for split in ['dev','test']:
  a=[r for r in state['cases'].values() if r['split']==split]
  counts={k:0 for k in ['TP','FP','TN','FN','ABSTAIN']}
  for x in a:
   if x.get('status')!='ok':counts['ABSTAIN']+=1;continue
   bad=x['gold']=='violation';p=x['predicted_violation']
   counts['TP' if bad and p else 'FP' if p else 'FN' if bad else 'TN']+=1
  out[split]={'n':len(a),'counts':counts,'precision':counts['TP']/(counts['TP']+counts['FP']) if counts['TP']+counts['FP'] else None,'recall':counts['TP']/(counts['TP']+counts['FN']) if counts['TP']+counts['FN'] else None}
 state['scores']=out;OUT.write_text(json.dumps(state,indent=2)+'\n');print('SCORES',out,flush=True)
if __name__=='__main__':main()
