#!/usr/bin/env python3
"""Real final HTML5 Firefox completion on different owner dock final output.

Separate isolated headless process, local file URLs, no production browser,
no upload. Intended as playback smoke test, not subjective perception judge.
"""
from pathlib import Path
import hashlib
import json
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

ROOT=Path('/data/var/perceptual-independent-20261024/owner_dock_cross_source_transfer_20261026')
GECKO='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
OUTPUT=ROOT/'firefox_transferred_output.json'
JS="""const v=document.querySelector('video');return v?{exists:true,currentTime:v.currentTime,duration:v.duration,ended:v.ended,paused:v.paused,readyState:v.readyState,videoWidth:v.videoWidth,videoHeight:v.videoHeight,error:v.error?{code:v.error.code,message:v.error.message}:null}:{exists:false};"""

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def play(driver,path):
 t=time.monotonic();driver.get(path.as_uri());s=driver.execute_script(JS)
 if not s.get('exists') or s['videoWidth']!=960 or s['videoHeight']!=540 or abs(s['duration']-14)>0.05:raise RuntimeError('wrong video dimensions/duration')
 result=driver.execute_async_script("const done=arguments[0];let v=document.querySelector('video');v.muted=true;v.currentTime=0;v.play().then(()=>done({ok:true})).catch(e=>done({ok:false,error:String(e)}));")
 if not result.get('ok'):raise RuntimeError(str(result))
 scores=[]
 for idx in range(52):
  v=driver.execute_script(JS);scores.append(v)
  if v['error']:raise RuntimeError('Media decoding '+str(v['error']))
  if v['ended']:break
  time.sleep(.4)
 last=scores[-1]
 return {'success':bool(last['ended'] and last['currentTime']>=13.9 and last['error'] is None),'first':s,'last':last,'wall_s':round(time.monotonic()-t,3),'samples':len(scores),'monotonic_media_time':all(b['currentTime']+0.001>=a['currentTime'] for a,b in zip(scores,scores[1:]))}

def main():
 if OUTPUT.exists():raise RuntimeError('Existing browser evidence; do not replay')
 encoded=json.loads((ROOT/'encodes.json').read_text());scored=json.loads((ROOT/'transfer_scored.json').read_text())
 assert scored['encoded_variants_SHA']==sha(ROOT/'encodes.json')
 options=Options();options.binary_location='/usr/bin/firefox';options.add_argument('-headless')
 options.set_preference('media.autoplay.default',0)
 options.set_preference('media.hardware-video-decoding.enabled',False)
 options.set_preference('gfx.webrender.software',True)
 options.set_preference('browser.cache.disk.enable',False)
 service=Service(executable_path=GECKO,log_output=str(ROOT/'firefox_geckodriver_isolated.log'))
 driver=None;results={};error=None
 try:
  driver=webdriver.Firefox(options=options,service=service)
  driver.set_window_size(1050,710);driver.set_page_load_timeout(20);driver.set_script_timeout(10)
  for name in ('clean_crf23','reversals_crf17'):
   item=encoded['variants'][name];p=Path(item['path']);assert sha(p)==item['sha256']
   results[name]={'SHA':item['sha256'],'playback':play(driver,p),'source_frame_faults':scored['cases'][name]['source_bound']['TP']+scored['cases'][name]['source_bound']['FP']}
   print('DOCK_TRANSFERENCE_REAL_HTML5_PLAY',name,'success',results[name]['playback']['success'],'time',results[name]['playback']['last']['currentTime'],'source_faults',results[name]['source_frame_faults'],flush=True)
 except Exception as exc:error=type(exc).__name__+': '+str(exc)[:500]
 finally:
  if driver:
   try:driver.quit()
   except Exception:pass
 report={'source_version':scored['reference_original_dock_MP4_SHA'],'original_protocol_SHA':sha(ROOT/'frozen_protocol.json'),'status':'PASS' if not error and len(results)==2 and all(x['playback']['success'] for x in results.values()) else 'REVIEW','error':error,'cases':results,'limits':'Headless Firefox native HTML5 local file playback verifies decoding/end state only, not visual quality or temporal frame correctness, no audio stream.'}
 tmp=OUTPUT.with_suffix('.writing');tmp.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');tmp.rename(OUTPUT)
 print('DOCK_TRANSFER_BROWSER_STATUS',report['status'],'sha',sha(OUTPUT))
if __name__=='__main__':main()
