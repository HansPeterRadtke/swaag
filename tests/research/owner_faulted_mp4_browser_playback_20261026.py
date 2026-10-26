#!/usr/bin/env python3
"""Real FF HTML5 user-facing playback on clean and injected-fault owner videos.

Uses isolated headless Firefox with no production profile and no external
server; directly opens H.264 files from /data/var research directory. Tests
player success, NOT subjective rendered continuity or 3D authenticity.
"""
import hashlib
import json
from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

BASE=Path('/data/var/perceptual-independent-20261024/owner_real_encoded_fault_injection_20261026')
EVID=BASE/'browser_faulted_video_and_clean_control_20261026.json'
GECKO='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
JAVA="""const v=document.querySelector('video');return v?{found:true,paused:v.paused,ended:v.ended,readyState:v.readyState,networkState:v.networkState,duration:v.duration,currentTime:v.currentTime,width:v.videoWidth,height:v.videoHeight,error:v.error?{code:v.error.code,message:v.error.message}:null}:{found:false};"""

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def put(path,obj):
 if path.exists():raise RuntimeError('frozen receipt already exists')
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.rename(path)

def play(driver,path):
 start=time.monotonic();driver.get(path.as_uri())
 first=driver.execute_script(JAVA)
 if not first['found'] or first['width']!=960 or first['height']!=540 or abs(first['duration']-14.)>.05:raise RuntimeError('not the expected actual video viewer/stream')
 result=driver.execute_async_script('''const done=arguments[0];const v=document.querySelector('video');v.muted=true;v.currentTime=0;v.play().then(()=>done({ok:true})).catch(e=>done({ok:false,reason:String(e)}));''')
 if not result.get('ok'):raise RuntimeError('HTML5 play refused '+str(result))
 sampled=[]
 for i in range(58):
  d=driver.execute_script(JAVA);d['elapsed_wall_s']=round(time.monotonic()-start,3);sampled.append(d)
  if d['error'] is not None:raise RuntimeError('browser media error '+str(d['error']))
  if d['ended']:break
  time.sleep(.4)
 last=sampled[-1]
 good=bool(last['ended'] and last['error'] is None and last['currentTime']>=13.9 and last['readyState']>=2)
 return {'success':good,'first':first,'last':last,'wall_s':round(time.monotonic()-start,3),'samples':len(sampled),'time_monotonic':all(a['currentTime']<=b['currentTime']+.001 for a,b in zip(sampled,sampled[1:])),'timeline_first_7':[{'time':round(q['currentTime'],2),'elapsed':q['elapsed_wall_s']} for q in sampled[:7]],'already_encoded_file_no_modification':True}

def main():
 if EVID.exists():raise RuntimeError('browser receipts already frozen')
 enc=json.loads((BASE/'encodes.json').read_text());scored=json.loads((BASE/'final_video_fault_score.json').read_text())
 assert enc['source_frame_receipt_sha256']==scored['source_decoded_original_PNG_SHA256']
 opts=Options();opts.binary_location='/usr/bin/firefox';opts.add_argument('-headless');opts.page_load_strategy='normal'
 opts.set_preference('media.autoplay.default',0)
 opts.set_preference('media.hardware-video-decoding.enabled',False)
 opts.set_preference('gfx.webrender.software',True)
 opts.set_preference('browser.cache.disk.enable',False)
 opts.set_preference('toolkit.telemetry.enabled',False)
 srv=Service(executable_path=GECKO,log_output=str(BASE/'research_firefox_geckodriver.log'))
 report={'source':'actual reencoded owner video and exact controlled reversed-frame version from same source, same browser','status':'REVIEW','kind':'native HTML5 file playback, no external upload, no production profile','gold_edits_from_prior_frozen_manifest':enc['variants']['short_reversals']['edit_positions'],'measurements':{}}
 browser=None
 try:
  browser=webdriver.Firefox(options=opts,service=srv)
  browser.set_window_size(1040,710)
  browser.set_page_load_timeout(20)
  browser.set_script_timeout(10)
  for name in ('clean_reencoded','short_reversals'):
   path=Path(enc['variants'][name]['path']);assert sha(path)==enc['variants'][name]['sha256']
   report['measurements'][name]={'sha256':sha(path),'playback':play(browser,path),'source_FIDELITY_flags':scored['cases'][name]['source_bound_detection']['TP']+scored['cases'][name]['source_bound_detection']['FP']}
   print('FINAL_BROWSER_FAULTED_VIDEO_PLAYBACK',name,report['measurements'][name],flush=True)
  report['status']='PASS' if all(x['playback']['success'] for x in report['measurements'].values()) else 'REVIEW'
 except Exception as e:
  report['status']='REVIEW';report['error']=type(e).__name__+': '+str(e)[:750]
 finally:
  if browser:
   try:browser.quit()
   except Exception as e:report['teardown_warning']=str(e)[:160]
  put(EVID,report)
 print('BROWSER_FAULTED_AND_CLEAN_MEDIA_DONE',report['status'],sha(EVID))

if __name__=='__main__':main()
