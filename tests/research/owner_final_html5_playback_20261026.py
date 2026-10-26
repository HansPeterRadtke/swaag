#!/usr/bin/env python3
"""Browser-level end-user playback smoke test for owner final encoded MP4.

Launches a NEW temporary isolated headless Firefox WebDriver profile. No
production web server/browser profile or GPU model is modified. Not a human
perceptual fidelity test; browser playback must finish without media error.
"""
from pathlib import Path
import hashlib
import json
import os
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

BASE=Path('/data/var/perceptual-independent-20261024/final_encoded_owner_mp4_20261026')
VIDEO=Path('/data/var/3d-publication-20261024/approved_warehouse_door_rear_corner_v47.mp4')
EXPECTED='e2ef866ec79318a6b3812865a51922141ad02be66b555900da311171f49b4adb'
GECKO='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
RESULT=BASE/'browser_html5_playback_20261026.json'

def hash_file(p):
 with p.open('rb') as f:
  h=hashlib.sha256()
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def write_once(p,o):
 if p.exists():raise RuntimeError('evidence already exists - not replaying')
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');tmp.rename(p)

SCRIPT="""const v=document.querySelector('video');if(!v)return {video_found:false,href:document.location.href.slice(0,12)};return {video_found:true,ready_state:v.readyState,network_state:v.networkState,video_width:v.videoWidth,video_height:v.videoHeight,duration:v.duration,current_time:v.currentTime,paused:v.paused,ended:v.ended,muted:v.muted,playback_rate:v.playbackRate,media_error:v.error?{code:v.error.code,message:v.error.message}:null};"""

def main():
 if RESULT.exists():raise RuntimeError('Do not repeat already-frozen browser results')
 if hash_file(VIDEO)!=EXPECTED:raise RuntimeError('Video SHA changed')
 d=json.loads((BASE/'decoded_frame_evidence.json').read_text());assert d['video_sha256']==EXPECTED and d['source_matched_frame_sequence_property']=='PASS'
 options=Options();options.add_argument('-headless');options.binary_location='/usr/bin/firefox'
 options.set_preference('media.autoplay.default',0)
 options.set_preference('media.hardware-video-decoding.enabled',False)
 options.set_preference('gfx.webrender.software',True)
 options.set_preference('browser.cache.disk.enable',False)
 options.set_preference('toolkit.telemetry.enabled',False)
 options.set_preference('app.update.enabled',False)
 options.page_load_strategy='normal'
 service=Service(executable_path=GECKO,log_output=str(BASE/'geckodriver_isolated.log'))
 evidence={'date':'2026-10-26','status':'REVIEW','source_sha256':EXPECTED,'decoded_video_source_fidelity_sha256':hash_file(BASE/'decoded_frame_evidence.json'),'browser':'Firefox isolated, headless, local file native HTML5 video viewer, no web server','controls':'Expected ended currentTime~14 s; stable 960x540; no media errors, nondecreasing times. Does not verify source camera details or visual film quality.','sampled_states':[],'screenshots_saved':[]}
 browser=None;start=time.perf_counter()
 try:
  browser=webdriver.Firefox(service=service,options=options)
  browser.set_window_size(1060,750)
  browser.set_page_load_timeout(20)
  browser.set_script_timeout(10)
  browser.get(VIDEO.as_uri())
  first=browser.execute_script(SCRIPT)
  evidence['first_view']=first
  if not first.get('video_found'):raise RuntimeError('No HTML5 video element in native Firefox file media viewer')
  if first['video_width']!=960 or first['video_height']!=540 or abs(first['duration']-14.)>.05:
   raise RuntimeError('Video intrinsic size or duration changed')
  play=browser.execute_async_script('''const done=arguments[0];const v=document.querySelector('video'); v.muted=true; v.currentTime=0; v.play().then(()=>done({ok:true})).catch(e=>done({ok:false,error:String(e)}));''')
  evidence['play_command']=play
  if not play.get('ok'):raise RuntimeError('HTML5 media playback rejected: '+str(play))
  old=-1;max_delta=0
  for i in range(45):
   state=browser.execute_script(SCRIPT);state['wall_elapsed_s']=round(time.perf_counter()-start,3)
   evidence['sampled_states'].append(state)
   t=state['current_time']
   if t+0.005<old:raise RuntimeError('HTML5 video currentTime decreased unexpectedly')
   max_delta=max(max_delta,t-old);old=t
   if len(evidence['screenshots_saved'])==0 and 5.0<=t<=9.0:
    p=BASE/'actual_browser_playing_warehouse_film.png'
    if p.exists():raise RuntimeError('Existing browser screenshot, refuse overwrite')
    browser.save_screenshot(str(p));evidence['screenshots_saved'].append({'file':str(p),'sha256':hash_file(p),'time_s':round(t,3)})
   if state['media_error'] is not None:raise RuntimeError('HTML5 decoder returned a media error: '+str(state['media_error']))
   if state['ended']:break
   time.sleep(.5)
  last=evidence['sampled_states'][-1]
  evidence['last_view']=last
  evidence['elapsed_wall_s']=round(time.perf_counter()-start,3)
  evidence['media_time_max_interval_s']=round(max_delta,3)
  evidence['sampled_states_n']=len(evidence['sampled_states'])
  evidence['all_media_times_non_decreasing']=True
  evidence['ended_successfully']=bool(last['ended'] and last['media_error'] is None and last['current_time']>=13.9 and last['ready_state']>=2)
  evidence['status']='PASS' if evidence['ended_successfully'] else 'REVIEW'
 except Exception as err:
  evidence['status']='REVIEW';evidence['failure']=type(err).__name__+': '+str(err)[:700];evidence['elapsed_wall_s']=round(time.perf_counter()-start,3)
 finally:
  if browser is not None:
   try:browser.quit()
   except Exception as e:evidence['teardown_warning']=str(e)[:200]
  write_once(RESULT,evidence)
 print('FIREFOX_USER_FACE_PLAYER',json.dumps({k:v for k,v in evidence.items() if k not in ('sampled_states','controls','first_view')},sort_keys=True)[:3400])

if __name__=='__main__':main()
