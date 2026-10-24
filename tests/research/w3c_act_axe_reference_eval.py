#!/usr/bin/env python3
"""Version-pinned real axe-core 4.14.0 on independent W3C ACT contrast test cases.

Source: W3C official afw4f7 minimum color contrast case labels. This script
compares a recognized standards-based analyzer, not toy hand-coded CSS rules.
No remote HTML scripts executed; two cases with source <script> are excluded
from assessed cases. No external resources or browser production sessions.
Raw per-case observations checkpoint in var, no repeated completed queries.
"""
from __future__ import annotations
from pathlib import Path
import datetime,hashlib,json,os,tempfile,time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
ROOT=Path('/data/var/perceptual-independent-20261024/w3c_act_contrast')
HTML_ROOT=ROOT
JS=ROOT/'axe-core-4.14.0.min.js'
DRIVER='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
OUTPUT=ROOT/'axe_core_4_14_0_evaluation.json'
CALL="""const done=arguments[arguments.length-1];
window.axe.run(document,{runOnly:{type:'rule',values:['color-contrast']},resultTypes:['violations','incomplete','passes','inapplicable']}).then(r=>{done({ok:true,violations:r.violations.map(x=>({id:x.id,n:x.nodes.length,impact:x.impact})),passes:r.passes.map(x=>({id:x.id,n:x.nodes.length})),incomplete:r.incomplete.map(x=>({id:x.id,n:x.nodes.length})),inapplicable:r.inapplicable.map(x=>({id:x.id,n:x.nodes.length})),version:r.testEngine.version});}).catch(e=>done({ok:false,error:String(e)}));"""
def project(res):
 if not res.get('ok'):return 'error'
 if any(x['id']=='color-contrast' and x['n'] for x in res['violations']):return 'failed'
 if any(x['id']=='color-contrast' and x['n'] for x in res['incomplete']):return 'cannot_tell'
 if any(x['id']=='color-contrast' and x['n'] for x in res['passes']):return 'passed'
 return 'inapplicable'
def main():
 source=json.loads((ROOT/'manifest.json').read_text())
 cases=sorted(source['cases'],key=lambda x:x['testcaseId'])
 assert len(cases)==34
 if not JS.exists():raise FileNotFoundError(JS)
 js=JS.read_text()
 sha=hashlib.sha256(JS.read_bytes()).hexdigest()
 result=json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {'date':'2026-10-24','tool':'axe-core 4.14.0','pinned_js_sha256':sha,'source':'W3C ACT rule afw4f7 independent 34 testcases','read_only_isolation':'Firefox headless isolated profile, local file HTML without source scripts, external network disabled; computed colors may differ if external images blocked','cases':{}}
 assert result['pinned_js_sha256']==sha
 opts=Options();opts.add_argument('-headless')
 opts.set_preference('network.proxy.type',1)
 for n in ['http','ssl']:
  opts.set_preference('network.proxy.'+n,'127.0.0.1');opts.set_preference('network.proxy.'+n+'_port',9)
 with tempfile.TemporaryDirectory(prefix='axe-contrast-w3c-',dir='/data/var/tmp') as td:
  opts.profile=td
  browser=webdriver.Firefox(service=Service(executable_path=DRIVER,log_output=os.devnull),options=opts)
  try:
   browser.set_script_timeout(24);browser.set_page_load_timeout(12)
   for i,c in enumerate(cases):
    key=c['testcaseId']
    if key in result['cases']:continue
    begin=time.monotonic();entry={'source_sha256':c['sha256'],'gold':c['expected'],'title':c['title'],'contains_original_scripts':c['has_script']}
    try:
     path=HTML_ROOT/(key+'.html')
     data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==c['sha256'],'source integrity'
     if b'<script' in data.lower():
      entry.update(status='excluded_untrusted_source_script',label='cannot_tell')
     else:
      browser.get(path.as_uri())
      browser.execute_script(js)
      value=browser.execute_async_script(CALL)
      entry.update(status='completed',label=project(value),result=value)
    except Exception as e:entry.update(status='execution_error',label='cannot_tell',error=type(e).__name__+': '+str(e)[:180])
    entry['seconds']=round(time.monotonic()-begin,3)
    result['cases'][key]=entry
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print('AXE',i+1,'of',len(cases),entry['title'],entry['status'],entry['label'],'sec',entry['seconds'],flush=True)
  finally:browser.quit()
 counts={}
 for x in result['cases'].values():counts[x['gold']+'->'+x['label']]=counts.get(x['gold']+'->'+x['label'],0)+1
 report={
  'cases_recorded':len(result['cases']),'cross_tab_expected_to_observed':counts,'coverage':sum(x['status']=='completed' for x in result['cases'].values()),'excluded_scripts':sum(x['status']=='excluded_untrusted_source_script' for x in result['cases'].values()),'errors':sum(x['status']=='execution_error' for x in result['cases'].values()),'total_elapsed_axe_seconds':round(sum(x['seconds'] for x in result['cases'].values()),2),
  'interpretation':'Axe runs a single WCAG criterion on W3C ACT conformance cases, not on full user GUIs; outcomes may be incomplete on gradient/transparent/image backgrounds. Script-bearing original cases excluded without replacement. No invisible source code injection beyond vetted axe bundle; local HTML does not call remote services.'}
 result['summary']=report
 OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
 print('AXE SUMMARY',report,flush=True)
if __name__=='__main__':main()
