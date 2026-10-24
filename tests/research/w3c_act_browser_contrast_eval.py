#!/usr/bin/env python3
"""Independent flat-color DOM contrast check on official W3C ACT afw4f7 cases.

Not axe-core, not ACT reference implementation: own WCAG 2.2 luminance formula
and large-text threshold, unknown backgrounds/alpha/images cause abstention.
Javascript disabled, network blocked; predeclared rule, no expected outcomes
supplied to the browser checker. Scores include inapplicable/abstain separately.
"""
from __future__ import annotations
import hashlib,json,os,tempfile,time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

ROOT=Path('/data/var/perceptual-independent-20261024/w3c_act_contrast')
OUT=ROOT/'independent_min_contrast_results.json'
DRIVER='/data/var/perceptual-independent-20261024/tools/geckodriver-v0.36.0/geckodriver'
JS=r'''const trace=[];let skipped=0;
const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
function rgb(css){if(!css) return null;let match=css.match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:[,\s\/]+([\d.]+))?\s*\)$/i);if(!match)return null;return [Number(match[1]),Number(match[2]),Number(match[3]),match[4]===undefined?1:Number(match[4])];}
function linear(n){let v=n/255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}
function lum(a){return .2126*linear(a[0])+.7152*linear(a[1])+.0722*linear(a[2]);}
function ratio(a,b){let x=lum(a),y=lum(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);}
function visible(node){let e=node.parentElement;if(!e)return false;for(let a=e;a;a=a.parentElement){let s=getComputedStyle(a);if(s.display==='none'||s.visibility==='hidden'||s.visibility==='collapse'||Number(s.opacity)===0||a.getAttribute('aria-hidden')==='true'||a.hasAttribute('hidden'))return false;}let range=document.createRange();range.selectNodeContents(node);return Array.from(range.getClientRects()).some(r=>r.width>0&&r.height>0);}
let node;
while(node=walker.nextNode()){
 const text=node.textContent.trim();if(!/[\p{L}\p{N}]/u.test(text))continue;
 const elem=node.parentElement; if(!elem||['SCRIPT','STYLE','NOSCRIPT','TEMPLATE'].includes(elem.tagName))continue;
 if(!visible(node))continue;
 let disabled=false;
 for(let a=elem;a;a=a.parentElement)if(a.hasAttribute('disabled')||a.getAttribute('aria-disabled')==='true')disabled=true;
 if(disabled)continue;
 const computed=getComputedStyle(elem),foreground=rgb(computed.color),weight=parseFloat(computed.fontWeight)||400,size=parseFloat(computed.fontSize)||16;
 const large=(size>=24||(size>=18.6666667&&weight>=700));const required=large?3:4.5;
 let back=null,uncertain=null;
 for(let a=elem;a;a=a.parentElement){let s=getComputedStyle(a);
  if(s.backgroundImage!=='none'){uncertain='background_image_or_gradient';break;}
  if(s.textShadow!=='none'){uncertain='text_shadow';break;}
  if(Number(s.opacity)<1){uncertain='ancestor_opacity';break;}
  let candidate=rgb(s.backgroundColor);if(candidate&&candidate[3]>=.999){back=candidate;break;}
  if(candidate&&candidate[3]>0.001){uncertain='translucent_background';break;}
 }
 if(!back&&!uncertain)back=[255,255,255,1];
 if(!foreground){uncertain='unparsed_foreground';}
 let effective=foreground?foreground.slice(0,3):null;
 if(foreground&&foreground[3]<.999&&back){effective=foreground.slice(0,3).map((v,i)=>v*foreground[3]+back[i]*(1-foreground[3]));}
 let r=(!uncertain&&effective&&back)?ratio(effective,back):null;
 trace.push({text_preview:text.slice(0,70),element:elem.tagName,font_px:size,weight,required,contrast:r,undecidable:uncertain,failed:r!==null&&r+1e-9<required});
}
const fails=trace.filter(x=>x.failed),inconclusive=trace.filter(x=>x.undecidable);
return {applicable_text_nodes:trace.length,failed_nodes:fails.length,uncertain_nodes:inconclusive.length,derived_status:fails.length?'failed':inconclusive.length?'undetermined':trace.length?'passed':'inapplicable',diagnostic_nodes:trace.slice(0,80)};'''

def main():
 if OUT.exists():print('PRIOR_COMPLETE_RESULT_NOT_RERUN');return
 original=json.loads((ROOT/'manifest.json').read_text())['cases'];assert len(original)==34 and all(x['status']=='ok' for x in original)
 opts=Options();opts.add_argument('-headless');opts.set_preference('javascript.enabled',False)
 opts.set_preference('network.proxy.type',1);opts.set_preference('network.proxy.http','127.0.0.1');opts.set_preference('network.proxy.http_port',9);opts.set_preference('network.proxy.ssl','127.0.0.1');opts.set_preference('network.proxy.ssl_port',9)
 with tempfile.TemporaryDirectory(prefix='w3c-contrast-browser-',dir='/data/var/tmp') as profile:
  opts.profile=profile
  browser=webdriver.Firefox(service=Service(executable_path=DRIVER,log_output=os.devnull),options=opts)
  try:
   browser.set_page_load_timeout(12)
   runs=[]
   for i,c in enumerate(original):
    t=time.monotonic();path=ROOT/(c['testcaseId']+'.html')
    entry={'testcaseId':c['testcaseId'],'source_sha256':c['sha256'],'has_script':c['has_script'],'has_external_image_resource':c['has_external_image_resource']}
    try:
     browser.get(path.as_uri());v=browser.execute_script(JS)
     entry.update(status='ok',metrics=v)
    except Exception as e:entry.update(status='error',error=type(e).__name__+': '+str(e)[:170])
    entry['seconds']=round(time.monotonic()-t,3)
    runs.append(entry)
    print('W3C',i+1,'/',len(original),entry['status'],entry.get('metrics',{}).get('derived_status'),entry['seconds'],flush=True)
  finally:browser.quit()
 # Ground truth first consulted AFTER all detector results are computed.
 label={x['testcaseId']:x['expected'] for x in original}
 counts={k:0 for k in ('TP','FP','FN','TN','ABSTAIN','INAPPLICABLE_CORRECT','INAPPLICABLE_WRONG')}
 conf={}
 for r in runs:
  gold=label[r['testcaseId']];pred=r.get('metrics',{}).get('derived_status','undetermined')
  conf[gold+'->'+pred]=conf.get(gold+'->'+pred,0)+1
  if pred=='undetermined':counts['ABSTAIN']+=1
  elif gold=='inapplicable':counts['INAPPLICABLE_CORRECT' if pred=='inapplicable' else 'INAPPLICABLE_WRONG']+=1
  elif pred=='inapplicable':counts['ABSTAIN']+=1
  elif gold=='failed':counts['TP' if pred=='failed' else 'FN']+=1
  elif gold=='passed':counts['FP' if pred=='failed' else 'TN']+=1
 result={'source':'W3C ACT afw4f7 official 34 independently authored minimal contrast test cases','detector':'own computed CSS color/background contrast WCAG 2.2 small 4.5 large 3.0, no axe code; Firefox 146.0','isolation':'archived local HTML, JavaScript globally disabled, no remote resources permitted; known dynamic or image cases cannot be fairly decided','config':'solid backgrounds only, skip disabled/hidden/decorative, uncertain on gradients/images/transparency/shadow; predefined before labels','cases':[{**r,'gold':label[r['testcaseId']]} for r in runs],'confusion':conf,'counts':counts,'limitations':'No reliable checking of CSS gradients/background images, pseudo element text, icon text, active scripts or rendered font pixels. ACT labels for document semantic applicability may not match static style approximation; abstention preferred. Case-specific text can be below viewport but its CSS still computed.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print('FINAL',counts,conf,flush=True)
if __name__=='__main__':main()
