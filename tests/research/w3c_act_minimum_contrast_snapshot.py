#!/usr/bin/env python3
"""Freeze independently authored W3C ACT afw4f7 minimum-contrast test cases.

Original HTML and expected labels live outside Git. Never execute fetched
scripts or external resources. Do not change existing downloaded source files.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,urllib.request,datetime
ROOT=Path('/data/var/perceptual-independent-20261024/w3c_act_contrast')
INDEX='https://www.w3.org/WAI/content-assets/wcag-act-rules/testcases.json'

def get(case):
 path=ROOT/(case['testcaseId']+'.html')
 try:
  if not path.exists():
   with urllib.request.urlopen(urllib.request.Request(case['url'],headers={'User-Agent':'visual-research/1.0'}),timeout=18) as response:data=response.read(600000)
   if len(data)>500000:raise ValueError('over 500KB cap')
   if not any(word in data.lower() for word in [b'<html',b'<p',b'<div',b'<body']):raise ValueError('no HTML detected')
   path.write_bytes(data)
  data=path.read_bytes()
  return {'testcaseId':case['testcaseId'],'title':case['testcaseTitle'],'expected':case['expected'],'source_url':case['url'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'has_script':b'<script' in data.lower(),'has_external_image_resource':b'url(' in data.lower() or b'<img' in data.lower(),'status':'ok'}
 except Exception as e:return {'testcaseId':case['testcaseId'],'status':'error','error':type(e).__name__+': '+str(e)[:180]}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with urllib.request.urlopen(INDEX,timeout=16) as response:idx=response.read(9000000)
 (ROOT/'testcases_index.json').write_bytes(idx) if not (ROOT/'testcases_index.json').exists() else None
 cases=[c for c in json.loads(idx)['testcases'] if c['ruleId']=='afw4f7'];assert len(cases)==34
 receipt=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  work={pool.submit(get,c):c for c in cases}
  for future in as_completed(work):
   r=future.result();receipt.append(r)
   print('W3C',r['title'] if 'title' in r else r['testcaseId'][:10],r['status'],flush=True)
 manifest={'source':'W3C ACT text has minimum contrast afw4f7 original testcases','rule':'https://www.w3.org/WAI/standards-guidelines/act/rules/afw4f7/','index_source':INDEX,'index_sha256':hashlib.sha256(idx).hexdigest(),'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':sorted(receipt,key=lambda x:x['testcaseId'])}
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('COMPLETE',len(receipt),'OK',sum(x['status']=='ok' for x in receipt),'SCRIPT',sum(x.get('has_script',False) for x in receipt))
if __name__=='__main__':main()
