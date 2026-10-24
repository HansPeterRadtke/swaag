#!/usr/bin/env python3
"""Bounded local ScreenVLM v2 real-web-source target screen PARSING (not defect QA).

CPU only, no model server or GPU. Four pairs selected from externally verified
UIJudgeBench sites; actual Firefox-captured element-centered screenshots.
Model gets ONLY pixels and official generic parsing instruction. Expected
source text and defect gold are attached after model outputs, not as input.
Timeout/partial model requests are permanently logged, never auto-replayed.
"""
import datetime,hashlib,json,time,resource,re,os
from pathlib import Path
from PIL import Image
ROOT=Path('/data/var/perceptual-independent-20261024')
MODEL=ROOT/'models/ScreenVLM'
DATA=ROOT/'visible-ui-real/anchored_manifest.json'
OUT=ROOT/'visible-ui-real/screenvlm_v2_first_run.json'
CASES=['real-epa-recycle--clip-overflow','real-epa-recycle','real-section508-manage--clip-overflow','real-section508-manage']
PROMPT='Generate the screen representation for this UI:'
NORM=500

def main():
 if not (MODEL/'research_manifest.json').exists():raise RuntimeError('Model exact source checkpoint not verified')
 provenance=json.loads((MODEL/'research_manifest.json').read_text())
 original=json.loads(DATA.read_text())['cases']
 todo=[]
 for name in CASES:
  matching=[(key,x) for key,x in original.items() if x['page_id']==name and x['criterion']=='redecheck:element-protrusion']
  assert len(matching)==1,(name,len(matching))
  todo.append(matching[0])
 state=json.loads(OUT.read_text()) if OUT.exists() else {'source':'UIJudgeBench independent real website screenshot, element explicitly scrolled into visible viewport and cropped','model':'ScreenVLM v2 main','model_revision':provenance['revision'],'model_files_manifest_sha256':hashlib.sha256((MODEL/'research_manifest.json').read_bytes()).hexdigest(),'hardware':'Nitro CPU-only fp32 one torch thread, no user-facing/production GPU inference','instruction':PROMPT,'model_task':'SCREEN REPRESENTATION markup, not classification of whether clipping/overlap is a defect','cases':[],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 completed={x['source_item_sha256'] for x in state['cases']}
 print('START_PREP',len(todo),'MODEL_REV',state['model_revision'],flush=True)
 import torch
 from transformers import AutoProcessor,AutoModelForVision2Seq
 torch.set_num_threads(1)
 processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
 print('PROCESSOR_READY',flush=True)
 load_start=time.monotonic()
 model=AutoModelForVision2Seq.from_pretrained(str(MODEL),torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cpu().eval()
 print('MODEL_READY',round(time.monotonic()-load_start,2),'s',flush=True)
 msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}]
 template=processor.apply_chat_template(msg,add_generation_prompt=True)
 for i,(key,entry) in enumerate(todo):
  source_sha=hashlib.sha256(key.encode()).hexdigest()
  if source_sha in completed:
   print('ALREADY_ATTEMPTED_NOT_REPEATED',i+1,flush=True);continue
  image=Image.open(entry['image_path']).convert('RGB')
  t=time.monotonic()
  row={'source_item_sha256':source_sha,'image_sha256':entry['image_sha256'],'prompt':PROMPT,'status':'pending_or_uncertain','input_image_size':list(image.size),'max_new_tokens':384}
  state['cases'].append(row);OUT.write_text(json.dumps(state,indent=2)+'\n')
  try:
   inputs=processor(text=template,images=[image],return_tensors='pt')
   print('TOKENS_INPUT',i+1,int(inputs.input_ids.shape[1]),flush=True)
   with torch.inference_mode():
    ids=model.generate(**inputs,max_new_tokens=384,do_sample=False,use_cache=True)
   offset=int(inputs.input_ids.shape[1]);raw=processor.batch_decode(ids[:,offset:],skip_special_tokens=False)[0]
   row.update(status='completed',raw_screen_tag=raw[:14000],output_chars=len(raw),location_tokens=len(re.findall(r'<loc_\d+>',raw)),markups_seen=len(re.findall(r'<(?:button|link|text|heading|paragraph|navigation_bar|list|image|input|form)\b',raw)))
  except Exception as e:row.update(status='failed_or_uncertain',error=type(e).__name__+': '+str(e)[:230])
  row['seconds']=round(time.monotonic()-t,2);row['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  # Gold only attached AFTER prediction; not used to choose stopping.
  row['external_gold_violation']=entry['gold_violation'];row['original_source_text']=entry['target_metadata']['text_content'][:1200]
  OUT.write_text(json.dumps(state,indent=2)+'\n')
  print('SCREENVLM_CASE',i+1,'status',row['status'],'sec',row['seconds'],'loc_tokens',row.get('location_tokens'),'out',row.get('output_chars'),'error',row.get('error'),flush=True)
  if row['status']!='completed':break
 print('CASES_RECORDED',len(state['cases']),'COMPLETED',sum(x['status']=='completed' for x in state['cases']),flush=True)
if __name__=='__main__':main()
