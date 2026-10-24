#!/usr/bin/env python3
"""Make fixed real-Firefox screenshots with source-authored gold, isolated profiles.

No access to production browser sessions, no GPU required, no model calls.
Existing successes are not reissued or overwritten. Local HTML only.
"""
import json,os,subprocess,tempfile,time
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]/'docs/research/perceptual-browser-20261024'
CASE_GOLD={
 'clean':('none','System ready','baseline visible text'),
 'clipped':('clipped_text','System ready for maintenance','white-space nowrap extends beyond fixed-width clipped panel without alternate access'),
 'overlap':('overlap','System ready','two absolute-positioned texts share painted region'),
 'low_contrast':('low_contrast','System ready','near-white foreground on white background'),
 'dark_theme':('none','System ready','white-on-dark retains contrast'),
 'right_aligned':('none','Ready','valid right-aligned visible short text'),
 'intentional_ellipsis':('none','System ready for maintenance','explicit visual truncation with title attribute exposing complete text'),
 'large_text_reflow':('none','System ready for maintenance','larger text fully wraps into taller box'),
}
BASE='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Fixture</title><style>
html,body{margin:0;background:#eeeeee;font-family:Arial,sans-serif}h1{font-size:24px;margin:14px 22px}
.box{position:relative;margin:15px 22px 0;padding:12px 8px;box-sizing:border-box;width:197px;height:53px;background:white;border:1px solid #999}
.text{color:#171717;font-size:19px;line-height:24px;white-space:nowrap}
</style>{STYLE}</head><body><h1>Operations control</h1><div class="box">{CONTENT}</div><div class="box" style="width:305px"><span class="text">Pressure: 240 kPa</span></div></body></html>'''
VARIANTS={
 'dark_theme':('<style>body{background:#151515;color:#fafafa}.box{background:#222;border-color:#777}.text{color:#fafafa}</style>','<span class="text">System ready</span>'),
 'right_aligned':('<style>.box{text-align:right}</style>','<span class="text">Ready</span>'),
 'intentional_ellipsis':('<style>.box{overflow:hidden}.text{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}</style>','<span class="text" title="System ready for maintenance">System ready for maintenance</span>'),
 'large_text_reflow':('<style>.box{height:106px;width:260px}.text{display:block;white-space:normal;font-size:23px;line-height:27px}</style>','<span class="text">System ready for maintenance</span>'),
}
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 for name,(style,content) in VARIANTS.items():
  source=ROOT/(name+'.html')
  if not source.exists():source.write_text(BASE.replace('{STYLE}',style).replace('{CONTENT}',content))
 manifest={'date':'2026-10-24','gold_is_from_authorized_fixture_source_not_prediction':True,'browser':'Firefox 146.0 headless isolated profiles','viewport_px':[540,310],'cases':[]}
 for name,(gold,expected,why) in CASE_GOLD.items():
  src=ROOT/(name+'.html');path=ROOT/(name+'-firefox.png');row={'case':name,'gold':gold,'expected_text':expected,'source':src.name,'image':path.name,'ground_reason':why}
  if path.exists():
   row['action']='existing_not_repeated'
  else:
   start=time.monotonic()
   with tempfile.TemporaryDirectory(prefix='visual-firefox-',dir='/data/var/tmp') as td:
    prof=Path(td)/'profile';prof.mkdir()
    cmd=['nice','-n','15','firefox','--headless','--no-remote','--profile',str(prof),'--window-size','540,310','--screenshot',str(path),src.as_uri()]
    try:
     c=subprocess.run(cmd,env={**os.environ,'MOZ_HEADLESS':'1'},text=True,capture_output=True,timeout=16,check=False)
     row.update(action='rendered',returncode=c.returncode,stderr=c.stderr[-180:],seconds=round(time.monotonic()-start,2))
    except subprocess.TimeoutExpired as e:row.update(action='timeout',seconds=round(time.monotonic()-start,2),error=str(e)[:100])
  if path.is_file():
   with Image.open(path) as img:row['image_pixels']=list(img.size)
   row['bytes']=path.stat().st_size
  manifest['cases'].append(row)
  print(name,row['action'],row.get('returncode'),row.get('image_pixels'),flush=True)
 (ROOT/'firefox_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('CAPTURED',sum('image_pixels' in r for r in manifest['cases']),'OF',len(CASE_GOLD))
if __name__=='__main__':main()
