import re

def reference_path(img):
 parts=img.split('/')
 if len(parts)<4 or parts[0]!='image':raise ValueError(img)
 app,stem=parts[-2:]
 m=re.match(r'^(.+)_(portrait|landscape)(?:_[A-Za-z0-9-]+_[0-9]+)?\.png$',stem)
 if not m:raise ValueError(img)
 return 'image/right/'+app+'/'+m.group(1)+'_'+m.group(2)+'.png'
