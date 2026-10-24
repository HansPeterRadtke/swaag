#!/usr/bin/env python3
"""Frozen evidence-scoped semantic subset of the October 24 live protocol."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
from run_live import PORTS, ROOT, SCHEMA_VERSION, fetch, run_trial, tasks

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=8);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not 1<=a.limit<=32:raise SystemExit('invalid bounded limit')
    rows=json.loads((ROOT/'fixtures.json').read_text());models={port:fetch(port,'/v1/models')['data'][0]['id'] for port in PORTS}
    trials=[t for t in tasks(rows) if t['kind'] in ('extraction','reasoning')]
    seen=set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            seen.add(json.loads(line)['trial'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    issued=0
    for t in trials:
        if t['id'] in seen:continue
        if issued==a.limit:break
        try:r=run_trial(t,models)
        except Exception as exc:
            # Capacity refusal is a separate non-inference outcome; never retry an uncertain request.
            r={'trial':t['id'],'failed_or_interrupted':type(exc).__name__,'reason':str(exc)[:120], 'timestamp':dt.datetime.now(dt.timezone.utc).isoformat()}
            with a.out.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({'trial':t['id'],'stopped':r['reason']}),flush=True)
            break
        r['timestamp']=dt.datetime.now(dt.timezone.utc).isoformat();r['study_revision']=SCHEMA_VERSION
        with a.out.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        issued+=1
        print(json.dumps({'trial':t['id'],'valid':r['valid'],'correct':r['correct'],'incomplete':r.get('incomplete',False),'seconds':round(r['total_elapsed_seconds'],3),'model':r['model']}),flush=True)
        if r.get('incomplete'):break
    print('new_trial_count',issued,flush=True)
if __name__=='__main__':main()
