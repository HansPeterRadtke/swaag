#!/usr/bin/env python3
"""Independent no-inference audit of a real private pending-media stop test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from verify_private_voice_durable_stop_evidence import pinned, require, matching, FIXED, WAV

D='stop-during-held-media'
CONV='w100-private-stop-pending-media'
TURN='turn-client-'+'a'*32
PINS={
 'verify_private_voice_stop_pending_stream.py':'fc64c42b89d4ca7937c3c3fd81d3b585e89df36bbfc7eced65ece2a5e0a4ac24',
 D+'/attempt.json':'8002bbb22e10729b5a02765a039dd14aaf1014b7ffc09f62c670daa7c67d363f',
 D+'/result.json':'3d8004ab9b4ec810f7fdb84b0be0c7532036a5ac29bdb3982197b5e7021a173f',
 D+'/events.jsonl':'b23cbff24034194da11793484bc1a4e500945689970eca2b77f03ad348ed04ab',
 D+'/voice-private.jsonl':'53eae71749bdfe6b6e41fd82108bafde510a81365f9d675ecad158c64543bd51',
 D+'/reply_fixture-request.json':'d7ada0e47ec16c26e3e0ec35d118dd65124632081a02c47af788c0ba5cd4e2a8',
 D+'/media_fixture-request.json':'6bb937bc5c7089daf3e08445c848e044a3b0b97f03f296e591ebb54e503374c5',
 D+'/conversations/'+CONV+'.json':'a37e5fc11aba7ffe83a208d545dc812550cd0c23edbe1311a4c65ce44fe1c774',
 D+'/reply-audio/ws-69a81264805243348da3be4c16888827/reply_turn-client-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_55878e0840314ae1899e5e15cfdbb7aa.wav':WAV,
 'durable-stop-candidate/cloudflare_ws_voice_server.py':FIXED,
}

def audit_events(events):
    h=matching(events,'hello');s=matching(events,'audio','start')
    c=matching(events,'audio','cancel');submit=matching(events,'turn','submitted')
    require([x.get('connection_generation') for _,x in h]==[1,2],'wrong sockets')
    require(len(s)==len(c)==len(submit)==1 and not matching(events,'audio','end'),'unexpected media or submit count')
    require(h[0][0]<submit[0][0]<s[0][0]<c[0][0]<h[1][0],'stop not before reconnect')
    require(submit[0][1].get('automatic') is False and s[0][1].get('streaming') is True
            and c[0][1]['reason']=='playback_stopped'
            and c[0][1]['stream_id']==s[0][1]['stream_id'],'not explicit pending-stream stop')
    require(all(x[1]['turn_id']==TURN for x in (s[0],c[0],submit[0])),'wrong turn')
    restored=matching(events,'history')[-1][1]['turns']
    require(len(restored)==1 and restored[0]['status']=='generated'
            and restored[0]['audio_stopped'] is True and restored[0]['audio_completed'] is False,
            'stop not durable before saved WAV')
    require(not any(x.get('type') in ('asr','turn_proposal','error') for x in events),'unexpected action/error')
    return c[0][1]['stream_id']

def verify(root):
    data={n:pinned(root,n,h) for n,h in PINS.items()}
    get=lambda n:json.loads(data[D+'/'+n])
    a=get('attempt.json');r=get('result.json')
    require(a['source_sha']==FIXED and a['real_model_or_agent_requests']==0 and a['no_lost_ACK_test'] is True,
            'wrong original test admission')
    require(r['status']=='verified_private_stop_while_pending_media_transport'
            and r['child_stopped'] is True and r['fixture_errors']==[]
            and r['local_fixture_requests']=={'reply_fixture':1,'media_fixture':1}
            and r['real_model_agent_phone_calls']==r['real_upstream_model_or_agent_calls']==0
            and r['full_retained_pcm_bytes']==174080,'wrong original fixture result')
    for k in ('stop_ack_before_final_frame','original_media_saved_after_stop','no_reconnect_restart',
              'durable_playback_stopped_not_completed'):
        require(r[k] is True,'failed original test: '+k)
    stream=audit_events([json.loads(x) for x in data[D+'/events.jsonl'].splitlines()])
    rows=get('conversations/'+CONV+'.json')['dialogue_history']
    require(len(rows)==1,'duplicate history');row=rows[0]
    require(row['turn_id']==TURN and row['audio_stopped_stream_id']==stream
            and row['audio_stopped'] is True and row['audio_completed'] is False
            and row['audio_transmitted'] is False and row['status']=='completed','late finalization lost stop')
    require(get('media_fixture-request.json')=={'text':row['assistant'],'sample_rate':16000},'wrong retained response')
    request=get('reply_fixture-request.json')
    require(request['conversation_id']==CONV and request['turn_id']==TURN and request['text']==row['submitted_text'],
            'wrong isolated reply fixture')
    return {'status':'independently_verified_pending_transport_stop',
            'actual_private_voice_process':True,'retained_original_PCM_bytes':174080,
            'no_new_model_calls':True,'physical_or_production_playback_tested':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.evidence),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
