#!/usr/bin/env python3
"""Read-only independent audit of the actual W100 playback stop/reconnect runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

BASE='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
FIXED='cd7ccfa7019e4dd63a2ed25e01ba257f405ebaffe3a415364f1724e931b134ea'
WAV='f8da0c638b6771faac445d831cdae16cd4315ecd86db0a63fa22e1290666dc4f'
A='baseline-stop-before-end'
B='corrected-stop-before-end'
C='stop-restart-and-normal-recovery'
SCRIPT_SHA={
 'prepare_private_voice_durable_audio_stop.py':'b2467bfd4847f578b7be6abcba0eb4a5f948f66b45a24f7426a7216cf60f11fe',
 'verify_private_voice_stop_reconnect.py':'a0e73783d96ef11fd20423a990be0204a40124bd25e06290441fedf652710da7',
 'verify_private_voice_durable_stop_followup.py':'535b109a6fb68bd69025a58d00af01ad353681e560fe95d57fb5af2dfbd6fc7c',
}
PINS={
 A+'/result.json':'94726e7e115efb4473c21c86cd296b3c5d272673c7901866a8963c9e86efa88d',
 A+'/attempt.json':'c90645adb51c9ac9e2306c9fb90732ec9a829bfddc491a16b4ac93ea3a5bbf24',
 A+'/events.jsonl':'1ec6e7a5027aef65b973f14da41b80f4339b26f5a621cb5d2a56391ec204b0b0',
 A+'/voice-private.jsonl':'a26b21982c9e01dcdb03edb2f1f48f41c5292739b40dacd78f6ea84d6d10e9a4',
 A+'/conversations/w100-private-stop-baseline-stop-before-end.json':'ec6cf8dc9d1f5bd143764ab1883e47eff254cb31400faea3e8249d9053a36968',
 B+'/result.json':'6520fe818e8672eb5e0fdcba7c16d50b36d547ea90ff2dc97c562a13fb273e0f',
 B+'/attempt.json':'5716b51266ad7002d5baf5078d3a34b3d691063b798f5edb173b260a69da9d6a',
 B+'/events.jsonl':'623b3163374ebc0f25d9a853b0daa94cc619acf69826a52a602119d5d712d4b6',
 B+'/voice-private.jsonl':'26d5db12d5c17ebe1f02f563ca0c13e7dfd9c72da498b44e549c88ed3d310a0c',
 B+'/conversations/w100-private-stop-corrected-stop-before-end.json':'90f0095d87eb459d46c65f2c24a83d8dc823f85a9532aefa217a0c6fd70dd22c',
 C+'/result.json':'fedf4ffebdf7883c8b1fc8dcaa068d2c4def12dcf5a0c44637a6ecb55c74554f',
 C+'/attempt.json':'74722b9afca212c273ab74d8383e4eb52bf03e89732f3f351a05b6b2108a92ea',
 C+'/followup-events.jsonl':'c2e2f13d5362433431ae1e5fa95d2b898acf6cfc049e2375fb091a59b56b34ce',
 C+'/voice-private.jsonl':'5f7f1fd55b53583c26cf706dc7b709a380bbd27cdca64f7257ea7ac31d65d7a7',
 C+'/conversations/w100-restarted-stopped-private.json':'92079084178da4a61391c1e2f748b02d78e06e29291cf88a542d4952cecad889',
 C+'/conversations/w100-normal-uninterrupted-recovery-private.json':'baf019e5da364421a3a2f41d48188bfd1693178d41d9622f7c885a266e8ae41f',
 C+'/conversations/w100-pure-storage-stop-boundary.json':'62b9dc8ca0830cc6877d1650e243e32e9f4707d81faedb847111eaa382306d79',
 **{x+'/reply-audio/previous-public-test-reply.wav':WAV for x in (A,B,C)},
 'durable-stop-candidate/cloudflare_ws_voice_server.py':FIXED,
}

def require(value,reason):
    if not value:raise ValueError(reason)

def pinned(root,name,want):
    p=root/name
    require(p.is_file() and not p.is_symlink(),'missing or symlinked evidence')
    try:p.resolve().relative_to(root.resolve())
    except ValueError as e:raise ValueError('outside evidence root') from e
    require(p.stat().st_size<1000000,'unbounded evidence')
    raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==want,'changed evidence: '+name)
    return raw

def matching(events,typ,event=None):
    return [(i,x) for i,x in enumerate(events) if x.get('type')==typ and
            (event is None or x.get('event')==event)]

def audit_stop(events,should_restart):
    hello=matching(events,'hello')
    starts=matching(events,'audio','start')
    cancel=matching(events,'audio','cancel')
    require([x.get('connection_generation') for _,x in hello]==[1,2],'bad connection sequence')
    require(len(cancel)==1 and len(starts)==(2 if should_restart else 1),'wrong start/cancel count')
    require(hello[0][0]<starts[0][0]<cancel[0][0]<hello[1][0]
            and cancel[0][1].get('reason')=='playback_stopped'
            and cancel[0][1].get('stream_id')==starts[0][1].get('stream_id'),'stop not bound before reconnect')
    require(not matching(events,'audio','end'),'not a stop-before-end experiment')
    if should_restart:
        require(starts[1][0]>hello[1][0] and starts[1][1].get('resumed') is True
                and starts[1][1].get('stream_id')!=starts[0][1].get('stream_id'),'baseline restart not observed')
    else:
        h=matching(events,'history')[-1][1]['turns']
        require(len(h)==1 and h[0].get('audio_stopped') is True
                and h[0].get('audio_completed') is False
                and h[0].get('status')=='completed','stopped state not recovered')
    require(not any(x.get('type') in ('turn','turn_proposal','asr','error') for x in events),'unexpected action/error')

def audit_followup(events):
    hello=matching(events,'hello')
    require([(x.get('conversation_id'),x.get('connection_generation')) for _,x in hello]==[
       ('w100-restarted-stopped-private',1),
       ('w100-normal-uninterrupted-recovery-private',1),
       ('w100-normal-uninterrupted-recovery-private',2)],'wrong followup connections')
    rs=matching(events,'replay','start');re=matching(events,'replay','end')
    starts=matching(events,'audio','start');ends=matching(events,'audio','end')
    require(len(rs)==len(re)==len(ends)==1 and len(starts)==2,'wrong explicit/automatic media counts')
    require(hello[0][0]<rs[0][0]<re[0][0]<hello[1][0]<starts[0][0]<hello[2][0]<starts[1][0]<ends[0][0],
            'media recovery order wrong')
    require(rs[0][1]['stream_id']==re[0][1]['stream_id']
            and starts[1][1].get('resumed') is True
            and starts[1][1]['stream_id']==ends[0][1]['stream_id'],'media stream identity changed')
    h=matching(events,'history')[-1][1]['turns']
    require(len(h)==1 and h[0]['audio_completed'] is True and not h[0]['audio_stopped'],'normal final ACK not durable')
    require(not any(x.get('type') in ('turn','turn_proposal','asr','error') for x in events),'unexpected action/error')

def verify(root):
    require(root.is_dir() and not root.is_symlink(),'wrong original root')
    d={n:pinned(root,n,h) for n,h in {**PINS,**SCRIPT_SHA}.items()}
    get=lambda x:json.loads(d[x])
    events=lambda x:[json.loads(l) for l in d[x].splitlines()]
    for label,want in ((A,BASE),(B,FIXED)):
        a=get(label+'/attempt.json');r=get(label+'/result.json')
        require(a['source_sha']==want and a['audio_sha']==WAV and a['no_lost_ACK_test'] is True
                and a['no_real_agent_or_model_calls'] is True,'wrong source/admission')
        require(r['status']=='observed_private_stop_reconnect' and r['stop_acknowledged_before_disconnect'] is True
                and r['initial_media_bytes']==1920 and r['new_agent_or_model_calls']==0
                and r['playback_completed_ACK_sent'] is False and r['child_stopped'] is True
                and r['original_audio_unchanged'] is True,'original stop observation changed')
        require(r['automatic_restart_observed'] is (label==A)
                and r['durable_audio_stopped'] is (label==B),'not a reproduced/fixed pair')
        audit_stop(events(label+'/events.jsonl'),label==A)
        logs=events(label+'/voice-private.jsonl')
        require(sum(x.get('event')=='server_start' for x in logs)==1,'not one process')
    r=get(C+'/result.json');a=get(C+'/attempt.json')
    require(a['source_sha']==FIXED and a['prior_stop_result_sha']==PINS[B+'/result.json']
            and a['no_model_or_agent_routes'] is True and a['no_lost_ACK_test'] is True,'wrong followup admission')
    require(r['status']=='verified_private_durable_stop_and_unmodified_normal_recovery'
            and r['model_agent_or_phone_calls']==0 and r['child_stopped'] is True
            and r['explicit_replay_bytes']==r['normal_pre_end_disconnect_recovery_bytes']==174080,
            'followup not complete')
    for k in ('stale_socket_rejected','wrong_stream_rejected','wrong_generation_rejected','genuine_ack_preserved',
              'late_ready_and_ack_cannot_resurrect','process_restart_suppresses_stopped_audio',
              'explicit_replay_preserves_original_stop','normal_ack_persisted'):
        require(r[k] is True,'missing real source verification: '+k)
    audit_followup(events(C+'/followup-events.jsonl'))
    for n in (B+'/conversations/w100-private-stop-corrected-stop-before-end.json',
              C+'/conversations/w100-restarted-stopped-private.json',
              C+'/conversations/w100-pure-storage-stop-boundary.json'):
        rows=get(n)['dialogue_history']
        require(len(rows)==1 and rows[0]['audio_stopped'] is True and rows[0]['audio_completed'] is False
                and rows[0]['audio_transmitted'] is False and rows[0]['status']=='completed','wrong durable stop row')
    rows=get(C+'/conversations/w100-normal-uninterrupted-recovery-private.json')['dialogue_history']
    require(len(rows)==1 and rows[0]['audio_completed'] is True and not rows[0].get('audio_stopped'),'ordinary delivery regressed')
    return {'status':'verified_actual_private_stop_reconnect_fix_and_regressions','baseline_restart':True,
            'candidate_restart':False,'durable_stop_across_new_process':True,
            'normal_recovery_and_explicit_replay_bytes':174080,'no_model_or_agent_calls':True,
            'production_deployed':False,'physical_phone_tested':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.evidence),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
