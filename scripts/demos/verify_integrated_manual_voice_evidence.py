#!/usr/bin/env python3
"""Offline evidence audit of the November 25 single private integrated run.

No network, inference, voice actions or mutation. The original transcript is
a public recorded utterance, not a claim about the world endorsed by this audit.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import wave

CONV='w99-german-public-manual-confirmed-integrated'
SOURCE='873e3d10e507b73d422bda818c0e03b1d03ae89c99813e479e9ea78dbe811acf'
SCRIPT='e32371ecebe9133456f4468c167648e0b15562c705b4256abf45a3538d027ff4'
WAV='ad33bba729df7624e8e0c97ffb6a1f06c81af96f59a202b2699f7953343cf4fd'
ALIAS='qwen3-coder-30b-a3b-instruct-q4-k-m'
PINNED={
    'attempt.json':'46d3d1dbe4b77691600d5abe07519fa0b2fa2e66e4db57c9684b7f3340fca9f3',
    'result.json':'d778c12e7e7f75961b1dfaa39773b995222d9fbcc88d1846d222e0b265a149b4',
    'actual-asr-input.wav':WAV,
    'asr-attempt.json':'01ba4e984991e26051477f862c3121ff79495387e10c6366eaf9cda2bce95977',
    'asr-raw.json':'593431afdb7a5c3144ea83b161cf7c74509b9c4879730a5969eda4f17d3f3d38',
    'coder-attempt.json':'e4cc1b9f557a558d9c4c4b448c5d2e4530a4f1eed9698b3d3afbd2ce0d1c5819',
    'coder-request.json':'4010bdb9467fb041620a0901bc6836df288410243c528ff5bda3f4c1b43cf2eb',
    'coder-raw.json':'a33a6523a019550520a54579cd858c523d99d7dbe43113044bb2fb4f51c3bb4d',
    'tts-attempt.json':'d479d121cfffd4682c2df317af4cbd8b3e10d1f62b73cf9ae6cbc6ccedd82773',
    'tts-request.json':'6bb937bc5c7089daf3e08445c848e044a3b0b97f03f296e591ebb54e503374c5',
    'tts-framed.bin':'519a46e88e5d1290feb28db83b69cee27557427ef6b08e60b5d6870e4e2c1e22',
    'tts-pcm.bin':'fb5abfdbec28d60897f407899541341a6c11afe8257ccae2f6706a00cb5ea561',
    'recovered-media.pcm':'fb5abfdbec28d60897f407899541341a6c11afe8257ccae2f6706a00cb5ea561',
    'ws-events.json':'b7d99c27bdef7889ab4c46517348247dd9b6fbf21d63c4c42e4fe40f39561088',
    'ws-events-progress.jsonl':'90f0e319b0579f953107209e22681b9f841c1d022ee607cd167173d07116c69b',
    'voice-private.jsonl':'3066b3484cfb6d4f9c91957759b6bdcc15585529c4ccf3e7cf07a8f92f260d98',
    'private-config.json':'e8c577c4875087e0c783150a5a2cb2952670c18ae47e79ebc298e677507c0189',
    'conversations/'+CONV+'.json':'13a6fd7ed6b6c7f497e24fd3699fd953c3cfcc02dd1043152454b91ee03269fa',
}

def require(ok, message):
    if not ok:raise ValueError(message)

def sha(b):
    return hashlib.sha256(b).hexdigest()

# Jetson's Python is older than the SWAAG core; avoid Path.is_relative_to on it.
def inside(root,p):
    try:p.resolve().relative_to(root.resolve());return True
    except ValueError:return False

def verified_file(root,path,want):
    p=root/path
    require(p.is_file() and not p.is_symlink() and inside(root,p),'missing or external evidence: '+str(path))
    require(p.stat().st_size<=3000000,'oversized evidence')
    b=p.read_bytes();require(sha(b)==want,'evidence altered: '+str(path));return b

def decode_wav(raw):
    with wave.open(io.BytesIO(raw),'rb') as w:
        require((w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000),'wrong WAV format')
        b=w.readframes(w.getnframes())
    require(bool(b),'empty WAV');return b

def decode_stream(raw):
    offset=0;pieces=[]
    while offset+4<=len(raw):
        n=int.from_bytes(raw[offset:offset+4],'big');offset+=4
        if n==0:
            require(offset==len(raw) and pieces,'invalid terminal frame')
            return b''.join(pieces)
        require(n%2==0 and 0<n<=1000000 and offset+n<=len(raw),'invalid PCM frame')
        pieces.append(raw[offset:offset+n]);offset+=n
    raise ValueError('stream lacks terminal frame')

def audit_events(events,turn,transcript,reply):
    def matching(typ,**kw):
        return [(i,x) for i,x in enumerate(events) if x.get('type')==typ and all(x.get(k)==v for k,v in kw.items())]
    hellos=matching('hello')
    require([x.get('connection_generation') for _,x in hellos]==[1,2,3],'wrong reconnect sequence')
    require(all(x.get('unverified_speaker_requires_review') is True for _,x in hellos),'speaker guard lost')
    finals=matching('asr',event='final')
    require(len(finals)==2 and all(x.get('turn_id')==turn and x.get('text')==transcript and x.get('auto_send') is False for _,x in finals)
            and finals[1][1].get('restored') is True,'not same manually reviewed transcript')
    submits=matching('turn',event='submitted')
    require(len(submits)==1 and submits[0][1].get('automatic') is False
            and submits[0][1].get('turn_id')==turn and submits[0][1].get('text')==transcript,'not exactly one manual submit')
    starts=matching('audio',event='start');ends=matching('audio',event='end')
    require(len(starts)==2 and len(ends)==1,'not interrupted then resumed media')
    i,s=submits[0];j,a=starts[0];k,b=starts[1];l,e=ends[0]
    require(hellos[0][0]<finals[0][0]<hellos[1][0]<finals[1][0]<i<j<hellos[2][0]<k<l,
            'event order invalid')
    require(a.get('stream_id')!=b.get('stream_id') and b.get('resumed') is True
            and b.get('stream_id')==e.get('stream_id') and b.get('generation')==e.get('generation')
            and all(x.get('turn_id')==turn for x in (a,b,e)),'stream binding invalid')
    require(not matching('turn_proposal') and not matching('error'),'unexpected action proposal or error')
    histories=matching('history')
    require(histories[0][1]['turns']==[],'initial history not empty')
    draft=histories[1][1]['turns']
    require(len(draft)==1 and draft[0].get('status')=='draft' and not draft[0].get('assistant')
            and draft[0].get('recognized_text')==transcript,'draft not durable before approval')
    last=histories[-1][1]['turns']
    require(len(last)==1 and last[0].get('assistant')==reply and last[0].get('audio_completed') is True
            and last[0].get('turn_id')==turn and last[0].get('status')=='completed','missing normal playback receipt')
    return b['stream_id']

def verify(root,script):
    require(root.is_dir() and not root.is_symlink(),'invalid evidence root')
    require(script.is_file() and not script.is_symlink() and sha(script.read_bytes())==SCRIPT,'as-run source changed')
    data={n:verified_file(root,n,h) for n,h in PINNED.items()}
    obj=lambda name:json.loads(data[name])
    attempt=obj('attempt.json');r=obj('result.json')
    require(attempt['source']==SOURCE and attempt['fixture']==WAV and attempt['max_calls_each']==1
            and attempt['no_live_agent_endpoint'] is True and attempt['no_phone'] is True
            and set(attempt['leases'])=={'jetson','nitro','thor'},'admission receipt mismatch')
    require(r.get('status')=='integrated_private_manual_voice_and_normal_reconnect_verified'
            and r.get('count')=={'asr':1,'coder':1,'tts':1} and r.get('relay_errors')==[]
            and r.get('child_stopped') is True and r.get('production_agent_called') is False
            and r.get('phone_tested') is False,'run receipt not a successful one-shot test')
    times=[datetime.datetime.fromisoformat(x['utc']) for x in
           (attempt,obj('asr-attempt.json'),obj('coder-attempt.json'),obj('tts-attempt.json'))]
    require(times==sorted(times) and len(set(times))==4,'attempt chronology mismatch')
    asr=obj('asr-raw.json');words=asr['text']
    require(asr.get('ok') is True and asr.get('engine')=='faster-whisper-large-v3'
            and 'amazon' in words.lower(),'ASR origin mismatch')
    require(obj('asr-attempt.json')['body_sha256']==WAV,'ASR admission body mismatch')
    coder=obj('coder-request.json');resp=obj('coder-raw.json')
    require(coder['model']==ALIAS and 'tools' not in coder and 'tool_choice' not in coder
            and json.loads(coder['messages'][-1]['content'])=={'public_recording_transcript':words},
            'wrong/no-tool model request')
    choice=resp['choices'][0];msg=choice['message'];reply=msg['content']
    require(resp['model']==ALIAS and len(resp['choices'])==1 and choice['finish_reason']=='stop'
            and not msg.get('tool_calls') and not msg.get('function_call') and reply==r['reply'],
            'no verified final tool-free response')
    tts=obj('tts-request.json')
    require(tts=={'text':reply,'sample_rate':16000},'spoken text differs from model reply')
    source_pcm=decode_wav(data['actual-asr-input.wav']);out=decode_stream(data['tts-framed.bin'])
    require(len(source_pcm)==288000 and len(out)==r['received_pcm_bytes']==174080
            and out==data['tts-pcm.bin']==data['recovered-media.pcm'],'recovered PCM differs')
    rows=obj('conversations/'+CONV+'.json')['dialogue_history']
    require(len(rows)==1,'duplicate durable turn');row=rows[0]
    turn=row['turn_id']
    require(row['operation_id']=='voice-'+sha((CONV+'\0'+turn).encode())
            and row['raw_asr']==row['recognized_text']==row['submitted_text']==words
            and row['assistant']==reply and row['requires_review'] is True
            and row['status']=='completed' and row['audio_transmitted'] is True
            and row['audio_completed'] is True and row['tts_engine']=='phonon_31m_easton_cpu_16k',
            'durable turn not source-bound and acknowledged')
    for key,want in (('assistant_audio_path','f8da0c638b6771faac445d831cdae16cd4315ecd86db0a63fa22e1290666dc4f'),
                     ('user_audio_path',WAV)):
        # Resolve using only the evidence-local suffix, so an offline copy is supported.
        path=Path(row[key]);category='reply-audio' if key.startswith('assistant') else 'turn-audio'
        require(category in path.parts,'wrong media path')
        rel=Path(*path.parts[path.parts.index(category):])
        pcm=decode_wav(verified_file(root,rel,want))
        require(pcm==(out if key.startswith('assistant') else source_pcm),'saved WAV mismatch')
    events=obj('ws-events.json')
    require(events==[json.loads(x) for x in data['ws-events-progress.jsonl'].splitlines()],'event journals differ')
    require(audit_events(events,turn,words,reply)==row['audio_stream_id'],'ACK wrong recovered stream')
    log=[json.loads(x) for x in data['voice-private.jsonl'].splitlines()]
    counts=collections.Counter(x['event'] for x in log)
    require(counts['server_start']==1 and counts['session_open']==3 and counts['session_close']==3
            and counts['asr_final']==1 and counts['tts_first_audio']==1,'server log not one integrated run')
    cfg=obj('private-config.json')
    require(cfg['server']['host']=='127.0.0.1' and cfg['interaction']['require_verified_speaker_for_auto_send'] is True
            and cfg['worker_events']['host']=='127.0.0.1' and cfg['worker_events']['port']==9,'isolation configuration changed')
    for group in ('asr','llm','tts'):
        require(all(not v or str(v).startswith('http://127.0.0.1:') for k,v in cfg[group].items()
                    if k.endswith('_url') or k=='url'),'nonlocal route in candidate')
    return {'status':'verified_original_integrated_private_noaction_evidence',
            'one_manual_submit':True,'real_local_model_calls_each':1,'socket_generations':3,
            'recovered_pcm_matches_saved_wav':True,'playback_ack_durable':True,
            'input_seconds':len(source_pcm)/32000,'output_seconds':len(out)/32000,
            'worker_events_intentionally_disconnected':True,'physical_playback_verified':False,
            'production_agent_execution_verified':False,'automatic_turn_detector_verified':False,
            'original_result_sha256':PINNED['result.json']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--as-run-script',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.evidence,a.as_run_script),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
