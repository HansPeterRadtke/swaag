#!/usr/bin/env python3
"""Offline evidence audit of W103 canonical English manual voice and reconnect.

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

CONV='w103-canonical-english-confirmed-integrated'
REVISION='a490b00155afabd7b5d4cb28f32024f5a86e8fbf'
SOURCE='85c70555e4ba5c2730b91557d43de0571e8e3da3735224d654559ef31a35c23b'
SCRIPT='f9615cb19cacc8556273fc0e1efc182058309d48ad728c58a49300630ac57ecb'
WAV='32319459d2f24f4c983a6986f685a125414d47bfbe436881bf449158a294dcba'
ALIAS='qwen3-coder-30b-a3b-instruct-q4-k-m'
PINNED={
    'attempt.json':'7bc351f29ada913b64f82afb307d12a5b2427f993c808fce53d7c34a16692600',
    'result.json':'7225cab773ba8e42b1ee9fdca569d7dde4eda81ab1554ca5f68cdd25e2b19355',
    'actual-asr-input.wav':WAV,
    'asr-attempt.json':'ed6260d42bbc4db763529ca072643704dd87923f2a373cdeab05a071ab18749a',
    'asr-raw.json':'a7a7b4db13ff84f503dcbaab968bce97e410355e7100800bd232436fb4c2fc1c',
    'coder-attempt.json':'033dcf45898447d1f1d134845d29cebc2ca1f930572b8e8554a2d5826541d19e',
    'coder-request.json':'232b031b349a7576935be6258cf0560abc7c0b0edbc28a76f28ab112357abc1b',
    'coder-raw.json':'82772703b7cdea2ca993c012b29b05605782c23e83eca382db7febc87a9406bd',
    'tts-attempt.json':'caafcb7303bb9b0cb01da261e3be77ac449d34fd9b8b07af8586cfe25a7089c3',
    'tts-request.json':'5193f8f2a935e4821668c32a043857f73e2e8e4f1543c7170b22b6b243eb3b86',
    'tts-framed.bin':'55de0c8d25094329e7441cbd05cd6306aa8fb1ac129ba36ae3ece638788e344c',
    'tts-pcm.bin':'43d975db746477a4770536af1dad09b02136d05efb703e7b677796a5284c947f',
    'recovered-media.pcm':'43d975db746477a4770536af1dad09b02136d05efb703e7b677796a5284c947f',
    'ws-events.json':'4057ce3856aed0d980fa3b30b2d793b0087886f2a0dd4990ed4d2ce185da3297',
    'ws-events-progress.jsonl':'be781c57c79247e18f9645c8310bc096a97f15057e68f514c3640ac52f69666a',
    'voice-private.jsonl':'50a52860579b3ab0b99662e7d74ff2046110981ee9ca83cdbaa57c9ecdcd40b0',
    'private-config.json':'2604a06d5c6a5e31b27886420eba51ef7a39e47b44bd16cd1de28b33d816d87c',
    'conversations/'+CONV+'.json':'7de18c42f7bccc07f05af17d451cbb4918adfbff727eb7234c81f96ed4c9326a',
}
SOURCE_PINS={
    'cloudflare_ws_voice_server.py':SOURCE,
    'voice_ws_config.py':'b3052253b3d3a22f0f3f69bf6cd264768d6dc0bab99fcd94fbb874b1a98c69ba',
    'voice_ws_core.py':'97e7b1309d675631c3655de1808821b8d11e5b063eed27bd7d4fc981e02d31b9',
    'voice_committed_speech.py':'81ad793b3e138724fe1607489013b2674ec1c2f0745ccdbb2b72a46d3a117c52',
    'voice_speech.py':'85a201a5ac805aba59536b7b527c477315788ef637df76749c1c6d72ed777bfb',
    'voice_swaag_transport.py':'089950509d14fecfdca96597f6a578136b73ae2d2ae2307fb0598cbac6d43309',
    'voice_log_privacy.py':'907caacd5156d1216c7ef2be4da109e99fb3bf28bb0d41047c46cb42d4342d36',
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
    require(len(finals)==3 and all(x.get('turn_id')==turn and x.get('text')==transcript and x.get('auto_send') is False for _,x in finals)
            and all(x.get('restored') is True for _,x in finals[1:]),'not same manually reviewed transcript')
    submits=matching('turn',event='submitted')
    require(len(submits)==1 and submits[0][1].get('automatic') is False
            and submits[0][1].get('turn_id')==turn and submits[0][1].get('text')==transcript,'not exactly one manual submit')
    starts=matching('audio',event='start');ends=matching('audio',event='end')
    require(len(starts)==2 and len(ends)==1,'not interrupted then resumed media')
    i,s=submits[0];j,a=starts[0];k,b=starts[1];l,e=ends[0]
    reviews=matching('state',state='awaiting_send',reason='speaker_identity_unverified')
    require(len(reviews)==3 and all(x.get('turn_id')==turn for _,x in reviews),
            'speaker review decisions missing')
    require(hellos[0][0]<finals[0][0]<reviews[0][0]<hellos[1][0]<finals[1][0]
            <reviews[1][0]<reviews[2][0]<finals[2][0]<i<j<hellos[2][0]<k<l,
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
    require(len(histories)==5 and reviews[2][0]<histories[2][0]<finals[2][0],
            'post-challenge history missing')
    guarded=histories[2][1]['turns']
    require(len(guarded)==1 and guarded[0].get('status')=='draft'
            and guarded[0].get('turn_id')==turn and not guarded[0].get('assistant')
            and guarded[0].get('recognized_text')==transcript,'forged commit changed draft')
    last=histories[-1][1]['turns']
    require(len(last)==1 and last[0].get('assistant')==reply and last[0].get('audio_completed') is True
            and last[0].get('turn_id')==turn and last[0].get('status')=='completed','missing normal playback receipt')
    return b['stream_id']

def verify(root,script,source):
    require(root.is_dir() and not root.is_symlink(),'invalid evidence root')
    require(script.is_file() and not script.is_symlink() and sha(script.read_bytes())==SCRIPT,'as-run source changed')
    require(source.is_dir() and not source.is_symlink(),'invalid canonical source root')
    for n,h in SOURCE_PINS.items():verified_file(source,n,h)
    data={n:verified_file(root,n,h) for n,h in PINNED.items()}
    obj=lambda name:json.loads(data[name])
    attempt=obj('attempt.json');r=obj('result.json')
    require(attempt['source']==SOURCE and attempt['canonical_revision']==REVISION and attempt['source_pins']==SOURCE_PINS and attempt['script_sha256']==SCRIPT and attempt['fixture']==WAV and attempt['max_calls_each']==1
            and attempt['no_live_agent_endpoint'] is True and attempt['no_phone'] is True
            and set(attempt['leases'])=={'jetson','nitro','thor'},'admission receipt mismatch')
    require(r.get('status')=='integrated_private_manual_voice_and_normal_reconnect_verified'
            and r.get('forged_owner_and_commit_refused') is True and r.get('count')=={'asr':1,'coder':1,'tts':1} and r.get('relay_errors')==[]
            and r.get('child_stopped') is True and r.get('production_agent_called') is False
            and r.get('phone_tested') is False,'run receipt not a successful one-shot test')
    times=[datetime.datetime.fromisoformat(x['utc']) for x in
           (attempt,obj('asr-attempt.json'),obj('coder-attempt.json'),obj('tts-attempt.json'))]
    require(times==sorted(times) and len(set(times))==4,'attempt chronology mismatch')
    asr=obj('asr-raw.json');words=asr['text']
    require(asr.get('ok') is True and asr.get('engine')=='faster-whisper-large-v3'
            and sha(words.encode())=='9ea8b7a4460d1440c61a0f19c77d9271277499d27133f937191083ef73343da2','ASR origin mismatch')
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
    require(len(source_pcm)==172800 and len(out)==r['received_pcm_bytes']==112640
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
    for key,want in (('assistant_audio_path','efa9e1b55e6a6e40649bf485999795481fbce03ad848522e98a7967c54fde07b'),
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
            and counts['asr_final']==1 and counts['tts_first_audio']==1 and counts['speaker_identity_unverified_manual_review']==3,'server log not one integrated run')
    review_logs=[x for x in log if x['event']=='speaker_identity_unverified_manual_review']
    require([x.get('stage') for x in review_logs][-2:]==['client_state','client_commit'],
            'wrong forged-owner/commit review stages')
    cfg=obj('private-config.json')
    require(cfg['server']['host']=='127.0.0.1' and cfg['interaction']['require_verified_speaker_for_auto_send'] is True
            and cfg['worker_events']['host']=='127.0.0.1' and cfg['worker_events']['port']==9,'isolation configuration changed')
    for group in ('asr','llm','tts'):
        require(all(not v or str(v).startswith('http://127.0.0.1:') for k,v in cfg[group].items()
                    if k.endswith('_url') or k=='url'),'nonlocal route in candidate')
    return {'status':'verified_original_integrated_private_noaction_evidence',
            'one_manual_submit':True,'forged_owner_and_commit_refused':True,'canonical_revision':REVISION,'real_local_model_calls_each':1,'socket_generations':3,
            'recovered_pcm_matches_saved_wav':True,'playback_ack_durable':True,
            'input_seconds':len(source_pcm)/32000,'output_seconds':len(out)/32000,
            'worker_events_intentionally_disconnected':True,'physical_playback_verified':False,
            'production_agent_execution_verified':False,'automatic_turn_detector_verified':False,
            'original_result_sha256':PINNED['result.json']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--as-run-script',type=Path,required=True)
    p.add_argument('--canonical-source',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.evidence,a.as_run_script,a.canonical_source),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
