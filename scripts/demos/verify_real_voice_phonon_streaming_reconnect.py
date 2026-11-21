#!/usr/bin/env python3
"""Real source-pinned no-action Whisper -> manual review -> live Phonon TTS -> reconnect.

This explicitly opt-in Jetson test uses one PREVIOUSLY verified tool-free
SWAAG reply and a pinned real source WAV. It intercepts the external
orchestrator request, without authorizing commands, and uses the actual live
VoiceSession.answer and its streaming Phonon TTS path. Isolated temporary
VoiceCore, localhost WebSocket, valid normal playback ACK and no physical phone.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,sys,time,unittest,wave
from pathlib import Path
from unittest.mock import patch
from websockets.legacy.client import connect
from websockets.legacy.server import serve
from verify_real_voice_phonon_reconnect import check_voice_source, load_verified_source

os.environ['SWAAG_REAL_ASR_NOACTION']='1'
os.environ['SWAAG_FULL_VOICE_NOACTION']='1'
VOICE_ROOT=Path('/data/src/worktrees/swaag-local-llm-noaction-20261119/voice_agent_live_runtime')
sys.path.insert(0,str(VOICE_ROOT))
import cloudflare_ws_voice_server as server
import test_voice_real_asr_reconnect as source_fixture

SOURCE=Path('/data/var/swaag-benchmarks/target-speaker-20261111/target.wav')
SOURCE_SHA='2b29bead44440ee10117f8dc6f2f20be552aa2ffbb39b36f10c854dc165376ee'
SOURCE_PROOF=Path('/data/var/swaag-benchmarks/real-canonical-reply-phonon-playback-20261121/verified-source-proof.json')
OUTPUT=None  # Set only after the explicit CLI --run admission.

class StreamedPhononNoAction(unittest.TestCase):
    setUp=source_fixture.RealAsrReconnectNoActionTests.setUp
    tearDown=source_fixture.RealAsrReconnectNoActionTests.tearDown
    def test_live_tts_disconnect_reconnect_exact(self):
        reply=load_verified_source(SOURCE_PROOF)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
        with wave.open(str(SOURCE),'rb') as wf:
            assert (wf.getframerate(),wf.getnchannels(),wf.getsampwidth())==(16000,1,2)
            pcm=wf.readframes(wf.getnframes())
        asr_calls=[];canonical_source_only=[]; sessions=[]
        real_asr=server.verify_transcript_large_v3_http
        def observed_asr(*args,**kwargs):
            assert bytes(args[1])==pcm
            t0=time.monotonic();text=real_asr(*args,**kwargs)
            asr_calls.append({'seconds':round(time.monotonic()-t0,3),'words':len(text.split())})
            return text
        def inert_verified_prior_answer(*args,**kwargs):
            # This is a prerecorded truthful canonical result, NOT a fresh SWAAG
            # operation. This stub has no tool/system action API.
            canonical_source_only.append((args[2],args[3]))
            return {'text':reply,'total_seconds':0,'first_token_seconds':0,'execution_status':'no_action'}
        async def prohibit(*args,**kwargs):raise AssertionError('No SWAAG side-effect or real tool execution')
        async def idle(*args,**kwargs):await asyncio.Future()
        real_hello=server.VoiceSession.hello
        async def record_hello(session):
            sessions.append(session)
            await real_hello(session)
        async def expect(phone,predicate,seconds=12):
            until=asyncio.get_running_loop().time()+seconds
            while True:
                left=until-asyncio.get_running_loop().time()
                self.assertGreater(left,0,'No expected WebSocket event')
                msg=await asyncio.wait_for(phone.recv(),left)
                if isinstance(msg,bytes):
                    if predicate('binary',msg):return msg
                    continue
                value=json.loads(msg)
                if value.get('type')=='turn' and value.get('event')=='submitted':
                    self.assertFalse(value.get('automatic'))
                if predicate('event',value):return value
        async def scenario():
            conversation='source-verified-live-phonon-noaction-reconnect'
            headers={'Authorization':'Bearer test-only-bearer-token'}
            assert server.cfg().tts.service_url=='http://127.0.0.1:15305/synthesize'
            with patch.object(server.VoiceSession,'hello',record_hello), \
                 patch.object(server,'verify_transcript_large_v3_http',observed_asr), \
                 patch.object(server,'receive_voice_run',inert_verified_prior_answer), \
                 patch.object(server,'cancel_voice_run',prohibit), \
                 patch.object(server.VoiceSession,'worker_events_loop',idle), \
                 patch.object(server.VoiceSession,'heartbeat_loop',idle), \
                 patch.object(server,'dependency_status',return_value={'server':True,'agent':False,'tts':True}):
                async with serve(server.handler,'127.0.0.1',0,process_request=server.process_request,
                                 close_timeout=1,max_queue=256) as listener:
                    port=listener.sockets[0].getsockname()[1]
                    base=f'ws://127.0.0.1:{port}/ws?conversation={conversation}&generation='
                    async with connect(base+'1',extra_headers=headers,close_timeout=1) as first:
                        await expect(first,lambda k,e:k=='event' and e.get('type')=='hello',8)
                        await first.send(json.dumps({'type':'client_state','auto_send':False,'auto_transcribe':False,'auto_commit_supported':False}))
                        await first.send(json.dumps({'type':'begin_manual_utterance'}))
                        await expect(first,lambda k,e:k=='event' and e.get('type')=='state' and e.get('source')=='manual_recording',8)
                        n=server.cfg().audio.frame_ms*32
                        for i in range(0,len(pcm),n):await first.send(pcm[i:i+n])
                        await first.send(json.dumps({'type':'end_utterance'}))
                        final=await expect(first,lambda k,e:k=='event' and e.get('type')=='asr' and e.get('event')=='final',60)
                        turn_id=final['turn_id']
                        self.assertFalse(final['auto_send'])
                        await expect(first,lambda k,e:k=='event' and e.get('type')=='state' and e.get('state')=='awaiting_send')
                        self.assertEqual('draft',server.get_turn(conversation,turn_id)['status'])
                        self.assertFalse(server.get_turn(conversation,turn_id)['submitted_text'])
                        self.assertEqual([],canonical_source_only)
                    async with connect(base+'2',extra_headers=headers,close_timeout=1) as second:
                        await expect(second,lambda k,e:k=='event' and e.get('type')=='hello',8)
                        await expect(second,lambda k,e:k=='event' and e.get('type')=='history',8)
                        restored=await expect(second,lambda k,e:k=='event' and e.get('type')=='asr' and e.get('event')=='final' and e.get('restored') is True,8)
                        self.assertEqual(turn_id,restored['turn_id'])
                        await expect(second,lambda k,e:k=='event' and e.get('type')=='state' and e.get('state')=='awaiting_send')
                        await second.send(json.dumps({'type':'client_state','auto_send':False}))
                        await second.send(json.dumps({'type':'submit','turn_id':turn_id,'text':restored['text']}))
                        await expect(second,lambda k,e:k=='event' and e.get('type')=='turn' and e.get('event')=='submitted',7)
                        start=await expect(second,lambda k,e:k=='event' and e.get('type')=='audio' and e.get('event')=='start' and e.get('streaming') is True,55)
                        self.assertEqual(turn_id,start['turn_id'])
                        partial=await expect(second,lambda k,e:k=='binary' and len(e)>0,12)
                        partial_bytes=len(partial)
                    until=asyncio.get_running_loop().time()+15
                    while server.get_turn(conversation,turn_id)['status']!='ready':
                        self.assertLess(asyncio.get_running_loop().time(),until,'Original stream failed to persist resumable WAV')
                        await asyncio.sleep(.05)
                    record=server.get_turn(conversation,turn_id)
                    saved=Path(record['assistant_audio_path']);self.assertTrue(saved.is_file())
                    full_pcm=server.core().wav_pcm(str(saved))
                    self.assertGreater(len(full_pcm),partial_bytes)
                    self.assertEqual(reply,record['assistant'])
                    async with connect(base+'3',extra_headers=headers,close_timeout=1) as third:
                        await expect(third,lambda k,e:k=='event' and e.get('type')=='hello',8)
                        await expect(third,lambda k,e:k=='event' and e.get('type')=='history',8)
                        start=await expect(third,lambda k,e:k=='event' and e.get('type')=='audio' and e.get('event')=='start',14)
                        self.assertTrue(start['resumed'])
                        stream=start['stream_id'];generation=start['generation']
                        recovered=bytearray()
                        stop=asyncio.get_running_loop().time()+22
                        while True:
                            left=stop-asyncio.get_running_loop().time()
                            self.assertGreater(left,0,'Did not finish audio playback')
                            msg=await asyncio.wait_for(third.recv(),left)
                            if isinstance(msg,bytes):recovered+=msg
                            else:
                                e=json.loads(msg)
                                if e.get('type')=='audio' and e.get('event')=='end':break
                        self.assertEqual(full_pcm,bytes(recovered))
                        await third.send(json.dumps({'type':'playback_completed','turn_id':turn_id,'stream_id':stream,'generation':generation}))
                        ack=await expect(third,lambda k,e:k=='event' and e.get('type')=='playback_ack',7)
                        self.assertEqual(turn_id,ack['turn_id'])
                        self.assertTrue(server.get_turn(conversation,turn_id)['audio_completed'])
                    async with connect(base+'4',extra_headers=headers,close_timeout=1) as fourth:
                        await expect(fourth,lambda k,e:k=='event' and e.get('type')=='hello',8)
                        history=await expect(fourth,lambda k,e:k=='event' and e.get('type')=='history',8)
                        await expect(fourth,lambda k,e:k=='event' and e.get('type')=='state' and e.get('state')=='listening',8)
                        self.assertEqual(1,len(history['turns']))
                        with self.assertRaises(asyncio.TimeoutError):await asyncio.wait_for(fourth.recv(),.35)
                    self.assertEqual(len(canonical_source_only),1)
                    self.assertEqual(1,len(asr_calls))
                    self.assertEqual(1,len(server._history_read(conversation)['dialogue_history']))
                    return {'status':'real_tts_streaming_to_ephemeral_voice_ws_with_prior_verified_answer',
                            'canonical_answer_not_fresh':True,'source_original_wav_sha256':SOURCE_SHA,
                            'tts_wav_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),
                            'generated_pcm_bytes':len(full_pcm),'partial_pcm_before_disconnect':partial_bytes,
                            'played_pcm_recovery_exact':True,'normal_ack':True,'no_replay_after_ack':True,
                            'model_action_calls':0,'source_stub_calls':len(canonical_source_only),
                            'real_Whisper_calls':len(asr_calls),'real_TTS_service':server.cfg().tts.service_engine,
                            'tts_wall_seconds':record.get('tts_seconds'),
                            'four_websocket_generations':[s.client_generation for s in sessions]}
        result=asyncio.run(asyncio.wait_for(scenario(),120))
        assert result['four_websocket_generations']==[1,2,3,4]
        assert result['model_action_calls']==0
        assert not OUTPUT.exists()
        OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
        OUTPUT.chmod(0o600)
        print('REAL_STREAMED_PHONON_NOACTION_RESULT',json.dumps(result,sort_keys=True),flush=True)

def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reply-proof',required=True,type=Path)
    cli.add_argument('--voice-source',required=True,type=Path)
    cli.add_argument('--evidence',required=True,type=Path)
    cli.add_argument('--run',action='store_true')
    args=cli.parse_args(argv)
    # Exact version and original WAV check, before enabling the real model.
    source=check_voice_source(args.voice_source)
    reply=load_verified_source(args.reply_proof)
    if args.voice_source.resolve()!=VOICE_ROOT.resolve():
        raise ValueError('This test uses only its pinned local Jetson source checkout')
    if not args.run:
        print(json.dumps({'status':'preflight_only','no_TTS_or_ASR_called':True,
                          'verified_source':source,'verified_reply_sha256':hashlib.sha256(reply.encode()).hexdigest()}))
        return 0
    if os.environ.get('SWAAG_STREAMED_PHONON_NOACTION')!='1':
        cli.error('Explicit SWAAG_STREAMED_PHONON_NOACTION=1 opt-in required')
    old=os.umask(0o077)
    try:
        args.evidence.mkdir(mode=0o700,parents=False,exist_ok=False)
    finally:
        os.umask(old)
    global SOURCE_PROOF,OUTPUT
    SOURCE_PROOF=args.reply_proof
    OUTPUT=args.evidence/'result.json'
    assert not OUTPUT.exists()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StreamedPhononNoAction)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun!=1 or result.skipped:
        raise RuntimeError('Real streamed TTS no-action acceptance failed')
    metrics=json.loads(OUTPUT.read_text())
    if (metrics.get('model_action_calls')!=0 or metrics.get('source_stub_calls')!=1
            or metrics.get('real_Whisper_calls')!=1
            or metrics.get('four_websocket_generations')!=[1,2,3,4]
            or not all(metrics.get(k) is True for k in ('played_pcm_recovery_exact','normal_ack','no_replay_after_ack'))):
        raise RuntimeError('Actual live streaming receipt failed independent acceptance')
    print('STREAMED_PHONON_NOACTION_VERIFIED',json.dumps({
        'status':metrics['status'],'four_generations':True,
        'exact_reconnect_pcm':True,'normal_ack':True,'SWAAG_tool_actions':0,
        'evidence':str(OUTPUT)},sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
