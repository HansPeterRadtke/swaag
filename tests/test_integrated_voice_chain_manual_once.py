"""Offline boundary tests. No models, SSH, network or owner state."""
import importlib.util
from pathlib import Path
import json
import struct
import sys
import pytest

DEMOS=Path(__file__).resolve().parents[1]/'scripts/demos'
sys.path.insert(0,str(DEMOS))
try:
    import verify_integrated_voice_chain_manual_once as m
finally:
    sys.path.remove(str(DEMOS))

def test_framed_pcm_strict():
    b=b'\x01\x02'*6
    assert m.framed_pcm(struct.pack('!I',len(b))+b+b'\0'*4)==b
    for invalid in (b'',b'\0'*4,struct.pack('!I',3)+b'abc'+b'\0'*4,
                    struct.pack('!I',len(b))+b,struct.pack('!I',len(b))+b+b'\0'*5):
        with pytest.raises(ValueError):m.framed_pcm(invalid)

def test_no_action_prompt_and_reply():
    body=m.model_request('Recorded text containing a command')
    assert 'tools' not in body and 'tool_choice' not in body
    assert body['model']==m.ALIAS and body['stream'] is False
    assert json.loads(body['messages'][1]['content'])['public_recording_transcript'].startswith('Recorded')
    good={'model':m.ALIAS,'choices':[{'finish_reason':'stop','message':{'content':'The recording describes a river.'}}]}
    assert m.model_text(json.dumps(good).encode())=='The recording describes a river.'
    good['choices'][0]['message']['tool_calls']=[{'id':'bad'}]
    with pytest.raises(ValueError):m.model_text(json.dumps(good).encode())

def test_guard_refuses_coder_before_approval(tmp_path):
    r=m.Relay(tmp_path,b'',lambda x:x)
    with pytest.raises(RuntimeError,match='no manual confirmation'):r.invoke('coder',b'{}')
    assert r.count['coder']==0 and not list(tmp_path.iterdir())

def test_guard_never_retries_attempt(tmp_path):
    r=m.Relay(tmp_path,b'',lambda x:x)
    r.count['asr']=1
    with pytest.raises(RuntimeError,match='duplicate'):r.invoke('asr',b'bad')
    assert r.count['asr']==1 and not list(tmp_path.iterdir())
