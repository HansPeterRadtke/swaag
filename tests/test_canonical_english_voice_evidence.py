"""Deterministic negative tests; never call the actual voice/model endpoints."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/demos"))
try:
    import verify_canonical_english_voice_evidence as audit
    import verify_canonical_english_voice_chain_once as runner
finally:
    sys.path.pop(0)


def trace():
    t,w,r="test-turn","test recorded words","test response"
    def hello(n):
        return dict(type="hello",connection_generation=n,unverified_speaker_requires_review=True)
    def draft():
        return dict(type="history",turns=[dict(turn_id=t,status="draft",recognized_text=w,assistant="")])
    asr=dict(type="asr",event="final",turn_id=t,text=w,auto_send=False)
    review=dict(type="state",state="awaiting_send",reason="speaker_identity_unverified",turn_id=t)
    audio=dict(type="audio",event="start",turn_id=t,stream_id="first",generation=1)
    return t,w,r,[
        hello(1),dict(type="history",turns=[]),asr,dict(review),
        hello(2),draft(),dict(asr,restored=True),dict(review),dict(review),draft(),
        dict(asr,restored=True),
        dict(type="turn",event="submitted",automatic=False,turn_id=t,text=w),
        audio,hello(3),draft(),
        dict(audio,stream_id="second",resumed=True),
        dict(audio,event="end",stream_id="second"),
        dict(type="history",turns=[dict(turn_id=t,assistant=r,status="completed",audio_completed=True)])]


def test_exact_canonical_guarded_recovery():
    t,w,r,e=trace()
    assert audit.audit_events(e,t,w,r)=="second"


@pytest.mark.parametrize("change",[
    lambda e:e.pop(7),
    lambda e:e[8].update(reason="not_reviewed"),
    lambda e:e[8].update(turn_id="other"),
    lambda e:e[9]["turns"][0].update(status="submitted"),
    lambda e:e[9]["turns"][0].update(assistant="premature"),
    lambda e:e[10].update(restored=False),
    lambda e:e[11].update(automatic=True),
    lambda e:e.insert(12,copy.deepcopy(e[11])),
    lambda e:e[15].update(resumed=False),
    lambda e:e[16].update(stream_id="wrong"),
    lambda e:e[17]["turns"][0].update(audio_completed=False),
])
def test_unsafe_or_incomplete_trace_rejected(change):
    t,w,r,e=trace();change(e)
    with pytest.raises(ValueError):
        audit.audit_events(e,t,w,r)


def test_exact_source_identity_matches_independent_pins():
    assert runner.SOURCE_PINS==audit.SOURCE_PINS
    assert runner.REVISION==audit.REVISION
    assert runner.WAV==audit.WAV


def test_relay_no_remote_redirect():
    with pytest.raises(ValueError,match="redirect refused"):
        runner.NoRedirect().redirect_request(None,None,302,"redirect",{},"http://example.invalid")


def test_relay_preapproval_and_duplicate_never_reach_upstream(tmp_path):
    r=runner.Relay(tmp_path,b"",lambda s:s)
    with pytest.raises(RuntimeError,match="no manual confirmation"):
        r.invoke("coder",b"{}")
    assert r.count=={"asr":0,"coder":0,"tts":0}
    r.count["asr"]=1
    with pytest.raises(RuntimeError,match="duplicate upstream"):
        r.invoke("asr",b"")
    assert list(tmp_path.iterdir())==[]
