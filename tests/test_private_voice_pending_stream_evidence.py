"""Pure evidence tests, not inference or live media tests."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    import verify_private_voice_stop_pending_stream_evidence as a
finally:
    sys.path.pop(0)


def trace():
    t=a.TURN
    return [dict(type='hello',connection_generation=1),
            dict(type='turn',event='submitted',turn_id=t,automatic=False),
            dict(type='audio',event='start',turn_id=t,stream_id='s',streaming=True),
            dict(type='audio',event='cancel',turn_id=t,stream_id='s',reason='playback_stopped'),
            dict(type='hello',connection_generation=2),
            dict(type='history',turns=[dict(status='generated',audio_stopped=True,audio_completed=False)])]


def test_held_media_stop_order():
    assert a.audit_events(trace())=='s'


@pytest.mark.parametrize('mutate',[
    lambda e:e[1].update(automatic=True),
    lambda e:e[2].update(streaming=False),
    lambda e:e[3].update(stream_id='different'),
    lambda e:e[3].update(turn_id='other-turn'),
    lambda e:e[5]['turns'][0].update(audio_stopped=False),
    lambda e:e[5]['turns'][0].update(audio_completed=True),
    lambda e:e.insert(5,dict(type='audio',event='start')),
    lambda e:e.insert(4,dict(type='audio',event='end')),
])
def test_invalid_pending_transport_evidence_refused(mutate):
    e=trace();mutate(e)
    with pytest.raises(ValueError):a.audit_events(e)
