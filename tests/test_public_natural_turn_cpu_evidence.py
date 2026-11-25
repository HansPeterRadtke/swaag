"""Offline source tests; not evidence of new model inference."""
from pathlib import Path
import copy
import io
import sys
import wave
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    import prepare_public_natural_turn_corpus as prep
    import run_p99_natural_turn_cpu_once as runner
    import verify_p99_natural_turn_evidence as verifier
finally:
    sys.path.pop(0)


def row():
    buf=io.BytesIO()
    with wave.open(buf,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(b'\0\0'*16000)
    return {'id':'public-case','language':'en','cohort':'general','duration':1.0,
            'audio':{'bytes':buf.getvalue()},
            'silence_spans':[{'start':0.1,'end':0.48,'label':'hold'},
                             {'start':0.6,'end':1.0,'label':'eot'}]}


def test_publisher_annotations_are_preserved():
    m,raw=prep.validate_row(row(),'en')
    assert m['wav_sha256']==prep.sha_bytes(raw)
    assert m['spans'][0]=={'index':0,'start_sample':1600,'end_sample':7680,'label':'hold'}
    assert runner.step_endpoints(m['spans'][0])==[5120,7680]
    assert runner.step_endpoints(m['spans'][1])==[12800,15360]


@pytest.mark.parametrize('mutate',[
    lambda r:r['silence_spans'][0].update(label='eot'),
    lambda r:r['silence_spans'][0].update(end=0.8),
    lambda r:r['silence_spans'][-1].update(end=2.0),
    lambda r:r.update(cohort='another'),
    lambda r:r.update(duration=3.0),
])
def test_bad_annotation_rejected(mutate):
    r=row();mutate(r)
    with pytest.raises(ValueError):prep.validate_row(r,'en')


def observed():
    g,_=prep.validate_row(row(),'en')
    x=copy.deepcopy(g)
    x.update(step_count=6,step_seconds=[0.002]*6)
    for s in x['spans']:
        s['points']=[[n,0.9] for n in runner.step_endpoints(s)]
    return g,x


def test_independent_verifier_accepts_exact_causal_trace():
    a,b=observed();verifier.validate_row(a,b)


@pytest.mark.parametrize('mutate',[
    lambda x:x['spans'][0]['points'].append([10240,0.95]),
    lambda x:x['spans'][0]['points'].pop(),
    lambda x:x['spans'][0]['points'][0].__setitem__(1,float('nan')),
    lambda x:x['spans'][0].update(label='eot'),
    lambda x:x.update(step_count=7),
    lambda x:x.update(wav_sha256='forged'),
])
def test_noncausal_or_tampered_trace_rejected(mutate):
    a,b=observed();mutate(b)
    with pytest.raises(ValueError):verifier.validate_row(a,b)


def test_independent_metric_agrees_and_high_score_is_not_authority():
    _,b=observed()
    d=copy.deepcopy(b);d.update(id='other',language='de')
    r=runner.summarize([b,d])
    assert r==verifier.independent_metrics([b,d])
    for lang in ('en','de'):
        assert r[lang]['0.8']['false_hold_spans']==1
        assert r[lang]['0.8']['eot_detected']==1
        assert r[lang]['0.95']['false_hold_spans']==0
        assert r[lang]['0.95']['eot_detected']==0


def test_no_padding_or_future_at_short_pause():
    assert runner.step_endpoints({'start_sample':0,'end_sample':3199})==[]
    assert runner.step_endpoints({'start_sample':0,'end_sample':5120})==[5120]
