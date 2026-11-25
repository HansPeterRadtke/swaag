"""Causal prefix and fixed fusion policy source tests, no model calls."""
from pathlib import Path
import copy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    import prepare_public_natural_turn_text as prep
    import verify_natural_turn_fusion_evidence as audit
finally:
    sys.path.pop(0)


def test_prefix_stops_at_crossing_word_and_does_not_peek_later():
    words=[{'start':0.0,'end':.1,'word':'please'},
           {'start':.2,'end':.5,'word':'continue'},
           {'start':.1,'end':.2,'word':'future bad alignment'}]
    assert prep.prefix(words,3200)=='please'
    assert prep.prefix(words,0)==''
    assert prep.prefix(words,8000)=='please continue future bad alignment'


def case():
    gold={'id':'x','language':'en','label':'hold','span_index':0,'cutoff_sample':0,
          'has_prefix':True,'text':'some words','text_sha256':prep.sha_bytes(b'some words')}
    o={k:v for k,v in gold.items() if k!='text'}
    o.update(status='valid',p_finished=.81,p_incomplete=.19,tokens=2,inference_seconds=.01)
    return gold,o


def test_text_receipt_valid():
    a,b=case();audit.validate_text_case(a,b)


@pytest.mark.parametrize('edit',[
    lambda r:r.update(text_sha256='other'),
    lambda r:r.update(p_finished=float('nan')),
    lambda r:r.update(p_incomplete=.5),
    lambda r:r.update(tokens=513),
    lambda r:r.update(status='no_aligned_prefix'),
])
def test_text_receipt_tampering_rejected(edit):
    a,b=case();edit(b)
    with pytest.raises(ValueError):audit.validate_text_case(a,b)


def test_fusion_must_not_turn_missing_text_into_authority():
    audio=[];text=[]
    for lang in ('en','de'):
        audio.append({'id':lang,'language':lang,'spans':[
            {'index':0,'label':'hold','start_sample':0,'points':[[5120,.95]]},
            {'index':1,'label':'eot','start_sample':10000,'points':[[15360,.95]]}]})
        text.extend([{'id':lang,'language':lang,'span_index':0,'label':'hold',
                      'cutoff_sample':0,'status':'no_aligned_prefix'},
                     {'id':lang,'language':lang,'span_index':1,'label':'eot',
                      'cutoff_sample':10000,'status':'valid','p_finished':.9}])
    result=audit.compare(audio,text)
    for lang in ('en','de'):
        assert result[lang]['audio_0.36']['false_hold_spans']==1
        assert result[lang]['and_0.36_0.8']['false_hold_spans']==0
        assert result[lang]['and_0.36_0.8']['eot_detected']==1
    bad=copy.deepcopy(text);bad[0]['cutoff_sample']=1
    with pytest.raises(ValueError):audit.compare(audio,bad)
