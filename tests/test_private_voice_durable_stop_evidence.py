"""Synthetic negative tests; real private process receipts are separate."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    import verify_private_voice_durable_stop_evidence as audit
    import prepare_private_voice_durable_audio_stop as prepare
finally:
    sys.path.pop(0)


def events(restart=False):
    h=lambda g:dict(type='hello',connection_generation=g)
    start=dict(type='audio',event='start',stream_id='one',resumed=False)
    e=[h(1),start,dict(type='audio',event='cancel',stream_id='one',
                      reason='playback_stopped'),h(2)]
    if restart:
        e.append(dict(start,stream_id='two',resumed=True))
    else:
        e.append(dict(type='history',turns=[dict(status='completed',
                         audio_stopped=True,audio_completed=False)]))
    return e


def test_baseline_and_corrected_traces_are_distinct():
    audit.audit_stop(events(True),True)
    audit.audit_stop(events(False),False)
    with pytest.raises(ValueError):audit.audit_stop(events(True),False)
    with pytest.raises(ValueError):audit.audit_stop(events(False),True)


@pytest.mark.parametrize('mutate',[
    lambda e:e[2].update(stream_id='wrong'),
    lambda e:e[2].update(reason='another_reason'),
    lambda e:e[3].update(connection_generation=1),
    lambda e:e[-1]['turns'][0].update(audio_completed=True),
    lambda e:e[-1]['turns'][0].update(audio_stopped=False),
    lambda e:e.insert(3,dict(type='audio',event='end')),
    lambda e:e.insert(3,dict(type='turn',event='submitted')),
])
def test_false_stopped_result_rejected(mutate):
    e=events(False);mutate(e)
    with pytest.raises(ValueError):audit.audit_stop(e,False)


def test_wrong_candidate_source_never_patched():
    with pytest.raises(ValueError,match='unknown W71 base'):
        prepare.patch(b'not the pinned source')


def test_evidence_hash_and_root_required(tmp_path):
    root=tmp_path/'root';root.mkdir()
    p=root/'result';p.write_bytes(b'test')
    assert audit.pinned(root,'result',prepare.sha(b'test'))==b'test'
    with pytest.raises(ValueError,match='changed'):
        audit.pinned(root,'result',prepare.sha(b'altered'))
    (tmp_path/'outside').write_bytes(b'test')
    with pytest.raises(ValueError,match='outside'):
        audit.pinned(root,'../outside',prepare.sha(b'test'))
