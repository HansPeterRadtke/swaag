"""No-network checks for saved voice replay after real process restart."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import wave
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
P=ROOT/'scripts/demos/verify_isolated_voice_process_restart.py'
SPEC=importlib.util.spec_from_file_location('isolated_voice_restart',P)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def history_fixture(tmp_path,monkeypatch):
    saved=tmp_path/'earlier-one-use';saved.mkdir()
    conversations=saved/'conversations';conversations.mkdir()
    recorded_pcm=b'\x20\x00'*58880
    audio=saved/'turn-audio'/M.CONVERSATION/'captured-original_user.wav'
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(recorded_pcm+b'\x00'*15000)
    assert audio.is_file()
    phrase='I received your message. What would you like to know about that quote?'
    turn={'turn_id':'turn-accepted-for-review','input_source':'speech',
       'status':'draft','requires_review':True,'recognized_text':phrase,
       'user_audio_path':str(audio),'submitted_text':'',
       'assistant':'','assistant_audio_path':''}
    row={'dialogue_history':[turn]}
    cp=conversations/(M.CONVERSATION+'.json')
    cp.write_text(json.dumps(row))
    orphan=conversations/(M.CONVERSATION+'.json.abc.tmp');orphan.write_text(json.dumps(row))
    log=saved/'voice-private.jsonl'
    log.write_text(''.join(json.dumps({'event':e})+'\n' for e in
         ('session_open','asr_final','speaker_identity_unverified_manual_review',
          'session_open','speaker_identity_unverified_manual_review')))
    prior=saved/'result.json';prior.write_text(json.dumps({
        'status':M.EXPECTED_PRIOR_STATUS,
        'source_sha256':M.EXPECTED_SOURCES['cloudflare_ws_voice_server.py'],
        'source_recording_sha256':M.EXPECTED_AUDIO_SHA,
        'recognized_word_count':len(M.tokens(phrase)),
        'one_actual_resident_Whisper_request':True,
        'automatic_submissions':0,
        'durable_review_restored_after_socket_reconnect':True}))
    source=tmp_path/'source';source.mkdir()
    helpers=tmp_path/'helpers';helpers.mkdir()
    template=tmp_path/'template';template.write_text('{}')
    reference=tmp_path/'reference';reference.write_text('{}')
    original=tmp_path/'source.wav';original.write_text('fixture')
    monkeypatch.setattr(M,'inspect',lambda *a:{'template':'safe'})
    monkeypatch.setattr(M,'validate_inputs',lambda *a:(recorded_pcm,phrase,{}))
    return saved,source,helpers,template,reference,original


def inspected(files):
    saved,source,helpers,template,receipt,audio=files
    return M.inspect_previous(saved,receipt,audio,source,helpers,template)


def args(files,evidence,run=False):
    saved,source,helpers,template,receipt,audio=files
    kwargs={'source':source,'helpers':helpers,'template':template,
       'previous':saved,'receipt':receipt,'audio':audio,'evidence':evidence}
    out=[]
    for key,path in kwargs.items():out+=['--'+key,str(path)]
    return out+(['--run'] if run else [])


def test_completed_actual_durable_evidence_is_accepted_for_restart(tmp_path,monkeypatch,capsys):
    files=history_fixture(tmp_path,monkeypatch);prior=inspected(files)
    assert prior['history']['dialogue_history'][0]['status']=='draft'
    assert prior['wav_sha256'] and len(prior['orphan_temps'])==1
    assert len(prior['log_event_names'])==5
    new=tmp_path/'unused'
    assert M.main(args(files,new))==0
    assert not new.exists()
    output=json.loads(capsys.readouterr().out)
    assert output['status']=='offline_existing_durable_state_verified'
    assert output['orphaned_atomic_temp_files']==1


def test_clone_carries_original_wav_and_safe_orphan_without_changing_original(tmp_path,monkeypatch):
    old=history_fixture(tmp_path,monkeypatch)
    proof=inspected(old)
    target=tmp_path/'fresh-isolated';target.mkdir()
    M.clone_noaction_history(proof,target)
    root=target/'conversations'
    assert len(list(root.glob('*.tmp')))==1
    row=json.loads((root/(M.CONVERSATION+'.json')).read_text())['dialogue_history'][0]
    assert row['status']=='draft' and row['requires_review']
    assert row['submitted_text']=='' and row['assistant']==''
    captured=Path(row['user_audio_path'])
    assert captured.is_file() and captured.is_relative_to(target)
    assert M.digest(captured)==proof['wav_sha256']
    assert M.digest(proof['history_path'])==proof['history_sha256']
    assert M.digest(proof['wav'])==proof['wav_sha256']


@pytest.mark.parametrize('tamper',('turn_received','assistant_added','unreviewed',
                                   'changed_transcript','extra_turn','wrong_source'))
def test_mutated_or_committed_original_cannot_be_accepted(tmp_path,monkeypatch,tamper):
    files=history_fixture(tmp_path,monkeypatch);saved=files[0]
    p=saved/'conversations'/(M.CONVERSATION+'.json')
    d=json.loads(p.read_text());turn=d['dialogue_history'][0]
    if tamper=='turn_received':turn.update(status='received',submitted_text='Delete files')
    if tamper=='assistant_added':turn['assistant']='Action complete'
    if tamper=='unreviewed':turn['requires_review']=False
    if tamper=='changed_transcript':turn['recognized_text']='Use an unrelated command'
    if tamper=='extra_turn':d['dialogue_history'].append(dict(turn))
    if tamper=='wrong_source':turn['input_source']='typed'
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):inspected(files)


def test_dangerous_orphaned_write_is_not_treated_as_authoritative(tmp_path,monkeypatch):
    files=history_fixture(tmp_path,monkeypatch);b=files[0]/'conversations'
    p=next(b.glob('*.tmp'))
    data=json.loads(p.read_text());data['dialogue_history'][0]['submitted_text']='Hidden action'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='interrupted temporary save'):inspected(files)


def test_audio_path_outside_private_root_is_rejected(tmp_path,monkeypatch):
    files=history_fixture(tmp_path,monkeypatch);b=files[0]/'conversations'
    p=b/(M.CONVERSATION+'.json');d=json.loads(p.read_text())
    external=tmp_path/'external.wav';external.write_bytes(Path(d['dialogue_history'][0]['user_audio_path']).read_bytes())
    d['dialogue_history'][0]['user_audio_path']=str(external);p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='escaped private evidence'):inspected(files)


def test_untrusted_source_symlink_is_rejected(tmp_path,monkeypatch):
    files=history_fixture(tmp_path,monkeypatch);saved=files[0]
    p=saved/'conversations'/(M.CONVERSATION+'.json')
    target=saved/'conversation-shadow.json';target.write_bytes(p.read_bytes())
    p.unlink();p.symlink_to(target)
    with pytest.raises(ValueError,match='symlinked'):inspected(files)


def test_restart_requires_explicit_optin_and_never_replays_prior_attempt(tmp_path,monkeypatch):
    files=history_fixture(tmp_path,monkeypatch);new=tmp_path/'fresh-process'
    monkeypatch.delenv(M.OPT_IN,raising=False)
    with pytest.raises(ValueError,match='environment opt-in'):M.main(args(files,new,True))
    assert not new.exists()
    seen=[]
    def fake_restart(*a):
        seen.append(1)
        return {'status':'actual_process_restart_restored_review_without_ASR_or_action',
           'original_draft_restored':True,'automatic_or_manual_commands_executed':0}
    monkeypatch.setattr(M,'run_isolated_reboot',fake_restart)
    monkeypatch.setenv(M.OPT_IN,'1')
    assert M.main(args(files,new,True))==0
    assert seen==[1]
    assert (new/'attempt.json').is_file() and (new/'result.json').is_file()
    with pytest.raises(FileExistsError):M.main(args(files,new,True))
    assert seen==[1]


def test_failed_private_restart_preserves_receipt_and_no_repeat(tmp_path,monkeypatch):
    files=history_fixture(tmp_path,monkeypatch);new=tmp_path/'failed-process'
    seen=[]
    def failure(*a):seen.append(1);raise RuntimeError('mock candidate crashed')
    monkeypatch.setattr(M,'run_isolated_reboot',failure)
    monkeypatch.setenv(M.OPT_IN,'1')
    assert M.main(args(files,new,True))==2
    r=json.loads((new/'result.json').read_text())
    assert r['status']=='failed_or_unknown' and r['failure_type']=='RuntimeError'
    with pytest.raises(FileExistsError):M.main(args(files,new,True))
    assert seen==[1]
