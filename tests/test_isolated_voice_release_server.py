"""Zero-network safety admission for private actual voice-server process probe."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import stat
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/demos/verify_isolated_voice_release_server.py'
SPEC=importlib.util.spec_from_file_location('private_release_probe',P)
MOD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def source_and_template(tmp_path,monkeypatch):
    source=tmp_path/'src';source.mkdir()
    contents={name:(name+' pinned source').encode() for name in MOD.EXPECTED_SOURCES}
    for name,data in contents.items():(source/name).write_bytes(data)
    monkeypatch.setattr(MOD,'EXPECTED_SOURCES',{name:MOD.sha(data) for name,data in contents.items()})
    original={'server':{},'storage':{},'asr':{},'llm':{},'tts':{},
              'interaction':{},'worker_events':{},'logging':{}}
    template=tmp_path/'template.json';template.write_text(json.dumps(original))
    monkeypatch.setattr(MOD,'EXPECTED_TEMPLATE_SHA',MOD.sha(template.read_bytes()))
    return source,template,original


def cli(source,template,evidence,run=False):
    args=['--source',str(source),'--template',str(template),'--evidence',str(evidence)]
    if run:args.append('--run')
    return args


def test_private_source_preflight_has_no_process_or_network_or_writes(tmp_path,monkeypatch,capsys):
    source,template,_=source_and_template(tmp_path,monkeypatch)
    def forbidden(*args):pytest.fail('Preflight must never run a child')
    monkeypatch.setattr(MOD,'run_once',forbidden)
    destination=tmp_path/'uncreated'
    assert MOD.main(cli(source,template,destination))==0
    assert not destination.exists()
    assert json.loads(capsys.readouterr().out)['candidate_files_valid']==len(MOD.EXPECTED_SOURCES)


@pytest.mark.parametrize('file_index',range(7))
def test_modifying_any_module_blocks_private_service_start(tmp_path,monkeypatch,file_index):
    source,template,_=source_and_template(tmp_path,monkeypatch)
    path=source/list(MOD.EXPECTED_SOURCES)[file_index]
    path.write_bytes(path.read_bytes()+b' tampered')
    with pytest.raises(ValueError,match='Candidate file differs'):
        MOD.inspect(source,template)


def test_modified_production_template_or_symlink_is_rejected(tmp_path,monkeypatch):
    source,template,_=source_and_template(tmp_path,monkeypatch)
    template.write_text('{"new-config":true}')
    with pytest.raises(ValueError,match='template drifted'):MOD.inspect(source,template)
    # The symlink guard is tested before parsing the intentionally bad template.
    fake=tmp_path/'symlink'
    fake.symlink_to(source)
    with pytest.raises(ValueError,match='Symlinked'):MOD.inspect(fake,template)


def test_scrubbed_config_restricts_every_external_provider_and_path(tmp_path):
    original={'server':{'host':'0.0.0.0','port':1234,'auth_token_file':'/private/live'},
     'storage':{'runtime_root':'/private/live','conversation_root':'/private/live',
                'log_path':'/private/live'},
     'asr':{'service_url':'http://thor/asr','secondary_service_url':'http://thor/asr2',
            'final_quality_url':'http://thor/asr3','health_url':'http://thor/health',
            'secondary_health_url':'http://thor/health2','reconcile_url':'http://thor/reconcile',
            'final_quality_enabled':True},
     'llm':{'url':'http://thor/qwen','health_url':'http://thor/qwenhealth'},
     'tts':{'service_url':'http://nitro/phonon','service_health_url':'http://nitro/health'},
     'interaction':{'auto_send_default':True,'semantic_interrupt_url':'http://thor/interrupt'},
     'worker_events':{'host':'thor','port':8080},
     'logging':{'include_content':True}}
    d=MOD.scrub_config(original,tmp_path,18783)
    assert d['server']['host']=='127.0.0.1' and d['server']['port']==18783
    assert d['server']['auth_token_file'].startswith(str(tmp_path))
    assert d['interaction']['require_verified_speaker_for_auto_send'] is True
    assert d['interaction']['require_client_commit'] is True
    assert d['worker_events']['host']=='127.0.0.1' and d['worker_events']['port']==9
    assert d['logging']['include_content'] is False
    for kind in ('asr','llm','tts'):
        assert all(str(v).startswith('http://127.0.0.1:9/')
                   for k,v in d[kind].items() if k.endswith('_url') or k=='url')
    assert '/private/live' not in json.dumps(d)
    assert '/phonon' not in json.dumps(d)
    assert original['server']['host']=='0.0.0.0'  # original was not mutated


def test_dual_optin_claim_and_real_codepath_one_use_only(tmp_path,monkeypatch,capsys):
    source,template,_=source_and_template(tmp_path,monkeypatch)
    dest=tmp_path/'one-use'
    monkeypatch.delenv(MOD.OPT_IN,raising=False)
    with pytest.raises(ValueError,match='explicit environment'):MOD.main(cli(source,template,dest,True))
    assert not dest.exists()
    called=[]
    def fake_run(src,cfg,output):
        called.append(str(output))
        return {'status':'private_unreleased_speaker_safe_server_boot_and_handshake_verified',
                'speaker_review_guard_default_on':True,'no_voice_or_user_turn_submitted':True,
                'unauthenticated_connection_http_status':401}
    monkeypatch.setattr(MOD,'run_once',fake_run)
    monkeypatch.setenv(MOD.OPT_IN,'1')
    assert MOD.main(cli(source,template,dest,True))==0
    assert len(called)==1
    assert json.loads((dest/'result.json').read_text())['speaker_review_guard_default_on']
    assert stat.S_IMODE((dest/'attempt.json').stat().st_mode)==0o600
    with pytest.raises(FileExistsError):MOD.main(cli(source,template,dest,True))
    assert len(called)==1


def test_failure_is_retained_and_unrepeatable(tmp_path,monkeypatch):
    source,template,_=source_and_template(tmp_path,monkeypatch)
    dest=tmp_path/'failure'
    monkeypatch.setenv(MOD.OPT_IN,'1')
    calls=[]
    def fake_failed(*args):
        calls.append(1);raise TimeoutError('private server failed')
    monkeypatch.setattr(MOD,'run_once',fake_failed)
    assert MOD.main(cli(source,template,dest,True))==2
    d=json.loads((dest/'result.json').read_text())
    assert d['status']=='failed_or_unknown' and d['error_type']=='TimeoutError'
    with pytest.raises(FileExistsError):MOD.main(cli(source,template,dest,True))
    assert len(calls)==1


def test_subprocess_uses_virtualenv_entrypoint_without_resolving_symlink(
        tmp_path,monkeypatch):
    import sys
    original=Path(sys.executable)
    link=tmp_path/'venv-python'
    link.symlink_to(original)
    monkeypatch.setattr(MOD.sys,'executable',str(link))
    assert MOD.runtime_python()==str(link)
    assert Path(MOD.runtime_python()).resolve()==original.resolve()
    assert MOD.runtime_python()!=str(original.resolve())
