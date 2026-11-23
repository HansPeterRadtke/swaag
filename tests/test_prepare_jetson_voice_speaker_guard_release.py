"""Zero-production-effects tests of the exact Jetson speaker release preview."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/demos'))
MODULE=ROOT/'scripts/demos/prepare_jetson_voice_speaker_guard_release.py'
SPEC=importlib.util.spec_from_file_location('release_noinstall',MODULE)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def unit(live=None, **changes):
    live=live or M.EXPECTED_LIVE_ROOT
    baseline={'ActiveState':'active','SubState':'running','MainPID':'3413413',
      'WorkingDirectory':str(live),'ExecStart':'{ path='+M.PYTHON+
       ' ; argv[]='+M.PYTHON+' '+str(live)+'/cloudflare_ws_voice_server.py --config '
       +M.EXPECTED_CONFIG+' ; }',
      'FragmentPath':'/etc/systemd/system/'+M.EXPECTED_UNIT,
      'DropInPaths':'/etc/systemd/system/jetson-voice-ws.service.d/30-voice-guideline-repair.conf '
       +M.EXPECTED_DROPIN_OLD,'ExecMainStartTimestamp':'2026-11-20 07:42:02 CET','NeedDaemonReload':'no'}
    baseline.update(changes)
    return baseline


def config(**edits):
    result={'server':{'host':'127.0.0.1','port':13483,'websocket_path':'/ws',
               'auth_token_file':'/etc/voice-agent/ws.token'},
      'llm':{'health_url':M.EXPECTED_HEALTH},
      'interaction':{'auto_send_default':True}}
    for k,v in edits.items():
        section,key=k.split('__',1)
        result[section][key]=v
    return result


def health(ready=True, classifier=True, sessions=0, answers=0):
    gw={'http_status':200 if classifier else 503,'body':{'ok':classifier,
       'fast_model':{'ok':True},'strong_model':{'ok':True},
       'classifier_model':{'ok':classifier},'swaag':{'ok':True}}}
    prod={'http_status':200,'body':{'ok':True,'live':True,'ready':ready,
       'components':{'agent':ready},'config_sha256':M.EXPECTED_TEMPLATE_SHA,
       'active_sessions':sessions,'active_conversations':0,
       'active_answer_jobs':answers}}
    return gw,prod


def test_pinned_original_unit_and_exact_security_config_admit_readonly():
    original=Path(M.EXPECTED_LIVE_ROOT)
    d=M.verify_unit(unit(),original)
    assert d['actual_main_pid']==3413413
    assert M.EXPECTED_DROPIN_OLD in d['overrides']
    M.verify_locked_config(config())


@pytest.mark.parametrize('changes',[
    {'ActiveState':'failed'},
    {'NeedDaemonReload':'yes'},
    {'MainPID':'0'},
    {'WorkingDirectory':'/data/unrelated'},
    {'ExecStart':'/bin/true'},
    {'FragmentPath':'/etc/systemd/system/other.service'},
    {'DropInPaths':'/etc/systemd/system/jetson-voice-ws.service.d/70-swaag-verified-speaker-review.conf'},
    {'DropInPaths':M.EXPECTED_DROPIN_OLD+' /etc/systemd/system/jetson-voice-ws.service.d/70-swaag-verified-speaker-review.conf'},
    {'DropInPaths':M.EXPECTED_DROPIN_OLD+' /tmp/adversarial.conf'},
])
def test_unexpected_effective_unit_never_qualifies_for_release(changes):
    with pytest.raises(ValueError):M.verify_unit(unit(**changes),Path(M.EXPECTED_LIVE_ROOT))


@pytest.mark.parametrize('options',[
 {'server__host':'0.0.0.0'},
 {'server__port':19877},
 {'server__websocket_path':'/not-ws'},
 {'llm__health_url':'http://unknown:13484/health'},
 {'interaction__require_verified_speaker_for_auto_send':False},
 {'interaction__require_verified_speaker_for_auto_send':'false'},
 {'interaction__auto_send_default':False},
 {'server__auth_token_file':'/tmp/insecure-token'},
 {'server__auth_token_file':'/etc/voice-agent/other.token'},
])
def test_config_change_and_dangerous_auto_disable_fail_closed(options):
    with pytest.raises(ValueError):M.verify_locked_config(config(**options))


def test_legacy_auto_send_without_explicit_guard_is_still_safe_only_after_patch():
    c=config()
    assert 'require_verified_speaker_for_auto_send' not in c['interaction']
    assert M.verify_locked_config(c) is None
    assert M.EXPECTED_SOURCES['cloudflare_ws_voice_server.py']!=M.EXPECTED_CURRENT_FILES['cloudflare_ws_voice_server.py']


def test_classifier_down_does_not_qualify_even_with_healthy_foreground():
    gw,server=health(ready=False,classifier=False)
    decision=M.classify_health(gw,server,unit())
    assert decision['release_gate']=='NO_GO'
    assert 'dedicated_turn_classifier_offline' in decision['reasons']
    assert 'voice_server_end_to_end_readiness_false' in decision['reasons']
    assert decision['backend_summary']['foreground_available']
    assert decision['backend_summary']['SWAAG_reachable']
    assert not decision['backend_summary']['turn_classifier_available']


def test_active_connection_or_answer_blocks_release_even_if_gateway_ready():
    for k in ('sessions','answers'):
        h={'sessions':0,'answers':0};h[k]=1
        gw,server=health(ready=True,classifier=True,**h)
        d=M.classify_health(gw,server,unit())
        assert d['release_gate']=='NO_GO'
        assert 'voice_active_'+('sessions' if k=='sessions' else 'answer_jobs')+'_not_idle' in d['reasons']
    gw,server=health(ready=True,classifier=True)
    d=M.classify_health(gw,server,unit())
    assert d['release_gate']=='TECHNICAL_PREFLIGHT_PASS_NOT_AUTHORIZED'
    assert d['reasons']==[]


def test_unmeasurable_sessions_and_config_drift_refuse_release():
    gw,prod=health(ready=True,classifier=True)
    prod['body']['active_sessions']=None
    prod['body']['config_sha256']='different'
    d=M.classify_health(gw,prod,unit())
    assert d['release_gate']=='NO_GO'
    assert 'voice_active_sessions_not_measurable' in d['reasons']
    assert 'running_server_config_drift' in d['reasons']


def test_no_release_is_ever_issued_by_this_script():
    import ast
    tree=ast.parse(MODULE.read_text())
    commands=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        fn=node.func
        if isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name):
            if fn.value.id=='os' and fn.attr in ('system','rename','replace'):
                pytest.fail('No shell/unit or deployment filesystem mutation allowed')
            if fn.value.id=='subprocess':
                assert fn.attr=='run',('Unexpected process action',fn.attr)
                assert node.args and isinstance(node.args[0],ast.List)
                vals=[x.value for x in node.args[0].elts if isinstance(x,ast.Constant)]
                assert vals[:2] in (['systemctl','show'],['systemd-analyze','verify']),vals
                commands.append(tuple(vals[:2]))
    assert sorted(commands)==[('systemctl','show'),('systemd-analyze','verify')]
    assert 'actual_release_deployed' in MODULE.read_text()


def test_preview_dropin_is_only_for_staging_and_has_no_header_injection(tmp_path):
    p=Path('/data/var/swaag-benchmarks/test-verify-guard-20261123/candidate')
    content=M.preview_override(p)
    assert '[Unit]' in content and 'RequiresMountsFor='+str(p) in content
    assert '[Service]' in content and 'ExecStart=\n' in content
    assert str(p/'cloudflare_ws_voice_server.py') in content
    assert 'PREVIEW ONLY' in content and 'NOT INSTALLED' in content
    for bad in ('/tmp/untrusted','/data/var/swaag-benchmarks/bad candidate',
                '/data/var/swaag-benchmarks/a\n[Service]'):
        with pytest.raises(ValueError):M.preview_override(Path(bad))


def fake_candidate(tmp_path,monkeypatch):
    p=tmp_path/'pinned-source';p.mkdir()
    m={name:(name+' exact untrusted replacement rejected').encode() for name in M.EXPECTED_SOURCES}
    for name,body in m.items():(p/name).write_bytes(body)
    monkeypatch.setattr(M,'EXPECTED_SOURCES',{name:M.sha(body) for name,body in m.items()})
    return p


def test_staging_private_only_with_exact_sha_and_rollback_manifest(tmp_path,monkeypatch):
    source=fake_candidate(tmp_path,monkeypatch)
    root=Path('/data/var/swaag-benchmarks/unit-test-only-release-candidate')
    # Use a path with known prefix, but all actual unit-test write effects go
    # into the test's private temporary directory.
    monkeypatch.setattr(M,'preview_override',lambda candidate:'[Unit]\n# PREVIEW ONLY\n[Service]\nExecStart=\n')
    monkeypatch.setattr(M,'verify_staged_preview_unit',lambda staged,evidence:{'offline_systemd_unit_parser_returncode':0,'service_installed':False})
    out=tmp_path/'private-evidence';out.mkdir()
    gate={'release_gate':'NO_GO'}
    data=M.stage_preview(source,out,config(),unit(),gate,{})
    assert data['release_gate']=='NO_GO'
    assert data['production_unit_config_or_service_changed'] is False
    assert data['independent_offline_systemd_parser']['service_installed'] is False
    assert len(list((out/'candidate').iterdir()))==len(M.EXPECTED_SOURCES)
    for name,expect in M.EXPECTED_SOURCES.items():
        assert M.sha((out/'candidate'/name).read_bytes())==expect
    rollback=json.loads((out/'rollback-manual.json').read_text())
    assert rollback['not_executable_or_authorizing']
    assert rollback['existing_verified_last_override']==M.EXPECTED_DROPIN_OLD
    assert rollback['no_unit_or_live_source_files_modified']
    assert (out/M.PREVIEW_DROPIN_NAME).is_file()


def test_tampered_candidate_is_rejected_before_private_stage(tmp_path,monkeypatch):
    source=fake_candidate(tmp_path,monkeypatch)
    (source/'cloudflare_ws_voice_server.py').write_bytes(b'bad')
    out=tmp_path/'unique';out.mkdir()
    monkeypatch.setattr(M,'preview_override',lambda p:'preview')
    with pytest.raises(ValueError,match='Source changed'):
        M.stage_preview(source,out,config(),unit(),{'release_gate':'NO_GO'}, {})


def test_source_only_prefight_and_single_use_no_effects(tmp_path,monkeypatch,capsys):
    cand=tmp_path/'candidate';live=tmp_path/'live';templ=Path(M.EXPECTED_CONFIG)
    fake_unit=unit()
    monkeypatch.setattr(M,'inspect_all',lambda *a:(config(),{'file':'hash'},{
        'identity':{'actual_main_pid':3413413},'unit':fake_unit}))
    monkeypatch.setattr(M,'health_one',lambda url:pytest.fail('Pure preflight is offline'))
    source=['--candidate',str(cand),'--live',str(live),'--config',str(templ),
            '--evidence',str(tmp_path/'new')]
    assert M.main(source)==0
    assert not (tmp_path/'new').exists()
    assert json.loads(capsys.readouterr().out)['status']=='read_only_exact_service_source_preflight'
    with pytest.raises(ValueError,match='Explicit one-use'):
        M.main(source+['--run'])
    assert not (tmp_path/'new').exists()


def test_failed_live_preflight_cannot_be_replayed_after_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'inspect_all',lambda *a:(config(),{},
          {'identity':{'actual_main_pid':1},'unit':unit(MainPID='1')}))
    monkeypatch.setattr(M,'health_one',lambda url:(_ for _ in ()).throw(TimeoutError('health blocked')))
    monkeypatch.setenv(M.OPT_IN,'1')
    cmd=['--candidate','/tmp/src','--live','/tmp/old','--config',M.EXPECTED_CONFIG,
         '--evidence',str(tmp_path/'one-time')]
    assert M.main(cmd+['--run'])==2
    d=json.loads((tmp_path/'one-time'/'result.json').read_text())
    assert d['status']=='failed_or_uncertain' and d['error_type']=='TimeoutError'
    with pytest.raises(FileExistsError):M.main(cmd+['--run'])


def test_health_lookup_never_uses_untrusted_destination():
    with pytest.raises(ValueError,match='fixed read-only health endpoints'):
        M.health_one('http://attacker.example/health')


def test_existing_private_bearer_is_stat_only_and_requires_exclusive_mode(tmp_path,monkeypatch):
    p=tmp_path/'private-staged-token';p.write_text('private bytes must not be read or copied')
    p.chmod(0o600)
    state=M.verify_private_token_stat({'server':{'auth_token_file':str(p)}})
    assert state['credential_mode']=='0o600' and state['bearer_contents_read_or_saved'] is False
    p.chmod(0o644)
    with pytest.raises(ValueError,match='exclusively readable'):
        M.verify_private_token_stat({'server':{'auth_token_file':str(p)}})
    p.chmod(0o600)
    q=tmp_path/'link';q.symlink_to(p)
    with pytest.raises(ValueError,match='symlinked'):
        M.verify_private_token_stat({'server':{'auth_token_file':str(q)}})


def test_staged_unit_verifier_checks_only_offline_parser_and_blocks_bad_syntax(tmp_path,monkeypatch):
    staged=tmp_path/'candidate';staged.mkdir()
    (staged/'cloudflare_ws_voice_server.py').write_text('pass')
    evidence=tmp_path/'private-evidence';evidence.mkdir()
    invoked=[]
    class Result:
        returncode=0
        stdout=''
        stderr='unit syntax verified locally'
    def fake_run(args,**kwargs):
        invoked.append(args)
        assert args[:2]==['systemd-analyze','verify']
        assert args[2].endswith('private-preview-grammar-only.service')
        return Result()
    monkeypatch.setattr(M.subprocess,'run',fake_run)
    result=M.verify_staged_preview_unit(staged,evidence)
    assert result['offline_systemd_unit_parser_returncode']==0
    assert result['service_installed'] is False
    assert invoked and (evidence/'private-unit-parser-output.log').exists()
    assert 'PrivateTmp=true' in (evidence/'private-preview-grammar-only.service').read_text()


def test_staged_unit_verifier_rejects_parser_failure_without_a_retry(tmp_path,monkeypatch):
    staged=tmp_path/'candidate';staged.mkdir()
    evidence=tmp_path/'private-evidence';evidence.mkdir()
    class Result:
        returncode=1
        stdout=''
        stderr='bad unit syntax'
    monkeypatch.setattr(M.subprocess,'run',lambda *a,**kw:Result())
    with pytest.raises(ValueError,match='syntax check failed'):
        M.verify_staged_preview_unit(staged,evidence)
    with pytest.raises(FileExistsError):M.verify_staged_preview_unit(staged,evidence)
