"""Offline one-use classifier-runner admission tests, no model or provider calls."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import hashlib
import stat
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/run_preinstalled_thor_classifier_isolated_once.py'
SPEC=importlib.util.spec_from_file_location('preinstalled_local_turn_model_oneuse',SCRIPT)
R=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'evidence';root.mkdir()
    model=tmp_path/'tiny-fake-test-only.gguf';model.write_bytes(b'TEST MODEL SOURCE BYTES ONLY')
    benchmark=root/'voice_classifier_acceptance.py';benchmark.write_bytes(b'TEST PUBLISHED BENCHMARK SOURCE')
    cases=[]
    for i in range(17):cases.append({'family':'turn_completion','case_id':f't{i}'})
    for i in range(18):cases.append({'family':'interruption','case_id':f'i{i}'})
    oracle=root/'voice_classifier_oracle_20261104.json';oracle.write_text(json.dumps({'cases':cases}))
    parser=tmp_path/'original-source';parser.mkdir()
    gateway=parser/'voice_agent_gateway.py';gateway.write_bytes(b'gateway')
    conf=parser/'voice_agent_gateway_config.py';conf.write_bytes(b'config')
    service_cfg=tmp_path/'private-config.toml';service_cfg.write_bytes(b'model = "classifier"')
    binary=tmp_path/'llama-server';binary.write_bytes(b'LOCAL BINARY STUB NOT EXECUTABLE MODEL')
    binary.chmod(0o700)
    monkeypatch.setattr(R,'ROOT',root)
    monkeypatch.setattr(R,'MODEL',model)
    monkeypatch.setattr(R,'MODEL_SHA',R.sha(model))
    monkeypatch.setattr(R,'MODEL_BYTES',model.stat().st_size)
    monkeypatch.setattr(R,'BENCH',benchmark)
    monkeypatch.setattr(R,'BENCH_SHA',R.sha(benchmark))
    monkeypatch.setattr(R,'ORACLE',oracle)
    monkeypatch.setattr(R,'ORACLE_SHA',R.sha(oracle))
    monkeypatch.setattr(R,'GATEWAY_DIR',parser)
    monkeypatch.setattr(R,'GATEWAY_SOURCE_SHA',R.sha(gateway))
    monkeypatch.setattr(R,'GATEWAY_CONFIG_SOURCE_SHA',R.sha(conf))
    monkeypatch.setattr(R,'SERVER',binary)
    monkeypatch.setattr(R,'CONFIG',service_cfg)
    # Use preflight without requiring a real model to be loaded or tested.
    return root,model,benchmark,oracle


def test_exact_real_model_and_loopback_source_contract():
    assert R.PORT==14921 and R.HOST=='127.0.0.1'
    assert R.MODEL_SHA=='d02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12'
    assert R.MODEL_BYTES==1929901056
    assert R.BENCH_SHA=='ed10fb53850003a463d9a01f99c06ea4daf17ddbfb55910cdb3e59522534e947'
    assert R.ORACLE_SHA=='929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e'
    assert R.URL=='http://127.0.0.1:14921'
    assert R.MODEL_ALIAS!='qwen3-vl-8b-hires'


def test_oneuse_preflight_fails_closed_when_original_source_changes(tmp_path,monkeypatch):
    root,model,script,oracle=fixture(tmp_path,monkeypatch)
    class FrozenSocket:
        def __enter__(self):return self
        def __exit__(self,*a):return None
        def settimeout(self,_):pass
        def connect_ex(self,_):return 1
    monkeypatch.setattr(R.socket,'socket',lambda:FrozenSocket())
    # /proc/meminfo is read-only; real test host must have >8GiB MemAvailable.
    p=R.preflight();assert p['oracle_cases']==35 and p['turn_completion_cases']==17 and p['interruption_cases']==18
    assert p['source_sha256']==R.sha(script)
    script.write_bytes(script.read_bytes()+b'CHANGED')
    with pytest.raises(RuntimeError,match='bytes changed'):
        R.preflight()


def test_preexisting_claim_or_result_denies_second_model_start(tmp_path,monkeypatch):
    root,*_=fixture(tmp_path,monkeypatch)
    marker=root/'attempt.json';R.exclusive_json(marker,{'model_run_began':True})
    with pytest.raises(RuntimeError,match='Existing/uncertain'):
        R.preflight()
    with pytest.raises(FileExistsError):
        R.exclusive_json(marker,{'corrupted':True})
    assert json.loads(marker.read_text())=={'model_run_began':True}
    assert marker.stat().st_mode & stat.S_IRWXG==0


def test_missing_or_symlinked_weight_refused(tmp_path,monkeypatch):
    root,model,*_=fixture(tmp_path,monkeypatch)
    model.unlink()
    with pytest.raises(RuntimeError,match='Missing/mismatched pinned model'):
        R.preflight()
    target=tmp_path/'elsewhere';target.write_bytes(b'TEST MODEL SOURCE BYTES ONLY')
    model.symlink_to(target)
    with pytest.raises(RuntimeError,match='Missing/mismatched pinned model'):
        R.preflight()


def test_wrong_model_oracle_distribution_never_admitted(tmp_path,monkeypatch):
    root,model,script,oracle=fixture(tmp_path,monkeypatch)
    original=json.loads(oracle.read_text())
    original['cases'][0]['family']='interruption'
    oracle.write_text(json.dumps(original))
    monkeypatch.setattr(R,'ORACLE_SHA',R.sha(oracle))
    with pytest.raises(RuntimeError,match='family counts'):
        R.preflight()


def test_live_model_port_already_bound_refuses_unsafe_start(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    class ExistingEndpoint:
        def __enter__(self):return self
        def __exit__(self,*a):return None
        def settimeout(self,_):pass
        def connect_ex(self,_):return 0
    monkeypatch.setattr(R.socket,'socket',lambda:ExistingEndpoint())
    with pytest.raises(RuntimeError,match='already occupied'):
        R.preflight()


def test_missing_explicit_broker_lease_and_optin_denies_model_start(tmp_path,monkeypatch):
    root,*_=fixture(tmp_path,monkeypatch)
    class EmptyPort:
        def __enter__(self):return self
        def __exit__(self,*a):return None
        def settimeout(self,_):pass
        def connect_ex(self,_):return 1
    monkeypatch.setattr(R.socket,'socket',lambda:EmptyPort())
    monkeypatch.delenv(R.OPTIN,raising=False)
    with pytest.raises(RuntimeError,match='opt-in'):
        R.run(R.argparse.Namespace(run=True,broker_lease_id='rb_valid'))
    assert not (root/'attempt.json').exists()
    monkeypatch.setenv(R.OPTIN,'1')
    with pytest.raises(RuntimeError,match='lease ID'):
        R.run(R.argparse.Namespace(run=True,broker_lease_id=''))
    assert not (root/'attempt.json').exists()


def test_binary_source_uses_only_owned_children_not_system_service_control():
    data=SCRIPT.read_text()
    assert 'subprocess.Popen' in data
    assert '.terminate()' in data
    assert 'server.poll()' in data
    assert '127.0.0.1' in data
    assert 'voice-classifier-35-case.json' in data
    assert 'systemctl restart' not in data
    assert 'killall' not in data
    assert 'SWAAG_ISOLATED_THOR_CLASSIFIER_NOACTION_W67' in data
    assert 'no_production_configuration_or_service_change' in data
