"""Offline one-use classifier-runner admission tests, no model or provider calls."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import hashlib
import stat
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/run_pinned_thor_qwen3_4b_text_classifier_once.py'
SPEC=importlib.util.spec_from_file_location('pinned_qwen3_4b_instruct_text_only_oneshot',SCRIPT)
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
    assert R.PORT==14922 and R.HOST=='127.0.0.1'
    assert R.MODEL_SHA=='2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e'
    assert R.MODEL_BYTES==2497280736
    assert R.BENCH_SHA=='ed10fb53850003a463d9a01f99c06ea4daf17ddbfb55910cdb3e59522534e947'
    assert R.ORACLE_SHA=='929f4fd01e21a12a2535db824034699fbe6359d9ecf872d978449f291cc9a26e'
    assert R.URL=='http://127.0.0.1:14922'
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
    assert 'SWAAG_ISOLATED_THOR_QWEN3_4B_TEXT_CLASSIFIER_W67' in data
    assert 'no_production_configuration_or_service_change' in data


# Model tests below are pure fixture-based broker authorization checks: no
# real lease request, GPU startup, model input, or provider effect is made.

def _broker(lease='rb_'+'a'*32, *, state='active', owner=None,
            gpu_gib=4, cpu_millicores=2000, expires_at=4000.0,
            snapshot_status='ok', snapshot_age=5.0):
    return {'machines':{'thor':{'snapshot_status':snapshot_status,
            'snapshot_age_seconds':snapshot_age,
            'gpu_memory_headroom_after_leases_bytes':0,
            'capacity':{'status':'ok','cuda_device_free_bytes':10*1024**3,
                        'configured_safety_reserve_bytes':4*1024**3},
            'requests':[{'request_id':lease,'state':state,
               'owner':owner if owner is not None else R.BROKER_OWNER,
               'gpu_bytes':int(gpu_gib*1024**3),
               'cpu_millicores':cpu_millicores,
               'expires_at':expires_at}]}}}


def test_actual_owned_granted_4g_two_core_lease_is_accepted_even_after_reserving_headroom():
    lease='rb_'+'a'*32
    data=R.verify_broker_status(_broker(),lease,now_epoch=2000.0)
    assert data['actual_broker_grant_verified'] is True
    assert data['host']=='thor' and data['owner']==R.BROKER_OWNER
    assert data['gpu_bytes']==4*1024**3
    assert data['cpu_millicores']==2000
    assert data['seconds_until_expiry']==2000


@pytest.mark.parametrize('changed,keyword',[
    ({'state':'denied'},'not active'),
    ({'state':'queued'},'not active'),
    ({'owner':'different-project'},'not active'),
    ({'gpu_gib':3},'budget'),
    ({'cpu_millicores':1000},'budget'),
    ({'expires_at':2200.0},'expiry'),
    ({'expires_at':2750.0},'expiry'),
    ({'snapshot_status':'stale'},'snapshot'),
    ({'snapshot_age':125.0},'snapshot'),
])
def test_denied_underreserved_expired_or_stale_thor_broker_disallows_real_model(changed,keyword):
    with pytest.raises(RuntimeError,match=keyword):
        R.verify_broker_status(_broker(**changed),'rb_'+'a'*32,now_epoch=2000)


def test_random_string_and_missing_or_duplicate_broker_lease_never_authorizes():
    with pytest.raises(RuntimeError,match='identifier'):
        R.verify_broker_status(_broker(),'rb_guess',now_epoch=2000)
    with pytest.raises(RuntimeError,match='unique actual active'):
        R.verify_broker_status(_broker(),'rb_'+'b'*32,now_epoch=2000)
    duplicate=_broker();duplicate['machines']['thor']['requests'].append(
        duplicate['machines']['thor']['requests'][0].copy())
    with pytest.raises(RuntimeError,match='unique actual active'):
        R.verify_broker_status(duplicate,'rb_'+'a'*32,now_epoch=2000)
    with pytest.raises(RuntimeError,match='host snapshot'):
        R.verify_broker_status({},'rb_'+'a'*32,now_epoch=2000)


def test_status_fetch_uses_readonly_strict_ssh_auth_and_validates_the_actual_remote_data(monkeypatch):
    capture=[]
    def mocked_ssh(argv,**kw):
        capture.append((argv,kw))
        return R.subprocess.CompletedProcess(argv,0,
            stdout=json.dumps(_broker(expires_at=R.time.time()+1000.0)),stderr='')
    monkeypatch.setattr(R.subprocess,'run',mocked_ssh)
    data=R.validate_real_broker_lease('rb_'+'a'*32)
    assert data['actual_broker_grant_verified'] is True
    assert len(capture)==1
    args,options=capture[0]
    assert args[0]=='ssh' and args[-2:]==['raspi','/data/bin/resource-broker status']
    assert 'BatchMode=yes' in args and 'StrictHostKeyChecking=yes' in args
    assert options['capture_output'] and options['check']
    assert options['timeout']<=9


def test_ssh_unreachable_does_not_synthesize_a_lease(monkeypatch):
    def cannot_reach(*args,**kwargs):
        raise R.subprocess.TimeoutExpired(args[0],timeout=9)
    monkeypatch.setattr(R.subprocess,'run',cannot_reach)
    with pytest.raises(RuntimeError,match='Cannot independently verify'):
        R.validate_real_broker_lease('rb_'+'a'*32)


def test_explicit_run_with_no_actual_active_lease_denied_without_attempt_or_child(tmp_path,monkeypatch):
    root,_,_,_=fixture(tmp_path,monkeypatch)
    class EmptyPort:
        def __enter__(self):return self
        def __exit__(self,*a):return None
        def settimeout(self,_):pass
        def connect_ex(self,_):return 1
    monkeypatch.setattr(R.socket,'socket',lambda:EmptyPort())
    monkeypatch.setenv(R.OPTIN,'1')
    def denied(lease_id):
        return R.verify_broker_status(_broker(state='denied'),lease_id,now_epoch=2000)
    monkeypatch.setattr(R,'validate_real_broker_lease',denied)
    with pytest.raises(RuntimeError,match='not active'):
        R.run(R.argparse.Namespace(run=True,broker_lease_id='rb_'+'a'*32))
    assert not (root/'attempt.json').exists()
    assert not (root/'isolated-model.log').exists()
    assert not (root/'result.json').exists()



def test_external_unmanaged_GPU_consumption_after_original_lease_blocks_start():
    # The central broker might have granted 4GiB when 10GiB CUDA was free.
    # Another unmanaged job may have consumed 5GiB before our model starts.
    # The reservation still appears active but is NOT safe to exercise.
    data=_broker()
    data['machines']['thor']['capacity']['cuda_device_free_bytes']=5*1024**3
    with pytest.raises(RuntimeError,match='Actual CUDA allocation headroom disappeared'):
        R.verify_broker_status(data,'rb_'+'a'*32,now_epoch=2000)


def test_reduced_safety_reserve_cannot_be_used_to_force_gpu_admission():
    data=_broker()
    data['machines']['thor']['capacity']['configured_safety_reserve_bytes']=1024**3
    with pytest.raises(RuntimeError,match='reserve was reduced'):
        R.verify_broker_status(data,'rb_'+'a'*32,now_epoch=2000)


def test_missing_cuda_allocator_free_bytes_rejected_before_gpu_model():
    data=_broker()
    data['machines']['thor']['capacity']['cuda_device_free_bytes']=None
    with pytest.raises(RuntimeError,match='Actual CUDA allocation headroom disappeared'):
        R.verify_broker_status(data,'rb_'+'a'*32,now_epoch=2000)
