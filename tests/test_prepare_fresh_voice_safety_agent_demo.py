"""Source-only admission tests. No model or external effect is ever performed."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tomllib
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/demos/prepare_fresh_voice_safety_agent_demo.py'
SPEC=importlib.util.spec_from_file_location('prepare_new_readonly_demo',PATH)
P=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_fixed_distinct_task_reads_exact_existing_two_voice_safety_sources():
    assert len(P.FILES)==2
    assert 'voice_classifier' in ' '.join(P.FILES)
    assert 'release_server' in ' '.join(P.FILES)
    assert not (ROOT/P.BROKEN_PATH).exists()
    assert P.MODEL_PROFILE=='qwen3-coder-30b-a3b-instruct-q4-k-m'
    assert P.MODEL_BASE=='http://10.8.0.7:14833'


def test_preparing_is_pure_source_read_plus_fresh_private_artifacts(tmp_path):
    dest=tmp_path/'private'
    result=P.prepare(ROOT,dest)
    assert result['no_model_request_preparation_only']
    assert result['no_side_effect_tools']
    assert result['original_source_hash_and_oracle']
    assert [r['top_level_test_count'] for r in result['original_source_hash_and_oracle'].values()]==[4,7]
    assert result['prior_historical_inputs_and_effects_not_repeated']
    conf=tomllib.loads((dest/'readonly-firstpass-config.toml').read_text())
    assert conf['tools']=={
       'enabled':['read_file','list_files','calculator'],
       'read_roots':[str(ROOT)],
       'allow_stateful_tools':True,
       'allow_side_effect_tools':False,
       'staged_discovery':False}
    assert conf['model']['max_retries']==0
    assert conf['runtime']['max_validation_recovery_cycles']>=1
    assert conf['runtime']['completion_evaluation_enabled'] is False
    assert conf['model']['base_url']=='http://10.8.0.7:14833'
    assert not list((dest/'sessions').iterdir())
    assert not list((dest/'agent-data').iterdir())
    assert not (dest/'operations.jsonl').exists()
    assert (dest/'fixed-new-task.txt').is_file()
    assert (dest/'manifest.json').is_file()
    assert (dest/'readonly-firstpass-config.toml').stat().st_mode & 0o777 == 0o600


def test_real_core_config_loads_generated_file_without_inference(tmp_path):
    from swaag.config import load_config
    dest=tmp_path/'test-safe'
    P.prepare(ROOT,dest)
    cfg=load_config([str(dest/'readonly-firstpass-config.toml')])
    assert cfg.model.profile_name==P.MODEL_PROFILE
    assert cfg.runtime.max_validation_recovery_cycles==1
    assert cfg.tools.allow_side_effect_tools is False
    assert cfg.tools.enabled==list(P.TOOLS)
    assert not list((dest/'sessions').iterdir())


def test_repeat_or_sneaky_overlap_preparation_is_refused(tmp_path):
    new=tmp_path/'new';P.prepare(ROOT,new)
    with pytest.raises(ValueError):P.prepare(ROOT,new)
    with pytest.raises(ValueError):P.prepare(ROOT,ROOT/'under-source-new')
    with pytest.raises(ValueError):P.prepare(ROOT,ROOT)


def test_private_destination_symlink_denied(tmp_path):
    real=tmp_path/'actual';real.mkdir()
    link=tmp_path/'symlink';link.symlink_to(real)
    with pytest.raises(ValueError):P.prepare(ROOT,link)
    assert not list(real.iterdir())


def test_missing_or_symlinked_original_files_refused_before_writes(tmp_path):
    repo=tmp_path/'isolated-repo';repo.mkdir()
    dest=tmp_path/'no-effects'
    with pytest.raises(ValueError,match='Missing'):
        P.prepare(repo,dest)
    assert not dest.exists()
    for name in P.FILES:
        place=repo/name;place.parent.mkdir(parents=True,exist_ok=True)
        place.symlink_to(ROOT/name)
    with pytest.raises(ValueError,match='symlinked'):
        P.prepare(repo,dest)
    assert not dest.exists()


def test_obsolete_path_existing_denied(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    for name in P.FILES:
        target=repo/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    old=repo/P.BROKEN_PATH;old.write_text('bad obsolete file present')
    with pytest.raises(ValueError,match='unexpectedly exists'):
        P.prepare(repo,tmp_path/'unused')


def test_manifest_freezes_source_hashes_and_user_prompt_sha(tmp_path):
    dest=tmp_path/'owned';P.prepare(ROOT,dest)
    m=json.loads((dest/'manifest.json').read_text())
    assert m['fixed_task_sha256']==P.sha((dest/'fixed-new-task.txt').read_bytes())
    assert m['configuration_sha256']==P.sha((dest/'readonly-firstpass-config.toml').read_bytes())
    for name,item in m['original_source_hash_and_oracle'].items():
        assert item['sha256']==P.sha((ROOT/name).read_bytes())
        assert item['required_exact_name'] in (ROOT/name).read_text()


def test_exact_original_pinned_source_bytes_are_required_before_any_model_call(tmp_path):
    assert P.PINNED_ORACLE_SOURCE_REVISION=='83a3ff051374409dafaa6491c9c0afa467c7b63f'
    assert set(P.EXPECTED_SOURCE_SHA256)==set(P.FILES)
    for name,want in P.EXPECTED_SOURCE_SHA256.items():
        assert P.sha((ROOT/name).read_bytes())==want
    # Byte drift must be refused before writing a new private output folder.
    repo=tmp_path/'different';repo.mkdir()
    for name in P.FILES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    original=repo/next(iter(P.FILES))
    original.write_bytes(original.read_bytes()+b'\n# unrelated changed byte\n')
    with pytest.raises(ValueError,match='historically pinned'):
        P.prepare(repo,tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_manifest_records_actual_runtime_and_original_source_blob_revision(tmp_path):
    output=tmp_path/'private';manifest=P.prepare(ROOT,output)
    assert manifest['original_source_blob_revision']==P.PINNED_ORACLE_SOURCE_REVISION
    assert manifest['source_revision']==P.subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
