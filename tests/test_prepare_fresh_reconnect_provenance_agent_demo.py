"""Pure source-preparation tests; not a model-inference demonstration."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/demos/prepare_fresh_reconnect_provenance_agent_demo.py'
SPEC=importlib.util.spec_from_file_location('prepare_wave66_real_agent_source',SCRIPT)
P=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_oracle_has_genuinely_new_original_source_files_and_checksum():
    assert P.ORIGINAL_SOURCE_REVISION=='c2a9e55008c8872abed290ac4e34e08c473083dc'
    assert P.BAD_PATH=='tests/test_replay_connection_legacy.py'
    assert len(P.ORIGINAL_SOURCES)==2
    assert not (ROOT/P.BAD_PATH).exists()
    for rel,(digest,name,count) in P.ORIGINAL_SOURCES.items():
        raw=(ROOT/rel).read_bytes()
        assert P.sha(raw)==digest
        tests=[n.name for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
        assert len(tests)==count
        assert name in tests
    assert tuple(v[2] for v in P.ORIGINAL_SOURCES.values())==(2,3)


def test_source_prepare_copies_fixed_inputs_without_model_requests(tmp_path):
    dest=tmp_path/'private'
    report=P.prepare(ROOT,dest)
    assert report['source_revision']==P.subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert report['original_source_blob_revision']==P.ORIGINAL_SOURCE_REVISION
    assert report['controlled_obsolete_missing_file']==P.BAD_PATH
    assert report['model_calls_or_tools_executed_during_preparation']==0
    assert report['distinct_from_all_previous_swaag_demonstration_source_inputs'] is True
    conf=tomllib.loads((dest/'read-only-config.toml').read_text())
    assert conf['tools']['enabled']==P.TOOLS
    assert conf['tools']['allow_side_effect_tools'] is False
    assert conf['model']['max_retries']==0
    assert conf['runtime']['completion_evaluation_enabled'] is False
    assert conf['runtime']['max_validation_recovery_cycles']==1
    assert not list((dest/'sessions').iterdir())
    assert not list((dest/'agent-data').iterdir())
    assert not (dest/'operations.jsonl').exists()
    assert (dest/'read-only-config.toml').stat().st_mode & 0o777 == 0o600
    assert P.sha((dest/'read-only-config.toml').read_bytes())==report['configuration_sha256']
    assert P.sha((dest/'fixed-new-reconnect-task.txt').read_bytes())==report['fixed_task_sha256']


def test_real_swaag_config_parses_prepared_private_source(tmp_path):
    from swaag.config import load_config
    dest=tmp_path/'private'
    P.prepare(ROOT,dest)
    c=load_config([str(dest/'read-only-config.toml')])
    assert c.tools.enabled==P.TOOLS
    assert c.tools.allow_side_effect_tools is False
    assert c.model.profile_name==P.MODEL_ALIAS
    assert c.model.max_retries==0
    assert c.runtime.max_validation_recovery_cycles==1


def test_frozen_task_explicit_one_obsolete_read_then_real_listing(tmp_path):
    d=tmp_path/'private'
    P.prepare(ROOT,d)
    task=(d/'fixed-new-reconnect-task.txt').read_text()
    assert 'exactly ONE' in task
    assert P.BAD_PATH in task
    assert task.index('read_file')<task.index('list_files')
    assert 'NEVER repeat' in task
    assert 'test_reconnect_replays_exact_answer_without_materializing_giant_journal' in task
    assert 'test_tool_message_is_separate_accounting_component_with_source_event' in task
    assert 'calculator' in task
    assert 'Do not add another user turn' in task


def test_existing_private_directory_and_symlink_refused(tmp_path):
    d=tmp_path/'one_use'
    P.prepare(ROOT,d)
    with pytest.raises(ValueError):P.prepare(ROOT,d)
    link=tmp_path/'linked'
    link.symlink_to(d)
    with pytest.raises(ValueError):P.prepare(ROOT,link)
    assert not list((d/'sessions').iterdir())


def test_source_repo_overlap_refused_even_when_new(tmp_path):
    with pytest.raises(ValueError):P.prepare(ROOT,ROOT/'new_unpublished_session')
    with pytest.raises(ValueError):P.prepare(ROOT,ROOT)


def test_wrong_source_sha_or_symlink_input_denied_before_any_file_write(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    target=tmp_path/'target'
    for rel in P.ORIGINAL_SOURCES:
        a=root/rel;a.parent.mkdir(parents=True,exist_ok=True)
        a.write_bytes((ROOT/rel).read_bytes())
    one=root/next(iter(P.ORIGINAL_SOURCES))
    one.write_bytes(one.read_bytes()+b'CORRUPT')
    with pytest.raises(ValueError,match='SHA mismatch'):
        P.prepare(root,target)
    assert not target.exists()
    one.unlink();one.symlink_to(ROOT/next(iter(P.ORIGINAL_SOURCES)))
    with pytest.raises(ValueError,match='symlinked'):
        P.prepare(root,target)
    assert not target.exists()


def test_obsolete_file_accidentally_exists_refused(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    for rel in P.ORIGINAL_SOURCES:
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/rel).read_bytes())
    (root/P.BAD_PATH).write_text('Do not accidentally turn fake obsolete into real source')
    with pytest.raises(ValueError,match='unexpectedly exists'):
        P.prepare(root,tmp_path/'private')
    assert not (tmp_path/'private').exists()
