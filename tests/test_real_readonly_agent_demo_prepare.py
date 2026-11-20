"""Generator tests do not themselves constitute an agent demonstration."""
from pathlib import Path
import importlib.util
import json
import os
import tomllib

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
 'readonly_demo_prepare',ROOT/'scripts/demos/prepare_real_readonly_agent_demo.py')
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_private_demo_config_reproduces_exact_source_inputs(tmp_path):
    dst=tmp_path/'new-test'
    m=module.prepare(ROOT,dst,'http://10.8.0.7:14835')
    assert m['no_model_called_by_prepare']
    assert set(m['real_sources_sha256'])==set(module.INPUTS)
    cfg=tomllib.loads((dst/'demo-readonly-config.toml').read_text())
    assert cfg['tools']['allow_side_effect_tools'] is False
    assert cfg['tools']['read_roots']==[str(ROOT)]
    assert cfg['model']['profile_name']=='qwen38-27b-q8-full-262k'
    assert cfg['tools']['enabled']==['list_files','read_file','calculator']
    assert cfg['sessions']['root']==str(dst/'sessions')
    assert (dst/'demo-task-v1.txt').read_bytes()==(
        ROOT/'scripts/demos/source_audit_prompt_v1.txt').read_bytes()
    assert (dst/'manifest.json').stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        module.prepare(ROOT,dst,'http://10.8.0.7:14835')


@pytest.mark.parametrize('url',['http://example.com:14835','https://10.8.0.7:14835',
                                  'http://0.0.0.0:14835','http://127.0.0.1'])
def test_generator_refuses_unapproved_model_destinations(url,tmp_path):
    with pytest.raises(ValueError):
        module.prepare(ROOT,tmp_path/'x',url)
    assert not (tmp_path/'x').exists()
