"""Fail-closed one-use voice runtime admission tests, no model or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/demos/claim_fresh_voice_noaction_session.py'
SPEC=importlib.util.spec_from_file_location('claim',SCRIPT)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_schema_refuses_shared_or_malformed_directory_names():
    assert M.CONFIG_PATTERN.fullmatch('/data/var/swaag-benchmarks/swaag-quality4/noaction-config.toml')
    assert M.CONFIG_PATTERN.fullmatch('/data/var/swaag-benchmarks/../other/noaction-config.toml') is None
    assert M.CONFIG_PATTERN.fullmatch('/data/infra/secrets/noaction-config.toml') is None
    assert M.SESSION_ROOT_PATTERN.fullmatch('/data/var/swaag-benchmarks/swaag-quality4/sessions')


def test_missing_or_wrong_hash_is_rejected_before_read(tmp_path):
    with pytest.raises(ValueError):
        M.validate_and_claim(tmp_path/'bad.toml','short')
    with pytest.raises(ValueError):
        M.validate_and_claim(tmp_path/'bad.toml','a'*64)


def test_matching_private_layout_requires_a_fresh_session_root(monkeypatch,tmp_path):
    """Use a private pytest root by replacing ONLY strict path grammar."""
    import re
    root=tmp_path/'swaag-test';root.mkdir()
    p=root/'noaction-config.toml';sessions=root/'sessions';sessions.mkdir()
    src='[tools]\nenabled=[]\nstaged_discovery=false\nallow_stateful_tools=false\nallow_side_effect_tools=false\n[sessions]\nroot="'+str(sessions)+'"\n'
    p.write_text(src)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(M,'CONFIG_PATTERN',re.compile(re.escape(str(p))))
    monkeypatch.setattr(M,'SESSION_ROOT_PATTERN',re.compile(re.escape(str(sessions))))
    a=M.validate_and_claim(p,digest)
    assert a['admitted'] is True
    assert a['prior_session_directories']==[]
    assert (sessions/M.CLAIM_NAME).stat().st_mode & 0o777 == 0o600
    assert json.loads((sessions/M.CLAIM_NAME).read_text())['admitted']
    with pytest.raises(FileExistsError):
        M.validate_and_claim(p,digest)


def test_existing_session_fails_closed_and_retains_claim(monkeypatch,tmp_path):
    import re
    root=tmp_path/'swaag-test';root.mkdir()
    p=root/'noaction-config.toml';sessions=root/'sessions';sessions.mkdir()
    (sessions/'session_existing').mkdir()
    p.write_text('[tools]\nenabled=[]\nstaged_discovery=false\nallow_stateful_tools=false\nallow_side_effect_tools=false\n[sessions]\nroot="'+str(sessions)+'"\n')
    monkeypatch.setattr(M,'CONFIG_PATTERN',re.compile(re.escape(str(p))))
    monkeypatch.setattr(M,'SESSION_ROOT_PATTERN',re.compile(re.escape(str(sessions))))
    with pytest.raises(ValueError,match='Existing source session'):
        M.validate_and_claim(p,hashlib.sha256(p.read_bytes()).hexdigest())
    data=json.loads((sessions/M.CLAIM_NAME).read_text())
    assert data['admitted'] is False
    assert data['prior_session_directories']==['session_existing']
    with pytest.raises(FileExistsError):
        M.validate_and_claim(p,hashlib.sha256(p.read_bytes()).hexdigest())


@pytest.mark.parametrize('bad_policy',[
  'enabled=["calculator"]','allow_side_effect_tools=true',
  'allow_stateful_tools=true','staged_discovery=true'])
def test_forbidden_tools_fail_before_claim(monkeypatch,tmp_path,bad_policy):
    import re
    root=tmp_path/'swaag-test';root.mkdir()
    p=root/'noaction-config.toml';sessions=root/'sessions';sessions.mkdir()
    t='enabled=[]\nstaged_discovery=false\nallow_stateful_tools=false\nallow_side_effect_tools=false'
    key=bad_policy.split('=')[0]
    t=t.replace(key+'='+('[]' if key=='enabled' else 'false'),bad_policy)
    p.write_text('[tools]\n'+t+'\n[sessions]\nroot="'+str(sessions)+'"\n')
    monkeypatch.setattr(M,'CONFIG_PATTERN',re.compile(re.escape(str(p))))
    monkeypatch.setattr(M,'SESSION_ROOT_PATTERN',re.compile(re.escape(str(sessions))))
    with pytest.raises(ValueError,match='unapproved system tool'):
        M.validate_and_claim(p,hashlib.sha256(p.read_bytes()).hexdigest())
    assert not (sessions/M.CLAIM_NAME).exists()
