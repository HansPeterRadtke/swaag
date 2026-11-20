"""Pure preflight policy tests; not a real agent or LLM execution test."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/demos/check_swaag_agent_runtime.py'
spec=importlib.util.spec_from_file_location('swaag_demo_preflight',P)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def available():
    return {key:True for key in module.REQUIRED_MODULES}


def test_supported_python311_sqlite3513_dependencies_is_ready():
    result=module.classify_env(python=(3,11),sqlite=(3,51,3),
        grammar_ok=True,available_modules=available())
    assert result['status']=='ready_for_deterministic_tests'
    assert result['problems']==[]


@pytest.mark.parametrize('broken,error',[
    ({'python':(3,10)},'unsupported_python_version'),
    ({'grammar_ok':False},'syntax_incompatible_with_claimed_python311'),
    ({'sqlite':(3,50,4)},'sqlite_runtime_older_than_required_3_51_3'),
    ({'available_modules':{'pytest':True}},'missing_dependency:opentelemetry.context'),
])
def test_failures_classified_without_claiming_test_failure(broken,error):
    args={'python':(3,11),'sqlite':(3,51,3),
          'grammar_ok':True,'available_modules':available()}
    args.update(broken)
    result=module.classify_env(**args)
    assert result['status']=='environment_blocked'
    assert error in result['problems']
    assert result['cannot_infer_live_agent_success']
