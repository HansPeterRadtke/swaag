"""Small delegation uses the real orchestration API without inventing results."""
from types import SimpleNamespace
import pytest
from swaag.tools.orchestration import OrchestrationDelegateTool
from swaag.tools.base import ToolValidationError
from swaag.schema_portability import assert_portable_json_schema


def _args():
    return dict(plan_objective='Finish the explicit user request', worker_objective='Compute the requested result',
                model_key='default', priority=1, finish_criteria='Verified answer',
                abort_criteria='Unrecoverable error', reporting_mode='completion_only')


def test_compact_delegation_applies_one_complete_durable_graph():
    class API:
        def __init__(self): self.calls=[]
        def execute(self, operation, payload):
            self.calls.append((operation,payload))
            return {'started_worker_ids':['worker_123'], 'plan':{'status':'running'}}
    tool=OrchestrationDelegateTool()
    api=API()
    original=_args()
    valid=tool.validate(original)
    result=tool.execute(valid, SimpleNamespace(runtime_capabilities={'orchestration':api}))
    assert len(api.calls)==1
    op,payload=api.calls[0]
    assert op=='plan.apply'
    graph=payload['plan_spec']
    assert graph['start'] is True and len(graph['nodes'])==1 and not graph['edges']
    assert graph['nodes'][0]['model_key']=='default'
    assert graph['nodes'][0]['objective']==original['worker_objective']
    assert graph['objective']==original['plan_objective']
    assert result.output['started_worker_ids']==['worker_123']
    assert original==_args()
    assert_portable_json_schema(tool.input_schema)


@pytest.mark.parametrize('change',[
    {'plan_objective':''},{'worker_objective':None},{'model_key':42},
    {'priority':0},{'priority':True},{'reporting_mode':'invented'},
    {'abort_criteria':[]},{'unknown':'field'},
])
def test_invalid_delegate_cannot_dispatch(change):
    values=_args()|change
    with pytest.raises(ToolValidationError):OrchestrationDelegateTool().validate(values)


def test_delegate_missing_capability_fails_closed():
    with pytest.raises(RuntimeError,match='No orchestration'):
        OrchestrationDelegateTool().execute(OrchestrationDelegateTool().validate(_args()),
                                            SimpleNamespace(runtime_capabilities={}))
