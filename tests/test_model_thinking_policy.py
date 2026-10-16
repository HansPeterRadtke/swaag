from types import SimpleNamespace
import pytest
from swaag.config import load_config
from swaag.model import LlamaCppClient, ModelClientError
from swaag.types import ContractSpec


def test_thinking_policy_matches_generation_tokenization_and_identity(make_config, monkeypatch):
    config=make_config(model__base_url='http://127.0.0.1:1234/v1',model__provider_name='vllm',model__thinking_mode='disabled')
    client=LlamaCppClient(config)
    contract=ContractSpec(name='probe',mode='json_schema',json_schema={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False})
    payload=client.build_completion_request('hello',max_tokens=32,contract=contract)
    requests=[]
    def post(url,**kw):
        requests.append(kw['json']);return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'count':17,'max_model_len':8192})
    monkeypatch.setattr('swaag.model.requests.post',post)
    assert client._remote_tokenize_messages([{'role':'user','content':'hello'}])==(17,8192)
    assert requests[0]['chat_template_kwargs']==payload['chat_template_kwargs']=={'enable_thinking':False}
    monkeypatch.setattr(client.__class__,'_remote_model',lambda self:{'id':'test-model'})
    assert client.cache_identity()['thinking_mode']=='disabled'
    assert client._prompt_protocol_identity()['thinking_mode']=='disabled'


def test_unsupported_thinking_override_is_explicit(make_config):
    with pytest.raises(ValueError,match='vLLM'):
        load_config(env={'SWAAG__MODEL__THINKING_MODE':'disabled','SWAAG__MODEL__PROVIDER_NAME':'unqualified'})
    client=LlamaCppClient(make_config(model__thinking_mode='enabled',model__provider_name='unqualified'))
    with pytest.raises(ModelClientError,match='vLLM'):client._chat_template_options()
    assert LlamaCppClient(make_config())._chat_template_options()=={}


def test_conflicting_template_and_thinking_policy_are_rejected(make_config):
    with pytest.raises(ValueError, match="conflicts"):
        load_config(env={"SWAAG__MODEL__PROVIDER_NAME":"vllm",
                         "SWAAG__MODEL__THINKING_MODE":"enabled",
                         "SWAAG__MODEL__CHAT_TEMPLATE_KWARGS":'{"enable_thinking":false}'})
    client = LlamaCppClient(make_config(model__provider_name="vllm", model__thinking_mode="enabled",
                                      model__chat_template_kwargs={"enable_thinking":False}))
    with pytest.raises(ModelClientError, match="conflicts"):
        client._chat_template_options()
