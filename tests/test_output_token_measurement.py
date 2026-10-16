from __future__ import annotations

from dataclasses import replace
import json
import pytest

from swaag.model import LlamaCppClient, ModelClientError
from swaag.model_cache import RecordReplayModelClient
from swaag.runtime import AgentRuntime
from swaag.tokens import ConservativeEstimator
from tests.test_tool_result_projection import _CharacterCountProjectionClient


@pytest.mark.parametrize("native_available", [True, False])
def test_selector_uses_native_output_count_when_fragment_estimate_is_low(make_config, native_available):
    class NativeOutputClient(_CharacterCountProjectionClient):
        def count_text(self, text):
            return ConservativeEstimator().count_text(text)

        def tokenize_output(self, text):
            if not native_available:
                raise ModelClientError("Provider has no output tokenizer")
            return len(text)

        def send_completion(self, payload, **kwargs):
            result = super().send_completion(payload, **kwargs)
            if len(result.text) > payload['n_predict']:
                return replace(result, text=result.text[:payload['n_predict']],
                    completion_tokens=payload['n_predict'], finish_reason='length')
            return result

    source = '\n'.join(f'FACT: required value {i:03d}.' for i in range(64))
    client = NativeOutputClient(['FACT:'])
    runtime = AgentRuntime(make_config(model__context_limit=16000), model_client=client)
    state = runtime.create_or_load_session()
    selected, semantic_ok = runtime._select_tool_result_verbatim_fragments(
        state, original_request='Keep every exact fact.', tool_name='reader', source_text=source)
    assert semantic_ok is native_available
    assert ''.join(selected) == source
    assert len(client.requests) == (1 if native_available else 2)


@pytest.mark.parametrize('payload,expected', [({'count': 137},137), ({'tokens':[3,4]},2), ({'count':True},None), ({'count':-1},None)])
def test_native_output_count_is_unwrapped_validated_and_closed(make_config, monkeypatch, payload, expected):
    config = make_config(model__provider_name='vllm', model__base_url='http://model:14834/v1',
                         model__completion_endpoint='/chat/completions')
    client = LlamaCppClient(config)
    calls = []
    class Response:
        closed = False
        def raise_for_status(self): pass
        def json(self): return payload
        def close(self): self.closed = True
    response = Response()
    def post(url, **kwargs):
        calls.append((url, kwargs['json']))
        return response
    monkeypatch.setattr('swaag.model.requests.post', post)
    text = '{"ids":["L00002C0001"]}'
    if expected is None:
        with pytest.raises(ModelClientError): client.tokenize_output(text)
    else:
        assert client.tokenize_output(text) == expected
    assert calls == [('http://model:14834/tokenize', {'model':config.model.profile_name,
        'prompt':text, 'add_special_tokens':False})]
    assert response.closed


def test_output_count_recording_replays_without_live_delegate(tmp_path):
    from tests.test_model_cache_exactness import Delegate
    class OutputDelegate(Delegate):
        def tokenize_output(self, text): return 137
    cassette = tmp_path / 'output-count.json'
    recorder = RecordReplayModelClient(cassette_path=cassette, mode='record',
        delegate=OutputDelegate(), request_metadata={'task':'output-count'})
    assert recorder.tokenize_output('exact output') == 137
    replay = RecordReplayModelClient(cassette_path=cassette, mode='replay',
        delegate=None, request_metadata=dict(recorder.request_metadata))
    assert replay.tokenize_output('exact output') == 137


def test_missing_output_tokenizer_is_not_recorded_as_an_exact_word_count(tmp_path):
    from types import SimpleNamespace
    recorder = RecordReplayModelClient(cassette_path=tmp_path / 'missing.json',
        mode='record', delegate=SimpleNamespace(), request_metadata={'task':'missing'})
    with pytest.raises(RuntimeError, match='Exact output tokenizer unavailable'):
        recorder.tokenize_output('nonword-tokenized output')


def test_unverified_output_adapter_does_not_make_an_http_request(make_config, monkeypatch):
    client = LlamaCppClient(make_config(model__provider_name='openai',
        model__base_url='http://model/v1', model__completion_endpoint='/chat/completions'))
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected HTTP')
    monkeypatch.setattr('swaag.model.requests.post', forbidden)
    with pytest.raises(ModelClientError, match='unavailable'):
        client.tokenize_output('opaque generated text')
