"""Exact native llama.cpp chat-token counting: never invent provider accuracy."""
from __future__ import annotations

import pytest

from swaag.grammar import yes_no_contract
from swaag.model import LlamaCppClient


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False
    def raise_for_status(self): pass
    def json(self): return self.payload
    def close(self): self.closed = True


def _mock_private_llama(monkeypatch, *, malformed=False, altered_template=False,
                        changed_props=False):
    recorded = []
    get_count = 0
    def get(url, **kwargs):
        nonlocal get_count
        assert url == "http://127.0.0.1:14833/props"
        get_count += 1
        return _Response({
            "chat_template": "Qwen template revision " + ("new" if changed_props and get_count > 1 else "original"),
            "chat_template_caps": {"tools": True},
            "model_alias": "test-model",
            "model_path": "/nonexistent/test.gguf",
            "default_generation_settings": {"n_ctx": 16384},
        })
    def post(url, *, json, **kwargs):
        recorded.append((url, json))
        if url == "http://127.0.0.1:14833/apply-template":
            assert json["chat_template_kwargs"] == {"enable_thinking": False}
            rendered = ("".join(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                                for msg in json["messages"])
                        + "<|im_start|>assistant\n")
            if altered_template:
                rendered = rendered.replace("Current user text exactly.", "Changed user text without authority.")
            return _Response({"prompt": rendered})
        assert url == "http://127.0.0.1:14833/tokenize"
        assert json.get("add_special") is False
        assert "<|im_start|>assistant\n" in json["content"]
        return _Response({"tokens": [1, 2, -1]} if malformed else {"tokens": list(range(31))})
    monkeypatch.setattr("swaag.model.requests.get", get)
    monkeypatch.setattr("swaag.model.requests.post", post)
    monkeypatch.setattr(LlamaCppClient, "_remote_model", lambda self: {"id": "test-model"})
    return recorded


def _client(make_config):
    return LlamaCppClient(make_config(
        model__base_url="http://127.0.0.1:14833",
        model__completion_endpoint="/chat/completions",
        model__provider_name="llama.cpp",
        model__profile_name="test-model",
        model__chat_template_kwargs={"enable_thinking": False},
    ))


def _history():
    return [
        {"role": "system", "content": "System must stay authoritative."},
        {"role": "user", "content": "First user says do not publish."},
        {"role": "assistant", "content": "Assistant verified earlier evidence."},
        {"role": "user", "content": "Current user text exactly."},
    ]


def test_private_llama_chat_uses_exact_server_template_and_native_tokenizer(make_config, monkeypatch):
    tracked = _mock_private_llama(monkeypatch)
    client = _client(make_config)
    history = _history()
    assembly = client.render_chat_prompt(history)
    assert assembly["input_token_strategy"] == "llama_cpp_native_chat_template_tokenize"
    assert assembly["input_token_count"] == "31"
    assert assembly["tokenizer_context_limit"] == "16384"
    assert assembly["prompt_serialization_exact"] == "false"  # Opaque accounting envelope stays labeled honestly.
    counted = client.count_text(assembly["prompt"])
    assert counted.exact and counted.tokens == 31
    assert [url.rsplit("/", 1)[-1] for url, _ in tracked] == ["apply-template", "tokenize"]
    assert tracked[0][1]["messages"] == history
    request = client.build_completion_request("synthetic accountability envelope",
        max_tokens=64, contract=yes_no_contract(), messages=history)
    assert request["messages"] == history
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("failure", ["bad_tokens", "altered_message", "model_template_changed"])
def test_private_llama_fails_conservatively_when_exact_serialization_unavailable(
    make_config, monkeypatch, failure,
):
    tracked = _mock_private_llama(monkeypatch,
        malformed=failure == "bad_tokens",
        altered_template=failure == "altered_message",
        changed_props=failure == "model_template_changed",
    )
    client = _client(make_config)
    result = client.render_chat_prompt(_history())
    assert result["input_token_strategy"] == "conservative_estimator_required"
    assert result["input_token_count"] == ""
    assert not client.count_text(result["prompt"]).exact
    assert tracked[0][0].endswith("/apply-template")
    if failure == "altered_message":
        assert len(tracked) == 1, "Do not call tokenizer for altered user content"


@pytest.mark.parametrize("url,provider,root", [
    ("http://127.0.0.1:14833", "llama.cpp", "http://127.0.0.1:14833"),
    ("http://10.8.0.7:14833/v1", "llama.cpp", "http://10.8.0.7:14833"),
    ("http://[::1]:14833/v1", "llama_cpp", "http://[::1]:14833"),
    ("https://inference.example/v1", "llama.cpp", None),
    ("http://127.0.0.1:14833", "vllm", None),
    ("http://10.8.0.7:14833/v2", "llama.cpp", None),
])
def test_only_known_private_llama_chat_transport_is_eligible(make_config, url, provider, root):
    client = LlamaCppClient(make_config(model__base_url=url,
        model__provider_name=provider, model__completion_endpoint="/chat/completions"))
    assert client._private_llama_chat_root() == root
