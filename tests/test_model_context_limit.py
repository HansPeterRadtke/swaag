from __future__ import annotations

import pytest

from swaag.grammar import yes_no_contract
from swaag.model import (
    LlamaCppClient,
    ModelClientError,
    stable_llama_server_properties,
)
from swaag.utils import sha256_text


def test_server_context_limit_comes_from_props(make_config, monkeypatch):
    config = make_config()
    client = LlamaCppClient(config)
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"default_generation_settings": {"n_ctx": 22016}}

    monkeypatch.setattr("swaag.model.requests.get", lambda *a, **k: Response())
    assert client.server_context_limit() == 22016
    assert client.context_limit_resolution() == (22016, "server_props:n_ctx")


def test_stable_server_properties_exclude_runtime_instance_noise() -> None:
    stable = {
        "model_alias": "model-a",
        "model_path": "/models/a.gguf",
        "model_ftype": "Q4_K_M",
        "default_generation_settings": {"n_ctx": 22016},
        "chat_template": "template-a",
        "build_info": "build-a",
    }
    first = stable_llama_server_properties(
        stable | {"is_sleeping": False, "media_marker": "instance-a"}
    )
    second = stable_llama_server_properties(
        stable | {"is_sleeping": True, "media_marker": "instance-b"}
    )

    assert first == second


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "22016"])
def test_server_context_limit_rejects_invalid_props(make_config, monkeypatch, value):
    client = LlamaCppClient(make_config())

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"default_generation_settings": {"n_ctx": value}}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ModelClientError, match="invalid n_ctx"):
        client.server_context_limit()


def test_server_context_limit_requires_generation_settings(make_config, monkeypatch):
    client = LlamaCppClient(make_config())

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ModelClientError, match="missing default_generation_settings"):
        client.server_context_limit()


def test_server_slot_count_comes_from_props(make_config, monkeypatch):
    client = LlamaCppClient(make_config())

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"total_slots": 3}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    assert client.server_slot_count() == 3


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "3"])
def test_server_slot_count_rejects_invalid_props(make_config, monkeypatch, value):
    client = LlamaCppClient(make_config())

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"total_slots": value}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ModelClientError, match="invalid total_slots"):
        client.server_slot_count()


def test_chat_prompt_uses_server_template_and_verifies_model_identity(
    make_config, monkeypatch
):
    client = LlamaCppClient(make_config())
    template = "template-v1"
    posts = []

    class PropsResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "chat_template": template,
                "model_path": "/models/model.gguf",
                "model_alias": "model-a",
                "build_info": "build-a",
            }

    class TemplateResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "prompt": "<s>system\nSYS</s>user\nUSR</s>assistant\n"
            }

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: PropsResponse())

    def post(url, *, json, timeout):
        posts.append((url, json, timeout))
        return TemplateResponse()

    monkeypatch.setattr("swaag.model.requests.post", post)
    rendered = client.render_chat_prompt(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ]
    )
    client.verify_prompt_protocol(rendered["prompt_protocol_sha256"])

    assert rendered["chat_template_sha256"] == sha256_text(template)
    assert len(rendered["prompt_protocol_sha256"]) == 64
    assert posts[0][0].endswith("/apply-template")
    assert posts[0][1]["messages"][1]["content"] == "USR"

    template = "template-v2"
    with pytest.raises(ModelClientError, match="changed after context compilation"):
        client.verify_prompt_protocol(rendered["prompt_protocol_sha256"])


def test_chat_prompt_rejects_model_switch_during_serialization(
    make_config, monkeypatch
):
    client = LlamaCppClient(make_config())
    calls = 0

    class PropsResponse:
        def __init__(self, version):
            self.version = version

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "chat_template": f"template-{self.version}",
                "model_path": f"/models/{self.version}.gguf",
            }

    class TemplateResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"prompt": "SYS USR"}

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return PropsResponse(calls)

    monkeypatch.setattr("swaag.model.requests.get", get)
    monkeypatch.setattr(
        "swaag.model.requests.post", lambda *args, **kwargs: TemplateResponse()
    )

    with pytest.raises(ModelClientError, match="changed while serializing"):
        client.render_chat_prompt(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USR"},
            ]
        )


def test_chat_completion_request_preserves_system_and_user_messages(
    make_config,
) -> None:
    client = LlamaCppClient(
        make_config(
            model__base_url="https://openrouter.ai/api/v1",
            model__completion_endpoint="/chat/completions",
        )
    )
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
    ]

    request = client.build_completion_request(
        "serialized prompt used for exact accounting",
        max_tokens=64,
        contract=yes_no_contract(),
        messages=messages,
    )

    assert request["messages"] == messages
    assert "stop" not in request


def test_completion_request_only_sends_explicit_model_stop_sequences(
    make_config,
) -> None:
    without_stops = LlamaCppClient(make_config(model__stop=[]))
    with_stops = LlamaCppClient(make_config(model__stop=["MODEL_STOP"]))

    assert "stop" not in without_stops.build_completion_request(
        "prompt", max_tokens=64, contract=yes_no_contract()
    )
    assert with_stops.build_completion_request(
        "prompt", max_tokens=64, contract=yes_no_contract()
    )["stop"] == ["MODEL_STOP"]
