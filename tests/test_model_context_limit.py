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
    assert request["provider"] == {"require_parameters": True}


def test_generic_chat_completion_does_not_send_openrouter_routing_fields(
    make_config,
) -> None:
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
        )
    )

    request = client.build_completion_request(
        "accounting envelope",
        max_tokens=64,
        contract=yes_no_contract(),
        messages=[{"role": "user", "content": "task"}],
    )

    assert "provider" not in request


def test_remote_context_and_identity_come_from_selected_model_metadata(
    make_config, monkeypatch
) -> None:
    requested_urls: list[str] = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "object": "list",
                "data": [
                    {
                        "id": "model-a",
                        "owned_by": "vllm",
                        "root": "/models/model-a",
                        "max_model_len": 32768,
                    },
                    {"id": "model-b", "max_model_len": 8192},
                ],
            }

    def get(url, **kwargs):
        requested_urls.append(url)
        return Response()

    monkeypatch.setattr("swaag.model.requests.get", get)
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
        )
    )

    assert client.context_limit_resolution() == (32768, "openai_models:max_model_len")
    identity = client.cache_identity()

    assert identity["status"] == "resolved"
    assert identity["model_alias"] == "model-a"
    assert identity["transport"] == "openai_chat_completions"
    assert requested_urls == [
        "https://inference.example/v1/models",
        "https://inference.example/v1/models",
    ]
    assert all("/props" not in url for url in requested_urls)
    assert client.server_slot_count() == 1


def test_remote_health_uses_standard_model_discovery_not_nonstandard_health(
    make_config, monkeypatch
) -> None:
    urls: list[str] = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "model-a", "max_model_len": 8192}]}

    def get(url, **kwargs):
        urls.append(url)
        return Response()

    monkeypatch.setattr("swaag.model.requests.get", get)
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
        )
    )

    assert client.health() == {
        "status": "ok",
        "transport": "openai_chat_completions",
        "model": "model-a",
    }
    assert urls == ["https://inference.example/v1/models"]


def test_remote_model_discovery_never_substitutes_a_different_single_model(
    make_config, monkeypatch
) -> None:
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "model-b", "max_model_len": 8192}]}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
        )
    )

    with pytest.raises(ModelClientError, match="not uniquely available"):
        client.context_limit_resolution()


def test_remote_context_uses_explicit_fallback_only_after_discovery_fails(
    make_config, monkeypatch
) -> None:
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "model-a", "owned_by": "openai"}]}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("unsupported")),
    )
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
            model__remote_context_limit_fallback=12288,
        )
    )

    assert client.context_limit_resolution() == (
        12288,
        "configured:model.remote_context_limit_fallback",
    )


def test_remote_context_never_uses_packaged_offline_fallback_implicitly(
    make_config, monkeypatch
) -> None:
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "model-a"}]}

    monkeypatch.setattr("swaag.model.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("unsupported")),
    )
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
            model__context_limit=2048,
            model__remote_context_limit_fallback=0,
        )
    )

    with pytest.raises(ModelClientError, match="exposed no context capacity"):
        client.context_limit_resolution()


def test_remote_prompt_accounting_uses_provider_message_tokenizer_when_available(
    make_config, monkeypatch
) -> None:
    requested_gets: list[str] = []
    requested_posts: list[tuple[str, dict, dict]] = []

    class ModelsResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {
                        "id": "model-a",
                        "max_model_len": 16384,
                        "root": "/models/revision-a",
                    }
                ]
            }

    class TokenizeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"count": 37, "max_model_len": 16384, "tokens": list(range(37))}

    def get(url, **kwargs):
        requested_gets.append(url)
        return ModelsResponse()

    def post(url, *, json, timeout, **kwargs):
        requested_posts.append((url, json, kwargs))
        return TokenizeResponse()

    monkeypatch.setattr("swaag.model.requests.get", get)
    monkeypatch.setattr("swaag.model.requests.post", post)
    monkeypatch.setenv("TEST_REMOTE_MODEL_TOKEN", "secret-value")
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
            model__api_key_env="TEST_REMOTE_MODEL_TOKEN",
        )
    )
    messages = [
        {"role": "system", "content": "Policy with a newline.\nKeep it."},
        {"role": "user", "content": "Task with a quote: \"exact\"."},
    ]

    rendered = client.render_chat_prompt(messages)
    client.verify_prompt_protocol(rendered["prompt_protocol_sha256"])

    assert rendered["prompt_serialization_exact"] == "false"
    assert rendered["input_token_strategy"] == "provider_tokenize_messages"
    assert rendered["input_token_count"] == "37"
    assert all(message["content"] in rendered["prompt"] for message in messages)
    assert client.tokenize(rendered["prompt"]) == 37
    with pytest.raises(ModelClientError, match="opaque OpenAI-compatible"):
        client.tokenize("an arbitrary fragment")
    assert requested_posts[0][0] == "https://inference.example/tokenize"
    assert requested_posts[0][1]["messages"] == messages
    assert requested_posts[0][2]["headers"] == {
        "Authorization": "Bearer secret-value"
    }
    assert all(url == "https://inference.example/v1/models" for url in requested_gets)


def test_remote_prompt_protocol_detects_model_metadata_drift(
    make_config, monkeypatch
) -> None:
    revision = "a"

    class ModelsResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {
                        "id": "model-a",
                        "root": f"/models/revision-{revision}",
                        "max_model_len": 8192,
                    }
                ]
            }

    monkeypatch.setattr(
        "swaag.model.requests.get", lambda *args, **kwargs: ModelsResponse()
    )
    monkeypatch.setattr(
        "swaag.model.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("unsupported")),
    )
    client = LlamaCppClient(
        make_config(
            model__base_url="https://inference.example/v1",
            model__completion_endpoint="/chat/completions",
            model__profile_name="model-a",
        )
    )
    rendered = client.render_chat_prompt(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "task"},
        ]
    )

    revision = "b"
    with pytest.raises(ModelClientError, match="changed after context compilation"):
        client.verify_prompt_protocol(rendered["prompt_protocol_sha256"])


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
