from __future__ import annotations

import pytest

from swaag.model import LlamaCppClient, ModelClientError


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
