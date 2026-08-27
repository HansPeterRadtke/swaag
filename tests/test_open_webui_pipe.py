from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_pipe_module(monkeypatch):
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **values):
            for name in self.__class__.__annotations__:
                setattr(self, name, values.get(name, getattr(self.__class__, name)))

    def field(*, default, **_):
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = field
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)
    path = Path(__file__).parents[1] / "integrations" / "open_webui_pipe.py"
    spec = importlib.util.spec_from_file_location("swaag_open_webui_pipe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_open_webui_pipe_returns_durable_result_and_emits_status(
    monkeypatch,
) -> None:
    module = _load_pipe_module(monkeypatch)
    calls: list[tuple[str, dict]] = []

    class FakeClient:
        def __init__(self, *_args):
            pass

        async def request(self, operation, params):
            calls.append((operation, params))
            if operation == "open_webui.send":
                return {
                    "return": None,
                    "events": [
                        {
                            "type": "status",
                            "data": {"description": "working", "done": False},
                        }
                    ],
                    "metadata": {"worker_id": "worker-1", "status": "working"},
                    "next_sequence": 4,
                }
            if operation == "task.events.wait":
                return {"next_sequence": 7, "terminal": True, "events": []}
            if operation == "open_webui.get":
                return {
                    "return": "exact durable answer",
                    "events": [
                        {
                            "type": "status",
                            "data": {"description": "complete", "done": True},
                        }
                    ],
                    "metadata": {"worker_id": "worker-1", "status": "completed"},
                }
            raise AssertionError(operation)

    monkeypatch.setattr(module, "_JsonLineClient", FakeClient)
    emitted: list[dict] = []

    async def emit(event):
        emitted.append(event)

    pipe = module.Pipe()
    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "Do the whole task."}]},
            __metadata__={"chat_id": "chat-1", "message_id": "request-1"},
            __event_emitter__=emit,
        )
    )

    assert result == "exact durable answer"
    assert [item[0] for item in calls] == [
        "open_webui.send",
        "task.events.wait",
        "open_webui.get",
    ]
    assert calls[1][1]["after_sequence"] == 4
    assert [item["data"]["done"] for item in emitted] == [False, True]


def test_open_webui_pipe_preserves_raw_images_files_and_remote_references(
    monkeypatch, tmp_path
) -> None:
    module = _load_pipe_module(monkeypatch)
    raw_file = tmp_path / "facts.bin"
    raw_file.write_bytes(b"exact file bytes")
    message, inline, references = module._latest_user_payload(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect both inputs."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,ZXhhY3QgaW1hZ2U="
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png"},
                        },
                    ],
                }
            ]
        }
    )
    files, file_references = asyncio.run(
        module._raw_file_attachments(
            [
                {
                    "name": "facts.bin",
                    "file": {
                        "path": str(raw_file),
                        "filename": "facts.bin",
                        "meta": {"content_type": "application/octet-stream"},
                    },
                },
                {
                    "name": "remote.pdf",
                    "file": {"path": "s3://private-bucket/remote.pdf"},
                },
            ]
        )
    )

    assert message == "Inspect both inputs."
    assert inline[0]["content_base64"] == "ZXhhY3QgaW1hZ2U="
    assert references == ["https://example.test/image.png"]
    assert files[0]["content_base64"] == "ZXhhY3QgZmlsZSBieXRlcw=="
    assert file_references == ["remote.pdf: s3://private-bucket/remote.pdf"]
