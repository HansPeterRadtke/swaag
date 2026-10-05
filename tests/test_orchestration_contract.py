from pathlib import Path


def test_latest_orchestration_authority_is_tracked_as_p0():
    todo = Path("docs/TODO.md").read_text()
    required = [
        "user-facing orchestrator",
        "durable dependency graph",
        "equal round-robin sharing",
        "dependency-gated runnable sets",
        "configurable worker priorities/weights",
        "model assignment",
        "durable orchestrator notifications",
        "Benchmark orchestration itself",
    ]
    missing = [item for item in required if item not in todo]
    assert not missing, f"untracked September 14 orchestration requirements: {missing}"


def test_docs_define_one_canonical_human_semantic_layer_and_demote_worker_apis():
    readme = Path("README.md").read_text()
    manual = Path("docs/manual.md").read_text()
    task_api = Path("docs/task-api.md").read_text()
    voice = Path("docs/voice-and-communication.md").read_text()
    design = Path("docs/design-principles.md").read_text()

    assert "one canonical semantic human/user conversation layer: the persistent orchestrator" in readme
    assert "Foreground `swaag ask` and `swaag chat` use that orchestrator directly in-process" in readme
    assert "developer/integration APIs" in readme

    assert "single canonical human/user semantic layer is the persistent orchestrator" in manual
    assert "Task API create/start/message/cancel" in manual
    assert "MUST NOT implement its own rule" in manual

    assert "Developer/integration API, not the ordinary user conversation port." in task_api
    assert "canonical human/user semantic layer is the persistent orchestrator" in task_api

    assert "exactly one official SWAAG entry point for ordinary voice/chat user messages" in voice
    assert "must not send ordinary speech to Task API" in voice

    assert "canonical external human-message operation is orchestrator.message" in design
    assert "developer/integration controls, not peer user-facing entrances" in design
