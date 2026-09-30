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
