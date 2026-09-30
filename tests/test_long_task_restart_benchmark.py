from pathlib import Path


def test_long_task_restart_benchmark_covers_restart_and_delayed_relevance():
    text = Path("src/swaag/benchmark/long_task_restart.py").read_text()
    assert "runtime2 = AgentRuntime(base)" in text
    assert "create_or_load_session(session_id)" in text
    assert "unrelated_turn_pairs" in text
    assert "_semantic_retrieval_probe(runtime2, retained)" in text
    assert "read_authoritative_messages(session_id)" in text
    assert "source_event_references" in text
    assert "ADVERSARIAL_DECOYS" in text


def test_long_task_restart_cli_is_registered():
    text = Path("src/swaag/benchmark/benchmark_runner.py").read_text()
    assert '"long-task-restart"' in text
    assert "run_long_task_restart_benchmark" in text
    assert "--unrelated-turn-pairs" in text
