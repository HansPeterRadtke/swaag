from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from swaag.config import AgentConfig, load_config
from swaag.live_runtime_profiles import get_live_runtime_recommendation
from swaag.model import LlamaCppClient
from swaag.runtime import AgentRuntime, TurnResult


pytestmark = pytest.mark.agent_test

_FAST_LIVE = get_live_runtime_recommendation("fast_live_tests")


def _live_base_url() -> str:
    return os.environ.get("SWAAG_LIVE_BASE_URL", "http://127.0.0.1:14829")


def _live_timeout() -> int:
    return int(os.environ.get("SWAAG_LIVE_TIMEOUT_SECONDS", str(_FAST_LIVE.timeout_seconds)))


def _live_connect_timeout() -> int:
    return int(os.environ.get("SWAAG_LIVE_CONNECT_TIMEOUT_SECONDS", str(_FAST_LIVE.connect_timeout_seconds)))


def _live_model_profile() -> str:
    return os.environ.get("SWAAG_LIVE_MODEL_PROFILE", _FAST_LIVE.model_profile)


def _live_structured_output_mode() -> str:
    return os.environ.get("SWAAG_LIVE_STRUCTURED_OUTPUT_MODE", _FAST_LIVE.structured_output_mode)


def _live_progress_poll_seconds() -> float:
    return float(os.environ.get("SWAAG_LIVE_PROGRESS_POLL_SECONDS", str(_FAST_LIVE.progress_poll_seconds)))


def _live_seeds() -> tuple[int, int, int]:
    raw = os.environ.get("SWAAG_LIVE_SEEDS", ",".join(str(seed) for seed in _FAST_LIVE.seeds))
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(seeds) != 3:
        raise ValueError(f"SWAAG_LIVE_SEEDS must contain exactly 3 comma-separated seeds, got: {raw!r}")
    return seeds


@pytest.fixture(scope="session")
def live_settings() -> dict[str, str | int | float]:
    if os.environ.get("SWAAG_RUN_LIVE") != "1":
        pytest.skip("Set SWAAG_RUN_LIVE=1 to run live llama.cpp tests")
    settings: dict[str, str | int | float] = {
        "base_url": _live_base_url(),
        "timeout": _live_timeout(),
        "connect_timeout": _live_connect_timeout(),
        "profile": _live_model_profile(),
        "structured_output_mode": _live_structured_output_mode(),
        "progress_poll_seconds": _live_progress_poll_seconds(),
        "seeds": list(_live_seeds()),
    }
    config = load_config(env={"SWAAG__MODEL__BASE_URL": str(settings["base_url"])})
    config.model.timeout_seconds = int(settings["timeout"])
    config.model.connect_timeout_seconds = int(settings["connect_timeout"])
    config.model.profile_name = str(settings["profile"])
    config.model.structured_output_mode = str(settings["structured_output_mode"])
    config.model.progress_poll_seconds = float(settings["progress_poll_seconds"])
    client = LlamaCppClient(config)
    try:
        health = client.health()
    except Exception as exc:  # pragma: no cover - exercised in live mode only
        pytest.fail(
            f"SWAAG_RUN_LIVE=1 but llama.cpp server is not reachable at {settings['base_url']}: {exc}"
        )
    if not isinstance(health, dict) or health.get("status") != "ok":
        pytest.fail(f"Live llama.cpp server at {settings['base_url']} returned unexpected health payload: {health!r}")
    return settings


def _make_runtime(
    tmp_path: Path,
    live_settings: dict[str, str | int | float],
    *,
    allow_side_effect_tools: bool = False,
    seed: int | None = None,
) -> tuple[AgentRuntime, object, AgentConfig]:
    config = load_config(
        env={
            "SWAAG__MODEL__BASE_URL": str(live_settings["base_url"]),
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__TOOLS__READ_ROOTS": json.dumps([str(tmp_path)]),
        }
    )
    config.model.timeout_seconds = int(live_settings["timeout"])
    config.model.connect_timeout_seconds = int(live_settings["connect_timeout"])
    config.model.profile_name = str(live_settings["profile"])
    config.model.structured_output_mode = str(live_settings["structured_output_mode"])
    config.model.progress_poll_seconds = float(live_settings["progress_poll_seconds"])
    if seed is not None:
        config.model.seed = int(seed)
    config.tools.allow_side_effect_tools = allow_side_effect_tools
    config.tools.allow_stateful_tools = True
    config.runtime.max_tool_steps = max(config.runtime.max_tool_steps, 8)
    config.runtime.max_reasoning_steps = max(config.runtime.max_reasoning_steps, 12)
    config.runtime.max_total_actions = max(config.runtime.max_total_actions, 14)
    config.runtime.tool_call_budget = max(config.runtime.tool_call_budget, 10)
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    return runtime, state, config


def _run_turn(
    tmp_path: Path,
    live_settings: dict[str, str | int | float],
    prompt: str,
    *,
    allow_side_effect_tools: bool = False,
    seed: int | None = None,
) -> tuple[AgentRuntime, object, TurnResult]:
    runtime, state, _ = _make_runtime(tmp_path, live_settings, allow_side_effect_tools=allow_side_effect_tools, seed=seed)
    turn = runtime.run_turn_in_session(state, prompt)
    return runtime, state, turn


def _run_across_seeds(
    tmp_path: Path,
    live_settings: dict[str, str | int | float],
    prompt: str,
    *,
    allow_side_effect_tools: bool = False,
    prepare_seed=None,
) -> list[tuple[int, AgentRuntime, object, TurnResult]]:
    runs: list[tuple[int, AgentRuntime, object, TurnResult]] = []
    for seed in live_settings["seeds"]:
        if prepare_seed is not None:
            prepare_seed(int(seed))
        runtime, state, turn = _run_turn(
            tmp_path,
            live_settings,
            prompt,
            allow_side_effect_tools=allow_side_effect_tools,
            seed=int(seed),
        )
        runs.append((int(seed), runtime, state, turn))
    return runs


def _event_types(runtime: AgentRuntime, session_id: str) -> list[str]:
    return [event.event_type for event in runtime.history.read_history(session_id)]


def _tool_names(runtime: AgentRuntime, session_id: str) -> list[str]:
    return [event.payload["tool_name"] for event in runtime.history.read_history(session_id) if event.event_type == "tool_called"]


def test_live_direct_response_without_tools(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        "Reply with exactly 17. Do not use any tools. Do not add any extra text.",
    )

    assert [seed for seed, *_ in runs] == list(live_settings["seeds"])
    assert all(turn.assistant_text.strip() == "17" for _seed, _runtime, _state, turn in runs)
    assert all(_tool_names(runtime, state.session_id) == [] for _seed, runtime, state, _turn in runs)


def test_live_calculator_tool_path(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        "Use the calculator tool to compute ((12345 + 6789) * 17) - 43. Reply with only the integer.",
    )

    assert all(turn.assistant_text.strip() == "325235" for _seed, _runtime, _state, turn in runs)
    assert all("calculator" in _tool_names(runtime, state.session_id) for _seed, runtime, state, _turn in runs)
    assert all("verification_passed" in _event_types(runtime, state.session_id) for _seed, runtime, state, _turn in runs)


def test_live_file_read_exact_extraction(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    facts = tmp_path / "facts.txt"
    facts.write_text("line1=ignore\nline2=ignore\nowner=carol\n", encoding="utf-8")

    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        f"Read {facts} and return exactly the full text on line 3. No extra words.",
    )

    assert all(turn.assistant_text.strip() == "owner=carol" for _seed, _runtime, _state, turn in runs)
    assert all(any(name in {"read_text", "read_file"} for name in _tool_names(runtime, state.session_id)) for _seed, runtime, state, _turn in runs)


def test_live_file_edit_exact_change(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    note = tmp_path / "note.txt"

    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        f"Edit {note} so the file content becomes exactly world followed by a newline. Reply exactly done.",
        allow_side_effect_tools=True,
        prepare_seed=lambda _seed: note.write_text("hello\n", encoding="utf-8"),
    )

    assert all(turn.assistant_text.strip() == "done" for _seed, _runtime, _state, turn in runs)
    assert note.read_text(encoding="utf-8") == "world\n"
    assert all(
        ("edit_applied" in _event_types(runtime, state.session_id) or "file_write_applied" in _event_types(runtime, state.session_id))
        for _seed, runtime, state, _turn in runs
    )


def test_live_multi_step_read_compute_write(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    numbers = tmp_path / "numbers.txt"
    result_file = tmp_path / "result.txt"

    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        (
            f"Read {numbers}. Use the calculator tool to add the two numbers. "
            f"Create {result_file} containing exactly sum=42 followed by a newline. Reply exactly written."
        ),
        allow_side_effect_tools=True,
        prepare_seed=lambda _seed: (
            numbers.write_text("19\n23\n", encoding="utf-8"),
            result_file.unlink() if result_file.exists() else None,
        ),
    )

    assert all(turn.assistant_text.strip() == "written" for _seed, _runtime, _state, turn in runs)
    assert result_file.read_text(encoding="utf-8") == "sum=42\n"
    for _seed, runtime, state, _turn in runs:
        tool_names = _tool_names(runtime, state.session_id)
        assert "calculator" in tool_names
        assert any(name in {"write_file", "edit_text"} for name in tool_names)


def test_live_run_tests_tool_path(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    module = tmp_path / "sample_math.py"
    test_file = tmp_path / "test_sample_math.py"
    module.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    test_file.write_text(
        "import unittest\n\n"
        "from sample_math import add\n\n\n"
        "class SampleMathTests(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(add(19, 23), 42)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )

    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        f"Run tests in {test_file}. If they pass, reply exactly passed.",
    )

    assert all(turn.assistant_text.strip() == "passed" for _seed, _runtime, _state, turn in runs)
    assert all("run_tests" in _tool_names(runtime, state.session_id) for _seed, runtime, state, _turn in runs)
    assert all("verification_passed" in _event_types(runtime, state.session_id) for _seed, runtime, state, _turn in runs)


def test_live_missing_file_returns_not_done(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        (
            f"Read {missing}. If the file does not exist or you cannot verify its contents, "
            "reply exactly not done. Do not guess."
        ),
        allow_side_effect_tools=True,
    )

    assert all(turn.assistant_text.strip().lower() == "not done" for _seed, _runtime, _state, turn in runs)
    assert all(
        (
            "step_failed" in _event_types(runtime, state.session_id)
            or "verification_failed" in _event_types(runtime, state.session_id)
            or "error" in _event_types(runtime, state.session_id)
        )
        for _seed, runtime, state, _turn in runs
    )


def test_live_verification_strength_rejects_unverified_claim(live_settings: dict[str, str | int | float], tmp_path: Path) -> None:
    facts = tmp_path / "facts.txt"
    facts.write_text("owner=carol\n", encoding="utf-8")

    runs = _run_across_seeds(
        tmp_path,
        live_settings,
        (
            f"Read {facts}. Only return owner=bob if the file proves it. "
            "If you cannot verify that, reply exactly not done. Do not guess."
        ),
    )

    assert all(turn.assistant_text.strip().lower() == "not done" for _seed, _runtime, _state, turn in runs)
    assert all(any(name in {"read_text", "read_file"} for name in _tool_names(runtime, state.session_id)) for _seed, runtime, state, _turn in runs)
