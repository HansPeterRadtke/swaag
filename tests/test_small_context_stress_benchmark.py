from pathlib import Path

from swaag.benchmark.small_context_stress import CASE_IDS, _find_boundary


class _WordCounterClient:
    @staticmethod
    def tokenize(text: str) -> int:
        return len(text.split()) if text.strip() else 0


def test_small_context_stress_covers_required_boundary_and_source_modes():
    assert set(CASE_IDS) == {
        "server_context_identity",
        "near_full_boundary",
        "exact_full_boundary",
        "one_token_over_boundary",
        "million_token_user_request_rejected",
        "irreducible_output_schema_rejected",
        "oversized_history_recovery",
        "oversized_tool_result_recovery",
        "oversized_attachment_recovery",
    }


def test_small_context_stress_cli_is_registered():
    text = Path("src/swaag/benchmark/benchmark_runner.py").read_text()
    assert '"small-context-stress"' in text
    assert "run_small_context_stress_benchmark" in text
    assert "--expected-context-limit" in text


def test_small_context_stress_uses_actual_server_identity_and_zero_generation_for_million_case():
    text = Path("src/swaag/benchmark/small_context_stress.py").read_text()
    assert "context_limit_resolution()" in text
    assert "actual_limit != int(expected_context_limit)" in text
    assert "1_000_000" in text
    assert "no_inference.send_calls != 0" in text
    assert "raw_tokens" in text
    assert 'row.get("projected") is True' in text
    assert 'row.get("exact_search_excerpted") is True' in text


def test_boundary_search_can_hit_exact_required_token_boundary(make_config, tmp_path):
    from swaag.benchmark.small_context_stress import _boundary_compilation

    config = make_config(
        model__context_limit=128,
        context__safety_margin_tokens=0,
    )
    client = _WordCounterClient()
    baseline = _boundary_compilation(config, client, repetitions=0, context_limit=128)
    target = baseline.report.required_tokens + 17
    repetitions, compilation = _find_boundary(
        config,
        client,
        target_required_tokens=target,
        context_limit=128,
    )
    assert repetitions == 17
    assert compilation.report.required_tokens == target


def test_small_context_stress_uses_real_history_stream_and_attachment_size_field():
    text = Path("src/swaag/benchmark/small_context_stress.py").read_text()
    assert "runtime.history.read_history(state.session_id)" in text
    assert "reference.size_bytes" in text
    assert "event_by_sequence" not in text
    assert "reference.bytes" not in text


def test_action_preparation_falls_back_to_full_fidelity_lean_before_reduction(make_config, tmp_path):
    from swaag.grammar import agent_action_contract
    from swaag.runtime import AgentRuntime
    from swaag.tokens import ExactTokenCounter
    from swaag.types import Message
    from swaag.utils import utc_now_iso

    class _Client:
        limit = 1500

        @staticmethod
        def tokenize(text: str) -> int:
            return len(text.split()) if text.strip() else 0

        def context_limit_resolution(self):
            return self.limit, "test:configured"

        @staticmethod
        def cache_identity():
            return {"status": "test"}

        @staticmethod
        def send_completion(*args, **kwargs):
            raise AssertionError("lean admission fallback must not invoke generation")

    config = make_config(
        model__context_limit=1500,
        tools__staged_discovery=False,
        runtime__completion_evaluation_enabled=False,
    )
    client = _Client()
    runtime = AgentRuntime(
        config,
        model_client=client,
        token_counter=ExactTokenCounter(client.tokenize),
    )
    state = runtime.create_or_load_session()
    request = "Explain the current state briefly."
    runtime._record_message(
        state, Message(role="user", content=request, created_at=utc_now_iso())
    )
    contract = agent_action_contract([])
    context_components = runtime._runtime_context_components(
        state, runtime._counter(state), projections={}
    )
    reports = {}
    for mode in ("standard", "lean"):
        assembly = runtime.prompts.build_agent_action_prompt(
            state.messages,
            [],
            original_request=request,
            pending_user_messages=[],
            prompt_mode=mode,
            context_components=context_components,
            capability_index=[],
            tool_result_projections={},
            validation_feedback="",
        )
        reports[mode] = runtime._compile_context(
            state, assembly, contract, minimum_output_tokens=64
        ).report
    assert reports["standard"].fits is False
    assert reports["lean"].fits is True

    prepared = runtime._prepare_action_call(
        state,
        original_request=request,
        pending_messages=[],
        tool_specs=[],
        capability_index=[],
        contract=contract,
        validation_feedback="",
        minimum_output_tokens=64,
    )
    assert prepared.prompt_mode == "lean"
    assert prepared.report.fits is True
    events = runtime.history.read_history(state.session_id)
    fallback = [e for e in events if e.event_type == "action_prompt_mode_fallback"]
    assert len(fallback) == 1
    assert fallback[0].payload["standard_budget_report"]["fits"] is False
    assert fallback[0].payload["lean_budget_report"]["fits"] is True
    assert not any(
        e.event_type in {"history_compressed", "history_reprojected"}
        for e in events
    )


def test_lean_overflow_fallback_obeys_runtime_setting(make_config):
    import pytest
    from swaag.runtime import BudgetExceededError
    from swaag.grammar import agent_action_contract
    from swaag.runtime import AgentRuntime
    from swaag.tokens import ExactTokenCounter

    class _Client:
        @staticmethod
        def tokenize(text: str) -> int:
            return len(text.split()) if text.strip() else 0

        @staticmethod
        def context_limit_resolution():
            return 1500, "test:configured"

        @staticmethod
        def cache_identity():
            return {"status": "test"}

    config = make_config(
        model__context_limit=1500,
        tools__staged_discovery=False,
        context__compact_on_overflow=False,
        runtime__lean_on_overflow=False,
    )
    runtime = AgentRuntime(
        config,
        model_client=_Client(),
        token_counter=ExactTokenCounter(_Client.tokenize),
    )
    state = runtime.create_or_load_session()
    contract = agent_action_contract([])

    with pytest.raises(BudgetExceededError):
        runtime._prepare_action_call(
            state,
            original_request="Explain the current state briefly.",
            pending_messages=[],
            tool_specs=[],
            capability_index=[],
            contract=contract,
            validation_feedback="",
            minimum_output_tokens=64,
        )

    events = runtime.history.read_history(state.session_id)
    assert not any(event.event_type == "action_prompt_mode_fallback" for event in events)
