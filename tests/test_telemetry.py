from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from swaag.telemetry import OperationalTelemetry
from swaag.types import (
    BudgetComponentReport,
    BudgetReport,
    CompletionResult,
)


def _collect_metrics(reader: InMemoryMetricReader) -> dict[str, object]:
    payload = reader.get_metrics_data()
    return {
        metric.name: metric
        for resource_metrics in payload.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def _telemetry_fixture():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    telemetry = OperationalTelemetry(
        tracer=tracer_provider.get_tracer("swaag-test"),
        meter=meter_provider.get_meter("swaag-test"),
    )
    return telemetry, exporter, metric_reader, tracer_provider, meter_provider


def test_genai_spans_metrics_and_context_accounting_follow_otel_conventions() -> None:
    telemetry, exporter, metric_reader, tracer_provider, meter_provider = (
        _telemetry_fixture()
    )
    report = BudgetReport(
        context_limit=2_000,
        input_tokens=1_200,
        reserved_response_tokens=256,
        safety_margin_tokens=32,
        required_tokens=1_488,
        non_context_tokens=40,
        fits=True,
        exact=True,
        breakdown=[
            BudgetComponentReport(
                name="task",
                category="task",
                tokens=500,
                exact=True,
                include_in_context=True,
                optional=False,
            )
        ],
    )
    completion = CompletionResult(
        text='{"action":"final"}',
        raw_request={"prompt": "private input"},
        raw_response={"content": "private output"},
        prompt_tokens=1_190,
        completion_tokens=18,
        finish_reason="stop",
        first_token_seconds=0.25,
    )

    with telemetry.agent_invocation(
        session_id="session-1",
        run_id="run-1",
        model_name="local-model",
    ):
        telemetry.record_context_compilation(
            call_kind="agent_action",
            context_limit_source="llama_cpp_props",
            report=report,
        )
        with telemetry.model_call(
            session_id="session-1",
            run_id="run-1",
            call_id="model-call-1",
            call_kind="agent_action",
            operation_name="text_completion",
            provider_name="llama.cpp",
            model_name="local-model",
            base_url="http://127.0.0.1:14829",
            max_tokens=256,
            cache_mode="disabled",
        ) as operation:
            operation.record_model_completion(completion, budget_report=report)
        with telemetry.tool_execution(
            session_id="session-1",
            run_id="run-1",
            call_id="tool-call-1",
            tool_name="read_file",
        ):
            pass

    tracer_provider.force_flush()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    agent = spans["invoke_agent swaag"]
    model = spans["text_completion local-model"]
    tool = spans["execute_tool read_file"]

    assert model.parent is not None and model.parent.span_id == agent.context.span_id
    assert tool.parent is not None and tool.parent.span_id == agent.context.span_id
    assert model.kind.name == "CLIENT"
    assert model.attributes["gen_ai.operation.name"] == "text_completion"
    assert model.attributes["gen_ai.provider.name"] == "llama.cpp"
    assert model.attributes["gen_ai.usage.input_tokens"] == 1_190
    assert model.attributes["gen_ai.usage.output_tokens"] == 18
    assert model.attributes["server.port"] == 14_829
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.call.id"] == "tool-call-1"
    assert all(
        key not in model.attributes
        for key in ("gen_ai.input.messages", "gen_ai.output.messages")
    )

    metrics = _collect_metrics(metric_reader)
    assert {
        "gen_ai.invoke_agent.duration",
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.execute_tool.duration",
        "swaag.context.compilation",
        "swaag.context.token.usage",
    }.issubset(metrics)
    token_points = metrics["gen_ai.client.token.usage"].data.data_points
    assert {
        point.attributes["gen_ai.token.type"]: point.sum for point in token_points
    } == {"input": 1_190, "output": 18}
    context_points = metrics["swaag.context.token.usage"].data.data_points
    assert any(
        point.attributes["swaag.context.token.type"] == "context_limit"
        and point.sum == 2_000
        for point in context_points
    )

    meter_provider.shutdown()
    tracer_provider.shutdown()


def test_operation_errors_set_span_status_and_low_cardinality_error_type() -> None:
    telemetry, exporter, metric_reader, tracer_provider, meter_provider = (
        _telemetry_fixture()
    )

    with pytest.raises(TimeoutError):
        with telemetry.tool_execution(
            session_id="session-2",
            run_id="run-2",
            call_id="tool-call-2",
            tool_name="shell_command",
        ):
            raise TimeoutError("tool timed out")

    tracer_provider.force_flush()
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "TimeoutError"
    duration = _collect_metrics(metric_reader)["gen_ai.execute_tool.duration"]
    assert duration.data.data_points[0].attributes["error.type"] == "TimeoutError"

    meter_provider.shutdown()
    tracer_provider.shutdown()
