from __future__ import annotations

import threading

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from swaag.model import LlamaCppClient
from swaag.runtime import AgentRuntime
from swaag.telemetry import OperationalTelemetry, trace_context_carrier
from swaag.types import (
    BudgetComponentReport,
    BudgetReport,
    CompletionResult,
)
from swaag.workers import WorkerManager


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
        telemetry.record_semantic_reduction(
            call_kind="summary",
            target_tokens=320,
            hierarchical_depth=2,
        )
        telemetry.record_history_compaction(
            source_message_count=7,
            hierarchical=True,
        )
        telemetry.record_inference_queued(
            call_kind="agent_action",
            source="worker",
            priority=0,
            queue_depth=2,
        )
        telemetry.record_inference_started(
            call_kind="agent_action",
            source="worker",
            priority=0,
            queue_wait_seconds=0.25,
            active_count=1,
            backend_capacity=2,
        )
        telemetry.record_inference_released(
            call_kind="agent_action",
            source="worker",
            priority=0,
            status="cancelled",
            cancellation_latency_seconds=0.1,
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
        "swaag.context.semantic_reduction",
        "swaag.context.semantic_reduction.target",
        "swaag.history.compaction",
        "swaag.inference.queue.depth",
        "swaag.inference.queue.wait",
        "swaag.inference.active",
        "swaag.inference.backend.slot.utilization",
        "swaag.inference.cancellation.latency",
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
    reduction_points = metrics["swaag.context.semantic_reduction"].data.data_points
    assert reduction_points[0].value == 1
    assert reduction_points[0].attributes["swaag.context.call_kind"] == "summary"
    assert reduction_points[0].attributes["swaag.context.hierarchical"] is True
    target_points = metrics[
        "swaag.context.semantic_reduction.target"
    ].data.data_points
    assert target_points[0].sum == 320
    compaction_points = metrics["swaag.history.compaction"].data.data_points
    assert compaction_points[0].value == 1
    assert compaction_points[0].attributes["swaag.history.hierarchical"] is True
    assert {
        event.name for event in agent.events
    } >= {
        "swaag.context.semantic_reduction",
        "swaag.history.compacted",
        "swaag.inference.queued",
        "swaag.inference.started",
        "swaag.inference.released",
    }
    active_points = metrics["swaag.inference.active"].data.data_points
    assert active_points[0].value == 0
    utilization = metrics[
        "swaag.inference.backend.slot.utilization"
    ].data.data_points
    assert utilization[0].sum == 0.5

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


def test_remote_trace_crosses_protocol_worker_and_model_boundaries(make_config) -> None:
    telemetry, exporter, _reader, tracer_provider, meter_provider = (
        _telemetry_fixture()
    )
    runtime = AgentRuntime(
        make_config(),
        model_client=object(),
        telemetry=telemetry,
    )
    workers = WorkerManager(runtime)
    finished = threading.Event()
    captured_headers: dict[str, str] = {}
    remote_trace_id = "1234567890abcdef1234567890abcdef"
    remote_parent_id = "1234567890abcdef"
    carrier = {
        "traceparent": f"00-{remote_trace_id}-{remote_parent_id}-01",
        "tracestate": "vendor=value",
    }

    def run_worker(_worker_id: str) -> None:
        with telemetry.agent_invocation(
            session_id="session-remote",
            run_id="run-remote",
            model_name="local-model",
        ):
            with telemetry.model_call(
                session_id="session-remote",
                run_id="run-remote",
                call_id="call-remote",
                call_kind="agent_action",
                operation_name="text_completion",
                provider_name="llama.cpp",
                model_name="local-model",
                base_url="http://127.0.0.1:14829",
                max_tokens=128,
                cache_mode="disabled",
            ):
                captured_headers.update(
                    LlamaCppClient(runtime.config)._request_headers_kwargs()[
                        "headers"
                    ]
                )
        finished.set()

    workers._run_worker = run_worker  # type: ignore[method-assign]
    with telemetry.protocol_server_request(
        protocol="swaag.jsonl",
        operation="task.create",
        carrier=carrier,
    ):
        workers._submit("worker-remote")
        assert finished.wait(timeout=5)
    workers.shutdown()
    tracer_provider.force_flush()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    protocol = spans["swaag.jsonl task.create"]
    agent = spans["invoke_agent swaag"]
    model = spans["text_completion local-model"]
    assert f"{protocol.context.trace_id:032x}" == remote_trace_id
    assert protocol.parent is not None
    assert f"{protocol.parent.span_id:016x}" == remote_parent_id
    assert agent.parent is not None and agent.parent.span_id == protocol.context.span_id
    assert model.parent is not None and model.parent.span_id == agent.context.span_id
    assert agent.attributes["swaag.worker.id"] == "worker-remote"
    assert model.attributes["swaag.worker.id"] == "worker-remote"
    propagated = captured_headers["traceparent"].split("-")
    assert propagated[1] == remote_trace_id
    assert propagated[2] == f"{model.context.span_id:016x}"
    assert captured_headers["tracestate"] == "vendor=value"

    meter_provider.shutdown()
    tracer_provider.shutdown()


def test_trace_carrier_rejects_unbounded_and_non_trace_fields() -> None:
    assert trace_context_carrier(
        {
            "TraceParent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01",
            "tracestate": "x" * 513,
            "authorization": "private",
        }
    ) == {
        "traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
    }
