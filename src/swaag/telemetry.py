from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from contextvars import ContextVar
from types import TracebackType
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit

from opentelemetry import context, metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

from swaag.types import BudgetReport, CompletionResult


_INSTRUMENTATION_NAME = "swaag.runtime"
_INSTRUMENTATION_VERSION = "0.1.0"
_TRACE_CONTEXT_FIELDS = frozenset({"traceparent", "tracestate"})
_TRACE_CONTEXT_MAX_FIELD_LENGTH = 512
_TRACE_CONTEXT_PROPAGATOR = TraceContextTextMapPropagator()
_ACTIVE_WORKER_ID: ContextVar[str] = ContextVar(
    "swaag_active_worker_id", default=""
)


def trace_context_carrier(values: Mapping[str, Any] | None) -> dict[str, str]:
    """Return only bounded W3C trace fields from an untrusted carrier."""
    if not isinstance(values, Mapping):
        return {}
    carrier: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip().casefold()
        if name not in _TRACE_CONTEXT_FIELDS or not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if value and len(value) <= _TRACE_CONTEXT_MAX_FIELD_LENGTH:
            carrier[name] = value
    return carrier


def inject_trace_context(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Inject the active W3C context while preserving unrelated headers."""
    carrier = dict(headers or {})
    _TRACE_CONTEXT_PROPAGATOR.inject(carrier=carrier)
    return carrier


def extract_trace_context(values: Mapping[str, Any] | None) -> context.Context:
    """Extract a valid remote context; malformed fields leave current context intact."""
    return _TRACE_CONTEXT_PROPAGATOR.extract(
        carrier=trace_context_carrier(values),
        context=context.get_current(),
    )


@contextmanager
def attached_trace_context(
    values: Mapping[str, Any] | None,
) -> Iterator[None]:
    token = context.attach(extract_trace_context(values))
    try:
        yield
    finally:
        context.detach(token)


def record_http_response_status(status_code: int) -> None:
    """Annotate the active server span at the response-writing boundary."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    status = int(status_code)
    span.set_attribute("http.response.status_code", status)
    if status >= 500:
        span.set_attribute("error.type", str(status))
        span.set_status(Status(StatusCode.ERROR))


def record_protocol_correlation(
    *,
    protocol: str,
    request_id: str = "",
    context_id: str = "",
    worker_id: str = "",
    session_id: str = "",
) -> None:
    """Add bounded mechanical adapter identifiers to the active protocol span."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    values = {
        "swaag.protocol.name": protocol,
        "swaag.protocol.request.id": request_id,
        "swaag.protocol.context.id": context_id,
        "swaag.worker.id": worker_id,
        "gen_ai.conversation.id": session_id,
    }
    for name, value in values.items():
        text = str(value).strip()
        if not text:
            continue
        if len(text) <= 256:
            span.set_attribute(name, text)
            continue
        span.set_attribute(name + ".sha256", hashlib.sha256(text.encode()).hexdigest())
        span.set_attribute(name + ".length", len(text))


@contextmanager
def worker_telemetry_context(worker_id: str) -> Iterator[None]:
    token = _ACTIVE_WORKER_ID.set(str(worker_id))
    try:
        yield
    finally:
        _ACTIVE_WORKER_ID.reset(token)


def _active_worker_attributes() -> dict[str, str]:
    worker_id = _ACTIVE_WORKER_ID.get().strip()
    return {"swaag.worker.id": worker_id} if worker_id else {}


def _http_route(path: str) -> str:
    if path in {
        "/.well-known/agent-card.json",
        "/a2a/v1",
        "/ag-ui",
        "/mcp",
        "/a2a/rest/message:send",
        "/a2a/rest/message:stream",
        "/a2a/rest/tasks",
    }:
        return path
    prefix = "/a2a/rest/tasks/"
    if not path.startswith(prefix):
        return ""
    task_segment = path.removeprefix(prefix)
    if not task_segment or "/" in task_segment:
        return ""
    if task_segment.endswith(":cancel"):
        return prefix + "{id}:cancel"
    if task_segment.endswith(":subscribe"):
        return prefix + "{id}:subscribe"
    return prefix + "{id}"


def _error_type(exc: BaseException) -> str:
    if exc.__class__.__name__ in {
        "ModelCallPreempted",
        "RunCancellationRequested",
    }:
        return "cancelled"
    return exc.__class__.__name__


class TelemetryOperation:
    """One active OTel operation without semantic input/output capture."""

    def __init__(
        self,
        *,
        span: trace.Span,
        duration_histogram: metrics.Histogram,
        metric_attributes: dict[str, Any],
        token_histogram: metrics.Histogram | None = None,
    ):
        self.span = span
        self._duration_histogram = duration_histogram
        self._metric_attributes = metric_attributes
        self._token_histogram = token_histogram
        self._started = time.monotonic()
        self._context_token: object | None = None
        self._finished = False
        self._recorded_error: tuple[str, str] | None = None
        self._retry_count = 0
        self._preemption_count = 0

    def __enter__(self) -> TelemetryOperation:
        self._context_token = context.attach(trace.set_span_in_context(self.span))
        return self

    def record_model_completion(
        self,
        completion: CompletionResult,
        *,
        budget_report: BudgetReport,
    ) -> None:
        input_tokens = completion.prompt_tokens
        if input_tokens is None and budget_report.exact:
            input_tokens = budget_report.input_tokens
        output_tokens = completion.completion_tokens
        if input_tokens is not None:
            self.span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
            if self._token_histogram is not None:
                self._token_histogram.record(
                    int(input_tokens),
                    {**self._metric_attributes, "gen_ai.token.type": "input"},
                )
        if output_tokens is not None:
            self.span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
            if self._token_histogram is not None:
                self._token_histogram.record(
                    int(output_tokens),
                    {**self._metric_attributes, "gen_ai.token.type": "output"},
                )
        if completion.finish_reason:
            self.span.set_attribute(
                "gen_ai.response.finish_reasons", [completion.finish_reason]
            )
        if completion.first_token_seconds is not None:
            self.span.set_attribute(
                "swaag.model.first_token_seconds",
                float(completion.first_token_seconds),
            )

    def record_error(self, error_type: str, description: str = "") -> None:
        self._recorded_error = (str(error_type), str(description))

    def record_retry(self) -> None:
        self._retry_count += 1

    def record_preemption(self) -> None:
        self._preemption_count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        if self._finished:
            return False
        attributes = dict(self._metric_attributes)
        if exc is not None or self._recorded_error is not None:
            error_type, description = (
                (_error_type(exc), str(exc))
                if exc is not None
                else self._recorded_error or ("_OTHER", "")
            )
            attributes["error.type"] = error_type
            self.span.set_attribute("error.type", error_type)
            if exc is not None:
                self.span.record_exception(exc)
            self.span.set_status(Status(StatusCode.ERROR, description))
        if self._retry_count:
            self.span.set_attribute("swaag.model.retry_count", self._retry_count)
        if self._preemption_count:
            self.span.set_attribute(
                "swaag.model.preemption_count", self._preemption_count
            )
        self._duration_histogram.record(
            max(0.0, time.monotonic() - self._started), attributes
        )
        if self._context_token is not None:
            context.detach(self._context_token)
        self.span.end()
        self._finished = True
        return False


class OperationalTelemetry:
    """OpenTelemetry API instrumentation; configured SDK/exporters remain external."""

    def __init__(
        self,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
    ):
        self.tracer = tracer or trace.get_tracer(
            _INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION
        )
        self.meter = meter or metrics.get_meter(
            _INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION
        )
        self._agent_duration = self.meter.create_histogram(
            "gen_ai.invoke_agent.duration",
            unit="s",
            description="End-to-end duration of an in-process agent invocation.",
        )
        self._model_duration = self.meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI client operation duration.",
        )
        self._tool_duration = self.meter.create_histogram(
            "gen_ai.execute_tool.duration",
            unit="s",
            description="Duration of one agent-side tool execution.",
        )
        self._token_usage = self.meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used.",
        )
        self._context_compilations = self.meter.create_counter(
            "swaag.context.compilation",
            unit="{compilation}",
            description="Number of mechanically measured context compilations.",
        )
        self._context_tokens = self.meter.create_histogram(
            "swaag.context.token.usage",
            unit="{token}",
            description="Measured context budget values and component sizes.",
        )
        self._semantic_reductions = self.meter.create_counter(
            "swaag.context.semantic_reduction",
            unit="{reduction}",
            description="Number of model-authored semantic reduction calls.",
        )
        self._semantic_reduction_targets = self.meter.create_histogram(
            "swaag.context.semantic_reduction.target",
            unit="{token}",
            description="Requested output size for semantic reduction calls.",
        )
        self._history_compactions = self.meter.create_counter(
            "swaag.history.compaction",
            unit="{compaction}",
            description="Number of durable history compaction operations.",
        )
        self._inference_queue_depth = self.meter.create_histogram(
            "swaag.inference.queue.depth",
            unit="{request}",
            description="Backend-local inference requests waiting for admission.",
        )
        self._inference_queue_wait = self.meter.create_histogram(
            "swaag.inference.queue.wait",
            unit="s",
            description="Time a model request waits for backend admission.",
        )
        self._inference_active = self.meter.create_up_down_counter(
            "swaag.inference.active",
            unit="{request}",
            description="Model requests currently admitted to a backend.",
        )
        self._inference_slot_utilization = self.meter.create_histogram(
            "swaag.inference.backend.slot.utilization",
            unit="1",
            description="Admitted request count divided by discovered backend capacity.",
        )
        self._inference_cancellation_latency = self.meter.create_histogram(
            "swaag.inference.cancellation.latency",
            unit="s",
            description="Time from a cancellation request to terminal acknowledgement.",
        )
        self._http_server_duration = self.meter.create_histogram(
            "http.server.request.duration",
            unit="s",
            description="Duration of an inbound HTTP request.",
        )
        self._protocol_server_duration = self.meter.create_histogram(
            "swaag.protocol.server.duration",
            unit="s",
            description="Duration of an inbound non-HTTP protocol request.",
        )

    def http_server_request(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, Any] | None = None,
    ) -> TelemetryOperation:
        normalized_method = str(method).upper()
        known_method = (
            normalized_method
            if normalized_method
            in {
                "CONNECT",
                "DELETE",
                "GET",
                "HEAD",
                "OPTIONS",
                "PATCH",
                "POST",
                "PUT",
                "TRACE",
            }
            else "_OTHER"
        )
        target_path = urlsplit(str(path)).path
        route = _http_route(target_path)
        protocol_name = (
            "ag_ui"
            if target_path == "/ag-ui"
            else "mcp"
            if target_path == "/mcp"
            else "a2a"
            if target_path == "/a2a/v1" or target_path.startswith("/a2a/rest/")
            else "discovery"
            if target_path == "/.well-known/agent-card.json"
            else "http"
        )
        attributes: dict[str, Any] = {
            "http.request.method": known_method,
            "url.path": target_path,
            "url.scheme": "http",
            "swaag.protocol.name": protocol_name,
        }
        if normalized_method != known_method:
            attributes["http.request.method_original"] = normalized_method
        if route:
            attributes["http.route"] = route
        parent = extract_trace_context(headers)
        return TelemetryOperation(
            span=self.tracer.start_span(
                f"{known_method if known_method != '_OTHER' else 'HTTP'}"
                + (f" {route}" if route else ""),
                context=parent,
                kind=SpanKind.SERVER,
                attributes=attributes,
            ),
            duration_histogram=self._http_server_duration,
            metric_attributes={
                "http.request.method": known_method,
                **({"http.route": route} if route else {}),
            },
        )

    def protocol_server_request(
        self,
        *,
        protocol: str,
        operation: str,
        carrier: Mapping[str, Any] | None = None,
    ) -> TelemetryOperation:
        protocol_name = str(protocol)
        operation_name = str(operation) or "unknown"
        attributes = {
            "swaag.protocol.name": protocol_name,
            "swaag.protocol.operation": operation_name,
        }
        return TelemetryOperation(
            span=self.tracer.start_span(
                f"{protocol_name} {operation_name}",
                context=extract_trace_context(carrier),
                kind=SpanKind.SERVER,
                attributes=attributes,
            ),
            duration_histogram=self._protocol_server_duration,
            metric_attributes=attributes,
        )

    def agent_invocation(
        self,
        *,
        session_id: str,
        run_id: str,
        model_name: str,
    ) -> TelemetryOperation:
        span_attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "swaag",
            "gen_ai.conversation.id": session_id,
            "gen_ai.output.type": "text",
            "swaag.run.id": run_id,
            **_active_worker_attributes(),
        }
        if model_name:
            span_attributes["gen_ai.request.model"] = model_name
        metric_attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "swaag",
        }
        return TelemetryOperation(
            span=self.tracer.start_span(
                "invoke_agent swaag",
                kind=SpanKind.INTERNAL,
                attributes=span_attributes,
            ),
            duration_histogram=self._agent_duration,
            metric_attributes=metric_attributes,
        )

    def model_call(
        self,
        *,
        session_id: str,
        run_id: str,
        call_id: str,
        call_kind: str,
        operation_name: str,
        provider_name: str,
        model_name: str,
        base_url: str,
        max_tokens: int,
        cache_mode: str,
    ) -> TelemetryOperation:
        endpoint = urlsplit(base_url)
        span_attributes: dict[str, Any] = {
            "gen_ai.operation.name": operation_name,
            "gen_ai.provider.name": provider_name,
            "gen_ai.request.max_tokens": int(max_tokens),
            "gen_ai.output.type": "json",
            "gen_ai.conversation.id": session_id,
            "swaag.run.id": run_id,
            "swaag.model.call.id": call_id,
            "swaag.model.call_kind": call_kind,
            "swaag.model.cache_mode": cache_mode,
            **_active_worker_attributes(),
        }
        if model_name:
            span_attributes["gen_ai.request.model"] = model_name
        if endpoint.hostname:
            span_attributes["server.address"] = endpoint.hostname
        if endpoint.port is not None:
            span_attributes["server.port"] = endpoint.port
        metric_attributes = {
            "gen_ai.operation.name": operation_name,
            "gen_ai.provider.name": provider_name,
            "swaag.model.call_kind": call_kind,
            "swaag.model.cache_mode": cache_mode,
        }
        if model_name:
            metric_attributes["gen_ai.request.model"] = model_name
        return TelemetryOperation(
            span=self.tracer.start_span(
                f"{operation_name} {model_name}" if model_name else operation_name,
                kind=SpanKind.CLIENT,
                attributes=span_attributes,
            ),
            duration_histogram=self._model_duration,
            metric_attributes=metric_attributes,
            token_histogram=self._token_usage,
        )

    def tool_execution(
        self,
        *,
        session_id: str,
        run_id: str,
        call_id: str,
        tool_name: str,
    ) -> TelemetryOperation:
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": call_id,
            "gen_ai.agent.name": "swaag",
            "gen_ai.conversation.id": session_id,
            "swaag.run.id": run_id,
            **_active_worker_attributes(),
        }
        metric_attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": "function",
            "gen_ai.agent.name": "swaag",
        }
        return TelemetryOperation(
            span=self.tracer.start_span(
                f"execute_tool {tool_name}",
                kind=SpanKind.INTERNAL,
                attributes=attributes,
            ),
            duration_histogram=self._tool_duration,
            metric_attributes=metric_attributes,
        )

    def record_context_compilation(
        self,
        *,
        call_kind: str,
        context_limit_source: str,
        report: BudgetReport,
    ) -> None:
        attributes = {
            "swaag.context.call_kind": call_kind,
            "swaag.context.limit_source": context_limit_source,
            "swaag.context.exact": report.exact,
            "swaag.context.fits": report.fits,
        }
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(
                "swaag.context.compiled",
                {
                    **attributes,
                    "swaag.context.context_limit": report.context_limit,
                    "swaag.context.input_tokens": report.input_tokens,
                    "swaag.context.reserved_output_tokens": report.reserved_response_tokens,
                    "swaag.context.safety_margin_tokens": report.safety_margin_tokens,
                    "swaag.context.required_tokens": report.required_tokens,
                    "swaag.context.overflow_tokens": max(
                        0, report.required_tokens - report.context_limit
                    ),
                },
            )
        self._context_compilations.add(1, attributes)
        values = {
            "context_limit": report.context_limit,
            "input": report.input_tokens,
            "reserved_output": report.reserved_response_tokens,
            "safety_margin": report.safety_margin_tokens,
            "required": report.required_tokens,
            "overflow": max(0, report.required_tokens - report.context_limit),
        }
        for token_type, value in values.items():
            self._context_tokens.record(
                int(value),
                {**attributes, "swaag.context.token.type": token_type},
            )
        for component in report.breakdown:
            self._context_tokens.record(
                int(component.tokens),
                {
                    **attributes,
                    "swaag.context.token.type": "component",
                    "swaag.context.component.category": component.category,
                    "swaag.context.component.in_context": component.include_in_context,
                    "swaag.context.component.optional": component.optional,
                },
            )

    def record_semantic_reduction(
        self,
        *,
        call_kind: str,
        target_tokens: int,
        hierarchical_depth: int,
    ) -> None:
        attributes = {
            "swaag.context.call_kind": str(call_kind),
            "swaag.context.hierarchical": int(hierarchical_depth) > 0,
        }
        self._semantic_reductions.add(1, attributes)
        self._semantic_reduction_targets.record(
            max(0, int(target_tokens)), attributes
        )
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(
                "swaag.context.semantic_reduction",
                {
                    **attributes,
                    "swaag.context.target_tokens": max(0, int(target_tokens)),
                    "swaag.context.hierarchical_depth": max(
                        0, int(hierarchical_depth)
                    ),
                },
            )

    def record_history_compaction(
        self,
        *,
        source_message_count: int,
        hierarchical: bool,
    ) -> None:
        attributes = {
            "swaag.history.hierarchical": bool(hierarchical),
        }
        self._history_compactions.add(1, attributes)
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(
                "swaag.history.compacted",
                {
                    **attributes,
                    "swaag.history.source_message_count": max(
                        0, int(source_message_count)
                    ),
                },
            )

    def record_inference_queued(
        self,
        *,
        call_kind: str,
        source: str,
        priority: int,
        queue_depth: int,
    ) -> None:
        attributes = {
            "swaag.model.call_kind": str(call_kind),
            "swaag.inference.source": str(source),
            "swaag.inference.priority": int(priority),
        }
        self._inference_queue_depth.record(max(0, int(queue_depth)), attributes)
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(
                "swaag.inference.queued",
                {**attributes, "swaag.inference.queue_depth": max(0, int(queue_depth))},
            )

    def record_inference_started(
        self,
        *,
        call_kind: str,
        source: str,
        priority: int,
        queue_wait_seconds: float,
        active_count: int,
        backend_capacity: int,
    ) -> None:
        attributes = {
            "swaag.model.call_kind": str(call_kind),
            "swaag.inference.source": str(source),
            "swaag.inference.priority": int(priority),
        }
        self._inference_queue_wait.record(
            max(0.0, float(queue_wait_seconds)), attributes
        )
        self._inference_active.add(1, attributes)
        self._inference_slot_utilization.record(
            max(0.0, float(active_count) / max(1, int(backend_capacity))),
            attributes,
        )
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(
                "swaag.inference.started",
                {
                    **attributes,
                    "swaag.inference.queue_wait_seconds": max(
                        0.0, float(queue_wait_seconds)
                    ),
                    "swaag.inference.active_count": max(0, int(active_count)),
                    "swaag.inference.backend_capacity": max(
                        1, int(backend_capacity)
                    ),
                },
            )

    def record_inference_released(
        self,
        *,
        call_kind: str,
        source: str,
        priority: int,
        status: str,
        cancellation_latency_seconds: float | None = None,
    ) -> None:
        active_attributes = {
            "swaag.model.call_kind": str(call_kind),
            "swaag.inference.source": str(source),
            "swaag.inference.priority": int(priority),
        }
        attributes = {
            **active_attributes,
            "swaag.inference.status": str(status),
        }
        self._inference_active.add(-1, active_attributes)
        if cancellation_latency_seconds is not None:
            self._inference_cancellation_latency.record(
                max(0.0, float(cancellation_latency_seconds)), attributes
            )
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event("swaag.inference.released", attributes)
