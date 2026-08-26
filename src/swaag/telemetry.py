from __future__ import annotations

import time
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from opentelemetry import context, metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from swaag.types import BudgetReport, CompletionResult


_INSTRUMENTATION_NAME = "swaag.runtime"
_INSTRUMENTATION_VERSION = "0.1.0"


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
