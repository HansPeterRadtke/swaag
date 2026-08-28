from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import os
from threading import Lock

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass(slots=True)
class OtlpExportRuntime:
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    _closed: bool = False
    _lock: Lock = field(default_factory=Lock)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.tracer_provider.force_flush()
        self.meter_provider.force_flush()
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def configure_otlp_export_from_environment() -> OtlpExportRuntime | None:
    """Configure the CLI host process when a standard OTLP endpoint is set."""
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().casefold() == "true":
        return None
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    traces_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""
    ).strip()
    metrics_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", ""
    ).strip()
    if not endpoint and not traces_endpoint and not metrics_endpoint:
        return None

    interval_millis: int | None = None
    if endpoint or metrics_endpoint:
        interval_text = os.environ.get(
            "OTEL_METRIC_EXPORT_INTERVAL", "60000"
        ).strip()
        try:
            interval_millis = int(interval_text)
        except ValueError as exc:
            raise ValueError(
                "OTEL_METRIC_EXPORT_INTERVAL must be an integer number of milliseconds"
            ) from exc
        if interval_millis <= 0:
            raise ValueError("OTEL_METRIC_EXPORT_INTERVAL must be positive")

    service_name = os.environ.get("OTEL_SERVICE_NAME", "swaag").strip() or "swaag"
    resource = Resource.create({SERVICE_NAME: service_name})
    tracer_provider = TracerProvider(resource=resource)
    if endpoint or traces_endpoint:
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    metric_readers: list[PeriodicExportingMetricReader] = []
    if interval_millis is not None:
        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(),
                export_interval_millis=interval_millis,
            )
        )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers,
    )

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    runtime = OtlpExportRuntime(tracer_provider, meter_provider)
    atexit.register(runtime.shutdown)
    return runtime
