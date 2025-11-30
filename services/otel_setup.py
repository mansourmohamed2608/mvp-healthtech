"""
Minimal, optional OpenTelemetry bootstrap for Python services.
Safe to import even when opentelemetry packages are not installed.
"""
import os
import logging

logger = logging.getLogger("otel")


def init_otel(service_name: str):
    """Initialize OTEL tracing if OTEL_ENABLED/ENABLE_OTEL=true and deps are available."""
    enabled = os.getenv("OTEL_ENABLED", os.getenv("ENABLE_OTEL", "")).lower() in ("true", "1")
    if not enabled:
        return
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
        logger.info("OTEL tracing initialized", extra={"service": service_name})
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OTEL tracing not initialized (missing deps/config)",
            extra={"service": service_name, "error": str(exc)},
        )
