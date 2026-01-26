import { Logger } from '@nestjs/common';

/**
 * Optional OpenTelemetry bootstrap. Safe to call even when OTEL
 * dependencies are not installed; in that case we just log a warning
 * and continue without tracing.
 */
export async function initOtel() {
  const enabled =
    (
      process.env.OTEL_ENABLED ||
      process.env.ENABLE_OTEL ||
      ''
    ).toLowerCase() === 'true' || process.env.OTEL_ENABLED === '1';
  if (!enabled) {
    return;
  }

  const logger = new Logger('OTEL');
  try {
    // Dynamic import to avoid hard dependency at runtime

    const { NodeSDK } = require('@opentelemetry/sdk-node');

    const {
      getNodeAutoInstrumentations,
    } = require('@opentelemetry/auto-instrumentations-node');

    const {
      OTLPTraceExporter,
    } = require('@opentelemetry/exporter-trace-otlp-http');

    const { Resource } = require('@opentelemetry/resources');

    const {
      SemanticResourceAttributes,
    } = require('@opentelemetry/semantic-conventions');
    const endpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
    const exporter = endpoint
      ? new OTLPTraceExporter({ url: endpoint })
      : new OTLPTraceExporter();
    const resource = new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]:
        process.env.OTEL_SERVICE_NAME || 'gateway',
    });
    const sdk = new NodeSDK({
      resource,
      instrumentations: [getNodeAutoInstrumentations()],
      traceExporter: exporter,
    });
    await sdk.start();
    logger.log('OpenTelemetry initialized (gateway)');
  } catch (err) {
    logger.warn(
      `OpenTelemetry not initialized (missing deps or config): ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}
