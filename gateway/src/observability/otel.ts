import { Logger } from '@nestjs/common';

/**
 * Optional OpenTelemetry bootstrap. Safe to call even when OTEL
 * dependencies are not installed; in that case we just log a warning
 * and continue without tracing.
 */
export async function initOtel() {
  const enabled = (process.env.OTEL_ENABLED || process.env.ENABLE_OTEL || '').toLowerCase() === 'true' || process.env.OTEL_ENABLED === '1';
  if (!enabled) {
    return;
  }

  const logger = new Logger('OTEL');
  try {
    // Dynamic import to avoid hard dependency at runtime
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { NodeSDK } = require('@opentelemetry/sdk-node');
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
    const sdk = new NodeSDK({
      serviceName: 'gateway',
      instrumentations: [getNodeAutoInstrumentations()],
      otlpExporter: undefined,
    });
    await sdk.start();
    logger.log('OpenTelemetry initialized (gateway)');
  } catch (err) {
    logger.warn(
      `OpenTelemetry not initialized (missing deps or config): ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}
