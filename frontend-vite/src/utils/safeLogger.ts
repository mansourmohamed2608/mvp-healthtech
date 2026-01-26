/**
 * Safe logger for frontend — suppresses console output in production
 * Prevents accidental PHI leakage via console
 */

const isDevelopment = import.meta.env.DEV || import.meta.env.MODE === 'development';

interface SafeLogger {
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
  debug: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
}

/**
 * Redact potentially sensitive data from log arguments
 */
function redactSensitive(arg: unknown): unknown {
  if (arg === null || arg === undefined) return arg;
  
  if (typeof arg === 'string') {
    // Redact anything that looks like PII/PHI
    return arg
      .replace(/\b\d{3}-\d{2}-\d{4}\b/g, '[SSN-REDACTED]')
      .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[EMAIL-REDACTED]')
      .replace(/\b\d{10,}\b/g, '[PHONE-REDACTED]');
  }
  
  if (typeof arg === 'object') {
    // Don't log objects in production — they may contain PHI
    return '[OBJECT-REDACTED]';
  }
  
  return arg;
}

function createSafeLogger(): SafeLogger {
  if (isDevelopment) {
    // In development, log with redaction
    return {
      log: (...args) => console.log('[DEV]', ...args.map(redactSensitive)),
      warn: (...args) => console.warn('[DEV]', ...args.map(redactSensitive)),
      error: (...args) => console.error('[DEV]', ...args.map(redactSensitive)),
      debug: (...args) => console.debug('[DEV]', ...args.map(redactSensitive)),
      info: (...args) => console.info('[DEV]', ...args.map(redactSensitive)),
    };
  }

  // In production, no-op all console methods
  const noop = () => {};
  return {
    log: noop,
    warn: noop,
    error: noop,
    debug: noop,
    info: noop,
  };
}

export const safeLogger = createSafeLogger();

/**
 * Use this instead of console.error for error reporting
 * In production, errors should go to an error tracking service (Sentry, etc.)
 */
export function reportError(error: Error, context?: Record<string, unknown>): void {
  if (isDevelopment) {
    console.error('[ERROR]', error.message, context ? redactSensitive(context) : '');
  }
  
  // In production, send to error tracking service
  // TODO: Integrate with Sentry or similar
  // if (!isDevelopment && window.Sentry) {
  //   window.Sentry.captureException(error, { extra: context });
  // }
}

export default safeLogger;
