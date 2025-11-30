import { LoggerService } from '@nestjs/common';

const SENSITIVE_KEYS = ['transcript', 'text', 'soap', 'payload', 'audio', 'body', 'message'];

function redact(value: any): any {
  if (value === null || value === undefined) return value;
  if (typeof value === 'string') {
    if (value.length > 0 && value.length <= 16) return '[[redacted]]';
    return '[[redacted]]';
  }
  if (Array.isArray(value)) return value.map(() => '[[redacted]]');
  if (typeof value === 'object') {
    const clone: Record<string, any> = {};
    for (const [k, v] of Object.entries(value)) {
      clone[k] = '[[redacted]]';
    }
    return clone;
  }
  return '[[redacted]]';
}

export function safeLog(
  logger: LoggerService,
  level: 'log' | 'warn' | 'error' | 'debug' | 'verbose',
  message: string,
  meta?: Record<string, any>,
) {
  const cleaned: Record<string, any> = {};
  for (const [k, v] of Object.entries(meta || {})) {
    if (SENSITIVE_KEYS.some((sk) => k.toLowerCase().includes(sk))) {
      cleaned[k] = redact(v);
    } else {
      cleaned[k] = v;
    }
  }
  (logger as any)[level](message, cleaned);
}
