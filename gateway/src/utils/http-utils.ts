import { HttpException, HttpStatus } from '@nestjs/common';
import { Request } from 'express';

export interface ErrorEnvelope {
  message: string;
  code?: string;
  correlationId?: string;
}

export function wrapError(err: any, req?: Request, defaultStatus: number = HttpStatus.INTERNAL_SERVER_ERROR): never {
  const correlationId = (req as any)?.correlationId;
  const message = err?.message || 'Unexpected error';
  const code = err?.code || err?.response?.status || undefined;
  const status = err?.status || err?.response?.status || defaultStatus;
  const envelope: ErrorEnvelope = { message, code, correlationId };
  throw new HttpException(envelope, status);
}

export function toCamel<T extends Record<string, any>>(obj: T): any {
  if (!obj || typeof obj !== 'object') return obj;
  if (Array.isArray(obj)) return obj.map(toCamel);
  const out: Record<string, any> = {};
  for (const [k, v] of Object.entries(obj)) {
    const camelKey = k.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
    out[camelKey] = toCamel(v);
  }
  return out;
}

export const camelResponse = <T extends Record<string, any>>(data: T) => toCamel(data);
