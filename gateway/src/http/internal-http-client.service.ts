// gateway/src/http/internal-http-client.service.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { Injectable, Logger } from '@nestjs/common';

export interface InternalClientOptions {
  baseUrl: string;
  serviceName: string;
  timeoutMs?: number;
  retries?: number;
}

@Injectable()
export class InternalHttpClient {
  private readonly logger = new Logger(InternalHttpClient.name);
  private clientCache = new Map<string, AxiosInstance>();
  private readonly defaultTimeout = Number(process.env.INTERNAL_HTTP_TIMEOUT_MS || 8000);
  private readonly defaultRetries = Number(process.env.INTERNAL_HTTP_RETRIES || 2);
  private readonly internalSecret = process.env.INTERNAL_SECRET || '';
   private readonly cbEnabled = (process.env.INTERNAL_HTTP_CB_ENABLED || 'false').toLowerCase() === 'true';
   private readonly cbThreshold = Number(process.env.INTERNAL_HTTP_CB_THRESHOLD || 5);
   private readonly cbCooldownMs = Number(process.env.INTERNAL_HTTP_CB_COOLDOWN_MS || 30000);
   private breakerState = new Map<string, { failures: number; openedUntil: number }>();

  getClient(opts: InternalClientOptions): AxiosInstance {
    const key = `${opts.baseUrl}:${opts.serviceName}`;
    if (this.clientCache.has(key)) return this.clientCache.get(key)!;

    const instance = axios.create({
      baseURL: opts.baseUrl,
      timeout: opts.timeoutMs || this.defaultTimeout,
    });

    instance.interceptors.request.use((config) => {
      if (this.cbEnabled) {
        const state = this.breakerState.get(opts.serviceName);
        const now = Date.now();
        if (state && state.openedUntil > now) {
          return Promise.reject({
            status: 503,
            message: 'circuit_open',
            service: opts.serviceName,
            correlationId: config.headers?.['x-correlation-id'],
            isRetryable: true,
          });
        } else if (state && state.openedUntil <= now) {
          this.breakerState.delete(opts.serviceName);
        }
      }
      config.headers = config.headers || {};
      if (this.internalSecret) {
        config.headers['x-internal-secret'] = this.internalSecret;
      }
      const corr = (config.headers['x-correlation-id'] as string) || (config.headers['X-Correlation-Id'] as string);
      if (!corr && (config as any).correlationId) {
        config.headers['x-correlation-id'] = (config as any).correlationId;
      }
      return config;
    });

    instance.interceptors.response.use(
      (resp) => {
        if (this.cbEnabled) {
          this.breakerState.delete(opts.serviceName);
        }
        return resp;
      },
      async (error) => {
      const retries = (opts.retries ?? this.defaultRetries);
      const cfg = error.config as AxiosRequestConfig & { __retryCount?: number };
      cfg.__retryCount = cfg.__retryCount || 0;
      if (cfg.__retryCount < retries && (!cfg.method || ['get', 'post'].includes(cfg.method))) {
        cfg.__retryCount += 1;
        const backoff = 100 * cfg.__retryCount;
        await new Promise((r) => setTimeout(r, backoff));
        return instance(cfg);
      }
      const status = error.response?.status || 500;
      const correlationId = error.response?.headers?.['x-correlation-id'];
      const message = error.response?.data?.message || 'downstream error';
      if (this.cbEnabled) {
        const state = this.breakerState.get(opts.serviceName) || { failures: 0, openedUntil: 0 };
        state.failures += 1;
        if (state.failures >= this.cbThreshold) {
          state.openedUntil = Date.now() + this.cbCooldownMs;
          this.logger.warn(`Circuit opened for ${opts.serviceName}`, { failures: state.failures });
        }
        this.breakerState.set(opts.serviceName, state);
      }
      const normalized = {
        statusCode: status,
        status,
        message,
        service: opts.serviceName,
        correlationId,
        isRetryable: status >= 500,
      };
      this.logger.warn(`${opts.serviceName} request failed`, normalized);
      return Promise.reject({
        ...normalized,
      });
    });

    this.clientCache.set(key, instance);
    return instance;
  }
}
