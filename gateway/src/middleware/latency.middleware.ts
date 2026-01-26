// gateway/src/middleware/latency.middleware.ts
import { Injectable, NestMiddleware, Logger } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { Histogram, Counter, register } from 'prom-client';

@Injectable()
export class LatencyMiddleware implements NestMiddleware {
  private readonly logger = new Logger(LatencyMiddleware.name);

  // Prometheus metrics
  private readonly requestDuration: Histogram<string>;
  private readonly slowRequests: Counter<string>;

  // Target: <20ms gateway overhead
  private readonly SLOW_REQUEST_THRESHOLD = 20; // ms

  constructor() {
    // Try to get existing metrics or create new ones
    try {
      this.requestDuration = register.getSingleMetric(
        'gateway_request_duration_ms',
      ) as Histogram<string>;
      if (!this.requestDuration) {
        throw new Error('Metric not found');
      }
    } catch {
      this.requestDuration = new Histogram({
        name: 'gateway_request_duration_ms',
        help: 'Gateway request processing duration in milliseconds',
        labelNames: ['method', 'route', 'status'],
        buckets: [1, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000],
      });
    }

    try {
      this.slowRequests = register.getSingleMetric(
        'gateway_slow_requests_total',
      ) as Counter<string>;
      if (!this.slowRequests) {
        throw new Error('Metric not found');
      }
    } catch {
      this.slowRequests = new Counter({
        name: 'gateway_slow_requests_total',
        help: 'Total number of slow requests (>20ms gateway overhead)',
        labelNames: ['method', 'route'],
      });
    }
  }

  use(req: Request, res: Response, next: NextFunction) {
    const startTime = process.hrtime.bigint();
    const { method, path } = req;

    // Add timing header BEFORE response is sent
    const originalSend = res.send;
    res.send = function (body) {
      const endTime = process.hrtime.bigint();
      const durationNs = endTime - startTime;
      const durationMs = Number(durationNs) / 1_000_000;

      // Set header before sending response
      if (!res.headersSent) {
        res.setHeader('X-Gateway-Time', `${durationMs.toFixed(2)}ms`);
      }

      return originalSend.call(this, body);
    };

    // Capture response finish event for metrics
    res.on('finish', () => {
      const endTime = process.hrtime.bigint();
      const durationNs = endTime - startTime;
      const durationMs = Number(durationNs) / 1_000_000; // Convert to milliseconds

      // Normalize route for metrics (remove IDs)
      const route = this.normalizeRoute(path);
      const status = res.statusCode.toString();

      // Record metrics
      this.requestDuration.observe({ method, route, status }, durationMs);

      // Log slow requests
      if (durationMs > this.SLOW_REQUEST_THRESHOLD) {
        this.slowRequests.inc({ method, route });
        this.logger.warn(
          `Slow request: ${method} ${path} took ${durationMs.toFixed(2)}ms (target: <${this.SLOW_REQUEST_THRESHOLD}ms)`,
        );
      }
    });

    next();
  }

  /**
   * Normalize route by removing IDs and dynamic segments
   * /users/123 -> /users/:id
   * /calls/CA123abc -> /calls/:callSid
   */
  private normalizeRoute(path: string): string {
    return path
      .replace(/\/[0-9a-f]{24}/g, '/:id') // MongoDB IDs
      .replace(/\/CA[0-9a-f]{32}/g, '/:callSid') // Twilio Call SIDs
      .replace(/\/\d+/g, '/:id') // Numeric IDs
      .replace(/\/[a-f0-9-]{36}/g, '/:uuid'); // UUIDs
  }

  /**
   * Get current latency metrics
   */
  getMetrics() {
    return {
      requestDuration: this.requestDuration,
      slowRequests: this.slowRequests,
    };
  }
}
