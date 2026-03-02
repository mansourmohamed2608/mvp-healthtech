/* eslint-disable @typescript-eslint/no-unsafe-call */
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe, Logger } from '@nestjs/common';
import morgan from 'morgan';
import helmet from 'helmet';
import { LatencyMiddleware } from './middleware/latency.middleware';
import { initOtel } from './observability/otel';
import { validateEnv } from './config/env.validation';
import { TwilioWsAdapter } from './adapters/twilio-ws.adapter';

async function bootstrap() {
  validateEnv();
  await initOtel();
  const app = await NestFactory.create(AppModule);

  // Use custom WebSocket adapter that handles any path (required for /twilio/{callSid})
  app.useWebSocketAdapter(new TwilioWsAdapter(app));

  // Security headers via Helmet
  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", 'data:', 'https:'],
          connectSrc: ["'self'", 'wss:', 'https:'],
          frameSrc: ["'none'"],
          objectSrc: ["'none'"],
        },
      },
      hsts: {
        maxAge: 31536000, // 1 year
        includeSubDomains: true,
        preload: true,
      },
      frameguard: { action: 'deny' },
      noSniff: true,
      xssFilter: true,
      referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
    }),
  );

  // Enable CORS for frontend — restrict in production
  const corsEnv = process.env.CORS_ALLOWED_ORIGINS;
  const corsOrigins = corsEnv
    ? corsEnv === '*'
      ? true // Allow all origins when * is specified
      : corsEnv.split(',').map((o) => o.trim())
    : [
        'http://localhost:3000',
        'http://localhost:5173',
        'http://localhost:3001',
      ];

  app.enableCors({
    origin: corsOrigins,
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'x-tenant-id',
      'x-correlation-id',
    ],
  });

  app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

  // Add latency measurement middleware (Week 1 requirement: <20ms gateway overhead)
  app.use(new LatencyMiddleware().use.bind(new LatencyMiddleware()));

  // Throttler guard is configured in app.module.ts
  const logger = new Logger('Bootstrap');
  const port = process.env.PORT || 3001;
  app.use(morgan('combined'));

  // ---------------------------------------------------------------
  // MED-8: Startup health gate — confirm DB + Redis before serving.
  // In production a failed health gate crashes the container so the
  // orchestrator can restart it rather than silently serving errors.
  // ---------------------------------------------------------------
  const { Pool } = await import('pg');
  const dbUrl = process.env.DATABASE_URL;
  if (dbUrl) {
    const probe = new Pool({ connectionString: dbUrl, max: 1, connectionTimeoutMillis: 5000 });
    try {
      await probe.query('SELECT 1');
      logger.log('Health gate: DB reachable');
    } catch (err) {
      logger.error('Health gate: DB unreachable — aborting startup', (err as Error).message);
      process.exit(1);
    } finally {
      await probe.end();
    }
  } else {
    logger.warn('Health gate: DATABASE_URL not set — skipping DB check');
  }

  const redisHost = process.env.REDIS_HOST;
  if (redisHost) {
    const { createClient } = await import('redis');
    const redisPort = parseInt(process.env.REDIS_PORT || '6379', 10);
    const client = createClient({ socket: { host: redisHost, port: redisPort, connectTimeout: 5000 } });
    try {
      await client.connect();
      await client.ping();
      logger.log('Health gate: Redis reachable');
    } catch (err) {
      logger.error('Health gate: Redis unreachable — aborting startup', (err as Error).message);
      process.exit(1);
    } finally {
      await client.disconnect().catch(() => undefined);
    }
  } else {
    logger.warn('Health gate: REDIS_HOST not set — skipping Redis check');
  }
  // ---------------------------------------------------------------

  await app.listen(port);
  logger.log(`Gateway listening on port ${port}`);
  logger.log(`CORS enabled for: ${corsOrigins === true ? '*' : (corsOrigins as string[]).join(', ')}`);
  logger.log(`Helmet security headers enabled`);
  logger.log(`Latency monitoring enabled (target: <20ms gateway overhead)`);
}
bootstrap();
