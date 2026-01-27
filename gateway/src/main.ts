/* eslint-disable @typescript-eslint/no-unsafe-call */
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe, Logger } from '@nestjs/common';
import morgan from 'morgan';
import helmet from 'helmet';
import { LatencyMiddleware } from './middleware/latency.middleware';
import { initOtel } from './observability/otel';
import { validateEnv } from './config/env.validation';

async function bootstrap() {
  validateEnv();
  await initOtel();
  const app = await NestFactory.create(AppModule);

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
  await app.listen(port);
  logger.log(`Gateway listening on port ${port}`);
  logger.log(`CORS enabled for: ${corsOrigins === true ? '*' : (corsOrigins as string[]).join(', ')}`);
  logger.log(`Helmet security headers enabled`);
  logger.log(`Latency monitoring enabled (target: <20ms gateway overhead)`);
}
bootstrap();
