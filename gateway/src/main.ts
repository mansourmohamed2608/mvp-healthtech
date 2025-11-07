/* eslint-disable @typescript-eslint/no-unsafe-call */
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe, Logger } from '@nestjs/common';
import morgan from 'morgan';
import { LatencyMiddleware } from './middleware/latency.middleware';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Enable CORS for frontend
  app.enableCors({
    origin: [
      'http://localhost:3000',
      'http://localhost:5173',
      'http://localhost:3001',
    ],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
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
  logger.log(`CORS enabled for localhost:3000, localhost:5173, localhost:3001`);
  logger.log(`Latency monitoring enabled (target: <20ms gateway overhead)`);
}
bootstrap();
