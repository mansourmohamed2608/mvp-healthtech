/* eslint-disable @typescript-eslint/no-unsafe-call */
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe, Logger } from '@nestjs/common';
import { ThrottlerGuard } from '@nestjs/throttler';
import * as morgan from 'morgan';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));
  app.useGlobalGuards(app.get(ThrottlerGuard));
  const logger = new Logger('Bootstrap');
  const port = process.env.PORT || 3000;
  app.use(morgan('combined'));
  await app.listen(port);
  logger.log(`Gateway listening on port ${port}`);
}
bootstrap();
