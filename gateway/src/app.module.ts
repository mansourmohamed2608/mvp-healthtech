import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { SessionModule } from './session/session.module';
import { TwilioModule } from './twilio/twilio.module';
import { HealthController } from './health.controller';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    SessionModule,
    TwilioModule,
  ],
  controllers: [HealthController],
})
export class AppModule {}
