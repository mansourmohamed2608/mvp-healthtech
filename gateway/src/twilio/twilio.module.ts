// gateway/src/twilio/twilio.module.ts
import { Module } from '@nestjs/common';
import { TwilioController } from './twilio.controller';
import { TwilioService } from './twilio.service';
import { CodecNegotiatorService } from './codec-negotiator.service';
import { SessionModule } from '../session/session.module';

@Module({
  imports: [SessionModule],
  controllers: [TwilioController],
  providers: [TwilioService, CodecNegotiatorService],
  exports: [TwilioService, CodecNegotiatorService],
})
export class TwilioModule {}
