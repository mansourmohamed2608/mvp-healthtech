import { Module } from '@nestjs/common';
import { TwilioController } from './twilio.controller';
@Module({ controllers: [TwilioController] })
export class TwilioModule {}
