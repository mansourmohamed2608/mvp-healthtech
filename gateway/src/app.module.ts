// gateway/src/app.module.ts
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerModule } from '@nestjs/throttler';
import { AuthModule } from './auth/auth.module';
import { SessionController } from './session/session.controller';
import { TwilioController } from './twilio/twilio.controller';
import { TwilioService } from './twilio/twilio.service';
import { AsrService } from './asr/asr.service';
import { LlmService } from './llm/llm.service';
import { ConversationService } from './conversation/conversation.service';
import { MetricsController } from './metrics/metrics.controller';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    ThrottlerModule.forRoot({
      throttlers: [
        { ttl: 60_000, limit: 50 }, // ttl in ms, limit = requests per ttl
      ],
    }),
    AuthModule,
  ],
  controllers: [SessionController, TwilioController, MetricsController],
  providers: [TwilioService, AsrService, LlmService, ConversationService],
})
export class AppModule {}
