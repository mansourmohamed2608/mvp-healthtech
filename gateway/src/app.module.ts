// gateway/src/app.module.ts
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';
import { AuthModule } from './auth/auth.module';
import { SessionModule } from './session/session.module';
import { TwilioModule } from './twilio/twilio.module';
import { VoiceModule } from './voice/voice.module';
import { AudioModule } from './audio/audio.module';
import { QueueModule } from './queue/queue.module';
import { ClinicalModule } from './clinical/clinical.module';
import { RAGModule } from './rag/rag.module';
import { MetricsController } from './metrics/metrics.controller';
import { AsrController } from './asr/asr.controller';
import { LlmController } from './llm/llm.controller';
import { TtsController } from './tts/tts.controller';
import { SoapController } from './soap/soap.controller';
import { FhirController } from './fhir/fhir.controller';
import { AsrService } from './asr/asr.service';
import { LlmService } from './llm/llm.service';
import { TtsService } from './tts/tts.service';
import { ConversationService } from './conversation/conversation.service';
import { VectorCacheService } from './cache/vector-cache.service';
import { KvCacheService } from './cache/kv-cache.service';

@Module({
  imports: [
    ConfigModule.forRoot({ 
      isGlobal: true,
      envFilePath: '../.env',  // Load from root .env
    }),
    ThrottlerModule.forRoot({
      throttlers: [
        { ttl: 60_000, limit: 50 }, // 50 requests per minute
      ],
    }),
    AuthModule,
    SessionModule,
    TwilioModule,
    VoiceModule,
    AudioModule,
    QueueModule,
    ClinicalModule,
    RAGModule,
  ],
  controllers: [
    MetricsController,
    AsrController,
    LlmController,
    TtsController,
    SoapController,
    FhirController,
  ],
  providers: [
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
    AsrService,
    LlmService,
    TtsService,
    ConversationService,
    VectorCacheService,
    KvCacheService,
  ],
})
export class AppModule {}

