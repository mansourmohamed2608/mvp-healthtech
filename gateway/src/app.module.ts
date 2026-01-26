// gateway/src/app.module.ts
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
import { HealthController } from './health.controller';
import { AsrService } from './asr/asr.service';
import { LlmService } from './llm/llm.service';
import { TtsService } from './tts/tts.service';
import { ConversationService } from './conversation/conversation.service';
import { ConversationController } from './conversation/conversation.controller';
import { VectorCacheService } from './cache/vector-cache.service';
import { KvCacheService } from './cache/kv-cache.service';
import { JwtAuthGuard } from './auth/jwt.guard';
import { RolesGuard } from './auth/roles.guard';
import { MiddlewareConsumer, Module, RequestMethod } from '@nestjs/common';
import { CorrelationMiddleware } from './middleware/correlation.middleware';
import { AuditService } from './audit/audit.service';
import { InternalHttpClient } from './http/internal-http-client.service';
import { VaModule } from './va/va.module';
import { join } from 'path';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      // Prefer .env.local for local dev, fallback to .env
      envFilePath: [
        join(__dirname, '..', '.env.local'),
        join(__dirname, '..', '.env'),
        join(__dirname, '..', '..', '.env.local'),
        join(__dirname, '..', '..', '.env'),
      ],
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
    VaModule,
  ],
  controllers: [
    MetricsController,
    HealthController,
    AsrController,
    LlmController,
    TtsController,
    SoapController,
    FhirController,
    ConversationController,
  ],
  providers: [
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
    {
      provide: APP_GUARD,
      useClass: RolesGuard,
    },
    JwtAuthGuard,
    AsrService,
    LlmService,
    TtsService,
    ConversationService,
    VectorCacheService,
    KvCacheService,
    AuditService,
    InternalHttpClient,
  ],
})
export class AppModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(CorrelationMiddleware)
      .forRoutes({ path: '*', method: RequestMethod.ALL });
  }
}
