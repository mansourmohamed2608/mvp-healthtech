// gateway/src/voice/voice.module.ts
import { Module } from '@nestjs/common';
import { VoiceGateway } from './voice.gateway';
import { ConversationModule } from '../conversation/conversation.module';
import { SessionModule } from '../session/session.module';
import { AsrService } from '../asr/asr.service';
import { InternalHttpClient } from '../http/internal-http-client.service';
import { AuthModule } from '../auth/auth.module';

@Module({
  imports: [ConversationModule, SessionModule, AuthModule],
  providers: [VoiceGateway, AsrService, InternalHttpClient],
  exports: [VoiceGateway],
})
export class VoiceModule {}
