// gateway/src/voice/voice.module.ts
import { Module } from '@nestjs/common';
import { VoiceGateway } from './voice.gateway';
import { ConversationModule } from '../conversation/conversation.module';
import { SessionModule } from '../session/session.module';

@Module({
  imports: [ConversationModule, SessionModule],
  providers: [VoiceGateway],
  exports: [VoiceGateway],
})
export class VoiceModule {}
