// gateway/src/conversation/conversation.module.ts
import { Module } from '@nestjs/common';
import { ConversationService } from './conversation.service';
import { LlmService } from '../llm/llm.service';
import { TtsService } from '../tts/tts.service';
import { AsrService } from '../asr/asr.service';

@Module({
  providers: [ConversationService, LlmService, TtsService, AsrService],
  exports: [ConversationService],
})
export class ConversationModule {}
