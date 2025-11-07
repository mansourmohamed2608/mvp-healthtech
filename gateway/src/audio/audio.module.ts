// gateway/src/audio/audio.module.ts
import { Module } from '@nestjs/common';
import { AudioProcessorService } from './audio-processor.service';

@Module({
  providers: [AudioProcessorService],
  exports: [AudioProcessorService],
})
export class AudioModule {}
