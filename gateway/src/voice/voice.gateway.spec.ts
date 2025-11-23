import { Test, TestingModule } from '@nestjs/testing';
import { VoiceGateway } from './voice.gateway';
import { ConversationService } from '../conversation/conversation.service';
import { SessionService } from '../session/session.service';

describe('VoiceGateway minimal contract', () => {
  it('extractCallSidFromUrl parses sid', () => {
    const gateway = new VoiceGateway({} as any, {} as any);
    const sid = (gateway as any).extractCallSidFromUrl('/twilio/ws/CA1234?token=abc');
    expect(sid).toBe('CA1234');
  });
});
