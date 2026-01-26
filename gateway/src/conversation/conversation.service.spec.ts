import { ConversationService } from './conversation.service';
import { LlmService } from '../llm/llm.service';
import { TtsService } from '../tts/tts.service';
import { AsrService } from '../asr/asr.service';

jest.mock('redis', () => ({
  createClient: jest.fn(() => ({
    on: jest.fn(),
    connect: jest.fn().mockRejectedValue(new Error('no redis in tests')),
  })),
}));

describe('ConversationService', () => {
  const llmMock: Partial<LlmService> = {
    orchestrate: jest
      .fn()
      .mockResolvedValue({ reply: 'assistant reply', intent: 'general' }),
  };
  const ttsMock: Partial<TtsService> = {
    synthesize: jest
      .fn()
      .mockResolvedValue({ audioBase64: 'YmFzZTY0', format: 'mulaw' }),
  };
  const asrMock: Partial<AsrService> = {
    transcribe: jest
      .fn()
      .mockResolvedValue({ text: 'user said hello', segments: [] }),
    stream: jest.fn(),
  };

  beforeAll(() => {
    // No outbound fetch needed; ASR is mocked
  });

  it('runs ASR -> /chat -> TTS and returns base64 audio', async () => {
    const svc = new ConversationService(
      llmMock as LlmService,
      ttsMock as TtsService,
      asrMock as AsrService,
    );
    // Force in-memory mode
    (svc as any).redisAvailable = false;

    const result = await svc.processVoiceInput({
      callSid: 'call-1',
      audio: Buffer.from('fake').toString('base64'),
      format: 'mulaw',
      sampleRate: 8000,
    });

    expect(result.transcript).toBeDefined();
    expect(result.response).toBe('assistant reply');
    expect(result.audioResponse).toBe('YmFzZTY0');
    expect(ttsMock.synthesize).toHaveBeenCalled();
    expect(llmMock.orchestrate).toHaveBeenCalledWith({
      transcript: 'user said hello',
      history: expect.any(Array),
      sessionId: 'call-1',
      mode: 'voice_agent_va',
      slots: expect.any(Object),
      dialect: expect.any(String),
    });
  });
});
