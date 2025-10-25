/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/require-await */
// gateway/src/twilio/twilio.controller.ts
import {
  Body,
  Controller,
  Headers,
  Post,
  Req,
  Res,
  UnauthorizedException,
} from '@nestjs/common';
import { TwilioService } from './twilio.service';
import { AsrService } from 'src/asr/asr.service';
import { ConversationService } from 'src/conversation/conversation.service';
import { LlmService } from 'src/llm/llm.service';

@Controller('twilio')
export class TwilioController {
  constructor(
    private readonly twilioService: TwilioService,
    private readonly asrService: AsrService,
    private readonly llmService: LlmService,
    private readonly conversationService: ConversationService,
  ) {}

  /**
   * Called by Twilio when a call starts.  Validates the request and returns
   * TwiML instructions to begin streaming audio.
   */
  @Post('voice/start')
  async start(
    @Headers() headers: any,
    @Body() body: any,
    @Req() req,
    @Res() res,
  ) {
    const signature =
      headers['x-twilio-signature'] || headers['X-Twilio-Signature'];
    const isValid = this.twilioService.validateTwilioRequest(
      req.protocol + '://' + req.get('host') + req.originalUrl,
      body,
      signature,
    );
    if (!isValid) {
      throw new UnauthorizedException('Invalid Twilio signature');
    }
    // Return TwiML instructions to start media streaming
    const response = `<Response><Start><Stream url="${process.env.GATEWAY_PUBLIC_URL}/twilio/voice/stream?callSid=${body.CallSid}"/></Start></Response>`;
    res.set('Content-Type', 'text/xml');
    return res.send(response);
  }

  /**
   * Twilio sends media frames as HTTP POST requests to the stream URL above.  Each
   * request contains a single audio chunk encoded in base64.
   */
  @Post('voice/stream')
  async stream(@Headers() headers: any, @Body() body: any) {
    const callSid = body.callSid || body.CallSid;
    const audio = body.media?.payload;
    if (!audio) return { ok: false };

    // 1. Get partial transcript from ASR
    const asrResponse = await this.asrService.stream(audio, callSid);
    const partial = asrResponse.partial;
    if (!partial) {
      return { ok: true };
    }

    // 2. Append user message to conversation history
    await this.conversationService.appendMessage(callSid, 'user', partial);

    // 3. Invoke the LLM service
    const llmResponse = await this.llmService.infer(partial, callSid);

    // 4. Append assistant reply to conversation history
    await this.conversationService.appendMessage(
      callSid,
      'assistant',
      llmResponse.reply,
    );

    // 5. (Week 3) send the reply back via TTS to the caller

    return { ok: true };
  }

  /**
   * Called by Twilio when the call ends.  Cleanup resources.
   */
  @Post('voice/stop')
  async stop(@Headers() headers: any, @Body() body: any) {
    const callSid = body.CallSid;
    await this.conversationService.clear(callSid);
    return { ok: true, event: 'stop' };
  }
}
