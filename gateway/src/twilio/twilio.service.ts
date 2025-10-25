/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
// gateway/src/twilio/twilio.service.ts
import { Injectable, Logger, UnauthorizedException } from '@nestjs/common';
import { createHmac } from 'crypto';
import axios from 'axios';

@Injectable()
export class TwilioService {
  private readonly logger = new Logger(TwilioService.name);

  /**
   * Validate the Twilio request using the signature and the auth token.
   * Docs: https://www.twilio.com/docs/usage/security#validating-requests
   */
  validateTwilioRequest(url: string, params: any, signature: string): boolean {
    const authToken = process.env.TWILIO_AUTH_TOKEN || '';
    const sorted = Object.keys(params)
      .sort()
      .map((k) => `${k}${params[k]}`)
      .join('');
    const data = url + sorted;
    const computedSignature = createHmac('sha1', authToken)
      .update(Buffer.from(data, 'utf-8'))
      .digest('base64');
    return computedSignature === signature;
  }

  async forwardAudioToAsr(mediaUrl: string, callSid: string) {
    // Download the media and send to ASR service
    const resp = await axios.get(mediaUrl, { responseType: 'arraybuffer' });
    const audio = Buffer.from(resp.data);
    await axios.post(
      process.env.ASR_SERVICE_URL + '/transcribe',
      { audio: audio.toString('base64'), callSid },
      { headers: { 'Content-Type': 'application/json' } },
    );
  }
}
