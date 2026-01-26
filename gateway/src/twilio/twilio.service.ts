// gateway/src/twilio/twilio.service.ts
import { Injectable, Logger } from '@nestjs/common';
import { createHmac } from 'crypto';
import * as twilio from 'twilio';
import { TwilioWebhookBody } from '../types/twilio';

const AccessToken = twilio.jwt.AccessToken;
const VoiceGrant = AccessToken.VoiceGrant;

@Injectable()
export class TwilioService {
  private readonly logger = new Logger(TwilioService.name);

  /**
   * Validate the Twilio request using the signature and the auth token.
   * Docs: https://www.twilio.com/docs/usage/security#validating-requests
   */
  validateTwilioRequest(
    url: string,
    params: Record<string, any>,
    signature: string,
  ): boolean {
    const authToken = process.env.TWILIO_AUTH_TOKEN || '';

    if (!authToken) {
      this.logger.warn('TWILIO_AUTH_TOKEN not configured');
      // In development, allow requests without validation
      if (process.env.NODE_ENV === 'development') {
        return true;
      }
      return false;
    }

    try {
      // Use Twilio's official validation method
      const isValid = twilio.validateRequest(authToken, signature, url, params);

      if (!isValid) {
        this.logger.warn(`Invalid Twilio signature for URL: ${url}`);
      }

      return isValid;
    } catch (error) {
      this.logger.error('Twilio signature validation error', error);
      return false;
    }
  }

  /**
   * Generate TwiML to start a media stream
   * Auth credentials (sig, ts) are passed via Custom Parameters, NOT query string
   */
  generateStreamTwiML(streamUrl: string, callSid: string): string {
    const response = new twilio.twiml.VoiceResponse();

    // Say a greeting
    response.say(
      {
        voice: 'Polly.Zeina',
        language: 'ar-AE',
      },
      'مرحبا بك في النظام الصحي',
    );

    // Start media streaming — pass HMAC auth via Custom Parameters (NOT query string)
    const start = response.start();
    const { sig, ts } = this.generateStreamAuth(callSid);
    start
      .stream({ url: streamUrl })
      .parameter({ name: 'callSid', value: callSid })
      .parameter({ name: 'sig', value: sig })
      .parameter({ name: 'ts', value: ts.toString() });

    // Pause to keep the stream open (10 minutes = 600 seconds)
    response.pause({ length: 600 });

    return response.toString();
  }

  /**
   * Generate HMAC signature and timestamp for WebSocket auth
   * These are passed as Custom Parameters, validated from 'start' message
   */
  private generateStreamAuth(callSid: string): { sig: string; ts: number } {
    const secret =
      process.env.TWILIO_AUTH_TOKEN || process.env.WS_SHARED_SECRET || '';
    const ts = Math.floor(Date.now() / 1000);
    if (!secret) {
      this.logger.warn('No secret configured for stream auth');
      return { sig: '', ts };
    }
    const sig = createHmac('sha256', secret)
      .update(`${callSid}:${ts}`)
      .digest('hex');
    return { sig, ts };
  }

  /**
   * Generate TwiML to end the call
   */
  generateHangupTwiML(message?: string): string {
    const response = new twilio.twiml.VoiceResponse();

    if (message) {
      response.say(
        {
          voice: 'Polly.Zeina',
          language: 'ar-AE',
        },
        message,
      );
    }

    response.hangup();
    return response.toString();
  }

  /**
   * Extract call metadata from webhook body
   */
  extractCallMetadata(body: TwilioWebhookBody) {
    return {
      callSid: body.CallSid,
      accountSid: body.AccountSid,
      from: body.From,
      to: body.To,
      callStatus: body.CallStatus,
      direction: body.Direction,
    };
  }

  /**
   * Generate Twilio Access Token for Voice SDK
   * Docs: https://www.twilio.com/docs/iam/access-tokens
   */
  generateAccessToken(identity: string): string {
    const accountSid = process.env.TWILIO_ACCOUNT_SID;
    const apiKey = process.env.TWILIO_API_KEY;
    const apiSecret = process.env.TWILIO_API_SECRET;
    const twimlAppSid = process.env.TWILIO_TWIML_APP_SID;

    if (!accountSid || !apiKey || !apiSecret || !twimlAppSid) {
      const missing: string[] = [];
      if (!accountSid) missing.push('TWILIO_ACCOUNT_SID');
      if (!apiKey) missing.push('TWILIO_API_KEY');
      if (!apiSecret) missing.push('TWILIO_API_SECRET');
      if (!twimlAppSid) missing.push('TWILIO_TWIML_APP_SID');

      this.logger.error(`Missing Twilio credentials: ${missing.join(', ')}`);
      throw new Error(
        `Missing Twilio environment variables: ${missing.join(', ')}`,
      );
    }

    // Create an access token
    const accessToken = new AccessToken(accountSid, apiKey, apiSecret, {
      identity,
      ttl: 3600, // 1 hour
    });

    // Create a Voice grant for this token
    const voiceGrant = new VoiceGrant({
      outgoingApplicationSid: twimlAppSid,
      incomingAllow: true, // Allow incoming calls
    });

    // Add the grant to the token
    accessToken.addGrant(voiceGrant);

    // Serialize the token to a JWT
    return accessToken.toJwt();
  }
}
