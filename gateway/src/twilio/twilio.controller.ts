// gateway/src/twilio/twilio.controller.ts
import {
  Body,
  Controller,
  Headers,
  Post,
  Req,
  Res,
  UnauthorizedException,
  Logger,
  HttpCode,
  HttpStatus,
  UseGuards,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import { TwilioService } from './twilio.service';
import { SessionService } from '../session/session.service';
import type { TwilioWebhookBody } from '../types/twilio';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { Roles } from '../auth/roles.decorator';

const maskPhone = (input?: string) => {
  if (!input) return input;
  const digits = input.replace(/\D/g, '');
  if (digits.length <= 4) return '*'.repeat(Math.max(0, digits.length - 2)) + digits.slice(-2);
  return `${'*'.repeat(Math.max(0, digits.length - 4))}${digits.slice(-4)}`;
};

@Controller('twilio')
export class TwilioController {
  private readonly logger = new Logger(TwilioController.name);

  constructor(
    private readonly twilioService: TwilioService,
    private readonly sessionService: SessionService,
  ) {}

  /**
   * Called by Twilio when a call starts. Validates the request and returns
   * TwiML instructions to begin streaming audio.
   */
  @Post('voice/start')
  @HttpCode(HttpStatus.OK)
  async start(
    @Headers() headers: Record<string, string>,
    @Body() body: TwilioWebhookBody,
    @Req() req: Request,
    @Res() res: Response,
  ) {
    const signature = headers['x-twilio-signature'] || headers['X-Twilio-Signature'];
    const url = `${req.protocol}://${req.get('host')}${req.originalUrl}`;

    // Validate Twilio signature
    const isValid = this.twilioService.validateTwilioRequest(
      url,
      body,
      signature || '',
    );

    if (!isValid && process.env.NODE_ENV !== 'development') {
      this.logger.warn(`Invalid Twilio signature for call ${body.CallSid}`);
      throw new UnauthorizedException('Invalid Twilio signature');
    }

    try {
      // Extract call metadata
      const metadata = this.twilioService.extractCallMetadata(body);
      this.logger.log(`Call started: ${metadata.callSid} from ${maskPhone(metadata.from)}`);

      // Create session for this call
      await this.sessionService.create({
        callSid: metadata.callSid,
        metadata,
      });

      // Generate WebSocket stream URL
      const streamUrl = `${process.env.GATEWAY_PUBLIC_URL || 'wss://your-domain.ngrok.io'}/twilio/ws/${metadata.callSid}`;

      // Return TwiML to start media streaming
      const twiml = this.twilioService.generateStreamTwiML(
        streamUrl,
        metadata.callSid,
      );

      res.set('Content-Type', 'text/xml');
      return res.send(twiml);
    } catch (error) {
      this.logger.error('Error starting call', error);
      const errorTwiml = this.twilioService.generateHangupTwiML(
        'عذرا، حدث خطأ في النظام',
      );
      res.set('Content-Type', 'text/xml');
      return res.send(errorTwiml);
    }
  }

  /**
   * Generate Twilio access token for frontend Voice SDK
   * Protected: requires authenticated user (clinician/internal).
   */
  @Post('token')
  @UseGuards(JwtAuthGuard)
  @Roles('clinician')
  @HttpCode(HttpStatus.OK)
  async getToken(@Req() req: any) {
    try {
      const userIdentity = req.user?.sub || `user-${Date.now()}`;
      const token = this.twilioService.generateAccessToken(userIdentity);

      this.logger.log(`Generated Twilio token for identity: ${userIdentity}`);
      // TODO: add rate limiting to this endpoint.
      return {
        token,
        identity: userIdentity,
      };
    } catch (error) {
      this.logger.error('Error generating Twilio token', error);
      throw error;
    }
  }

  /**
   * Called by Twilio for call status updates
   */
  @Post('voice/status')
  @HttpCode(HttpStatus.OK)
  async status(
    @Headers() headers: Record<string, string>,
    @Body() body: TwilioWebhookBody,
  ) {
    const signature = headers['x-twilio-signature'] || headers['X-Twilio-Signature'];
    const metadata = this.twilioService.extractCallMetadata(body);

    this.logger.log(
      `Call status update: ${metadata.callSid} - ${metadata.callStatus}`,
    );

    // Update session based on call status
    if (metadata.callStatus === 'completed' || metadata.callStatus === 'failed') {
      try {
        await this.sessionService.delete(metadata.callSid);
        this.logger.log(`Session cleaned up for call: ${metadata.callSid}`);
      } catch (error) {
        this.logger.warn(`Failed to cleanup session: ${metadata.callSid}`, error);
      }
    }

    return { ok: true, status: metadata.callStatus };
  }

  /**
   * Called by Twilio when the call ends. Cleanup resources.
   */
  @Post('voice/stop')
  @HttpCode(HttpStatus.OK)
  async stop(
    @Headers() headers: Record<string, string>,
    @Body() body: TwilioWebhookBody,
  ) {
    const metadata = this.twilioService.extractCallMetadata(body);
    this.logger.log(`Call ended: ${metadata.callSid}`);

    try {
      await this.sessionService.delete(metadata.callSid);
    } catch (error) {
      this.logger.warn(`Failed to cleanup session on stop: ${metadata.callSid}`, error);
    }

    return { ok: true, event: 'stop' };
  }
}
