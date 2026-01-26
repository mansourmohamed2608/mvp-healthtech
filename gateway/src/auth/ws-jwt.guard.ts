import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { createHmac, timingSafeEqual } from 'crypto';

/**
 * WebSocket JWT Guard for Twilio Media Streams
 *
 * Auth credentials are passed via TwiML Custom Parameters (NOT query string).
 * Validation happens on the 'start' event message, NOT the upgrade request.
 *
 * Flow:
 * 1. WS connects without auth (guard allows initial connection)
 * 2. 'start' message contains customParameters: { callSid, sig, ts }
 * 3. Guard validates HMAC(secret, callSid:ts) using constant-time compare
 * 4. If invalid, connection is closed
 */
@Injectable()
export class WsJwtGuard implements CanActivate {
  constructor(private readonly jwtService: JwtService) {}

  canActivate(context: ExecutionContext): boolean {
    const client = context.switchToWs().getClient();
    const data = context.switchToWs().getData();

    try {
      // Try to extract JWT token from URL (for non-Twilio WS clients)
      const jwtToken = this.extractTokenFromUrl(client);

      // For Twilio streams, auth happens via 'start' message custom parameters
      // Allow initial connection, full validation happens in handleStreamStart
      if (!jwtToken) {
        // Mark as pending auth - will be validated on 'start' message
        client.pendingTwilioAuth = true;
        client.user = { sub: 'twilio_pending', roles: ['twilio'] };
        return true;
      }

      // Validate JWT for non-Twilio clients
      const payload = this.jwtService.verify(jwtToken);
      client.user = payload;
      return true;
    } catch (_e) {
      return false;
    }
  }

  /**
   * Validate Twilio stream auth from 'start' message custom parameters
   * Called from voice.gateway.ts when 'start' event is received
   */
  static validateTwilioStreamAuth(
    customParameters: Record<string, string> | undefined,
  ): { valid: boolean; callSid: string | null; reason?: string } {
    if (!customParameters) {
      return {
        valid: false,
        callSid: null,
        reason: 'Missing customParameters',
      };
    }

    const callSid = customParameters.callSid;
    const sig = customParameters.sig;
    const tsStr = customParameters.ts;

    if (!callSid || !sig || !tsStr) {
      return {
        valid: false,
        callSid,
        reason: 'Missing sig/ts/callSid in customParameters',
      };
    }

    const ts = Number(tsStr);
    if (isNaN(ts)) {
      return { valid: false, callSid, reason: 'Invalid timestamp' };
    }

    // Validate timestamp is within 5 minute window
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - ts) > 300) {
      return { valid: false, callSid, reason: 'Timestamp expired (>5min)' };
    }

    // Validate HMAC using constant-time comparison
    const secret =
      process.env.TWILIO_AUTH_TOKEN || process.env.WS_SHARED_SECRET || '';
    if (!secret) {
      return { valid: false, callSid, reason: 'No secret configured' };
    }

    const expected = createHmac('sha256', secret)
      .update(`${callSid}:${ts}`)
      .digest('hex');

    // Constant-time comparison to prevent timing attacks
    const sigBuffer = Buffer.from(sig, 'utf-8');
    const expectedBuffer = Buffer.from(expected, 'utf-8');

    if (sigBuffer.length !== expectedBuffer.length) {
      return { valid: false, callSid, reason: 'Invalid signature length' };
    }

    if (!timingSafeEqual(sigBuffer, expectedBuffer)) {
      return { valid: false, callSid, reason: 'Invalid signature' };
    }

    return { valid: true, callSid };
  }

  private extractTokenFromUrl(client: any): string | null {
    const url: string = client?.upgradeReq?.url || client?.url || '';
    const match = url.match(/token=([^&]+)/);
    if (match) return decodeURIComponent(match[1]);
    return null;
  }
}
