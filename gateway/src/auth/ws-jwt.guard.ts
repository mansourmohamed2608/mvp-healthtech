import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { createHmac } from 'crypto';

@Injectable()
export class WsJwtGuard implements CanActivate {
  constructor(private readonly jwtService: JwtService) {}

  canActivate(context: ExecutionContext): boolean {
    const client = context.switchToWs().getClient();
    const req = context.switchToWs().getData();
    try {
      const jwtToken = this.extractToken(client, req);
      const { sig, ts } = this.extractSigAndTs(client);
      const url: string = client?.upgradeReq?.url || client?.url || '';
      const callSid = this.extractCallSid(url);
      if (!sig || !ts || !callSid) throw new UnauthorizedException();
      if (!this.validateHmac(sig, ts, callSid)) throw new UnauthorizedException();
      if (jwtToken) {
        const payload = this.jwtService.verify(jwtToken);
        (client as any).user = payload;
      } else {
        (client as any).user = { sub: 'twilio', roles: ['twilio'] };
        (client as any).twilio = true;
      }
      (client as any).callSid = callSid;
      return true;
    } catch (_e) {
      return false; // Keep WS handshake lean; caller will get closed connection
    }
  }

  private extractToken(client: any, data: any): string | null {
    // Expect token in query string as ?token= or in a headers object if provided
    const url: string = client?.upgradeReq?.url || client?.url || '';
    const match = url.match(/token=([^&]+)/);
    if (match) return decodeURIComponent(match[1]);
    if (data && typeof data === 'object' && data.token) return data.token;
    return null;
  }

  private extractSigAndTs(client: any): { sig: string | null; ts: number | null } {
    const url: string = client?.upgradeReq?.url || client?.url || '';
    const sigMatch = url.match(/sig=([^&]+)/);
    const tsMatch = url.match(/ts=([^&]+)/);
    return {
      sig: sigMatch ? decodeURIComponent(sigMatch[1]) : null,
      ts: tsMatch ? Number(tsMatch[1]) : null,
    };
  }

  private extractCallSid(url: string): string | null {
    const match = url.match(/twilio\/ws\/([^?]+)/);
    return match ? match[1] : null;
  }

  /**
   * Twilio-style HMAC: HMAC(secret, `${callSid}:${ts}`)
   * secret comes from TWILIO_AUTH_TOKEN (preferred) or WS_SHARED_SECRET.
   * ts must be within 5 minutes to prevent replay.
   * Token generation example:
   *   ts = Math.floor(Date.now()/1000)
   *   sig = HMAC_SHA256(secret, `${callSid}:${ts}`)
   *   ws://.../twilio/ws/<callSid>?token=<jwt>&sig=<sig>&ts=<ts>
   */
  private validateHmac(sig: string, ts: number | null, callSid: string): boolean {
    const secret = process.env.TWILIO_AUTH_TOKEN || process.env.WS_SHARED_SECRET || '';
    if (!secret || !ts) return false;
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - ts) > 300) return false; // 5 minute window
    const expected = createHmac('sha256', secret).update(`${callSid}:${ts}`).digest('hex');
    return sig === expected;
  }
}
