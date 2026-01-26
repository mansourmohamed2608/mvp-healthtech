import { JwtService } from '@nestjs/jwt';
import { createHmac } from 'crypto';
import { WsJwtGuard } from './ws-jwt.guard';
import { ExecutionContext } from '@nestjs/common';

const buildContext = (url: string): ExecutionContext =>
  ({
    switchToWs: () => ({
      getClient: () => ({ url }),
      getData: () => ({}),
    }),
  }) as unknown as ExecutionContext;

describe('WsJwtGuard (HMAC + JWT)', () => {
  const jwtSecret = 'test-jwt';
  const wsSecret = 'test-ws';
  const callSid = 'CA123';
  let guard: WsJwtGuard;
  let jwtService: JwtService;

  beforeAll(() => {
    process.env.JWT_SECRET = jwtSecret;
    process.env.WS_SHARED_SECRET = wsSecret;
    jwtService = new JwtService({ secret: jwtSecret });
    guard = new WsJwtGuard(jwtService);
  });

  const makeUrl = (sig: string, ts: number, token: string) =>
    `/twilio/ws/${callSid}?token=${encodeURIComponent(token)}&sig=${encodeURIComponent(sig)}&ts=${ts}`;

  it('allows valid token + sig within window', () => {
    const ts = Math.floor(Date.now() / 1000);
    const sig = createHmac('sha256', wsSecret)
      .update(`${callSid}:${ts}`)
      .digest('hex');
    const token = jwtService.sign({ sub: 'user1' });
    const ctx = buildContext(makeUrl(sig, ts, token));
    expect(guard.canActivate(ctx)).toBe(true);
  });

  it('rejects invalid signature', () => {
    const ts = Math.floor(Date.now() / 1000);
    const token = jwtService.sign({ sub: 'user1' });
    const ctx = buildContext(makeUrl('badsig', ts, token));
    expect(guard.canActivate(ctx)).toBe(false);
  });

  it('rejects expired timestamp', () => {
    const ts = Math.floor(Date.now() / 1000) - 10000; // beyond 5 minute window
    const sig = createHmac('sha256', wsSecret)
      .update(`${callSid}:${ts}`)
      .digest('hex');
    const token = jwtService.sign({ sub: 'user1' });
    const ctx = buildContext(makeUrl(sig, ts, token));
    expect(guard.canActivate(ctx)).toBe(false);
  });

  it('rejects missing params', () => {
    const ctx = buildContext(`/twilio/ws/${callSid}`);
    expect(guard.canActivate(ctx)).toBe(false);
  });
});
