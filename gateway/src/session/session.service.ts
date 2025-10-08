import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
@Injectable()
export class SessionService {
  create() {
    return { sessionId: randomUUID(), issuedAt: new Date().toISOString() };
  }
}
