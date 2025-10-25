/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
import { Injectable } from '@nestjs/common';
import { createClient } from 'redis';

@Injectable()
export class ConversationService {
  private readonly client = createClient({
    url: `redis://${process.env.REDIS_HOST}:${process.env.REDIS_PORT}`,
  });
  constructor() {
    this.client.connect();
  }
  async appendMessage(sessionId: string, role: string, text: string) {
    await this.client.rPush(
      `conv:${sessionId}`,
      JSON.stringify({ role, text }),
    );
  }
  async getMessages(sessionId: string): Promise<any[]> {
    const items = await this.client.lRange(`conv:${sessionId}`, 0, -1);
    return items.map((x) => JSON.parse(x));
  }
  async clear(sessionId: string) {
    await this.client.del(`conv:${sessionId}`);
  }
}
