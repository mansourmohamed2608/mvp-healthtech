/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
// gateway/src/session/session.controller.ts
import { Controller, Get, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { v4 as uuidv4 } from 'uuid';

@Controller('session')
export class SessionController {
  @UseGuards(JwtAuthGuard)
  @Get()
  createSession(@Request() req) {
    // In a real implementation you would store session metadata in Redis or DB
    const sessionId = uuidv4();
    return { sessionId, user: req.user.sub };
  }
}
