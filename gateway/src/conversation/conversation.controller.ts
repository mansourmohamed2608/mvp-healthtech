import { Body, Controller, Get, Param, Post, Query, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { Roles } from '../auth/roles.decorator';
import { ConversationService } from './conversation.service';

@UseGuards(JwtAuthGuard)
@Roles('clinician')
@Controller('conversation')
export class ConversationController {
  constructor(private readonly conversationService: ConversationService) {}

  @Get(':sessionId/messages')
  async getMessages(
    @Param('sessionId') sessionId: string,
    @Query('limit') limit?: string,
  ) {
    const parsed = Number(limit || 50);
    const messages = await this.conversationService.getHistory(sessionId, Number.isFinite(parsed) ? parsed : 50);
    return { messages };
  }

  @Post(':sessionId/preferences')
  async updatePreferences(
    @Param('sessionId') sessionId: string,
    @Body() body: { dialect?: string; voice?: string },
  ) {
    const state = await this.conversationService.getState(sessionId);
    const context = (state?.context || {}) as Record<string, any>;
    const preferences = {
      ...(context.preferences || {}),
      ...(body.dialect !== undefined ? { dialect: body.dialect } : {}),
    } as Record<string, any>;
    const voiceRaw = typeof body.voice === 'string' ? body.voice.trim() : body.voice;
    if (voiceRaw === undefined) {
      // keep existing
    } else if (!voiceRaw || voiceRaw.toLowerCase() === 'auto') {
      delete preferences.voice;
    } else {
      preferences.voice = voiceRaw;
    }
    await this.conversationService.updateContext(sessionId, {
      ...context,
      preferences,
    });
    return { ok: true, preferences };
  }
}
