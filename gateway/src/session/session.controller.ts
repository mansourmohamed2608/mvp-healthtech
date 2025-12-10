// gateway/src/session/session.controller.ts
import {
  Controller,
  Post,
  Get,
  Delete,
  Param,
  Body,
  HttpCode,
  HttpStatus,
  Patch,
  UseGuards,
  Request,
} from '@nestjs/common';
import { SessionService } from './session.service';
import { CreateSessionDto } from './dto/create-session.dto';
import {
  SessionResponseDto,
  CreateSessionResponseDto,
} from './dto/session-response.dto';
import { JwtAuthGuard } from '../auth/jwt.guard';

@UseGuards(JwtAuthGuard)
@Controller('sessions')
export class SessionController {
  constructor(private readonly sessionService: SessionService) {}

  @Post()
  @HttpCode(HttpStatus.CREATED)
  async create(
    @Body() createSessionDto: CreateSessionDto,
  ): Promise<CreateSessionResponseDto> {
    return this.sessionService.create(createSessionDto);
  }

  @Get(':id')
  async findOne(@Param('id') id: string): Promise<SessionResponseDto> {
    return this.sessionService.findById(id);
  }

  @Patch(':id/extend')
  async extend(@Param('id') id: string): Promise<SessionResponseDto> {
    return this.sessionService.extendSession(id);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  async remove(@Param('id') id: string): Promise<void> {
    return this.sessionService.delete(id);
  }

  @Get(':id/status')
  async getStatus(@Param('id') id: string): Promise<{ isActive: boolean }> {
    const isActive = await this.sessionService.isActive(id);
    return { isActive };
  }

  // Protected endpoint example - requires JWT
  @UseGuards(JwtAuthGuard)
  @Post('authenticated')
  @HttpCode(HttpStatus.CREATED)
  async createAuthenticated(
    @Request() req: any,
    @Body() createSessionDto: CreateSessionDto,
  ): Promise<CreateSessionResponseDto> {
    const dto: CreateSessionDto = {
      ...createSessionDto,
      userId: req.user.sub,
    };
    return this.sessionService.create(dto);
  }
}
