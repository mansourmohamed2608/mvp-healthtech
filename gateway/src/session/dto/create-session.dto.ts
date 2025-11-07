import { IsString, IsOptional, IsObject } from 'class-validator';

export class CreateSessionDto {
  @IsString()
  @IsOptional()
  userId?: string;

  @IsString()
  @IsOptional()
  callSid?: string;

  @IsObject()
  @IsOptional()
  metadata?: Record<string, any>;
}
