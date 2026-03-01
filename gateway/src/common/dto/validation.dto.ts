import {
  IsString,
  IsOptional,
  IsEnum,
  IsArray,
  IsNumber,
  IsBoolean,
  IsNotEmpty,
  MaxLength,
  MinLength,
  Min,
  Max,
  ValidateNested,
  ArrayMaxSize,
  IsUUID,
  Matches,
} from 'class-validator';
import { Type, Transform } from 'class-transformer';

/**
 * Base DTO with common validation rules
 */
export abstract class BaseDto {
  @IsOptional()
  @IsString()
  @IsUUID('4')
  tenantId?: string;

  @IsOptional()
  @IsString()
  @MaxLength(100)
  correlationId?: string;
}

/**
 * Speaker role enum for type safety
 */
export enum SpeakerRole {
  DOCTOR = 'doctor',
  PATIENT = 'patient',
  NURSE = 'nurse',
  UNKNOWN = 'unknown',
}

/**
 * Audio format enum
 */
export enum AudioFormat {
  WAV = 'wav',
  MP3 = 'mp3',
  WEBM = 'webm',
  OGG = 'ogg',
  FLAC = 'flac',
}

/**
 * Language enum for supported languages
 */
export enum Language {
  EN = 'en',
  AR = 'ar',
  AR_EG = 'ar-eg',
  AR_SA = 'ar-sa',
}

/**
 * Transcription request DTO
 */
export class TranscribeRequestDto extends BaseDto {
  @IsOptional()
  @IsEnum(Language)
  language?: Language = Language.EN;

  @IsOptional()
  @IsBoolean()
  enableDiarization?: boolean = true;

  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(10)
  numSpeakers?: number = 2;

  @IsOptional()
  @IsString()
  @MaxLength(500)
  context?: string;

  @IsOptional()
  @IsBoolean()
  identifySpeakerRoles?: boolean = true;
}

/**
 * Speaker segment in transcription
 */
export class SpeakerSegmentDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(50)
  speaker: string;

  @IsEnum(SpeakerRole)
  role: SpeakerRole;

  @IsString()
  @MaxLength(10000)
  text: string;

  @IsNumber()
  @Min(0)
  start: number;

  @IsNumber()
  @Min(0)
  end: number;
}

/**
 * Transcription response DTO
 */
export class TranscriptionResponseDto {
  @IsString()
  @MaxLength(50000)
  transcript: string;

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => SpeakerSegmentDto)
  @ArrayMaxSize(1000)
  segments: SpeakerSegmentDto[];

  @IsNumber()
  @Min(0)
  duration: number;

  @IsEnum(Language)
  language: Language;
}

/**
 * LLM inference request DTO
 */
export class LlmInferRequestDto extends BaseDto {
  @IsString()
  @IsNotEmpty()
  @MinLength(1)
  @MaxLength(50000)
  prompt: string;

  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(4096)
  maxTokens?: number = 1024;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(2)
  temperature?: number = 0.7;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(1)
  topP?: number = 0.9;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(10)
  @IsString({ each: true })
  @MaxLength(50, { each: true })
  stopSequences?: string[];
}

/**
 * Chat message DTO
 */
export class ChatMessageDto {
  @IsEnum(['user', 'assistant', 'system'])
  role: 'user' | 'assistant' | 'system';

  @IsString()
  @IsNotEmpty()
  @MaxLength(50000)
  content: string;
}

/**
 * LLM chat request DTO
 */
export class LlmChatRequestDto extends BaseDto {
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => ChatMessageDto)
  @ArrayMaxSize(100)
  messages: ChatMessageDto[];

  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(4096)
  maxTokens?: number = 1024;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(2)
  temperature?: number = 0.7;

  @IsOptional()
  @IsBoolean()
  stream?: boolean = false;
}

/**
 * SOAP note generation request DTO
 */
export class GenerateSoapRequestDto extends BaseDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(50000)
  transcript: string;

  @IsOptional()
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => SpeakerSegmentDto)
  @ArrayMaxSize(1000)
  segments?: SpeakerSegmentDto[];

  @IsOptional()
  @IsString()
  @MaxLength(500)
  specialty?: string;

  @IsOptional()
  @IsString()
  @IsUUID('4')
  encounterId?: string;

  @IsOptional()
  @IsString()
  @IsUUID('4')
  patientId?: string;

  @IsOptional()
  @IsString()
  @MaxLength(50)
  @Matches(/^[a-zA-Z0-9-]+$/, { message: 'Template must be alphanumeric with dashes only' })
  template?: string = 'default';
}

/**
 * SOAP note response DTO
 */
export class SoapNoteDto {
  @IsString()
  @MaxLength(10000)
  subjective: string;

  @IsString()
  @MaxLength(10000)
  objective: string;

  @IsString()
  @MaxLength(10000)
  assessment: string;

  @IsString()
  @MaxLength(10000)
  plan: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  @ArrayMaxSize(50)
  icdCodes?: string[];

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  @ArrayMaxSize(50)
  cptCodes?: string[];
}

/**
 * TTS synthesis request DTO
 */
export class TtsSynthesizeRequestDto extends BaseDto {
  @IsString()
  @IsNotEmpty()
  @MinLength(1)
  @MaxLength(5000)
  text: string;

  @IsOptional()
  @IsString()
  @MaxLength(100)
  @Matches(/^[a-zA-Z0-9_-]+$/, { message: 'Voice ID must be alphanumeric with underscores/dashes' })
  voice?: string = 'default';

  @IsOptional()
  @IsEnum(Language)
  language?: Language = Language.EN;

  @IsOptional()
  @IsEnum(AudioFormat)
  format?: AudioFormat = AudioFormat.WAV;

  @IsOptional()
  @IsNumber()
  @Min(0.5)
  @Max(2.0)
  speed?: number = 1.0;
}

/**
 * FHIR resource push request DTO
 */
export class FhirPushRequestDto extends BaseDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(100)
  @Matches(/^[A-Z][a-zA-Z]+$/, { message: 'Resource type must be a valid FHIR resource name' })
  resourceType: string;

  @IsObject()
  resource: Record<string, any>;

  @IsOptional()
  @IsString()
  @MaxLength(100)
  patientId?: string;

  @IsOptional()
  @IsString()
  @MaxLength(100)
  encounterId?: string;
}

/**
 * Pagination request DTO
 */
export class PaginationDto {
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(100)
  @Type(() => Number)
  limit?: number = 20;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Type(() => Number)
  offset?: number = 0;

  @IsOptional()
  @IsString()
  @MaxLength(50)
  @Matches(/^[a-zA-Z_]+$/, { message: 'Sort field must be alphabetic with underscores' })
  sortBy?: string;

  @IsOptional()
  @IsEnum(['asc', 'desc'])
  sortOrder?: 'asc' | 'desc' = 'desc';
}

/**
 * Search request DTO
 */
export class SearchRequestDto extends PaginationDto {
  @IsOptional()
  @IsString()
  @MaxLength(200)
  @Transform(({ value }) => value?.trim())
  query?: string;

  @IsOptional()
  @IsString()
  @MaxLength(50)
  dateFrom?: string;

  @IsOptional()
  @IsString()
  @MaxLength(50)
  dateTo?: string;
}

// Helper function for IsObject decorator
function IsObject() {
  return function (target: any, propertyKey: string) {
    // Custom object validator can be added here
  };
}
