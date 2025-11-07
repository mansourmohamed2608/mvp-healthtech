export class SessionResponseDto {
  sessionId: string;
  userId?: string;
  callSid?: string;
  metadata?: Record<string, any>;
  createdAt: string;
  expiresAt?: string;
  isActive: boolean;
}

export class CreateSessionResponseDto {
  sessionId: string;
  issuedAt: string;
  expiresAt: string;
}
