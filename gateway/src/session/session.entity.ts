export interface Session {
  sessionId: string;
  userId?: string;
  callSid?: string;
  metadata?: Record<string, any>;
  createdAt: Date;
  expiresAt?: Date;
  isActive: boolean;
}

export interface SessionData {
  sessionId: string;
  userId?: string;
  callSid?: string;
  metadata?: Record<string, any>;
  createdAt: string;
  expiresAt?: string;
  isActive: boolean;
}
