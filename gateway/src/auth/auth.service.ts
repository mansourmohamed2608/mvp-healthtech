import { Injectable, Logger } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

export interface JwtPayload {
  sub: string; // User ID
  username?: string;
  email?: string;
  roles?: string[];
  isClinician?: boolean;
  iat?: number;
  exp?: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);

  constructor(private readonly jwtService: JwtService) {}

  /**
   * In a production system you would verify the user against a DB or OIDC provider.
   * For the MVP we accept any username/password combination and return a signed JWT.
   */
  async validateUser(username: string, password: string): Promise<boolean> {
    return !!username && !!password;
  }

  async validateJwtPayload(payload: JwtPayload): Promise<any> {
    // In production, validate against user database
    // For now, just return the payload
    return {
      userId: payload.sub,
      username: payload.username,
      email: payload.email,
      roles: payload.roles || [],
    };
  }

  async login(user: {
    sub: string;
    username: string;
  }): Promise<{ access_token: string }> {
    const payload: JwtPayload = { sub: user.sub, username: user.username };
    return { access_token: this.jwtService.sign(payload) };
  }

  async generateToken(userId: string, metadata?: any): Promise<TokenResponse> {
    const payload: JwtPayload = {
      sub: userId,
      username: metadata?.username,
      email: metadata?.email,
      roles: metadata?.roles || ['user'],
    };

    const expiresIn = process.env.JWT_EXPIRES_IN || '7d';
    const accessToken = this.jwtService.sign(payload);

    this.logger.log(`Generated token for user: ${userId}`);

    return {
      access_token: accessToken,
      token_type: 'Bearer',
      expires_in: this.parseExpiresIn(expiresIn),
    };
  }

  async verifyToken(token: string): Promise<JwtPayload> {
    try {
      return this.jwtService.verify(token);
    } catch (error) {
      this.logger.error('Token verification failed', error);
      throw error;
    }
  }

  private parseExpiresIn(expiresIn: string): number {
    // Convert JWT expiration string to seconds
    const match = expiresIn.match(/^(\d+)([dhms])$/);
    if (!match) return 604800; // Default 7 days

    const value = parseInt(match[1], 10);
    const unit = match[2];

    switch (unit) {
      case 'd':
        return value * 86400;
      case 'h':
        return value * 3600;
      case 'm':
        return value * 60;
      case 's':
        return value;
      default:
        return 604800;
    }
  }
}
