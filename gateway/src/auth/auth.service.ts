import { Injectable, Inject, Logger, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { Pool } from 'pg';
import * as bcrypt from 'bcrypt';

export interface JwtPayload {
  sub: string; // User ID
  username?: string;
  email?: string;
  roles?: string[];
  tenant_id?: string;
  isClinician?: boolean;
  iat?: number;
  exp?: number;
}

/** Claims returned on successful credential validation */
export interface UserClaims {
  username: string;
  tenantId: string;
  roles: string[];
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);

  constructor(
    private readonly jwtService: JwtService,
    @Inject('PG_POOL') private readonly pool: Pool | null,
  ) {}

  /**
   * Validate credentials and return user claims.
   * Primary path: users table with bcrypt (works in all envs incl. production).
   * Fallback: DEV_AUTH_USERS env var — non-production only, for first-boot before seed.
   *
   * @param tenantId   Tenant to scope the lookup (defaults to 'default' for single-tenant).
   */
  async validateUser(
    username: string,
    password: string,
    tenantId = 'default',
  ): Promise<UserClaims | false> {
    if (!username || !password) return false;

    // --- Primary path: check the users table with bcrypt ---
    if (this.pool) {
      try {
        const result = await this.pool.query(
          `SELECT password_hash, active, tenant_id, roles FROM users
           WHERE tenant_id = $1 AND username = $2 LIMIT 1`,
          [tenantId, username],
        );
        if (result.rows.length > 0) {
          const row = result.rows[0] as {
            password_hash: string;
            active: boolean;
            tenant_id: string;
            roles: string[];
          };
          if (!row.active) return false;
          const valid = await bcrypt.compare(password, row.password_hash);
          if (!valid) return false;
          return { username, tenantId: row.tenant_id, roles: row.roles || ['user'] };
        }
      } catch (err) {
        this.logger.error('DB lookup in validateUser failed', (err as Error).message);
        // Fall through to env-var path so a DB hiccup doesn't lock everyone out in dev
      }
    }

    // --- Fallback: DEV_AUTH_USERS env var (non-production only) ---
    if (process.env.NODE_ENV === 'production') {
      return false; // Never allow env-var auth in production
    }
    const allowedUsers = (process.env.DEV_AUTH_USERS || 'dev:changeme')
      .split(',')
      .map((p) => p.trim());
    const matched = allowedUsers.some((pair) => {
      const sep = pair.indexOf(':');
      if (sep < 0) return false;
      return pair.slice(0, sep) === username && pair.slice(sep + 1) === password;
    });
    return matched ? { username, tenantId: 'default', roles: ['user', 'clinician'] } : false;
  }

  async validateJwtPayload(payload: JwtPayload): Promise<any> {
    return {
      userId: payload.sub,
      username: payload.username,
      email: payload.email,
      roles: payload.roles || [],
      tenant_id: payload.tenant_id, // forwarded to req.user so TenantGuard can read it
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
      tenant_id: metadata?.tenantId,
    };

    const expiresIn = process.env.JWT_EXPIRES_IN || '1h';
    const accessToken = this.jwtService.sign(payload, { expiresIn });

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

  /**
   * Issue a long-lived refresh token signed with a separate secret.
   * Intended for storage as an httpOnly cookie — never accessible to JS.
   */
  async generateRefreshToken(userId: string): Promise<string> {
    const secret = this.getRefreshSecret();
    const expiresIn = process.env.REFRESH_TOKEN_EXPIRES_IN || '7d';
    return this.jwtService.sign(
      { sub: userId, type: 'refresh' },
      { secret, expiresIn },
    );
  }

  /** Verify a refresh token from the httpOnly cookie. Throws on invalid/expired. */
  async verifyRefreshToken(token: string): Promise<{ sub: string }> {
    const secret = this.getRefreshSecret();
    try {
      const payload = this.jwtService.verify<{ sub: string; type: string }>(
        token,
        { secret },
      );
      if (payload.type !== 'refresh') throw new Error('not a refresh token');
      return { sub: payload.sub };
    } catch {
      throw new UnauthorizedException('Invalid or expired refresh token');
    }
  }

  private getRefreshSecret(): string {
    const jwtSecret = process.env.JWT_SECRET;
    if (!jwtSecret) throw new Error('JWT_SECRET not set');
    // REFRESH_TOKEN_SECRET must differ from JWT_SECRET in production
    return process.env.REFRESH_TOKEN_SECRET || `${jwtSecret}_refresh`;
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
