// gateway/src/auth/auth.controller.ts
import {
  Controller,
  Get,
  Post,
  UseGuards,
  Req,
  Res,
  Logger,
  HttpCode,
  HttpStatus,
  UnauthorizedException,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AuthService } from './auth.service';
import type { Request, Response } from 'express';
import { AuditService } from '../audit/audit.service';

/** Parse a named cookie value from a raw Cookie header string (no dependency needed). */
function parseCookie(header: string | undefined, name: string): string | undefined {
  if (!header) return undefined;
  const match = header.split(';').find((p) => p.trim().startsWith(`${name}=`));
  return match ? decodeURIComponent(match.split('=').slice(1).join('=').trim()) : undefined;
}

interface LoginBody {
  userId: string;
  password: string;
  tenantId?: string; // optional; defaults to 'default' for single-tenant deployments
  metadata?: { roles?: string[] };
}

@Controller('auth')
export class AuthController {
  private readonly logger = new Logger(AuthController.name);

  constructor(
    private readonly authService: AuthService,
    private readonly auditService: AuditService,
  ) {}

  /**
   * JWT Authentication - Login endpoint.
   * In production: validates against the users table (bcrypt hash).
   * In dev/test: falls back to DEV_AUTH_USERS env var when DB has no matching user.
   */
  @Post('login')
  @HttpCode(HttpStatus.OK)
  async login(@Req() req: Request, @Res({ passthrough: true }) res: Response) {
    const body = req.body as LoginBody;
    const { userId, password, metadata } = body;
    if (!userId || !password) {
      throw new UnauthorizedException('userId and password are required');
    }

    const valid = await this.authService.validateUser(userId, password);
    if (!valid) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Merge any caller-supplied roles; default to clinician for backward-compat
    const mergedRoles = metadata?.roles?.length ? metadata.roles : ['user', 'clinician'];
    const devMetadata = {
      ...((metadata as Record<string, unknown>) || {}),
      roles: mergedRoles,
    };
    const token = await this.authService.generateToken(userId, devMetadata);
    this.logger.log(`JWT token generated for user: ${userId}`);

    // Issue refresh token as httpOnly cookie (never accessible to JS)
    const refreshToken = await this.authService.generateRefreshToken(userId);
    res.cookie('refresh_token', refreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
      path: '/auth',
    });

    // For login events, use 'system' tenant since user is authenticating
    await this.auditService.log({
      tenantId: 'system',
      actorId: userId,
      action: 'LOGIN',
      resourceType: 'user',
      resourceId: userId,
      metadata: { method: 'password', roles: metadata?.roles || [] },
    });

    return token;
  }

  /**
   * Refresh access token using the httpOnly refresh-token cookie.
   * Implements single-use rotation: issues a new refresh token on every call.
   */
  @Post('refresh')
  @HttpCode(HttpStatus.OK)
  async refresh(@Req() req: Request, @Res({ passthrough: true }) res: Response) {
    const rawCookie = req.headers.cookie as string | undefined;
    const refreshToken = parseCookie(rawCookie, 'refresh_token');
    if (!refreshToken) {
      throw new UnauthorizedException('Refresh token missing');
    }

    const payload = await this.authService.verifyRefreshToken(refreshToken);
    const newAccessToken = await this.authService.generateToken(payload.sub, {});

    // Rotate: replace old refresh token with a new one
    const newRefreshToken = await this.authService.generateRefreshToken(payload.sub);
    res.cookie('refresh_token', newRefreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000,
      path: '/auth',
    });

    return newAccessToken;
  }

  /** Clear refresh token cookie and log the user out. */
  @Post('logout')
  @HttpCode(HttpStatus.NO_CONTENT)
  async logout(@Res({ passthrough: true }) res: Response) {
    res.clearCookie('refresh_token', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/auth',
    });
  }

  /**
   * OIDC Authentication - Initiate login
   */
  @Get('oidc/login')
  @UseGuards(AuthGuard('oidc'))
  async oidcLogin() {
    // This route initiates the OIDC flow
    // User will be redirected to the identity provider
  }

  /**
   * OIDC Authentication - Callback endpoint
   */
  @Get('oidc/callback')
  @UseGuards(AuthGuard('oidc'))
  async oidcCallback(
    @Req() req: Request & { user: Record<string, unknown> },
    @Res() res: Response,
  ) {
    try {
      const user = req.user;
      this.logger.log(`OIDC user authenticated: ${String(user.email)}`);

      // Generate JWT token for the authenticated user — include tenant + roles
      const token = await this.authService.generateToken(
        user.oidcId as string,
        {
          email: user.email,
          name: user.displayName,
          provider: user.provider,
          tenantId: user.tenantId,
          roles: user.roles,
        },
      );
      // For OIDC login, use user's tenant from claims or 'system'
      const userTenantId = (user.tenantId || user.tenant_id || 'system') as string;
      await this.auditService.log({
        tenantId: userTenantId,
        actorId: user.oidcId as string,
        action: 'LOGIN',
        resourceType: 'user',
        resourceId: user.oidcId as string,
        metadata: {
          method: 'oidc',
          provider: user.provider as string,
          roles: (user.roles as string[]) || [],
        },
      });

      // Use hash fragment so the token is not sent in Referer header and
      // is not recorded in server access logs (hash is browser-only).
      const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:5173';
      return res.redirect(
        `${frontendUrl}/auth/callback#token=${token.access_token}`,
      );
    } catch (error) {
      this.logger.error('OIDC callback error', error);
      return res.redirect(
        `${process.env.FRONTEND_URL || 'http://localhost:5173'}/auth/error`,
      );
    }
  }

  /**
   * Get current user info (requires valid JWT or OIDC session)
   */
  @Get('me')
  @UseGuards(AuthGuard('jwt'))
  getCurrentUser(
    @Req() req: Request & { user: { userId: string; metadata: unknown } },
  ) {
    const user = req.user;
    this.logger.log(`User info requested: ${user.userId}`);

    return {
      userId: user.userId,
      metadata: user.metadata,
    };
  }

  /**
   * Health check endpoint
   */
  @Get('health')
  health() {
    return {
      status: 'ok',
      oidcConfigured: !!(
        process.env.OIDC_CLIENT_ID && process.env.OIDC_CLIENT_SECRET
      ),
      jwtConfigured: !!process.env.JWT_SECRET,
    };
  }
}
