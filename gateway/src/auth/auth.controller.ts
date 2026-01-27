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

interface LoginBody {
  userId: string;
  password: string;
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
   * JWT Authentication - Login endpoint
   */
  @Post('login')
  @HttpCode(HttpStatus.OK)
  async login(@Req() req: Request) {
    // DEV-ONLY fallback auth — DISABLED in production
    if (process.env.NODE_ENV === 'production') {
      this.logger.warn('Dev auth login attempt blocked in production');
      throw new UnauthorizedException(
        'Dev auth is disabled in production. Use OIDC.',
      );
    }

    const body = req.body as LoginBody;
    const { userId, password, metadata } = body;
    if (!userId || !password) {
      throw new UnauthorizedException('userId and password are required');
    }

    const allowedUsers = (process.env.DEV_AUTH_USERS || 'dev:changeme')
      .split(',')
      .map((pair) => pair.trim());
    const valid = allowedUsers.some((pair) => {
      const [user, pass] = pair.split(':');
      return user === userId && pass === password;
    });

    if (!valid) {
      throw new UnauthorizedException('Invalid credentials (dev fallback)');
    }

    // Dev auth users get clinician role to access all features in demo
    const devMetadata = {
      ...((metadata as Record<string, unknown>) || {}),
      roles: ['user', 'clinician'],
    };
    const token = await this.authService.generateToken(userId, devMetadata);
    this.logger.log(`JWT token generated for user: ${userId} (DEV MODE)`);
    // For login events, use 'system' tenant since user is authenticating
    await this.auditService.log({
      tenantId: 'system',
      actorId: userId,
      action: 'LOGIN',
      resourceType: 'user',
      resourceId: userId,
      metadata: { method: 'password_dev', roles: metadata?.roles || [] },
    });

    return token;
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

      // Generate JWT token for the authenticated user
      const token = await this.authService.generateToken(
        user.oidcId as string,
        {
          email: user.email,
          name: user.displayName,
          provider: user.provider,
        },
      );
      // For OIDC login, use user's tenant from claims or 'system'
      const userTenantId = (user.tenant_id || user.tenantId || 'system') as string;
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

      // In production, redirect to frontend with token
      const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:5173';
      return res.redirect(
        `${frontendUrl}/auth/callback?token=${token.access_token}`,
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
