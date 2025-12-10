// gateway/src/auth/auth.controller.ts
import { Controller, Get, Post, UseGuards, Req, Res, Logger, HttpCode, HttpStatus, UnauthorizedException } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AuthService } from './auth.service';
import type { Request, Response } from 'express';
import { AuditService } from '../audit/audit.service';

@Controller('auth')
export class AuthController {
  private readonly logger = new Logger(AuthController.name);

  constructor(private readonly authService: AuthService, private readonly auditService: AuditService) {}

  /**
   * JWT Authentication - Login endpoint
   */
  @Post('login')
  @HttpCode(HttpStatus.OK)
  async login(@Req() req: Request) {
    // DEV-ONLY fallback auth. Replace with real IdP/OIDC before production.
    const { userId, password, metadata } = req.body as any;
    if (!userId || !password) {
      throw new UnauthorizedException('userId and password are required');
    }

    const allowedUsers = (process.env.DEV_AUTH_USERS || 'dev:changeme').split(',').map((pair) => pair.trim());
    const valid = allowedUsers.some((pair) => {
      const [user, pass] = pair.split(':');
      return user === userId && pass === password;
    });

    if (!valid) {
      throw new UnauthorizedException('Invalid credentials (dev fallback)');
    }

    const token = await this.authService.generateToken(userId, metadata);
    this.logger.log(`JWT token generated for user: ${userId}`);
    await this.auditService.log({
      actorId: userId,
      action: 'LOGIN',
      resourceType: 'user',
      resourceId: userId,
      metadata: { method: 'password', roles: metadata?.roles || [] },
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
  async oidcCallback(@Req() req: any, @Res() res: Response) {
    try {
      const user = req.user;
      this.logger.log(`OIDC user authenticated: ${user.email}`);

      // Generate JWT token for the authenticated user
      const token = await this.authService.generateToken(user.oidcId, {
        email: user.email,
        name: user.displayName,
        provider: user.provider,
      });
      await this.auditService.log({
        actorId: user.oidcId,
        action: 'LOGIN',
        resourceType: 'user',
        resourceId: user.oidcId,
        metadata: { method: 'oidc', provider: user.provider, roles: user.roles || [] },
      });

      // In production, redirect to frontend with token
      const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:5173';
      return res.redirect(`${frontendUrl}/auth/callback?token=${token.access_token}`);
    } catch (error) {
      this.logger.error('OIDC callback error', error);
      return res.redirect(`${process.env.FRONTEND_URL || 'http://localhost:5173'}/auth/error`);
    }
  }

  /**
   * Get current user info (requires valid JWT or OIDC session)
   */
  @Get('me')
  @UseGuards(AuthGuard('jwt'))
  async getCurrentUser(@Req() req: any) {
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
      oidcConfigured: !!(process.env.OIDC_CLIENT_ID && process.env.OIDC_CLIENT_SECRET),
      jwtConfigured: !!process.env.JWT_SECRET,
    };
  }
}
