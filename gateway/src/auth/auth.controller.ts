// gateway/src/auth/auth.controller.ts
import { Controller, Get, Post, UseGuards, Req, Res, Logger, HttpCode, HttpStatus } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AuthService } from './auth.service';
import type { Request, Response } from 'express';

@Controller('auth')
export class AuthController {
  private readonly logger = new Logger(AuthController.name);

  constructor(private readonly authService: AuthService) {}

  /**
   * JWT Authentication - Login endpoint
   */
  @Post('login')
  @HttpCode(HttpStatus.OK)
  async login(@Req() req: Request) {
    const { userId, metadata } = req.body;
    
    if (!userId) {
      return { error: 'userId is required' };
    }

    const token = await this.authService.generateToken(userId, metadata);
    this.logger.log(`JWT token generated for user: ${userId}`);
    
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
