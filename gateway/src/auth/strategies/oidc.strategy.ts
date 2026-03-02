// gateway/src/auth/strategies/oidc.strategy.ts
import { Injectable, UnauthorizedException, Logger, Inject } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy, StrategyOptions } from 'passport-openidconnect';
import { ConfigService } from '@nestjs/config';
import { Pool } from 'pg';

@Injectable()
export class OidcStrategy extends PassportStrategy(Strategy, 'oidc') {
  private readonly logger = new Logger(OidcStrategy.name);

  constructor(
    private configService: ConfigService,
    @Inject('PG_POOL') private readonly pool: Pool | null,
  ) {
    const clientID =
      configService.get<string>('OIDC_CLIENT_ID') || 'dummy-client-id';
    const clientSecret =
      configService.get<string>('OIDC_CLIENT_SECRET') || 'dummy-secret';

    const options: StrategyOptions = {
      issuer:
        configService.get<string>('OIDC_ISSUER') ||
        'https://accounts.google.com',
      authorizationURL:
        configService.get<string>('OIDC_AUTHORIZATION_URL') ||
        'https://accounts.google.com/o/oauth2/v2/auth',
      tokenURL:
        configService.get<string>('OIDC_TOKEN_URL') ||
        'https://oauth2.googleapis.com/token',
      userInfoURL:
        configService.get<string>('OIDC_USERINFO_URL') ||
        'https://openidconnect.googleapis.com/v1/userinfo',
      clientID,
      clientSecret,
      callbackURL:
        configService.get<string>('OIDC_CALLBACK_URL') ||
        'http://localhost:3001/auth/oidc/callback',
      scope: ['openid', 'profile', 'email'],
    };

    super(options);

    if (clientID === 'dummy-client-id' || clientSecret === 'dummy-secret') {
      this.logger.warn(
        '⚠️  OIDC credentials not configured. OIDC authentication disabled (using dummy values).',
      );
    } else {
      this.logger.log('✅ OIDC Strategy initialized with valid credentials');
    }
  }

  async validate(
    issuer: string,
    profile: any,
    done: (error: any, user?: any) => void,
  ): Promise<any> {
    try {
      this.logger.log(`OIDC user validated: ${profile.id}`);

      const email = profile.emails?.[0]?.value as string | undefined;

      // Look up the user in the DB by email to resolve tenant_id and roles.
      // Falls back to safe defaults so OIDC still works before the user record exists.
      let tenantId = 'default';
      let roles: string[] = ['user'];

      if (this.pool && email) {
        try {
          const result = await this.pool.query(
            `SELECT tenant_id, roles FROM users
             WHERE email = $1 AND active = true LIMIT 1`,
            [email],
          );
          if (result.rows.length > 0) {
            const row = result.rows[0] as { tenant_id: string; roles: string[] };
            tenantId = row.tenant_id;
            roles = row.roles || ['user'];
          } else {
            this.logger.warn(`OIDC user ${email} has no matching users row — using defaults`);
          }
        } catch (err) {
          this.logger.error('DB lookup for OIDC user failed', (err as Error).message);
        }
      }

      const user = {
        oidcId: profile.id,
        email,
        displayName: profile.displayName,
        firstName: profile.name?.givenName,
        lastName: profile.name?.familyName,
        photo: profile.photos?.[0]?.value,
        provider: profile.provider,
        tenantId,
        roles,
      };

      return done(null, user);
    } catch (error) {
      this.logger.error('Error validating OIDC user', error);
      return done(new UnauthorizedException('Invalid OIDC token'), false);
    }
  }
}


@Injectable()
export class OidcStrategy extends PassportStrategy(Strategy, 'oidc') {
  private readonly logger = new Logger(OidcStrategy.name);

  constructor(private configService: ConfigService) {
    const clientID =
      configService.get<string>('OIDC_CLIENT_ID') || 'dummy-client-id';
    const clientSecret =
      configService.get<string>('OIDC_CLIENT_SECRET') || 'dummy-secret';

    const options: StrategyOptions = {
      issuer:
        configService.get<string>('OIDC_ISSUER') ||
        'https://accounts.google.com',
      authorizationURL:
        configService.get<string>('OIDC_AUTHORIZATION_URL') ||
        'https://accounts.google.com/o/oauth2/v2/auth',
      tokenURL:
        configService.get<string>('OIDC_TOKEN_URL') ||
        'https://oauth2.googleapis.com/token',
      userInfoURL:
        configService.get<string>('OIDC_USERINFO_URL') ||
        'https://openidconnect.googleapis.com/v1/userinfo',
      clientID,
      clientSecret,
      callbackURL:
        configService.get<string>('OIDC_CALLBACK_URL') ||
        'http://localhost:3001/auth/oidc/callback',
      scope: ['openid', 'profile', 'email'],
    };

    super(options);

    if (clientID === 'dummy-client-id' || clientSecret === 'dummy-secret') {
      this.logger.warn(
        '⚠️  OIDC credentials not configured. OIDC authentication disabled (using dummy values).',
      );
    } else {
      this.logger.log('✅ OIDC Strategy initialized with valid credentials');
    }
  }

  async validate(
    issuer: string,
    profile: any,
    done: (error: any, user?: any) => void,
  ): Promise<any> {
    try {
      this.logger.log(`OIDC user validated: ${profile.id}`);

      // Extract user info from OIDC profile
      const user = {
        oidcId: profile.id,
        email: profile.emails?.[0]?.value,
        displayName: profile.displayName,
        firstName: profile.name?.givenName,
        lastName: profile.name?.familyName,
        photo: profile.photos?.[0]?.value,
        provider: profile.provider,
      };

      return done(null, user);
    } catch (error) {
      this.logger.error('Error validating OIDC user', error);
      return done(new UnauthorizedException('Invalid OIDC token'), false);
    }
  }
}
