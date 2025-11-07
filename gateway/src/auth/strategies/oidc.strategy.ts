// gateway/src/auth/strategies/oidc.strategy.ts
import { Injectable, UnauthorizedException, Logger } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy, StrategyOptions } from 'passport-openidconnect';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class OidcStrategy extends PassportStrategy(Strategy, 'oidc') {
  private readonly logger = new Logger(OidcStrategy.name);

  constructor(private configService: ConfigService) {
    const clientID = configService.get<string>('OIDC_CLIENT_ID') || 'dummy-client-id';
    const clientSecret = configService.get<string>('OIDC_CLIENT_SECRET') || 'dummy-secret';

    const options: StrategyOptions = {
      issuer: configService.get<string>('OIDC_ISSUER') || 'https://accounts.google.com',
      authorizationURL: configService.get<string>('OIDC_AUTHORIZATION_URL') || 'https://accounts.google.com/o/oauth2/v2/auth',
      tokenURL: configService.get<string>('OIDC_TOKEN_URL') || 'https://oauth2.googleapis.com/token',
      userInfoURL: configService.get<string>('OIDC_USERINFO_URL') || 'https://openidconnect.googleapis.com/v1/userinfo',
      clientID,
      clientSecret,
      callbackURL: configService.get<string>('OIDC_CALLBACK_URL') || 'http://localhost:3001/auth/oidc/callback',
      scope: ['openid', 'profile', 'email'],
    };

    super(options);

    if (clientID === 'dummy-client-id' || clientSecret === 'dummy-secret') {
      this.logger.warn('⚠️  OIDC credentials not configured. OIDC authentication disabled (using dummy values).');
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
