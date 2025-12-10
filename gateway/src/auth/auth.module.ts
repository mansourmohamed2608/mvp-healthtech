import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { AuthService } from './auth.service';
import { JwtStrategy } from './jwt.strategy';
import { OidcStrategy } from './strategies/oidc.strategy';
import { AuthController } from './auth.controller';
import { WsJwtGuard } from './ws-jwt.guard';
import { AuditService } from '../audit/audit.service';

@Module({
  imports: [
    ConfigModule,
    PassportModule,
    JwtModule.registerAsync({
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: (configService: ConfigService) => {
        const secret = configService.get<string>('JWT_SECRET');
        if (!secret) throw new Error('JWT_SECRET not set');
        return {
          secret,
          signOptions: { expiresIn: '1h' },
        };
      },
    }),
  ],
  controllers: [AuthController],
  providers: [AuthService, JwtStrategy, OidcStrategy, WsJwtGuard, AuditService],
  exports: [AuthService, WsJwtGuard, JwtModule, AuditService],
})
export class AuthModule {}
