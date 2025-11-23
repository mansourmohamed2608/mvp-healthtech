import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { AuthService } from './auth.service';
import { JwtStrategy } from './jwt.strategy';
import { OidcStrategy } from './strategies/oidc.strategy';
import { AuthController } from './auth.controller';
import { WsJwtGuard } from './ws-jwt.guard';

@Module({
  imports: [
    PassportModule,
    JwtModule.register({
      secret: (() => {
        if (!process.env.JWT_SECRET) throw new Error('JWT_SECRET not set');
        return process.env.JWT_SECRET;
      })(),
      signOptions: { expiresIn: '1h' },
    }),
  ],
  controllers: [AuthController],
  providers: [AuthService, JwtStrategy, OidcStrategy, WsJwtGuard],
  exports: [AuthService, WsJwtGuard],
})
export class AuthModule {}
