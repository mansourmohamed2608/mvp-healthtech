/* eslint-disable @typescript-eslint/require-await */
import { Injectable } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

export interface JwtPayload {
  sub: string;
  username: string;
}

@Injectable()
export class AuthService {
  constructor(private readonly jwtService: JwtService) {}

  /**
   * In a production system you would verify the user against a DB or OIDC provider.
   * For the MVP we accept any username/password combination and return a signed JWT.
   */
  async validateUser(username: string, password: string): Promise<boolean> {
    return !!username && !!password;
  }

  async login(user: {
    sub: string;
    username: string;
  }): Promise<{ access_token: string }> {
    const payload: JwtPayload = { sub: user.sub, username: user.username };
    return { access_token: this.jwtService.sign(payload) };
  }
}
