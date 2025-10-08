// import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common';
// import * as jwt from 'jsonwebtoken';
// @Injectable()
// export class JwtGuard implements CanActivate {
//   canActivate(context: ExecutionContext): boolean {
//     const req = context.switchToHttp().getRequest();
//     const token = (req.headers.authorization || '').replace('Bearer ', '');
//     try {
//       jwt.verify(token, process.env.JWT_SECRET || 'devsecret');
//       return true;
//     } catch {
//       throw new UnauthorizedException();
//     }
//   }
// }
