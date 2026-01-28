import {
  CanActivate,
  ExecutionContext,
  Injectable,
  Logger,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ROLES_KEY } from './roles.decorator';

@Injectable()
export class RolesGuard implements CanActivate {
  private readonly logger = new Logger(RolesGuard.name);

  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredRoles = this.reflector.getAllAndOverride<string[]>(
      ROLES_KEY,
      [context.getHandler(), context.getClass()],
    );
    if (!requiredRoles || requiredRoles.length === 0) {
      return true;
    }
    const request = context.switchToHttp().getRequest();
    const user = request.user;
    if (!user) {
      this.logger.warn(
        `RolesGuard: No user on request, required roles: ${requiredRoles.join(', ')}`,
      );
      return false;
    }
    const roles = user.roles || [];
    const hasRole = requiredRoles.some((r) => roles.includes(r));
    if (!hasRole) {
      this.logger.warn(
        `RolesGuard: User ${user.userId || user.sub} has roles [${roles.join(', ')}] but needs one of [${requiredRoles.join(', ')}]`,
      );
    }
    return hasRole;
  }
}
