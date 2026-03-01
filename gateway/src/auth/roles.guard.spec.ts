// gateway/src/auth/roles.guard.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { RolesGuard } from './roles.guard';
import { ROLES_KEY } from './roles.decorator';

describe('RolesGuard', () => {
  let guard: RolesGuard;
  let reflector: jest.Mocked<Reflector>;

  const createMockContext = (user: any): ExecutionContext => {
    return {
      switchToHttp: () => ({
        getRequest: () => ({ user }),
      }),
      getHandler: () => jest.fn(),
      getClass: () => jest.fn(),
    } as unknown as ExecutionContext;
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        RolesGuard,
        {
          provide: Reflector,
          useValue: {
            getAllAndOverride: jest.fn(),
          },
        },
      ],
    }).compile();

    guard = module.get<RolesGuard>(RolesGuard);
    reflector = module.get(Reflector);
  });

  it('should be defined', () => {
    expect(guard).toBeDefined();
  });

  describe('canActivate', () => {
    it('should return true when no roles are required', () => {
      reflector.getAllAndOverride.mockReturnValue(undefined);

      const context = createMockContext({ userId: 'user1', roles: [] });
      const result = guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should return true when required roles is empty array', () => {
      reflector.getAllAndOverride.mockReturnValue([]);

      const context = createMockContext({ userId: 'user1', roles: ['admin'] });
      const result = guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should return true when user has required role', () => {
      reflector.getAllAndOverride.mockReturnValue(['clinician']);

      const context = createMockContext({
        userId: 'user1',
        roles: ['clinician', 'viewer'],
      });
      const result = guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should return true when user has one of multiple required roles', () => {
      reflector.getAllAndOverride.mockReturnValue(['admin', 'superuser']);

      const context = createMockContext({
        userId: 'user1',
        roles: ['admin'],
      });
      const result = guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should return false when user lacks required role', () => {
      reflector.getAllAndOverride.mockReturnValue(['admin']);

      const context = createMockContext({
        userId: 'user1',
        roles: ['viewer'],
      });
      const result = guard.canActivate(context);

      expect(result).toBe(false);
    });

    it('should return false when no user is present', () => {
      reflector.getAllAndOverride.mockReturnValue(['admin']);

      const context = createMockContext(null);
      const result = guard.canActivate(context);

      expect(result).toBe(false);
    });

    it('should return false when user has no roles array', () => {
      reflector.getAllAndOverride.mockReturnValue(['clinician']);

      const context = createMockContext({
        userId: 'user1',
        // No roles property
      });
      const result = guard.canActivate(context);

      expect(result).toBe(false);
    });

    it('should handle user with empty roles array', () => {
      reflector.getAllAndOverride.mockReturnValue(['admin']);

      const context = createMockContext({
        userId: 'user1',
        roles: [],
      });
      const result = guard.canActivate(context);

      expect(result).toBe(false);
    });
  });
});
