// gateway/src/auth/auth.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { JwtService } from '@nestjs/jwt';
import { AuthService, JwtPayload } from './auth.service';

describe('AuthService', () => {
  let service: AuthService;
  let jwtService: jest.Mocked<JwtService>;

  beforeEach(async () => {
    const mockJwtService = {
      sign: jest.fn(),
      verify: jest.fn(),
      decode: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AuthService,
        {
          provide: JwtService,
          useValue: mockJwtService,
        },
      ],
    }).compile();

    service = module.get<AuthService>(AuthService);
    jwtService = module.get(JwtService);
  });

  describe('validateUser', () => {
    it('should return true for valid username and password', async () => {
      const result = await service.validateUser('testuser', 'password123');
      expect(result).toBe(true);
    });

    it('should return false for empty username', async () => {
      const result = await service.validateUser('', 'password');
      expect(result).toBe(false);
    });

    it('should return false for empty password', async () => {
      const result = await service.validateUser('user', '');
      expect(result).toBe(false);
    });

    it('should return false for both empty', async () => {
      const result = await service.validateUser('', '');
      expect(result).toBe(false);
    });
  });

  describe('validateJwtPayload', () => {
    it('should return user object from JWT payload', async () => {
      const payload: JwtPayload = {
        sub: 'user-123',
        username: 'drsmith',
        email: 'dr.smith@clinic.com',
        roles: ['clinician', 'admin'],
      };

      const result = await service.validateJwtPayload(payload);

      expect(result.userId).toBe('user-123');
      expect(result.username).toBe('drsmith');
      expect(result.roles).toContain('clinician');
    });

    it('should default roles to empty array', async () => {
      const payload: JwtPayload = {
        sub: 'user-456',
      };

      const result = await service.validateJwtPayload(payload);

      expect(result.roles).toEqual([]);
    });
  });

  describe('login', () => {
    it('should return access token', async () => {
      const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
      jwtService.sign.mockReturnValue(mockToken);

      const result = await service.login({
        sub: 'user-789',
        username: 'nurse1',
      });

      expect(result.access_token).toBe(mockToken);
      expect(jwtService.sign).toHaveBeenCalledWith(
        expect.objectContaining({
          sub: 'user-789',
          username: 'nurse1',
        }),
      );
    });
  });

  describe('generateToken', () => {
    it('should generate token with metadata', async () => {
      const mockToken = 'generated-token';
      jwtService.sign.mockReturnValue(mockToken);

      const result = await service.generateToken('user-abc', {
        username: 'admin',
        email: 'admin@clinic.com',
        roles: ['admin', 'superuser'],
      });

      expect(result.access_token).toBe(mockToken);
      expect(result.token_type).toBe('Bearer');
      expect(result.expires_in).toBeGreaterThan(0);
    });

    it('should use default roles if not provided', async () => {
      jwtService.sign.mockReturnValue('token');

      await service.generateToken('user-def');

      expect(jwtService.sign).toHaveBeenCalledWith(
        expect.objectContaining({
          roles: ['user'],
        }),
      );
    });
  });
});
