// Set required env vars for tests
process.env.JWT_SECRET = process.env.JWT_SECRET || 'test-jwt-secret';
process.env.INTERNAL_SECRET = process.env.INTERNAL_SECRET || 'test-internal-secret';

// Mock pg and uuid globally (ts-jest will pick up from moduleNameMapper)
jest.mock('pg');
jest.mock('uuid');
jest.mock('redis', () => ({
  createClient: jest.fn(() => ({
    on: jest.fn(),
    connect: jest.fn().mockResolvedValue(null),
    quit: jest.fn(),
    lRange: jest.fn().mockResolvedValue([]),
    exists: jest.fn().mockResolvedValue(0),
    set: jest.fn(),
    get: jest.fn().mockResolvedValue(null),
    del: jest.fn(),
  })),
}));

// Silence noisy logs during tests
jest.spyOn(console, 'warn').mockImplementation(() => {});
jest.spyOn(console, 'error').mockImplementation(() => {});
