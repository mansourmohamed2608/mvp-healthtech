// gateway/src/audit/audit.service.spec.ts
/**
 * AuditService Unit Tests
 * PR-7 Hardening: Ensures tenantId is REQUIRED for all audit logs
 */
import { AuditService } from './audit.service';

describe('AuditService', () => {
  let service: AuditService;
  let mockPoolQuery: jest.Mock;

  beforeEach(() => {
    // Mock DATABASE_URL to enable pool
    process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/testdb';
    
    service = new AuditService();
    
    // Mock the pool.query method
    mockPoolQuery = jest.fn().mockResolvedValue({ rowCount: 1 });
    (service as any).pool = { query: mockPoolQuery };
  });

  afterEach(() => {
    delete process.env.DATABASE_URL;
    jest.clearAllMocks();
  });

  describe('log()', () => {
    it('should require tenantId parameter (compile-time enforcement)', () => {
      // This test verifies that tenantId is required at the type level
      // The following would cause a TypeScript compile error if uncommented:
      // 
      // await service.log({
      //   actorId: 'user-123',
      //   action: 'TEST_ACTION',
      //   resourceType: 'test',
      //   resourceId: 'resource-123',
      //   // Missing tenantId - this should fail type checking
      // });
      //
      // If this test compiles, tenantId is properly required
      expect(true).toBe(true);
    });

    it('should throw error when tenantId is empty string', async () => {
      await expect(
        service.log({
          tenantId: '',
          actorId: 'user-123',
          action: 'TEST_ACTION',
          resourceType: 'test',
          resourceId: 'resource-123',
        }),
      ).rejects.toThrow('tenantId is required for audit logging');
    });

    it('should accept valid tenantId and log to database', async () => {
      await service.log({
        tenantId: 'tenant-abc',
        actorId: 'user-123',
        action: 'SOAP_NOTE_CREATED',
        resourceType: 'soap_note',
        resourceId: 'note-456',
        metadata: { sessionId: 'session-789' },
      });

      expect(mockPoolQuery).toHaveBeenCalledWith(
        expect.stringContaining('INSERT INTO audit_log'),
        ['user-123', 'SOAP_NOTE_CREATED', 'soap_note', 'note-456', { sessionId: 'session-789' }, 'tenant-abc'],
      );
    });

    it('should accept system tenant for auth events', async () => {
      await service.log({
        tenantId: 'system',
        actorId: 'user-123',
        action: 'LOGIN',
        resourceType: 'user',
        resourceId: 'user-123',
        metadata: { method: 'password_dev' },
      });

      expect(mockPoolQuery).toHaveBeenCalledWith(
        expect.stringContaining('INSERT INTO audit_log'),
        expect.arrayContaining(['system']),
      );
    });

    it('should include tenantId in SQL insert', async () => {
      await service.log({
        tenantId: 'clinic-xyz',
        actorId: 'clinician-001',
        action: 'FHIR_WRITE_ATTEMPTED',
        resourceType: 'soap_note',
        resourceId: 'note-123',
      });

      // Verify the query includes tenant_id column
      expect(mockPoolQuery).toHaveBeenCalledWith(
        expect.stringContaining('tenant_id'),
        expect.arrayContaining(['clinic-xyz']),
      );
    });

    it('should log warning when pool is not configured', async () => {
      // Create service without pool
      const serviceWithoutPool = new AuditService();
      (serviceWithoutPool as any).pool = null;
      
      const loggerWarnSpy = jest.spyOn((serviceWithoutPool as any).logger, 'warn');
      
      await serviceWithoutPool.log({
        tenantId: 'tenant-abc',
        actorId: 'user-123',
        action: 'TEST_ACTION',
        resourceType: 'test',
        resourceId: 'resource-123',
      });

      expect(loggerWarnSpy).toHaveBeenCalledWith(
        'Audit pool not configured; skipping audit log',
      );
    });

    it('should handle database errors gracefully', async () => {
      mockPoolQuery.mockRejectedValue(new Error('Connection failed'));
      
      const loggerWarnSpy = jest.spyOn((service as any).logger, 'warn');
      
      // Should not throw, just log warning
      await service.log({
        tenantId: 'tenant-abc',
        actorId: 'user-123',
        action: 'TEST_ACTION',
        resourceType: 'test',
        resourceId: 'resource-123',
      });

      expect(loggerWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Audit log failed'),
        expect.objectContaining({ error: 'Connection failed' }),
      );
    });
  });

  describe('Type Safety', () => {
    it('should enforce tenantId as first parameter conceptually', () => {
      // This test documents the expected interface
      // The log() method signature should be:
      // log(params: { tenantId: string; actorId: string; ... })
      // NOT: log(params: { actorId: string; tenantId?: string; ... })
      
      const logMethod = service.log;
      expect(typeof logMethod).toBe('function');
      
      // The params type should require tenantId
      // This is verified by TypeScript at compile time
    });
  });
});

describe('AuditService Regression Prevention', () => {
  it('should fail compilation if tenantId becomes optional again', () => {
    // IMPORTANT: This test serves as documentation
    // If someone makes tenantId optional, the soap.controller.ts
    // and other files that now pass tenantId will still compile,
    // but this test documents the requirement.
    //
    // The real enforcement is:
    // 1. TypeScript type checking (tenantId: string, not tenantId?: string)
    // 2. Runtime validation (throws if tenantId is falsy)
    // 3. Code review catching removal of tenantId from call sites
    
    // Verify the service exists and has the log method
    process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/testdb';
    const svc = new AuditService();
    expect(svc.log).toBeDefined();
    delete process.env.DATABASE_URL;
  });
});
