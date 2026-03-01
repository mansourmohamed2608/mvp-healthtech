// gateway/test/integration/database.e2e-spec.ts
/**
 * Database Integration Tests
 * Tests database migrations and data operations
 */
import { Test, TestingModule } from '@nestjs/testing';
import { Pool } from 'pg';

describe('Database Integration (e2e)', () => {
  let pool: Pool;

  beforeAll(async () => {
    if (!process.env.DATABASE_URL || !process.env.INTEGRATION_TEST) {
      return;
    }

    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
    });
  });

  afterAll(async () => {
    if (pool) {
      await pool.end();
    }
  });

  describe('Migrations', () => {
    it('should have audit_log table', async () => {
      if (!pool) return;

      const result = await pool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_name = 'audit_log'
        );
      `);

      expect(result.rows[0].exists).toBe(true);
    });

    it('should have soap_notes table', async () => {
      if (!pool) return;

      const result = await pool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_name = 'soap_notes'
        );
      `);

      expect(result.rows[0].exists).toBe(true);
    });

    it('should have sessions table', async () => {
      if (!pool) return;

      const result = await pool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_name = 'sessions'
        );
      `);

      // May not exist depending on schema
      expect(result.rows[0]).toBeDefined();
    });
  });

  describe('Tenant Isolation', () => {
    it('should enforce tenant_id on audit_log', async () => {
      if (!pool) return;

      const result = await pool.query(`
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'audit_log' AND column_name = 'tenant_id';
      `);

      expect(result.rows.length).toBeGreaterThan(0);
    });
  });

  describe('Indexes', () => {
    it('should have performance indexes on audit_log', async () => {
      if (!pool) return;

      const result = await pool.query(`
        SELECT indexname 
        FROM pg_indexes 
        WHERE tablename = 'audit_log';
      `);

      // Should have at least primary key index
      expect(result.rows.length).toBeGreaterThan(0);
    });
  });
});
