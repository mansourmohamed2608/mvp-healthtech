// gateway/src/db/db.module.ts
/**
 * DbModule — single global Pool shared across all services.
 *
 * Replaces the pattern of each controller/service constructing its own
 * `new Pool({ connectionString: process.env.DATABASE_URL })`, which leads to
 * unbounded connection growth under load.
 *
 * Inject the pool with:  @Inject(PG_POOL) private readonly pool: Pool
 * Inject tenant-aware queries: @Inject(TENANT_DB) private readonly db: TenantDbService
 */
import { Global, Module, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Pool } from 'pg';
import { TenantDbService } from './tenant-db.service';
import { PG_POOL, TENANT_DB } from './db.constants';

export { PG_POOL } from './db.constants';

@Global()
@Module({
  providers: [
    {
      provide: PG_POOL,
      inject: [ConfigService],
      useFactory: (config: ConfigService) => {
        const connectionString = config.get<string>('DATABASE_URL');
        if (!connectionString) {
          new Logger('DbModule').warn(
            'DATABASE_URL not set — pg Pool will not be initialised',
          );
          return null;
        }
        const pool = new Pool({
          connectionString,
          max: 20,                   // max simultaneous connections per gateway instance
          idleTimeoutMillis: 30_000, // release idle connections after 30 s
          connectionTimeoutMillis: 5_000,
        });
        pool.on('error', (err) =>
          new Logger('PgPool').error('Unexpected pool client error', err.message),
        );
        return pool;
      },
    },
    TenantDbService,
    {
      provide: TENANT_DB,
      useExisting: TenantDbService,
    },
  ],
  exports: [PG_POOL, TenantDbService, TENANT_DB],
})
export class DbModule {}
