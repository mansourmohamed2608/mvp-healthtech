#!/usr/bin/env node
/**
 * scripts/seed-user.js
 *
 * Creates a user in the `users` table with a bcrypt-hashed password.
 * Run this after applying migration 004 to set up dev/demo accounts.
 *
 * Usage:
 *   node scripts/seed-user.js --username dev --password changeme --roles clinician,admin
 *   node scripts/seed-user.js --username nurse1 --password securepass123 --tenant clinic-a
 *
 * Requires:
 *   DATABASE_URL env var (or set inline: DATABASE_URL=postgres://... node scripts/seed-user.js ...)
 *   npm packages: pg, bcrypt  (already in gateway/package.json — run from repo root or gateway/)
 */

'use strict';

const { Pool } = require('pg');
const bcrypt = require('bcrypt');

// --- Parse CLI args -----------------------------------------------------------
const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, arr) => {
    if (arg.startsWith('--')) pairs.push([arg.slice(2), arr[i + 1]]);
    return pairs;
  }, []),
);

const username  = args.username  || 'dev';
const password  = args.password  || 'changeme';
const email     = args.email     || null;
const tenantId  = args.tenant    || 'default';
const rolesRaw  = args.roles     || 'clinician';
const roles     = rolesRaw.split(',').map((r) => r.trim());
const active    = args.active !== 'false';
const COST      = 12; // bcrypt work factor — increase to 14 for production

// --- Validate -----------------------------------------------------------------
if (!process.env.DATABASE_URL) {
  console.error('ERROR: DATABASE_URL environment variable not set.');
  console.error('  Example: DATABASE_URL=postgres://postgres:pass@localhost:5432/healthtech node scripts/seed-user.js');
  process.exit(1);
}

if (password === 'changeme' && process.env.NODE_ENV === 'production') {
  console.error('ERROR: default password "changeme" is not allowed in production. Pass --password <strong-password>.');
  process.exit(1);
}

// --- Main ---------------------------------------------------------------------
(async () => {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 1 });
  try {
    console.log(`Hashing password for "${username}" (cost=${COST}) — this takes ~1 second...`);
    const password_hash = await bcrypt.hash(password, COST);

    const result = await pool.query(
      `INSERT INTO users (tenant_id, username, email, password_hash, roles, active)
       VALUES ($1, $2, $3, $4, $5, $6)
       ON CONFLICT (tenant_id, username)
       DO UPDATE SET
         password_hash = EXCLUDED.password_hash,
         email         = EXCLUDED.email,
         roles         = EXCLUDED.roles,
         active        = EXCLUDED.active,
         updated_at    = now()
       RETURNING id, tenant_id, username, email, roles, active, created_at`,
      [tenantId, username, email, password_hash, roles, active],
    );

    const user = result.rows[0];
    console.log('✅  User upserted successfully:');
    console.log(`    id:        ${user.id}`);
    console.log(`    tenant:    ${user.tenant_id}`);
    console.log(`    username:  ${user.username}`);
    console.log(`    email:     ${user.email ?? '(none)'}`);
    console.log(`    roles:     ${user.roles.join(', ')}`);
    console.log(`    active:    ${user.active}`);
    console.log(`    created:   ${user.created_at}`);
  } catch (err) {
    console.error('ERROR seeding user:', err.message);
    process.exit(1);
  } finally {
    await pool.end();
  }
})();
