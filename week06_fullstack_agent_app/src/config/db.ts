import { Pool } from 'pg';

export const pool = new Pool({
  host: process.env['DB_HOST'] || 'localhost',
  port: Number(process.env['DB_PORT']) || 5432,
  database: process.env['DB_NAME'] || 'ecommerce_dashboard',
  user: process.env['DB_USER'] || 'postgres',
  password: process.env['DB_PASSWORD'] || '',
});

pool.on('error', (err) => {
  console.error('Unexpected database error', err);
  process.exit(1);
});
