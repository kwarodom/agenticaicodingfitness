import app from './app';
import { pool } from './config/db';
import { redisClient } from './config/redis';

const PORT = Number(process.env['PORT']) || 3000;

async function start(): Promise<void> {
  await redisClient.connect();
  await pool.query('SELECT 1'); // verify DB connection
  app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
}

start().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
