import { createClient } from 'redis';

export const redisClient = createClient({
  socket: {
    host: process.env['REDIS_HOST'] || 'localhost',
    port: Number(process.env['REDIS_PORT']) || 6379,
  },
});

redisClient.on('error', (err) => {
  console.error('Redis error:', err);
});
