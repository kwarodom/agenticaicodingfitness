import { Request, Response, NextFunction } from 'express';
import { pool } from '../config/db';
import { redisClient } from '../config/redis';

const CACHE_TTL = 300; // 5 minutes

async function getCached<T>(key: string, fetch: () => Promise<T>): Promise<T> {
  const cached = await redisClient.get(key);
  if (cached) return JSON.parse(cached) as T;
  const data = await fetch();
  await redisClient.setEx(key, CACHE_TTL, JSON.stringify(data));
  return data;
}

export async function getStats(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const range = (req.query['range'] as string) || '30d';
    const days = range === '7d' ? 7 : range === '90d' ? 90 : 30;
    const cacheKey = `dashboard:stats:${days}`;

    const stats = await getCached(cacheKey, async () => {
      const [revenue, orders, customers, topProducts] = await Promise.all([
        pool.query(
          `SELECT
             COALESCE(SUM(total), 0) as total_revenue,
             COALESCE(SUM(total) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day' * $1 / 2), 0) as recent_revenue,
             COALESCE(SUM(total) FILTER (WHERE created_at < NOW() - INTERVAL '1 day' * $1 / 2 AND created_at >= NOW() - INTERVAL '1 day' * $1), 0) as previous_revenue
           FROM orders
           WHERE status != 'cancelled' AND created_at >= NOW() - INTERVAL '1 day' * $1`,
          [days]
        ),
        pool.query(
          `SELECT
             COUNT(*) as total,
             COUNT(*) FILTER (WHERE status = 'pending') as pending,
             COUNT(*) FILTER (WHERE status = 'processing') as processing,
             COUNT(*) FILTER (WHERE status = 'shipped') as shipped,
             COUNT(*) FILTER (WHERE status = 'delivered') as delivered,
             COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled
           FROM orders
           WHERE created_at >= NOW() - INTERVAL '1 day' * $1`,
          [days]
        ),
        pool.query(
          `SELECT COUNT(*) as new_customers
           FROM users
           WHERE role = 'user' AND created_at >= NOW() - INTERVAL '1 day' * $1`,
          [days]
        ),
        pool.query(
          `SELECT p.id, p.name, SUM(oi.quantity) as units_sold, SUM(oi.quantity * oi.unit_price) as revenue
           FROM order_items oi
           JOIN products p ON oi.product_id = p.id
           JOIN orders o ON oi.order_id = o.id
           WHERE o.status != 'cancelled' AND o.created_at >= NOW() - INTERVAL '1 day' * $1
           GROUP BY p.id, p.name
           ORDER BY revenue DESC
           LIMIT 5`,
          [days]
        ),
      ]);

      return {
        revenue: revenue.rows[0],
        orders: orders.rows[0],
        customers: customers.rows[0],
        top_products: topProducts.rows,
      };
    });

    res.json(stats);
  } catch (err) {
    next(err);
  }
}

export async function getRevenueChart(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const range = (req.query['range'] as string) || '30d';
    const days = range === '7d' ? 7 : range === '90d' ? 90 : 30;
    const cacheKey = `dashboard:revenue-chart:${days}`;

    const data = await getCached(cacheKey, async () => {
      const result = await pool.query(
        `SELECT
           DATE(created_at) as date,
           COALESCE(SUM(total), 0) as revenue,
           COUNT(*) as order_count
         FROM orders
         WHERE status != 'cancelled' AND created_at >= NOW() - INTERVAL '1 day' * $1
         GROUP BY DATE(created_at)
         ORDER BY date ASC`,
        [days]
      );
      return result.rows;
    });

    res.json(data);
  } catch (err) {
    next(err);
  }
}
