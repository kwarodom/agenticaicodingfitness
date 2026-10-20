import { Request, Response, NextFunction } from 'express';
import { pool } from '../config/db';

export async function listCustomers(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const page = Math.max(1, Number(req.query['page']) || 1);
    const limit = Math.min(100, Math.max(1, Number(req.query['limit']) || 20));
    const offset = (page - 1) * limit;
    const search = req.query['search'] ? `%${req.query['search']}%` : null;

    const params: (number | string)[] = [limit, offset];
    const whereClause = search ? 'WHERE u.email ILIKE $3 OR u.name ILIKE $3' : '';
    if (search) params.push(search);

    const [rows, count] = await Promise.all([
      pool.query(
        `SELECT u.id, u.email, u.name, u.created_at,
                COUNT(o.id) as order_count,
                COALESCE(SUM(o.total), 0) as total_spent
         FROM users u
         LEFT JOIN orders o ON u.id = o.user_id
         ${whereClause}
         GROUP BY u.id
         ORDER BY u.created_at DESC
         LIMIT $1 OFFSET $2`,
        params
      ),
      pool.query(`SELECT COUNT(*) FROM users u ${whereClause}`, search ? [search] : []),
    ]);

    res.json({ data: rows.rows, total: Number(count.rows[0].count), page, limit });
  } catch (err) {
    next(err);
  }
}

export async function getCustomer(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const [user, orders] = await Promise.all([
      pool.query(
        `SELECT u.id, u.email, u.name, u.created_at,
                COUNT(o.id) as order_count,
                COALESCE(SUM(o.total), 0) as total_spent
         FROM users u
         LEFT JOIN orders o ON u.id = o.user_id
         WHERE u.id = $1
         GROUP BY u.id`,
        [req.params['id']]
      ),
      pool.query(
        'SELECT id, status, total, created_at FROM orders WHERE user_id = $1 ORDER BY created_at DESC LIMIT 10',
        [req.params['id']]
      ),
    ]);

    if (user.rowCount === 0) {
      res.status(404).json({ error: 'Customer not found' });
      return;
    }

    res.json({ ...user.rows[0], recent_orders: orders.rows });
  } catch (err) {
    next(err);
  }
}
