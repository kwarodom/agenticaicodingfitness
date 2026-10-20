import { Request, Response, NextFunction } from 'express';
import { pool } from '../config/db';
import { AuthRequest } from '../middleware/auth';

const VALID_STATUSES = ['pending', 'processing', 'shipped', 'delivered', 'cancelled'] as const;
type OrderStatus = typeof VALID_STATUSES[number];

export async function listOrders(req: AuthRequest, res: Response, next: NextFunction): Promise<void> {
  try {
    const page = Math.max(1, Number(req.query['page']) || 1);
    const limit = Math.min(100, Math.max(1, Number(req.query['limit']) || 20));
    const offset = (page - 1) * limit;
    const status = req.query['status'] as string | undefined;

    const params: (number | string)[] = [limit, offset];
    let whereClause = req.userRole !== 'admin' ? 'WHERE o.user_id = $3' : '';
    if (req.userRole !== 'admin') params.push(req.userId!);

    if (status) {
      whereClause += whereClause ? ' AND o.status = $' : 'WHERE o.status = $';
      whereClause += params.length + 1;
      params.push(status);
    }

    const [rows, count] = await Promise.all([
      pool.query(
        `SELECT o.id, o.status, o.total, o.created_at, u.email as customer_email
         FROM orders o JOIN users u ON o.user_id = u.id
         ${whereClause}
         ORDER BY o.created_at DESC LIMIT $1 OFFSET $2`,
        params
      ),
      pool.query(`SELECT COUNT(*) FROM orders o ${whereClause}`, params.slice(2)),
    ]);

    res.json({ data: rows.rows, total: Number(count.rows[0].count), page, limit });
  } catch (err) {
    next(err);
  }
}

export async function getOrder(req: AuthRequest, res: Response, next: NextFunction): Promise<void> {
  try {
    const orderResult = await pool.query(
      `SELECT o.id, o.status, o.total, o.created_at, o.updated_at, u.email as customer_email
       FROM orders o JOIN users u ON o.user_id = u.id
       WHERE o.id = $1`,
      [req.params['id']]
    );

    if (orderResult.rowCount === 0) {
      res.status(404).json({ error: 'Order not found' });
      return;
    }

    const order = orderResult.rows[0];

    // Non-admins can only view their own orders
    if (req.userRole !== 'admin') {
      const ownerCheck = await pool.query('SELECT user_id FROM orders WHERE id = $1', [req.params['id']]);
      if (ownerCheck.rows[0]?.user_id !== req.userId) {
        res.status(403).json({ error: 'Access denied' });
        return;
      }
    }

    const items = await pool.query(
      `SELECT oi.quantity, oi.unit_price, p.id as product_id, p.name as product_name
       FROM order_items oi JOIN products p ON oi.product_id = p.id
       WHERE oi.order_id = $1`,
      [req.params['id']]
    );

    res.json({ ...order, items: items.rows });
  } catch (err) {
    next(err);
  }
}

export async function createOrder(req: AuthRequest, res: Response, next: NextFunction): Promise<void> {
  const client = await pool.connect();
  try {
    const { items } = req.body as { items: { productId: number; quantity: number }[] };

    await client.query('BEGIN');

    let total = 0;
    const resolvedItems: { productId: number; quantity: number; unitPrice: number }[] = [];

    for (const item of items) {
      const product = await client.query(
        'SELECT id, price, stock FROM products WHERE id = $1 FOR UPDATE',
        [item.productId]
      );
      if (product.rowCount === 0) {
        await client.query('ROLLBACK');
        res.status(400).json({ error: `Product ${item.productId} not found` });
        return;
      }
      const p = product.rows[0];
      if (p.stock < item.quantity) {
        await client.query('ROLLBACK');
        res.status(400).json({ error: `Insufficient stock for product ${item.productId}` });
        return;
      }
      total += p.price * item.quantity;
      resolvedItems.push({ productId: item.productId, quantity: item.quantity, unitPrice: p.price });

      await client.query('UPDATE products SET stock = stock - $1 WHERE id = $2', [item.quantity, item.productId]);
    }

    const order = await client.query(
      'INSERT INTO orders (user_id, status, total) VALUES ($1, $2, $3) RETURNING id, status, total, created_at',
      [req.userId, 'pending', total]
    );
    const orderId = order.rows[0].id;

    for (const item of resolvedItems) {
      await client.query(
        'INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES ($1, $2, $3, $4)',
        [orderId, item.productId, item.quantity, item.unitPrice]
      );
    }

    await client.query('COMMIT');
    res.status(201).json({ ...order.rows[0], items: resolvedItems });
  } catch (err) {
    await client.query('ROLLBACK');
    next(err);
  } finally {
    client.release();
  }
}

export async function updateOrderStatus(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { status } = req.body as { status: OrderStatus };

    if (!VALID_STATUSES.includes(status)) {
      res.status(400).json({ error: `Invalid status. Must be one of: ${VALID_STATUSES.join(', ')}` });
      return;
    }

    const result = await pool.query(
      'UPDATE orders SET status = $1, updated_at = NOW() WHERE id = $2 RETURNING *',
      [status, req.params['id']]
    );

    if (result.rowCount === 0) {
      res.status(404).json({ error: 'Order not found' });
      return;
    }

    res.json(result.rows[0]);
  } catch (err) {
    next(err);
  }
}
