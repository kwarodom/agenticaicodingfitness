import { Request, Response, NextFunction } from 'express';
import { pool } from '../config/db';

export async function listProducts(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const page = Math.max(1, Number(req.query['page']) || 1);
    const limit = Math.min(100, Math.max(1, Number(req.query['limit']) || 20));
    const offset = (page - 1) * limit;
    const search = req.query['search'] ? `%${req.query['search']}%` : null;

    const baseWhere = search ? 'WHERE name ILIKE $3 OR description ILIKE $3' : '';
    const params: (number | string)[] = [limit, offset];
    if (search) params.push(search);

    const [rows, count] = await Promise.all([
      pool.query(`SELECT id, name, description, price, stock, category, created_at FROM products ${baseWhere} ORDER BY created_at DESC LIMIT $1 OFFSET $2`, params),
      pool.query(`SELECT COUNT(*) FROM products ${baseWhere}`, search ? [search] : []),
    ]);

    res.json({
      data: rows.rows,
      total: Number(count.rows[0].count),
      page,
      limit,
    });
  } catch (err) {
    next(err);
  }
}

export async function getProduct(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const result = await pool.query(
      'SELECT id, name, description, price, stock, category, created_at, updated_at FROM products WHERE id = $1',
      [req.params['id']]
    );
    if (result.rowCount === 0) {
      res.status(404).json({ error: 'Product not found' });
      return;
    }
    res.json(result.rows[0]);
  } catch (err) {
    next(err);
  }
}

export async function createProduct(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { name, description, price, stock, category } = req.body as {
      name: string;
      description: string;
      price: number;
      stock: number;
      category: string;
    };

    const result = await pool.query(
      'INSERT INTO products (name, description, price, stock, category) VALUES ($1, $2, $3, $4, $5) RETURNING *',
      [name, description, price, stock, category]
    );

    res.status(201).json(result.rows[0]);
  } catch (err) {
    next(err);
  }
}

export async function updateProduct(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { name, description, price, stock, category } = req.body as {
      name?: string;
      description?: string;
      price?: number;
      stock?: number;
      category?: string;
    };

    const result = await pool.query(
      `UPDATE products
       SET name = COALESCE($1, name),
           description = COALESCE($2, description),
           price = COALESCE($3, price),
           stock = COALESCE($4, stock),
           category = COALESCE($5, category),
           updated_at = NOW()
       WHERE id = $6
       RETURNING *`,
      [name, description, price, stock, category, req.params['id']]
    );

    if (result.rowCount === 0) {
      res.status(404).json({ error: 'Product not found' });
      return;
    }

    res.json(result.rows[0]);
  } catch (err) {
    next(err);
  }
}

export async function deleteProduct(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const result = await pool.query('DELETE FROM products WHERE id = $1', [req.params['id']]);
    if (result.rowCount === 0) {
      res.status(404).json({ error: 'Product not found' });
      return;
    }
    res.status(204).send();
  } catch (err) {
    next(err);
  }
}
