/**
 * Integration-style API tests (no live DB — pool is mocked).
 * Covers: auth validation, product CRUD access control, order auth,
 * customer admin guard, dashboard admin guard, and the Swagger UI route.
 */
import request from 'supertest';
import app from '../app';
import { pool } from '../config/db';
import { redisClient } from '../config/redis';

// ── Mocks ────────────────────────────────────────────────────────────────────

jest.mock('../config/db', () => ({ pool: { query: jest.fn(), connect: jest.fn() } }));
jest.mock('../config/redis', () => ({
  redisClient: { get: jest.fn(), setEx: jest.fn(), connect: jest.fn() },
}));
jest.mock('bcryptjs', () => ({
  hash: jest.fn().mockResolvedValue('hashed'),
  compare: jest.fn().mockResolvedValue(true),
}));

const mockQuery = pool.query as jest.Mock;
const mockRedisGet = redisClient.get as jest.Mock;

// ── Helpers ──────────────────────────────────────────────────────────────────

import jwt from 'jsonwebtoken';

function makeToken(userId: number, role: 'user' | 'admin'): string {
  return jwt.sign({ userId, role }, 'dev-secret-change-in-production', { expiresIn: '1h' });
}

const userToken = makeToken(1, 'user');
const adminToken = makeToken(2, 'admin');

// ── Health ───────────────────────────────────────────────────────────────────

describe('GET /health', () => {
  it('returns 200 { status: ok }', async () => {
    const res = await request(app).get('/health');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: 'ok' });
  });
});

// ── Swagger UI ───────────────────────────────────────────────────────────────

describe('GET /api/docs', () => {
  it('returns 200 HTML', async () => {
    const res = await request(app).get('/api/docs/');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/html/);
  });
});

// ── Auth: POST /api/auth/register ────────────────────────────────────────────

describe('POST /api/auth/register', () => {
  beforeEach(() => mockQuery.mockReset());

  it('returns 400 for invalid email', async () => {
    const res = await request(app).post('/api/auth/register')
      .send({ email: 'not-valid', password: 'password123', name: 'Test' });
    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty('errors');
  });

  it('returns 400 for short password', async () => {
    const res = await request(app).post('/api/auth/register')
      .send({ email: 'test@example.com', password: 'short', name: 'Test' });
    expect(res.status).toBe(400);
  });

  it('returns 400 for missing name', async () => {
    const res = await request(app).post('/api/auth/register')
      .send({ email: 'test@example.com', password: 'password123' });
    expect(res.status).toBe(400);
  });

  it('returns 409 when email already exists', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [{ id: 1 }], rowCount: 1 });
    const res = await request(app).post('/api/auth/register')
      .send({ email: 'existing@example.com', password: 'password123', name: 'Test' });
    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/already registered/i);
  });

  it('returns 201 with token on success', async () => {
    mockQuery
      .mockResolvedValueOnce({ rows: [], rowCount: 0 })
      .mockResolvedValueOnce({ rows: [{ id: 1, email: 'new@example.com', name: 'New', role: 'user' }], rowCount: 1 });
    const res = await request(app).post('/api/auth/register')
      .send({ email: 'new@example.com', password: 'password123', name: 'New' });
    expect(res.status).toBe(201);
    expect(res.body).toHaveProperty('token');
    expect(res.body).toHaveProperty('refreshToken');
    expect(res.body.user.email).toBe('new@example.com');
  });
});

// ── Auth: POST /api/auth/login ───────────────────────────────────────────────

describe('POST /api/auth/login', () => {
  beforeEach(() => mockQuery.mockReset());

  it('returns 400 for missing email', async () => {
    const res = await request(app).post('/api/auth/login').send({ password: 'pw' });
    expect(res.status).toBe(400);
  });

  it('returns 401 for unknown user', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [], rowCount: 0 });
    const res = await request(app).post('/api/auth/login')
      .send({ email: 'ghost@example.com', password: 'password123' });
    expect(res.status).toBe(401);
    expect(res.body.error).toMatch(/invalid credentials/i);
  });

  it('returns 200 with token on valid credentials', async () => {
    mockQuery.mockResolvedValueOnce({
      rows: [{ id: 1, email: 'user@example.com', name: 'User', role: 'user', password_hash: 'hashed' }],
      rowCount: 1,
    });
    const res = await request(app).post('/api/auth/login')
      .send({ email: 'user@example.com', password: 'password123' });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('token');
  });
});

// ── Auth: POST /api/auth/logout ──────────────────────────────────────────────

describe('POST /api/auth/logout', () => {
  it('returns 200', async () => {
    const res = await request(app).post('/api/auth/logout');
    expect(res.status).toBe(200);
    expect(res.body.message).toBe('Logged out');
  });
});

// ── Products ─────────────────────────────────────────────────────────────────

describe('GET /api/products', () => {
  beforeEach(() => mockQuery.mockReset());

  it('returns 200 with paginated shape (public)', async () => {
    mockQuery
      .mockResolvedValueOnce({ rows: [{ id: 1, name: 'Widget', price: '9.99', stock: 100, category: 'misc', created_at: new Date() }], rowCount: 1 })
      .mockResolvedValueOnce({ rows: [{ count: '1' }] });
    const res = await request(app).get('/api/products');
    expect(res.status).toBe(200);
    expect(res.body.data).toHaveLength(1);
    expect(res.body.total).toBe(1);
    expect(res.body.page).toBe(1);
  });

  it('applies pagination params', async () => {
    mockQuery
      .mockResolvedValueOnce({ rows: [], rowCount: 0 })
      .mockResolvedValueOnce({ rows: [{ count: '50' }] });
    const res = await request(app).get('/api/products?page=3&limit=10');
    expect(res.status).toBe(200);
    expect(res.body.page).toBe(3);
    expect(res.body.limit).toBe(10);
  });
});

describe('GET /api/products/:id', () => {
  beforeEach(() => mockQuery.mockReset());

  it('returns 404 for missing product', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [], rowCount: 0 });
    const res = await request(app).get('/api/products/999');
    expect(res.status).toBe(404);
  });

  it('returns 200 for existing product', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [{ id: 1, name: 'Widget', price: '9.99' }], rowCount: 1 });
    const res = await request(app).get('/api/products/1');
    expect(res.status).toBe(200);
    expect(res.body.id).toBe(1);
  });
});

describe('POST /api/products', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).post('/api/products').send({ name: 'X', price: 1, stock: 1, category: 'Y' });
    expect(res.status).toBe(401);
  });

  it('returns 403 for non-admin user', async () => {
    const res = await request(app).post('/api/products')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ name: 'X', price: 1, stock: 1, category: 'Y' });
    expect(res.status).toBe(403);
  });

  it('returns 201 for admin with valid body', async () => {
    mockQuery.mockResolvedValueOnce({
      rows: [{ id: 2, name: 'Gizmo', price: '29.99', stock: 50, category: 'Gadgets', created_at: new Date() }],
    });
    const res = await request(app).post('/api/products')
      .set('Authorization', `Bearer ${adminToken}`)
      .send({ name: 'Gizmo', description: 'A cool gadget', price: 29.99, stock: 50, category: 'Gadgets' });
    expect(res.status).toBe(201);
    expect(res.body.name).toBe('Gizmo');
  });
});

describe('PATCH /api/products/:id', () => {
  beforeEach(() => mockQuery.mockReset());

  it('returns 401 without token', async () => {
    const res = await request(app).patch('/api/products/1').send({ price: 5 });
    expect(res.status).toBe(401);
  });

  it('returns 404 for non-existent product (admin)', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [], rowCount: 0 });
    const res = await request(app).patch('/api/products/999')
      .set('Authorization', `Bearer ${adminToken}`)
      .send({ price: 5 });
    expect(res.status).toBe(404);
  });
});

describe('DELETE /api/products/:id', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).delete('/api/products/1');
    expect(res.status).toBe(401);
  });

  it('returns 403 for non-admin', async () => {
    const res = await request(app).delete('/api/products/1')
      .set('Authorization', `Bearer ${userToken}`);
    expect(res.status).toBe(403);
  });

  it('returns 204 on successful delete (admin)', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [], rowCount: 1 });
    const res = await request(app).delete('/api/products/1')
      .set('Authorization', `Bearer ${adminToken}`);
    expect(res.status).toBe(204);
  });
});

// ── Orders ───────────────────────────────────────────────────────────────────

describe('GET /api/orders', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).get('/api/orders');
    expect(res.status).toBe(401);
  });

  it('returns 200 for authenticated user', async () => {
    mockQuery
      .mockResolvedValueOnce({ rows: [], rowCount: 0 })
      .mockResolvedValueOnce({ rows: [{ count: '0' }] });
    const res = await request(app).get('/api/orders')
      .set('Authorization', `Bearer ${userToken}`);
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('data');
  });
});

describe('POST /api/orders', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).post('/api/orders').send({ items: [{ productId: 1, quantity: 1 }] });
    expect(res.status).toBe(401);
  });

  it('returns 400 for empty items array', async () => {
    const res = await request(app).post('/api/orders')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ items: [] });
    expect(res.status).toBe(400);
  });

  it('returns 400 for non-integer quantity', async () => {
    const res = await request(app).post('/api/orders')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ items: [{ productId: 1, quantity: 0 }] });
    expect(res.status).toBe(400);
  });
});

describe('PATCH /api/orders/:id/status', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).patch('/api/orders/1/status').send({ status: 'shipped' });
    expect(res.status).toBe(401);
  });

  it('returns 403 for non-admin', async () => {
    const res = await request(app).patch('/api/orders/1/status')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ status: 'shipped' });
    expect(res.status).toBe(403);
  });

  it('returns 400 for invalid status (admin)', async () => {
    const res = await request(app).patch('/api/orders/1/status')
      .set('Authorization', `Bearer ${adminToken}`)
      .send({ status: 'flying' });
    expect(res.status).toBe(400);
  });

  it('returns 200 for valid status update (admin)', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [{ id: 1, status: 'shipped' }], rowCount: 1 });
    const res = await request(app).patch('/api/orders/1/status')
      .set('Authorization', `Bearer ${adminToken}`)
      .send({ status: 'shipped' });
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('shipped');
  });
});

// ── Customers ─────────────────────────────────────────────────────────────────

describe('GET /api/customers', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).get('/api/customers');
    expect(res.status).toBe(401);
  });

  it('returns 403 for non-admin user', async () => {
    const res = await request(app).get('/api/customers')
      .set('Authorization', `Bearer ${userToken}`);
    expect(res.status).toBe(403);
  });

  it('returns 200 for admin', async () => {
    mockQuery
      .mockResolvedValueOnce({ rows: [{ id: 1, email: 'a@b.com', name: 'A', order_count: 2, total_spent: '50' }], rowCount: 1 })
      .mockResolvedValueOnce({ rows: [{ count: '1' }] });
    const res = await request(app).get('/api/customers')
      .set('Authorization', `Bearer ${adminToken}`);
    expect(res.status).toBe(200);
    expect(res.body.data).toHaveLength(1);
  });
});

describe('GET /api/customers/:id', () => {
  it('returns 401 without token', async () => {
    const res = await request(app).get('/api/customers/1');
    expect(res.status).toBe(401);
  });

  it('returns 404 for missing customer (admin)', async () => {
    mockQuery
      .mockResolvedValueOnce({ rows: [], rowCount: 0 })
      .mockResolvedValueOnce({ rows: [] });
    const res = await request(app).get('/api/customers/999')
      .set('Authorization', `Bearer ${adminToken}`);
    expect(res.status).toBe(404);
  });
});

// ── Dashboard ─────────────────────────────────────────────────────────────────

describe('GET /api/dashboard/stats', () => {
  beforeEach(() => { mockQuery.mockReset(); mockRedisGet.mockReset(); });

  it('returns 401 without token', async () => {
    const res = await request(app).get('/api/dashboard/stats');
    expect(res.status).toBe(401);
  });

  it('returns 403 for non-admin', async () => {
    const res = await request(app).get('/api/dashboard/stats')
      .set('Authorization', `Bearer ${userToken}`);
    expect(res.status).toBe(403);
  });

  it('returns cached response when Redis hit', async () => {
    const cached = { revenue: {}, orders: {}, customers: {}, top_products: [] };
    mockRedisGet.mockResolvedValueOnce(JSON.stringify(cached));
    const res = await request(app).get('/api/dashboard/stats')
      .set('Authorization', `Bearer ${adminToken}`);
    expect(res.status).toBe(200);
    expect(mockQuery).not.toHaveBeenCalled();
  });

  it('queries DB and returns 200 on cache miss', async () => {
    mockRedisGet.mockResolvedValueOnce(null);
    const statsRow = { total_revenue: '5000', recent_revenue: '2500', previous_revenue: '2500' };
    const ordersRow = { total: '100', pending: '10', processing: '20', shipped: '30', delivered: '35', cancelled: '5' };
    const customersRow = { new_customers: '15' };
    mockQuery
      .mockResolvedValueOnce({ rows: [statsRow] })
      .mockResolvedValueOnce({ rows: [ordersRow] })
      .mockResolvedValueOnce({ rows: [customersRow] })
      .mockResolvedValueOnce({ rows: [] });
    const res = await request(app).get('/api/dashboard/stats')
      .set('Authorization', `Bearer ${adminToken}`);
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('revenue');
    expect(res.body).toHaveProperty('orders');
    expect(res.body).toHaveProperty('top_products');
  });
});

describe('GET /api/dashboard/revenue-chart', () => {
  beforeEach(() => { mockQuery.mockReset(); mockRedisGet.mockReset(); });

  it('returns 401 without token', async () => {
    const res = await request(app).get('/api/dashboard/revenue-chart');
    expect(res.status).toBe(401);
  });

  it('returns chart data for admin', async () => {
    mockRedisGet.mockResolvedValueOnce(null);
    mockQuery.mockResolvedValueOnce({ rows: [{ date: '2026-03-01', revenue: '1200', order_count: '10' }] });
    const res = await request(app).get('/api/dashboard/revenue-chart?range=7d')
      .set('Authorization', `Bearer ${adminToken}`);
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
  });
});
