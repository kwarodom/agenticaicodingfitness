import request from 'supertest';
import app from '../app';
import { pool } from '../config/db';

jest.mock('../config/db.js', () => ({
  pool: {
    query: jest.fn(),
  },
}));

const mockPool = pool as jest.Mocked<typeof pool>;

describe('GET /api/products', () => {
  it('returns 200 with paginated shape', async () => {
    (mockPool.query as jest.Mock)
      .mockResolvedValueOnce({ rows: [], rowCount: 0 })   // products query
      .mockResolvedValueOnce({ rows: [{ count: '0' }] }); // count query

    const res = await request(app).get('/api/products');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('data');
    expect(res.body).toHaveProperty('total');
    expect(res.body).toHaveProperty('page');
    expect(res.body).toHaveProperty('limit');
  });
});

describe('POST /api/products', () => {
  it('returns 401 without auth token', async () => {
    const res = await request(app)
      .post('/api/products')
      .send({ name: 'Widget', price: 9.99, stock: 100, category: 'gadgets' });
    expect(res.status).toBe(401);
  });
});

describe('DELETE /api/products/:id', () => {
  it('returns 401 without auth token', async () => {
    const res = await request(app).delete('/api/products/1');
    expect(res.status).toBe(401);
  });
});
