import request from 'supertest';
import app from '../app';

describe('GET /api/orders', () => {
  it('returns 401 without auth token', async () => {
    const res = await request(app).get('/api/orders');
    expect(res.status).toBe(401);
  });
});

describe('POST /api/orders', () => {
  it('returns 401 without auth token', async () => {
    const res = await request(app)
      .post('/api/orders')
      .send({ items: [{ productId: 1, quantity: 2 }] });
    expect(res.status).toBe(401);
  });

  it('returns 401 for empty items without auth', async () => {
    const res = await request(app)
      .post('/api/orders')
      .send({ items: [] });
    expect(res.status).toBe(401);
  });
});

describe('PATCH /api/orders/:id/status', () => {
  it('returns 401 without auth token', async () => {
    const res = await request(app)
      .patch('/api/orders/1/status')
      .send({ status: 'shipped' });
    expect(res.status).toBe(401);
  });
});
