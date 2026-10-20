# Project Memory Notes — E-Commerce Dashboard API

## Status
API layer complete. Frontend (React/TypeScript) not yet started.

---

## What's Been Built

### Backend API (Node.js + TypeScript + Express)

#### Auth — `/api/auth`
| Method | Path | Auth | Notes |
|--------|------|------|-------|
| POST | `/register` | None | Returns JWT + refresh token |
| POST | `/login` | None | Returns JWT + refresh token |
| POST | `/refresh` | None | Accepts refresh token, returns new JWT |
| POST | `/logout` | None | Stateless — client discards tokens |

JWT access token: 15min expiry. Refresh token: 7 days.
Passwords hashed with bcrypt (cost factor 12).

#### Products — `/api/products`
| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/` | None | Paginated, supports `?search=`, `?page=`, `?limit=` |
| GET | `/:id` | None | Single product |
| POST | `/` | Admin | Create product |
| PATCH | `/:id` | Admin | Partial update via COALESCE |
| DELETE | `/:id` | Admin | Hard delete |

#### Orders — `/api/orders`
| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/` | User | Admins see all; users see own only |
| GET | `/:id` | User | Includes line items; users can only view own |
| POST | `/` | User | Transactional: checks stock, decrements atomically |
| PATCH | `/:id/status` | Admin | Valid statuses: pending, processing, shipped, delivered, cancelled |

#### Customers — `/api/customers`
| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/` | Admin | Paginated with order_count and total_spent aggregates |
| GET | `/:id` | Admin | Includes last 10 orders |

#### Dashboard — `/api/dashboard`
| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/stats` | Admin | Revenue, order counts by status, new customers, top 5 products |
| GET | `/revenue-chart` | Admin | Daily revenue + order count for chart |

Both dashboard endpoints support `?range=7d|30d|90d` and are Redis-cached for 5 minutes.

---

## Architecture Decisions

- **JWT stateless auth** — access token 15m, refresh 7d. No server-side session store.
- **Redis caching** — dashboard endpoints only (hot-path aggregates). TTL 300s. Cache key pattern: `dashboard:<endpoint>:<days>`.
- **Transactional order creation** — uses `pg` client (not pool) with `BEGIN/COMMIT/ROLLBACK` to prevent overselling.
- **COALESCE updates** — PATCH endpoints use `COALESCE($n, column)` so only provided fields are updated.
- **Role-based access** — two roles: `user` and `admin`. `requireAdmin` middleware guards write/admin endpoints.
- **Input validation** — `express-validator` on all POST routes + `validate` middleware (`src/middleware/validate.ts`) that checks `validationResult` and returns 400 with error array before hitting the controller.

---

## Database Schema

Tables: `users`, `products`, `orders`, `order_items`

Key indexes:
- `orders(user_id)`, `orders(status)`, `orders(created_at)`
- `order_items(order_id)`
- `products(category)`

Migration file: `src/migrations/001_init.sql`

---

## File Structure
```
src/
  app.ts                  # Express app setup
  server.ts               # Entry point, DB/Redis connect
  config/
    db.ts                 # pg Pool
    redis.ts              # Redis client
  middleware/
    auth.ts               # authenticate + requireAdmin
    validate.ts           # express-validator result checker → 400
    errorHandler.ts       # Central error handler
  controllers/
    authController.ts
    productController.ts
    orderController.ts
    customerController.ts
    dashboardController.ts
  routes/
    auth.ts
    products.ts
    orders.ts
    customers.ts
    dashboard.ts
  migrations/
    001_init.sql
  __tests__/
    auth.test.ts
    products.test.ts
    orders.test.ts
```

---

## TODO / Next Steps
- [ ] Stripe webhook endpoint (`/api/webhooks/stripe`) for payment events
- [ ] Frontend React/TypeScript app in `/src/components`, `/src/pages`, `/src/hooks`
- [ ] Integration tests with a real test DB (per AGENTS.md: no mocking the DB)
- [ ] Rate limiting middleware (e.g. `express-rate-limit`)
- [ ] Swagger/OpenAPI documentation

---

## Environment
- Copy `.env.example` → `.env` and fill in values before running
- Run migration: `psql -d ecommerce_dashboard -f src/migrations/001_init.sql`
- Dev server: `npm run dev` (port 3000)
