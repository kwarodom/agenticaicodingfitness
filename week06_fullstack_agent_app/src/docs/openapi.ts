export const openApiSpec = {
  openapi: '3.0.3',
  info: {
    title: 'E-Commerce Dashboard API',
    version: '1.0.0',
    description: 'REST API for the e-commerce dashboard. Provides endpoints for authentication, product management, order processing, customer management, and analytics.',
    contact: { name: 'API Support', email: 'support@example.com' },
  },
  servers: [
    { url: 'http://localhost:3000', description: 'Development' },
  ],
  tags: [
    { name: 'Auth', description: 'Authentication and token management' },
    { name: 'Products', description: 'Product catalog management' },
    { name: 'Orders', description: 'Order lifecycle management' },
    { name: 'Customers', description: 'Customer data (admin only)' },
    { name: 'Dashboard', description: 'Analytics and KPIs (admin only)' },
  ],
  components: {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT',
        description: 'JWT access token (15 min expiry). Obtain via /api/auth/login.',
      },
    },
    schemas: {
      Error: {
        type: 'object',
        properties: {
          error: { type: 'string', example: 'Resource not found' },
        },
      },
      ValidationError: {
        type: 'object',
        properties: {
          errors: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                field: { type: 'string' },
                message: { type: 'string' },
              },
            },
          },
        },
      },
      User: {
        type: 'object',
        properties: {
          id: { type: 'integer', example: 1 },
          email: { type: 'string', format: 'email', example: 'user@example.com' },
          name: { type: 'string', example: 'Jane Smith' },
          role: { type: 'string', enum: ['user', 'admin'], example: 'user' },
        },
      },
      AuthResponse: {
        type: 'object',
        properties: {
          token: { type: 'string', description: 'JWT access token (15 min)' },
          refreshToken: { type: 'string', description: 'Refresh token (7 days)' },
          user: { $ref: '#/components/schemas/User' },
        },
      },
      Product: {
        type: 'object',
        properties: {
          id: { type: 'integer', example: 1 },
          name: { type: 'string', example: 'Wireless Headphones' },
          description: { type: 'string', example: 'Premium noise-cancelling headphones' },
          price: { type: 'number', format: 'float', example: 99.99 },
          stock: { type: 'integer', example: 150 },
          category: { type: 'string', example: 'Electronics' },
          created_at: { type: 'string', format: 'date-time' },
          updated_at: { type: 'string', format: 'date-time' },
        },
      },
      PaginatedProducts: {
        type: 'object',
        properties: {
          data: { type: 'array', items: { $ref: '#/components/schemas/Product' } },
          total: { type: 'integer', example: 42 },
          page: { type: 'integer', example: 1 },
          limit: { type: 'integer', example: 20 },
        },
      },
      OrderItem: {
        type: 'object',
        properties: {
          product_id: { type: 'integer', example: 1 },
          product_name: { type: 'string', example: 'Wireless Headphones' },
          quantity: { type: 'integer', example: 2 },
          unit_price: { type: 'number', format: 'float', example: 99.99 },
        },
      },
      Order: {
        type: 'object',
        properties: {
          id: { type: 'integer', example: 1 },
          status: {
            type: 'string',
            enum: ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
            example: 'pending',
          },
          total: { type: 'number', format: 'float', example: 199.98 },
          customer_email: { type: 'string', format: 'email', example: 'user@example.com' },
          created_at: { type: 'string', format: 'date-time' },
          updated_at: { type: 'string', format: 'date-time' },
        },
      },
      OrderDetail: {
        allOf: [
          { $ref: '#/components/schemas/Order' },
          {
            type: 'object',
            properties: {
              items: { type: 'array', items: { $ref: '#/components/schemas/OrderItem' } },
            },
          },
        ],
      },
      Customer: {
        type: 'object',
        properties: {
          id: { type: 'integer', example: 1 },
          email: { type: 'string', format: 'email', example: 'user@example.com' },
          name: { type: 'string', example: 'Jane Smith' },
          order_count: { type: 'integer', example: 5 },
          total_spent: { type: 'number', format: 'float', example: 549.95 },
          created_at: { type: 'string', format: 'date-time' },
        },
      },
      DashboardStats: {
        type: 'object',
        properties: {
          revenue: {
            type: 'object',
            properties: {
              total_revenue: { type: 'number', example: 12500.00 },
              recent_revenue: { type: 'number', example: 6200.00 },
              previous_revenue: { type: 'number', example: 6300.00 },
            },
          },
          orders: {
            type: 'object',
            properties: {
              total: { type: 'integer', example: 120 },
              pending: { type: 'integer', example: 15 },
              processing: { type: 'integer', example: 20 },
              shipped: { type: 'integer', example: 35 },
              delivered: { type: 'integer', example: 45 },
              cancelled: { type: 'integer', example: 5 },
            },
          },
          customers: {
            type: 'object',
            properties: { new_customers: { type: 'integer', example: 28 } },
          },
          top_products: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'integer' },
                name: { type: 'string' },
                units_sold: { type: 'integer' },
                revenue: { type: 'number' },
              },
            },
          },
        },
      },
    },
    responses: {
      Unauthorized: {
        description: 'Missing or invalid access token',
        content: { 'application/json': { schema: { $ref: '#/components/schemas/Error' } } },
      },
      Forbidden: {
        description: 'Insufficient permissions',
        content: { 'application/json': { schema: { $ref: '#/components/schemas/Error' } } },
      },
      NotFound: {
        description: 'Resource not found',
        content: { 'application/json': { schema: { $ref: '#/components/schemas/Error' } } },
      },
      BadRequest: {
        description: 'Validation error',
        content: { 'application/json': { schema: { $ref: '#/components/schemas/ValidationError' } } },
      },
    },
  },
  paths: {
    '/health': {
      get: {
        summary: 'Health check',
        tags: ['Auth'],
        responses: {
          200: { description: 'OK', content: { 'application/json': { schema: { type: 'object', properties: { status: { type: 'string', example: 'ok' } } } } } },
        },
      },
    },
    '/api/auth/register': {
      post: {
        summary: 'Register a new user',
        tags: ['Auth'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['email', 'password', 'name'],
                properties: {
                  email: { type: 'string', format: 'email', example: 'user@example.com' },
                  password: { type: 'string', minLength: 8, example: 'password123' },
                  name: { type: 'string', example: 'Jane Smith' },
                },
              },
            },
          },
        },
        responses: {
          201: { description: 'User registered', content: { 'application/json': { schema: { $ref: '#/components/schemas/AuthResponse' } } } },
          400: { $ref: '#/components/responses/BadRequest' },
          409: { description: 'Email already registered', content: { 'application/json': { schema: { $ref: '#/components/schemas/Error' } } } },
        },
      },
    },
    '/api/auth/login': {
      post: {
        summary: 'Login and obtain tokens',
        tags: ['Auth'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['email', 'password'],
                properties: {
                  email: { type: 'string', format: 'email', example: 'user@example.com' },
                  password: { type: 'string', example: 'password123' },
                },
              },
            },
          },
        },
        responses: {
          200: { description: 'Login successful', content: { 'application/json': { schema: { $ref: '#/components/schemas/AuthResponse' } } } },
          400: { $ref: '#/components/responses/BadRequest' },
          401: { description: 'Invalid credentials', content: { 'application/json': { schema: { $ref: '#/components/schemas/Error' } } } },
        },
      },
    },
    '/api/auth/refresh': {
      post: {
        summary: 'Refresh access token',
        tags: ['Auth'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['refreshToken'],
                properties: { refreshToken: { type: 'string' } },
              },
            },
          },
        },
        responses: {
          200: { description: 'New access token', content: { 'application/json': { schema: { type: 'object', properties: { token: { type: 'string' } } } } } },
          401: { $ref: '#/components/responses/Unauthorized' },
        },
      },
    },
    '/api/auth/logout': {
      post: {
        summary: 'Logout (client discards tokens)',
        tags: ['Auth'],
        responses: {
          200: { description: 'Logged out', content: { 'application/json': { schema: { type: 'object', properties: { message: { type: 'string', example: 'Logged out' } } } } } },
        },
      },
    },
    '/api/products': {
      get: {
        summary: 'List products',
        tags: ['Products'],
        parameters: [
          { name: 'page', in: 'query', schema: { type: 'integer', default: 1 } },
          { name: 'limit', in: 'query', schema: { type: 'integer', default: 20, maximum: 100 } },
          { name: 'search', in: 'query', schema: { type: 'string' }, description: 'Search by name or description' },
        ],
        responses: {
          200: { description: 'Paginated product list', content: { 'application/json': { schema: { $ref: '#/components/schemas/PaginatedProducts' } } } },
        },
      },
      post: {
        summary: 'Create a product (admin)',
        tags: ['Products'],
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['name', 'price', 'stock', 'category'],
                properties: {
                  name: { type: 'string', example: 'Wireless Headphones' },
                  description: { type: 'string', example: 'Premium noise-cancelling headphones' },
                  price: { type: 'number', minimum: 0, example: 99.99 },
                  stock: { type: 'integer', minimum: 0, example: 150 },
                  category: { type: 'string', example: 'Electronics' },
                },
              },
            },
          },
        },
        responses: {
          201: { description: 'Product created', content: { 'application/json': { schema: { $ref: '#/components/schemas/Product' } } } },
          400: { $ref: '#/components/responses/BadRequest' },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
        },
      },
    },
    '/api/products/{id}': {
      get: {
        summary: 'Get a product by ID',
        tags: ['Products'],
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'integer' } }],
        responses: {
          200: { description: 'Product detail', content: { 'application/json': { schema: { $ref: '#/components/schemas/Product' } } } },
          404: { $ref: '#/components/responses/NotFound' },
        },
      },
      patch: {
        summary: 'Update a product (admin)',
        tags: ['Products'],
        security: [{ bearerAuth: [] }],
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'integer' } }],
        requestBody: {
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  description: { type: 'string' },
                  price: { type: 'number', minimum: 0 },
                  stock: { type: 'integer', minimum: 0 },
                  category: { type: 'string' },
                },
              },
            },
          },
        },
        responses: {
          200: { description: 'Product updated', content: { 'application/json': { schema: { $ref: '#/components/schemas/Product' } } } },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
          404: { $ref: '#/components/responses/NotFound' },
        },
      },
      delete: {
        summary: 'Delete a product (admin)',
        tags: ['Products'],
        security: [{ bearerAuth: [] }],
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'integer' } }],
        responses: {
          204: { description: 'Product deleted' },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
          404: { $ref: '#/components/responses/NotFound' },
        },
      },
    },
    '/api/orders': {
      get: {
        summary: 'List orders (own orders for users; all for admins)',
        tags: ['Orders'],
        security: [{ bearerAuth: [] }],
        parameters: [
          { name: 'page', in: 'query', schema: { type: 'integer', default: 1 } },
          { name: 'limit', in: 'query', schema: { type: 'integer', default: 20, maximum: 100 } },
          { name: 'status', in: 'query', schema: { type: 'string', enum: ['pending', 'processing', 'shipped', 'delivered', 'cancelled'] } },
        ],
        responses: {
          200: {
            description: 'Paginated order list',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    data: { type: 'array', items: { $ref: '#/components/schemas/Order' } },
                    total: { type: 'integer' },
                    page: { type: 'integer' },
                    limit: { type: 'integer' },
                  },
                },
              },
            },
          },
          401: { $ref: '#/components/responses/Unauthorized' },
        },
      },
      post: {
        summary: 'Create an order',
        tags: ['Orders'],
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['items'],
                properties: {
                  items: {
                    type: 'array',
                    minItems: 1,
                    items: {
                      type: 'object',
                      required: ['productId', 'quantity'],
                      properties: {
                        productId: { type: 'integer', example: 1 },
                        quantity: { type: 'integer', minimum: 1, example: 2 },
                      },
                    },
                  },
                },
              },
            },
          },
        },
        responses: {
          201: { description: 'Order created', content: { 'application/json': { schema: { $ref: '#/components/schemas/OrderDetail' } } } },
          400: { description: 'Validation error or insufficient stock', content: { 'application/json': { schema: { $ref: '#/components/schemas/Error' } } } },
          401: { $ref: '#/components/responses/Unauthorized' },
        },
      },
    },
    '/api/orders/{id}': {
      get: {
        summary: 'Get order by ID',
        tags: ['Orders'],
        security: [{ bearerAuth: [] }],
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'integer' } }],
        responses: {
          200: { description: 'Order detail with line items', content: { 'application/json': { schema: { $ref: '#/components/schemas/OrderDetail' } } } },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
          404: { $ref: '#/components/responses/NotFound' },
        },
      },
    },
    '/api/orders/{id}/status': {
      patch: {
        summary: 'Update order status (admin)',
        tags: ['Orders'],
        security: [{ bearerAuth: [] }],
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'integer' } }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['status'],
                properties: {
                  status: { type: 'string', enum: ['pending', 'processing', 'shipped', 'delivered', 'cancelled'] },
                },
              },
            },
          },
        },
        responses: {
          200: { description: 'Status updated', content: { 'application/json': { schema: { $ref: '#/components/schemas/Order' } } } },
          400: { $ref: '#/components/responses/BadRequest' },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
          404: { $ref: '#/components/responses/NotFound' },
        },
      },
    },
    '/api/customers': {
      get: {
        summary: 'List customers with aggregates (admin)',
        tags: ['Customers'],
        security: [{ bearerAuth: [] }],
        parameters: [
          { name: 'page', in: 'query', schema: { type: 'integer', default: 1 } },
          { name: 'limit', in: 'query', schema: { type: 'integer', default: 20, maximum: 100 } },
          { name: 'search', in: 'query', schema: { type: 'string' }, description: 'Search by email or name' },
        ],
        responses: {
          200: {
            description: 'Paginated customer list',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    data: { type: 'array', items: { $ref: '#/components/schemas/Customer' } },
                    total: { type: 'integer' },
                    page: { type: 'integer' },
                    limit: { type: 'integer' },
                  },
                },
              },
            },
          },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
        },
      },
    },
    '/api/customers/{id}': {
      get: {
        summary: 'Get customer with recent orders (admin)',
        tags: ['Customers'],
        security: [{ bearerAuth: [] }],
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'integer' } }],
        responses: {
          200: {
            description: 'Customer detail',
            content: {
              'application/json': {
                schema: {
                  allOf: [
                    { $ref: '#/components/schemas/Customer' },
                    { type: 'object', properties: { recent_orders: { type: 'array', items: { $ref: '#/components/schemas/Order' } } } },
                  ],
                },
              },
            },
          },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
          404: { $ref: '#/components/responses/NotFound' },
        },
      },
    },
    '/api/dashboard/stats': {
      get: {
        summary: 'Dashboard KPI stats (admin, cached 5min)',
        tags: ['Dashboard'],
        security: [{ bearerAuth: [] }],
        parameters: [
          { name: 'range', in: 'query', schema: { type: 'string', enum: ['7d', '30d', '90d'], default: '30d' } },
        ],
        responses: {
          200: { description: 'Dashboard statistics', content: { 'application/json': { schema: { $ref: '#/components/schemas/DashboardStats' } } } },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
        },
      },
    },
    '/api/dashboard/revenue-chart': {
      get: {
        summary: 'Daily revenue chart data (admin, cached 5min)',
        tags: ['Dashboard'],
        security: [{ bearerAuth: [] }],
        parameters: [
          { name: 'range', in: 'query', schema: { type: 'string', enum: ['7d', '30d', '90d'], default: '30d' } },
        ],
        responses: {
          200: {
            description: 'Daily revenue series',
            content: {
              'application/json': {
                schema: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      date: { type: 'string', format: 'date', example: '2026-03-01' },
                      revenue: { type: 'number', example: 1250.00 },
                      order_count: { type: 'integer', example: 12 },
                    },
                  },
                },
              },
            },
          },
          401: { $ref: '#/components/responses/Unauthorized' },
          403: { $ref: '#/components/responses/Forbidden' },
        },
      },
    },
  },
};
