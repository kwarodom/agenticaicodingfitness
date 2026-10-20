import { Router } from 'express';
import { body } from 'express-validator';
import { authenticate, requireAdmin } from '../middleware/auth';
import { validate } from '../middleware/validate';
import { listOrders, getOrder, createOrder, updateOrderStatus } from '../controllers/orderController';

const router = Router();

const createOrderRules = [
  body('items').isArray({ min: 1 }),
  body('items.*.productId').isInt({ min: 1 }),
  body('items.*.quantity').isInt({ min: 1 }),
];

router.get('/', authenticate, listOrders);
router.get('/:id', authenticate, getOrder);
router.post('/', authenticate, createOrderRules, validate, createOrder);
router.patch('/:id/status', authenticate, requireAdmin, body('status').notEmpty(), validate, updateOrderStatus);

export default router;
