import { Router } from 'express';
import { authenticate, requireAdmin } from '../middleware/auth';
import { listCustomers, getCustomer } from '../controllers/customerController';

const router = Router();

router.get('/', authenticate, requireAdmin, listCustomers);
router.get('/:id', authenticate, requireAdmin, getCustomer);

export default router;
