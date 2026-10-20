import { Router } from 'express';
import { authenticate, requireAdmin } from '../middleware/auth';
import { getStats, getRevenueChart } from '../controllers/dashboardController';

const router = Router();

router.get('/stats', authenticate, requireAdmin, getStats);
router.get('/revenue-chart', authenticate, requireAdmin, getRevenueChart);

export default router;
