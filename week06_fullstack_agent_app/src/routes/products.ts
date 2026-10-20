import { Router } from 'express';
import { body } from 'express-validator';
import { authenticate, requireAdmin } from '../middleware/auth';
import { listProducts, getProduct, createProduct, updateProduct, deleteProduct } from '../controllers/productController';

const router = Router();

const productRules = [
  body('name').trim().notEmpty(),
  body('price').isFloat({ min: 0 }),
  body('stock').isInt({ min: 0 }),
  body('category').trim().notEmpty(),
];

router.get('/', listProducts);
router.get('/:id', getProduct);
router.post('/', authenticate, requireAdmin, productRules, createProduct);
router.patch('/:id', authenticate, requireAdmin, updateProduct);
router.delete('/:id', authenticate, requireAdmin, deleteProduct);

export default router;
