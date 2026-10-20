import { Router } from 'express';
import { body } from 'express-validator';
import { register, login, refreshToken, logout } from '../controllers/authController';
import { validate } from '../middleware/validate';

const router = Router();

const registerRules = [
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 8 }),
  body('name').trim().notEmpty(),
];

const loginRules = [
  body('email').isEmail().normalizeEmail(),
  body('password').notEmpty(),
];

router.post('/register', registerRules, validate, register);
router.post('/login', loginRules, validate, login);
router.post('/refresh', refreshToken);
router.post('/logout', logout);

export default router;
