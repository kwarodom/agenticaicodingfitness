import { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { pool } from '../config/db';

function signToken(userId: number, role: string): string {
  const secret = process.env['JWT_SECRET']!;
  return jwt.sign({ userId, role }, secret, { expiresIn: '15m' });
}

function signRefreshToken(userId: number): string {
  const secret = process.env['JWT_REFRESH_SECRET']!;
  return jwt.sign({ userId }, secret, { expiresIn: '7d' });
}

export async function register(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { email, password, name } = req.body as { email: string; password: string; name: string };

    const existing = await pool.query('SELECT id FROM users WHERE email = $1', [email]);
    if ((existing.rowCount ?? 0) > 0) {
      res.status(409).json({ error: 'Email already registered' });
      return;
    }

    const hash = await bcrypt.hash(password, 12);
    const result = await pool.query(
      'INSERT INTO users (email, password_hash, name, role) VALUES ($1, $2, $3, $4) RETURNING id, email, name, role',
      [email, hash, name, 'user']
    );

    const user = result.rows[0];
    const token = signToken(user.id, user.role);
    const refreshToken = signRefreshToken(user.id);

    res.status(201).json({ token, refreshToken, user: { id: user.id, email: user.email, name: user.name, role: user.role } });
  } catch (err) {
    next(err);
  }
}

export async function login(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { email, password } = req.body as { email: string; password: string };

    const result = await pool.query('SELECT id, email, name, role, password_hash FROM users WHERE email = $1', [email]);
    const user = result.rows[0];

    if (!user || !(await bcrypt.compare(password, user.password_hash))) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    const token = signToken(user.id, user.role);
    const refreshToken = signRefreshToken(user.id);

    res.json({ token, refreshToken, user: { id: user.id, email: user.email, name: user.name, role: user.role } });
  } catch (err) {
    next(err);
  }
}

export async function refreshToken(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { refreshToken: token } = req.body as { refreshToken: string };
    if (!token) {
      res.status(400).json({ error: 'Refresh token required' });
      return;
    }

    const secret = process.env['JWT_REFRESH_SECRET']!;
    const payload = jwt.verify(token, secret) as { userId: number };

    const result = await pool.query('SELECT id, role FROM users WHERE id = $1', [payload.userId]);
    const user = result.rows[0];
    if (!user) {
      res.status(401).json({ error: 'User not found' });
      return;
    }

    const newToken = signToken(user.id, user.role);
    res.json({ token: newToken });
  } catch {
    res.status(401).json({ error: 'Invalid or expired refresh token' });
  }
}

export async function logout(_req: Request, res: Response): Promise<void> {
  // Client discards tokens; stateless JWT — optionally add to a blocklist via Redis
  res.json({ message: 'Logged out' });
}
