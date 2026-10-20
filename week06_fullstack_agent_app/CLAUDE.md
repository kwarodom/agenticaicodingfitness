# CLAUDE.md - Project Memory Configuration

## Project Overview
- **Name**: E-Commerce Dashboard
- **Tech Stack**: React + TypeScript + Node.js
- **Database**: PostgreSQL
- **API Base**: http://localhost:3000

## Code Conventions
- Use functional components with hooks
- Prefer async/await over promises
- Follow ESLint rules strictly
- Write tests for all API endpoints

## Important Context
- User authentication uses JWT tokens
- Database migrations run automatically on startup
- Redis cache on port 6379
- Stripe integration for payments

## File Structure
/src
  /components    # Reusable UI components
  /pages         # Route components
  /hooks         # Custom React hooks
  /utils         # Helper functions
  /api           # API client functions

## Common Commands
- npm run dev      # Start development server
- npm test         # Run test suite
- npm run build    # Production build
- npm run migrate  # Run database migrations