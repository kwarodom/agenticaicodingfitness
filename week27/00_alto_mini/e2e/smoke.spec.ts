// Optional Playwright smoke (needs `npm i -D @playwright/test` and `make dev`). The QA lane adds forensics guards.
import { test, expect } from '@playwright/test';
test('dashboard renders rooms, alerts and history', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('http://127.0.0.1:8127/');
  await expect(page.getByTestId('room-1201')).toBeVisible();
  await expect(page.getByTestId('alerts').locator('tbody tr')).toHaveCount(3);
  await expect(page.locator('#hist')).not.toContainText('loading');
  expect(errors).toEqual([]);
});
