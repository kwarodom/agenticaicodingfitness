// Forensics fixture for the QA lane (Lab 03 §4). Copy next to your specs and `import { test, expect } from './report-fixture'`.
// Fails the test when any collector fired: console errors, uncaught page errors, 5xx responses, 401/403 on auth pages.
// Screenshots per named step land in the test output dir; the QA lane copies them to docs/qa/report/pr-<P>/.
import { test as base, expect } from '@playwright/test';
export const test = base.extend<{ report: { step: (n: string) => Promise<void>; errors: string[] } }>({
  report: async ({ page }, use, testInfo) => {
    const errors: string[] = [];
    page.on('console', m => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });
    page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
    page.on('response', r => {
      if (r.status() >= 500) errors.push(`5xx: ${r.status()} ${r.url()}`);
      if ((r.status() === 401 || r.status() === 403) && /auth|login|me\b/.test(r.url())) errors.push(`auth: ${r.status()} ${r.url()}`);
    });
    let i = 0;
    await use({ errors, step: async (name) => { await page.screenshot({ path: testInfo.outputPath(`${++i}-${name}.jpg`) }); } });
    expect(errors, 'forensics guard').toEqual([]);
  },
});
export { expect };
