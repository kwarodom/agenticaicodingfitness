import { expect, test } from 'claude-code/testing'

test('wide recursive deletes are refused, lab deletes run', async ($, on) => {
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran: ' + e.command }))
  for (const cmd of ['rm -rf /', 'rm -rf ~/projects', 'rm -fr ../other', 'rm -r $HOME']) {
    const r = await $.tool.call({ tool: 'Bash', command: cmd })
    expect(r.deny).toContain('refused')
  }
  const ok = await $.tool.call({ tool: 'Bash', command: 'rm -rf build/' })
  expect(ok.result).toBe('ran: rm -rf build/')
})
