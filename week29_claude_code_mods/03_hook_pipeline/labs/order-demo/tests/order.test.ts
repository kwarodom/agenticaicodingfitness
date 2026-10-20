import { expect, test } from 'claude-code/testing'

test('the first hook registered is the outermost', async ($, on) => {
  // stands in for Claude Code's own Bash tool
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran [' + e.command + ']' }))
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  expect(r.result).toBe('ran [ls >A >B] <B <A')
  const w = await $.tool.call({ tool: 'Bash', command: 'whoami' })
  expect(w.result).toBe('coach (answered by C) <B <A')
})
