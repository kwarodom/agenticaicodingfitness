import { expect, test } from 'claude-code/testing'

test('a force push is refused when no one can answer the question', async ($, on) => {
  // stands in for Claude Code's Bash tool. Nothing answers AskUserQuestion,
  // so $.ui.ask rejects, as it does in `claude -p`.
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'pushed' }))
  const r = await $.tool.call({ tool: 'Bash', command: 'git push --force origin main' })
  expect(r).toEqual({ deny: 'hello-mod: the force-push guard failed, so the command was not run.' })
  const ok = await $.tool.call({ tool: 'Bash', command: 'git push origin feature' })
  expect(ok.result).toBe('pushed')
})
