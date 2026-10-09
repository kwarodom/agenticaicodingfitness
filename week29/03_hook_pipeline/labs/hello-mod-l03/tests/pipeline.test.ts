import { expect, test } from 'claude-code/testing'

// The engine fills these fields in a real session; a test spells them out.
const ORIGIN = { kind: 'composer' } as const

test('wrap: the tool result comes back unchanged and the status counts it', async ($, on) => {
  const status: string[] = []
  on('ui.status', ($, e) => { status.push(String(e.text)); return { value: undefined } })
  on('tool.call', { tool: 'Bash' }, ($, e) => ({ result: 'ran: ' + e.command }))
  const r = await $.tool.call({ tool: 'Bash', command: 'ls' })
  await $.tool.call({ tool: 'Bash', command: 'pwd' })
  expect(r.result).toBe('ran: ls')
  expect(status).toEqual(['hello-mod · 1 tool calls', 'hello-mod · 2 tool calls'])
})

test('rewrite: the prompt reaches Claude Code with one more line of context', async ($, on) => {
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  const r = await $.prompt.submit({ text: 'plan my leg day', wait: false, origin: ORIGIN })
  expect(r.text).toBe('plan my leg day')
  expect(r.context).toEqual(['Reply like a fitness coach.'])
})
