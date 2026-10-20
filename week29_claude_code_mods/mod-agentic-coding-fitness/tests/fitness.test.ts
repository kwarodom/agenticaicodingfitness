import { expect, test } from 'claude-code/testing'

// Runs with `claude plugin test` from the plugin directory: no session, sign-in, or network.
// Stubs answer in Claude Code's place. Register every stub before the first call on `$`.

test('/fitness-log records a lab and reports pass/fail', async ($, on) => {
  const store: Record<string, unknown> = {}
  on('store.get', (e) => ({ value: store[e.key] }))
  on('store.set', (e) => { store[e.key] = e.value; return { value: undefined } })
  on('command.register', () => ({ value: undefined }))
  on('tool.register', () => ({ value: undefined }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.invalidate', () => ({ value: undefined }))

  await $.session.start({})
  const ok = await $.command.run({ command: 'fitness-log', args: '05 agent-loops 0.86' })
  expect(ok.text).toContain('05/agent-loops → 0.86 (pass)')

  const low = await $.command.run({ command: 'fitness-log', args: '06 agent-memory-graphs 0.5' })
  expect(low.text).toContain('(below 0.8)')

  const bad = await $.command.run({ command: 'fitness-log', args: 'nope' })
  expect(bad.text).toContain('Usage:')
})

test('a passing weekNN pytest run is auto-logged and announced after the turn', async ($, on) => {
  const store: Record<string, unknown> = {}
  on('store.get', (e) => ({ value: store[e.key] }))
  on('store.set', (e) => { store[e.key] = e.value; return { value: undefined } })
  on('command.register', () => ({ value: undefined }))
  on('tool.register', () => ({ value: undefined }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.invalidate', () => ({ value: undefined }))
  on('tool.call', () => ({ result: '===== 7 passed in 0.42s =====' }))

  await $.session.start({})
  await $.tool.call({ tool: 'Bash', command: 'uv run pytest week05/tests -q' })
  const done = await $.turn.complete({ turnId: 't1', answer: 'ok', durationMs: 10, usage: {} })
  expect(done.text).toContain('Lab auto-logged: 05/agent-loops')
})

test('/fitness-grade uses the model stub', async ($, on) => {
  on('store.get', () => ({ value: undefined }))
  on('store.set', () => ({ value: undefined }))
  on('command.register', () => ({ value: undefined }))
  on('tool.register', () => ({ value: undefined }))
  on('ui.status', () => ({ value: undefined }))
  on('model.complete', () => ({
    value: {
      isAnswered: true,
      text: 'PASS\nNames the loop and the token trade-off.',
      usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
    },
  }))
  await $.session.start({})
  const r = await $.command.run({ command: 'fitness-grade', args: 'I used a while loop with a stop tool; it costs tokens per step.' })
  expect(r.text).toContain('PASS')
})
