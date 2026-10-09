import { expect, test } from 'claude-code/testing'

test('a turn with 6 tool calls raises one toast', async ($, on) => {
  const toasts: string[] = []
  on('ui.toast', ($, e) => { toasts.push(String(e.text)); return { value: undefined } })
  on('prompt.submit', ($, e) => ({ text: e.text }))
  on('tool.call', () => ({ result: 'ok' }))
  on('turn.complete', ($, e) => ({ text: e.answer }))

  await $.prompt.submit({ text: 'refactor everything', wait: false, origin: { kind: 'composer' } })
  for (let i = 0; i < 6; i++) await $.tool.call({ tool: 'Read', file_path: '/lab/f' + i })
  await $.turn.complete({ turnId: 't1', answer: 'done', durationMs: 900, isAborted: false, reason: 'answer' })
  expect(toasts).toEqual(['Heavy set: 6 tool calls in one turn'])
})
