import type { Register } from 'claude-code'

export const register: Register = on => {
  // L02 · rewrite what Claude Code draws
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )

  // L03 · wrap: let every tool run, then count it
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    $.ui.status('hello-mod · ' + calls + ' tool calls')
    return r
  })

  // L03 · rewrite: add a line of context Claude reads with the prompt
  on('prompt.submit', ($, e, next) =>
    next({ ...e, context: [...(e.context ?? []), 'Reply like a fitness coach.'] }),
  )

  // L03 · observe: look at the finished turn, change nothing
  on('turn.complete', ($, e, next) => {
    if (e.durationMs > 120_000) $.ui.toast('Long set: ' + Math.round(e.durationMs / 1000) + 's')
    return next(e)
  })
}
