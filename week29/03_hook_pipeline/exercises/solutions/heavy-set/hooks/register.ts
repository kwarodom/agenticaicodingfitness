import type { Register } from 'claude-code'

export const register: Register = on => {
  let turnCalls = 0

  // observe: a new prompt starts a new set
  on('prompt.submit', ($, e, next) => {
    turnCalls = 0
    return next(e)
  })

  // wrap: count each tool call of the turn
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    turnCalls += 1
    return r
  })

  // observe the end of the turn; toast after a heavy one
  on('turn.complete', ($, e, next) => {
    if (turnCalls > 5) $.ui.toast('Heavy set: ' + turnCalls + ' tool calls in one turn')
    return next(e)
  })
}
