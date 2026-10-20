import type { Register } from 'claude-code'

// Three hooks on the same event. Each one marks the way in (the command)
// and the way out (the result), so a test can read the order.
export const register: Register = on => {
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const r = await next({ ...e, command: e.command + ' >A' })
    return { result: String(r.result) + ' <A' }
  })
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const r = await next({ ...e, command: e.command + ' >B' })
    return { result: String(r.result) + ' <B' }
  })
  // answer: C never calls next, so nothing beneath it runs
  on('tool.call', { tool: 'Bash' }, ($, e, next) =>
    e.command.startsWith('whoami') ? { result: 'coach (answered by C)' } : next(e),
  )
}
