import type { Register } from 'claude-code'

const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/

export const register: Register = on => {
  // L03 · a catch-all observer, registered FIRST and made "fail-open" with .catch
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    return r
  }).catch(($, e, next) => next(e))

  // L04 · the guard, registered SECOND
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (FORCE_PUSH.test(e.command)) {
      const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
      if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    }
    return next(e)
  }).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))
}
