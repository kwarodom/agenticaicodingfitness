# ▶ Mod Gym Lesson 04 — Guards: deny, hold and catch

> Part of Week 29 · Claude Code mods. You learn to change how Claude Code itself looks and behaves, from inside the terminal you code in every day. These tutorials are the long form; **🏋 Mod Gym**, the panel inside Claude Code, is the short form with checks. Through the lessons you grow one lab mod, `hello-mod`; the finished version is in `week29_claude_code_mods/reference/hello-mod`.

**What you'll actually do**
- Stop a tool call before it runs by **answering** `tool.call` with `{ deny }`.
- **Hold** a risky call and ask the person first with `$.ui.ask`, and learn what happens when nobody can answer (`claude -p`).
- Make a guard **fail closed** with `.catch` and `next.called`, and read the `gating hook with .catch` line in `claude plugin validate`.
- Reproduce a real failure we measured while building this course: a harmless-looking observer let `git push --force` through. Then fix it three ways, and prove each fix with a test.
- Connect it to the `blast-radius` sample mod and to a mutation gate for the AltoTech Reef sandbox.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** this laptop only

**Sources:** [React to events](https://code.claude.com/docs/en/plugins/mods/events) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · [Manage mods](https://code.claude.com/docs/en/plugins/mods/admin) · [claude-code-playground sample mods (`blast-radius`)](https://github.com/anthropics/claude-code-playground/tree/main/claude-code/mods) · the API declarations Claude Code 2.1.292 writes (`.claude-plugin/types/claude-code/index.d.ts`, the authority whenever the docs and the code differ)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lesson 03 | you can say which hook is outermost when two hooks share an event | this lesson's bug is an ordering bug |
| `hello-mod` with the Lesson 03 hooks | `claude plugin test ~/mod-gym-lab/hello-mod` passes | you add the guard to it |
| A scratch git repo (optional) | `git init /tmp/push-lab` | to try the guard live without touching a real remote |

The labs are already built in the repo. Three of them are **meant to fail**: they show the bug.

```text
week29_claude_code_mods/04_guards/
  labs/guard-without-catch/   a guard with no .catch            (test fails: the push runs)
  labs/guard-before/          observer with .catch, then guard  (test fails: the push runs)
  labs/guard-after/           guard first, observer without it  (test passes)
  labs/two-guards/            two guards stacked on Bash        (refuses, but with the wrong reason)
  exercises/solutions/one-guard/
```

## 1 · Deny: a guard is a hook that answers

In Lesson 03 a hook that returns without calling `next` **answers** for Claude Code. On `tool.call`, the answer `{ deny: 'reason' }` means the tool never runs. There is no permission prompt, and Claude receives your reason as the tool's result, so it can change course.

```ts
const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/

on('tool.call', { tool: 'Bash' }, ($, e, next) =>
  FORCE_PUSH.test(e.command)
    ? { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    : next(e),
)
```

Write the reason for Claude, not for you: say what was blocked and what to do instead. A bare "denied" makes Claude retry the same thing in a different spelling.

The matcher `{ tool: 'Bash' }` matters. Inside the hook `e.command` is typed as a string only because the matcher narrowed `e` to the Bash tool's input.

✓ Checkpoint: you can say what Claude sees when this guard denies a call (the reason, as the tool result).

## 2 · Hold: ask before you refuse

Refusing every force push is blunt. Sometimes the student really means it. `$.ui.ask` holds the call and shows Claude Code's own question dialog:

```ts
on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
  if (FORCE_PUSH.test(e.command)) {
    const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
    if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
  }
  return next(e)
})
```

Three facts about `$.ui.ask`, all from its declaration in the 2.1.292 types:

- It resolves to the label the person chose, or to free text typed under "Other". That is why the code compares with `'Push anyway'` exactly and treats everything else as no.
- It is itself a **tool call**: a `tool.call` of `AskUserQuestion`, drawn by `ui.render` on `AskUserQuestion`. So other hooks on `tool.call` see it go by. Section 5 is about exactly that.
- It **rejects** when the person dismisses the dialog, and always in a `claude -p` run, where there is no one to ask.

This is the pattern of the `blast-radius` sample mod in `claude-code-playground`: it holds `rm -rf` and force pushes, shows what they would touch, and offers Proceed / Cancel.

**Try it live (optional).** Add the hook to your `hello-mod`, start `claude --plugin-dir ~/mod-gym-lab/hello-mod` in a scratch repo, and ask Claude to run `git push --force`. You get the question before anything runs. Pick **Cancel**: the push does not run, and Claude receives your deny reason as the tool result.

✓ Checkpoint: you can name the two situations in which `$.ui.ask` rejects instead of returning a label.

## 3 · When the guard itself fails

A hook that throws is **skipped**, and the chain carries on without it. For an observer that is exactly right. For a guard it is a hole: if `$.ui.ask` rejects, the guard throws, it is skipped, and the push runs.

`labs/guard-without-catch` has the guard above with no `.catch`. Its test stands in for Claude Code's Bash tool with `() => ({ result: 'pushed' })` and answers nothing else, so `$.ui.ask` rejects as it would under `claude -p`:

```ts
test('a force push is refused when no one can answer the question', async ($, on) => {
  // stands in for Claude Code's Bash tool. Nothing answers AskUserQuestion,
  // so $.ui.ask rejects, as it does in `claude -p`.
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'pushed' }))
  const r = await $.tool.call({ tool: 'Bash', command: 'git push --force origin main' })
  expect(r).toEqual({ deny: 'hello-mod: the force-push guard failed, so the command was not run.' })
  const ok = await $.tool.call({ tool: 'Bash', command: 'git push origin feature' })
  expect(ok.result).toBe('pushed')
})
```

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/guard-without-catch
```

**Expected output** (captured on this Mac; this lab is meant to fail)

```
tests/guard.test.ts:
(fail) a force push is refused when no one can answer the question [16.00ms]
  AssertionError: expect(received).toEqual()
  
  Expected: {
    deny: "hello-mod: the force-push guard failed, so the command was not run."
  }
  Received: { result: "pushed" }
  
  the engine reported:
    hello-mod's tool.call hook was skipped: hello-mod: no implementation for tool.call

 0 pass
 1 fail
```

`Received: { result: "pushed" }`: the guard was skipped and the force push ran.

The fix is `.catch` on the registration. Its handler runs **in place of** the hook when the hook fails:

```ts
on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
  …
}).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))
```

Read the handler as two cases:

| `next.called` | what it means | handler does |
|---|---|---|
| `false` | the guard failed **before** it let the call through | refuse: `{ deny }` (fail closed) |
| `true` | the guard had already called `next`, so it had approved | `next(e)` replays what that call settled to; nothing runs twice |

That is why a guard judges **before** it calls `next`: then "had it called next?" is the same question as "had it approved?".

Validate tells you which gating hooks have a handler:

```bash
# on: laptop
claude plugin validate week29_claude_code_mods/04_guards/labs/guard-after
```

**Expected output** (captured on this Mac; the manifest lines are trimmed)

```
…
  ❯ ./register.ts hooks: tool.call{tool=Bash}, tool.call
  ❯ ./register.ts gating hook with .catch: tool.call{tool=Bash}
  ❯ ./register.ts gating hook without .catch: tool.call
  ❯ ./register.ts calls: $.ui.ask
✔ Validation passed with warnings
```

`gating hook with .catch: tool.call{tool=Bash}` is the guard. `gating hook without .catch: tool.call` is the counter, and Section 5 explains why that one should **not** have a `.catch`.

✓ Checkpoint: you can say what `next.called` is in the handler when `$.ui.ask` rejects (`false`), and what the handler returns.

## 4 · The audit view: gating hooks

`claude plugin validate` lists a `gating hook` for every hook at a site whose answer can refuse what was asked: `tool.call`, `tool.check`, `prompt.submit`, `command.run` and others. It is a fact, not a warning. Before you install anyone's mod, read those lines next to `calls:`:

- A gating hook **with** `.catch` is somebody's guard. Read the deny reasons.
- A gating hook **without** `.catch` either fails open on purpose (an observer) or is a guard with a hole.
- A `calls:` line with `$.process` or `$.http` you did not expect deserves a question before anything else. An organisation can enforce rules like these with its own policy mod in the `prepend` tier, above every user plugin (see *Manage mods*).

✓ Checkpoint: in the output above, you can say which gating hook is the guard and which is the observer.

## 5 · The bug we measured: a fail-open observer above a guard

While building this course we gave the Lesson 03 counter what looked like a harmless safety net, `.catch(($, e, next) => next(e))`, "if the counter breaks, let the tool run anyway", and we registered it **before** the guard. `labs/guard-before`:

```ts
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
```

The guard has its fail-closed `.catch`. Same test as Section 3:

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/guard-before
```

**Expected output** (captured on this Mac; this lab is meant to fail)

```
tests/guard.test.ts:
(fail) a force push is refused when no one can answer the question [17.82ms]
  AssertionError: expect(received).toEqual()
  
  Expected: {
    deny: "hello-mod: the force-push guard failed, so the command was not run."
  }
  Received: { result: "pushed" }
  
  the engine reported:
    hook failed closed: hello-mod: errorKind=HooksError errorChars=31 (tool.call; its .catch answered)
    hello-mod's tool.call hook was skipped: hello-mod: no implementation for tool.call

 0 pass
 1 fail
```

The engine even reports `hook failed closed … its .catch answered`, and the push still ran. What happened, step by step:

```text
Bash "git push --force" ──► counter (outer, .catch → next(e))
                                 └─► guard ── $.ui.ask ──► tool.call AskUserQuestion
                                                             └─► passes through the counter too (no matcher)
                                                                  └─► nothing can answer → the call fails
counter's .catch says "next(e)" → the Bash call goes on without the guard's refusal → "pushed"
```

The question the guard asked is itself a `tool.call` (Section 2), and the catch-all counter sits on every `tool.call`. When that call failed, the counter's fail-open handler answered, and it answered **for the Bash call it was wrapping**. In a real terminal the dialog normally works, so you would only see this where nobody can answer: a `claude -p` run, a CI job, a dismissed dialog.

Three fixes, each measured with the same test:

| fix | change | result |
|---|---|---|
| **A · guard first** (`labs/guard-after`) | register the guard before the observer, and drop the observer's `.catch` | ✓ refused |
| **B · no fail-open catch** | keep the order, just delete `.catch(($, e, next) => next(e))` from the observer | ✓ refused (we ran it) |
| **C · don't match the inner call** | keep both, but give the observer the matcher `{ tool: '!AskUserQuestion' }` | ✓ refused (we ran it) |

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/guard-after
```

**Expected output** (captured on this Mac)

```
tests/guard.test.ts:
(pass) a force push is refused when no one can answer the question [17.10ms]

 1 pass
 0 fail
Ran 1 test across 1 file. [0.15s]
```

Use A. B shows that the observer's `.catch` added nothing: a hook that throws is already skipped, so an observer already fails open without one. C works but depends on you knowing every `$` call every guard makes. `week29_claude_code_mods/reference/hello-mod` and Mod Gym itself both follow A: the guard is registered first, and the catch-all observers have no `.catch`.

**One more measured case: two guards on one tool.** `labs/two-guards` adds a second guard on Bash (refuse recursive deletes outside the lab), registered above the force-push guard, both with fail-closed handlers:

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/labs/two-guards
```

**Expected output** (captured on this Mac; trimmed)

```
…
(fail) a force push is refused when no one can answer the question [17.92ms]
  AssertionError: expect(received).toEqual()
  
  Expected: {
    deny: "hello-mod: the force-push guard failed, so the command was not run."
  }
  Received: { deny: "hello-mod: the delete guard failed." }
…
 1 pass
 1 fail
```

Still refused, because both handlers fail closed, but with the outer guard's reason, so Claude is told the wrong story. Keep **one guard per tool**, with every rule for that tool inside it. That is the exercise.

✓ Checkpoint: you can explain, in two sentences, why the counter's `.catch` let the push run, and which fix you would use.

## 6 · AltoTech tie-in: a mutation gate for Reef

In Week 26 the Reef Lab Runner only let a lesson change the NemoClaw sandbox through one gate per session. A guard gives you the same gate at the agent level, inside Claude Code: one `tool.call` hook on `Bash`, registered first, that holds any command that would change the OpenShell sandbox or its policy, asks with `$.ui.ask` which sandbox and why, and refuses with a reason Claude can act on. Built with this lesson's rules it fails closed when nobody can answer, so a headless agent run cannot change a sandbox at all. Read-only commands go straight to `next(e)`.

## 7 · In the Mod Gym panel

1. Type `/mod-gym 4`, then `/mod-gym` and stay on the *Lesson* tab.
2. Add the guard (Sections 2 and 3) to `~/mod-gym-lab/hello-mod`, registered **before** your Lesson 03 counter.
3. Press **c**. Mod Gym checks that your module returns a `deny`, has a `.catch(` handler, and that `claude plugin validate` reports `gating hook with .catch`.
4. Press **2** and ask Claude to run `git push --force` in a scratch repo, then pick **Cancel**. Whether the Bash row on the Events tape reads `→ denied` depends on which plugin is stacked above the other (Lesson 03, between plugins): if your guard sits above Mod Gym, Mod Gym never sees the call at all.
5. Press **a** for a coaching prompt: Claude reviews your guard like a security reviewer.

✓ Checkpoint: the three Lesson 04 checks are ✓ in the panel.

## Try it yourself

Add a second rule to your guard: refuse, without asking, any recursive `rm` whose target starts at `/`, `~`, `$HOME` or `..`. A delete inside the lab (`rm -rf build/`) must still run. Put the rule **inside the same Bash guard** (Section 5, two guards), keep the guard registered first, and write a test for both cases.

<details><summary>Solution — <code>exercises/solutions/one-guard</code></summary>

```ts
import type { Register } from 'claude-code'

const FORCE_PUSH = /git\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)/
const WIDE_DELETE = /\brm\s+-[a-z]*r[a-z]*\s+(\/|~|\$HOME|\.\.)/

export const register: Register = on => {
  // L04 · one guard for Bash, registered first; every Bash rule lives here
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    if (WIDE_DELETE.test(e.command)) {
      return { deny: 'hello-mod: recursive delete outside the lab folder refused: ' + e.command }
    }
    if (FORCE_PUSH.test(e.command)) {
      const pick = await $.ui.ask('Force push from a class repo?', ['Cancel', 'Push anyway'])
      if (pick !== 'Push anyway') return { deny: 'Force push blocked by hello-mod. Push to a new branch instead.' }
    }
    return next(e)
  }).catch(($, e, next) => (next.called ? next(e) : { deny: 'hello-mod: the force-push guard failed, so the command was not run.' }))

  // L03 · the observer, second, with no .catch
  let calls = 0
  on('tool.call', async ($, e, next) => {
    const r = await next(e)
    calls += 1
    return r
  })
}
```

```ts
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
```

```bash
# on: laptop
claude plugin test week29_claude_code_mods/04_guards/exercises/solutions/one-guard
```

**Expected output** (captured on this Mac)

```
(pass) wide recursive deletes are refused, lab deletes run [15.77ms]
…
(pass) a force push is refused when no one can answer the question [16.95ms]

 2 pass
 0 fail
```

The regex is a speed bump, not a sandbox: `rm -rf "$PWD/../x"`, `find / -delete` or a script that deletes for you all slip past it. A guard that must really hold works on resolved paths (`$.fs.stat(path, { resolve: true })` answers the `realPath`) and on an allow-list of places, not a deny-list of spellings.

</details>

✓ Checkpoint: both tests in your solution pass, and the force-push test still passes with the delete rule added.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The force push runs in `claude -p` but is held in the terminal | `$.ui.ask` rejects headless; the guard has no `.catch`, or a fail-open observer sits above it | add the fail-closed `.catch`; register the guard first; drop `.catch(next(e))` from observers |
| `labs/guard-before` and `labs/guard-without-catch` fail | they are meant to | compare with `labs/guard-after` |
| The deny reason is from a different guard | two guards with `.catch` stacked on one tool | merge them into one guard per tool |
| `e.command` is not typed | the hook has no `{ tool: 'Bash' }` matcher | add the matcher |

## Next

[Lesson 05 — Draw: panes, bands, status, toast](../05_drawing/TUTORIAL.md): so far your mod has only spoken through the status line and Claude's tool results. Lesson 05 draws your own band above the prompt and a pane like Mod Gym's, with elements from `$.ui.resolve(e)`.
