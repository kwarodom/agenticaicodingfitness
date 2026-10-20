# ▶ Mod Gym Lesson 08 — Test, validate, ship (capstone)

> Part of Week 29 · Claude Code mods. You learn mods by changing the coding agent you use every week. The **Mod Gym** panel (`/mod-gym`) walks you through eight lessons and checks your lab mod, `hello-mod`, as it grows. This last lesson makes the mod trustworthy and shareable: tests that run with no session, a strict type-check, `claude plugin validate`, and a marketplace file so a classmate can install it with one line. The capstone audits a real starter mod that was written from the docs but never run.

**What you'll actually do**
- Write tests with the `claude-code/testing` kit: stubs that sit beneath your plugin and answer in Claude Code's place.
- Learn the four rules that made every first test in this course fail, and the messages that tell you which one you broke.
- Type-check the mod with the declarations Claude Code writes for your build.
- Read `claude plugin validate` as a reviewer, add `marketplace.json`, and ship: GitHub install line, folder marketplace, version bumps.
- **Capstone:** take the research starter `week29/mod-agentic-coding-fitness`, run it through test, validate and `tsc`, and fix what is actually broken.

**Time** ~75 min · **Difficulty** intermediate · **Hardware** this laptop only

**Sources:** [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) · [Host a marketplace](https://code.claude.com/docs/en/plugins/host-marketplace) · [Plugin evals](https://code.claude.com/docs/en/plugin-evals) · the API declarations Claude Code 2.1.292 writes (`claude-code.d.ts`), which win wherever they and the docs differ

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lessons 01–07 | `hello-mod` has `/reps` and `mcp__hello-mod__log_reps` | the tests below call both |
| Claude Code 2.1.287+ | `claude --version` → `2.1.292 (Claude Code)` here | `claude plugin test` ships with mods |
| Node (for `tsc`) | `npx --version` | section 4 runs TypeScript through `npx` |
| A GitHub repo (optional) | `gh auth status` | only to publish in section 6 |

The finished lab mod, tests included, is `week29/reference/hello-mod`. Compare against it whenever you are stuck.

## 1 · The test kit: you play Claude Code

`claude plugin test <dir>` runs every `tests/*.test.ts` in an environment like the one your hooks run in, but with no session, no sign-in, no network and no file system. The engine loads your plugin as a session would. Underneath it there is **nothing**: the hooks you register with the test's `on` stand in for Claude Code.

```ts
// tests/hello-mod.test.ts
import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

// The engine fills these fields in a real session; a test spells them out.
const START = { cwd: '/home/student', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

// Every hook `on` adds here sits beneath the plugin and answers in Claude Code's place.
function engine(on: On, saved: Record<string, unknown> = {}) {
  mock.store(on, saved)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
}

test('/reps starts at zero and log_reps adds to it', async ($, on) => {
  engine(on)
  await $.session.start(START)
  expect((await $.command.run(cmd('reps', ''))).text).toContain('0 reps')
  const r = await $.tool.call({ tool: 'mcp__hello-mod__log_reps', n: 10 })
  expect(String(r.result)).toContain('Total: 10')
  expect((await $.command.run(cmd('reps', ''))).text).toContain('10 reps')
})

test('reps saved by an earlier session are loaded from $.store', async ($, on) => {
  engine(on, { reps: 7 })
  await $.session.start(START)
  expect((await $.command.run(cmd('reps', ''))).text).toContain('7 reps')
})
```

The kit gives you three ready-made stand-ins: `mock.store(on, entries?)` (an in-memory `$.store`), `mock.clock(on)` (a clock you advance by hand) and `mock.env(on, { HOME: … })`. Everything else you stub yourself.

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/hello-mod
```

**Expected output** (captured on this Mac with `week29/reference/hello-mod`, which has two more tests: the guard and the prompt context)

```
tests/hello-mod.test.ts:
(pass) /reps starts at zero and log_reps adds to it [22.22ms]
(pass) reps saved by an earlier session are loaded from $.store [9.41ms]
(pass) a force push is refused when no one can answer the question (the guard fails closed) [11.02ms]
(pass) the prompt gets the coach context [9.24ms]

 4 pass
 0 fail
Ran 4 tests across 1 file. [0.19s]
```

✓ Checkpoint: your own `claude plugin test` is green.

## 2 · Four rules, and the message for each

Every first test in this course failed. These are the reasons, with the exact message the kit printed.

**Rule 1 · A stub is a hook: `($, e)`, not `(e)`.** The stand-ins take the same arguments as your plugin's hooks. A one-argument stub receives `$`, so `e.key` is `undefined`. Nothing fails loudly: the starter in the capstone stored every value under the key `"undefined"` and its tests still passed. We proved it with a stub that records what it was asked:

```
  Expected: ["progress"]
  Received: [undefined]
```

**Rule 2 · `session.start` needs a `cwd`, and something must answer it.** `$.session.start({})` fails, and so does a test with no `session.start` stub:

```
  HooksError: agentic-coding-fitness: next() passed an argument with no { cwd }
```

```
  HooksError: no implementation for session.start

  nothing beneath the plugins answers session.start: a test answers it with on('session.start', ...)
```

The second message is the general rule: **"no implementation for X" means stub X.** It names the event; add `on('X', …)`. One more catch: register every stub *before* the first call on `$`. A late one throws `on("ui.close") after the test first called $`.

**Rule 3 · A plugin's refusal comes back as `{ deny }`.** When your guard refuses a call that the test made through `$.tool.call`, the result is `{ deny: '…' }`, not `isError`. The reference guard test:

```ts
test('a force push is refused when no one can answer the question (the guard fails closed)', async ($, on) => {
  engine(on)
  // $.ui.ask is a tool.call of AskUserQuestion; with nothing here to answer it, it rejects
  on('tool.call', { tool: 'Bash' }, () => ({ result: 'pushed' }))
  await $.session.start(START)
  const r = await $.tool.call({ tool: 'Bash', command: 'git push --force origin main' })
  expect(r.deny).toContain('force-push guard failed')
  const ok = await $.tool.call({ tool: 'Bash', command: 'git push origin feature' })
  expect(ok.result).toBe('pushed')
})
```

This test is how Lesson 04's ordering rule was found: with a catch-all observer registered first and given `.catch(($, e, next) => next(e))`, the same test returned `{ result: "pushed" }`.

**Rule 4 · A missing stub can be silent.** A hook that throws is skipped. Delete the `tool.register` stub from the reference tests and all four still pass: no test depended on the tool having been registered. If registration matters to you, assert on it.

✓ Checkpoint: you can name the fix for `no implementation for turn.complete` without looking (add `on('turn.complete', ($, e) => ({ text: e.answer }))`).

## 3 · UI tests: mount the pane and press its buttons

`$.ui.mount` draws a component through your plugin on a surface you name and lets you act on it by `key`. Mod Gym's own test (`week29/mod-gym/tests/mod-gym.test.ts`) loops over two surfaces:

```ts
for (const surface of ['terminal', 'desktop'] as const) {
  await $.command.run(cmd('mod-gym', '1'))
  const ui = await $.ui.mount({ plugin: 'mod-gym', surface, ...PANE })
  expect(await ui.find({ type: 'Text', text: /Lesson 01 of 08/ })).toBeDefined()
  await ui.press({ key: 'next' })
  expect(await ui.find({ type: 'Text', text: /Lesson 02 of 08 · Your first mod/ })).toBeDefined()
  // …
  await ui.unmount()
}
```

`PANE` holds the full `Pane` props: `title`, `isFocused`, `bodyColumns`, `placement`, `scroll` and `view`. A press runs the button's real `onPress`, so a bug in a handler shows up here. The capstone finds one this way.

## 4 · Type-check: the cheapest reviewer

Once Claude Code has loaded your mod from its folder (`claude --plugin-dir`, or even one `claude -p --plugin-dir … "/reps"`), it lays this build's declarations in `.claude-plugin/types/` and a `tsconfig.json` next to your mod that extends them. Then:

```bash
# on: laptop
npx -y -p typescript@5.6 tsc -p ~/mod-gym-lab/hello-mod; echo "tsc exit $?"
```

**Expected output** (captured on this Mac, the reference mod)

```
tsc exit 0
```

The laid `tsconfig.json` is `strict`, with `noUncheckedIndexedAccess`, and it includes `hooks`, `types` and `tests`. Type-check before you run tests: on the starter's original test file `tsc` reports 20 errors, and the first ones are exactly Rules 1 and 2:

```
orig/tests/fitness.test.ts(8,44): error TS2339: Property 'key' does not exist on type 'EngineInterface'.
…
orig/tests/fitness.test.ts(15,25): error TS2345: Argument of type '{}' is not assignable to parameter of type 'SessionStartInput'.
```

Writing JavaScript? Put a JSDoc type on the entry point, `/** @param {import('claude-code').On} on */`, and `tsc` with `allowJs`/`checkJs` will type every `$`, `e` and `next` under it.

✓ Checkpoint: `tsc` exits 0 on your mod.

## 5 · Validate: read it as a reviewer would

`claude plugin validate` reads the marketplace file, the manifest and the hooks module the way the engine will, without running anything. Read three kinds of line:

- **`hooks:`**: every event and matcher. A `?` (as in `tool.call{tool=?}`) means the matcher is not a literal and the reader could not see it.
- **`calls:`**: everything the mod can reach through `$`. Anything you did not expect under `$.process`, `$.http` or `$.fs` is a question for the author.
- **`gating hook with / without .catch:`**: the hooks whose failure decides whether something goes through.

Add the marketplace file next to `plugin.json` (`.claude-plugin/marketplace.json`):

```json
{
  "name": "hello-mod",
  "owner": { "name": "Agentic Coding Fitness student" },
  "plugins": [{ "name": "hello-mod", "source": "./" }]
}
```

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/hello-mod
```

**Expected output** (captured on this Mac, the top of the report)

```
Validating marketplace manifest: …/hello-mod/.claude-plugin/marketplace.json

⚠ Found 2 warnings:

  ❯ description: No marketplace description provided. Adding a description helps users understand what this marketplace offers
  ❯ plugins[0] plugin.json → author: No author information provided. Consider adding author details for plugin attribution

Validating plugin: …/hello-mod/.claude-plugin/plugin.json
…
```

Warnings do not fail validation; errors do. Two more errors you met in earlier lessons: a `$.state` key missing from `types/index.d.ts` (`hello-mod.reps is not declared`), and a plugin name that passes as Anthropic's own (`claude-…`, `anthropic-…`).

✓ Checkpoint: validate reads all three files and ends with `✔ Validation passed` (with or without warnings).

## 6 · Ship it

**From GitHub.** Push the mod folder as its own repository (the marketplace file's `"source": "./"` points at the repo root). A classmate types, at a terminal session's prompt:

```text
/plugin install hello-mod --marketplace <you>/<repo>
```

They answer `y` to add the marketplace and pick a scope. A mod that is only a hooks module is active at once.

**From a folder (how Mod Gym is installed on the teacher's machine).** A marketplace can be a folder. `week29/.claude-plugin/marketplace.json` is one, listing `mod-gym` (`./mod-gym`) and the reference `hello-mod` (`./reference/hello-mod`):

```bash
# on: laptop — do this once, from the repo root
claude plugin marketplace add ./week29
claude plugin install mod-gym@fitness-week29 --scope user
claude plugin list
```

**Expected output** (captured on this Mac, the `mod-gym` entry of `claude plugin list`)

```
  ❯ mod-gym@fitness-week29
    Version: 0.1.0
    Read from: /Users/altodev/Desktop/agenticaicodingfitness/week29/mod-gym
    Scope: user
    Status: ✔ enabled
```

`Read from:` is the folder itself, not a copy. Edit a file there and run `/reload-plugins` in your session: no new version, no reinstall.

**Version bumps.** Every other install (a GitHub or git source) runs the copy made at install time, cached by `version`. Edits reach your classmates only when you raise `version` in `plugin.json` and they run `/plugin marketplace update`, then `claude plugin update hello-mod`, then `/reload-plugins`.

**The class marketplace.** The repo root's `.claude-plugin/marketplace.json` (marketplace `agentic-coding-fitness`, the one students already added) now lists two plugins:

```
[('agentic-coding-fitness', './plugins/agentic-coding-fitness'), ('mod-gym', './week29/mod-gym')]
```

Once that is pushed, students run `/plugin marketplace update agentic-coding-fitness`, then `/plugin install mod-gym@agentic-coding-fitness`.

**One more gate (optional).** `claude plugin eval` compares Claude with and without your plugin on cases in `evals/`; `claude plugin eval init` drafts them in the format your build accepts. The starter has one case, `evals/coach-explains-mods/`.

✓ Checkpoint: `marketplace.json` exists, validate passes, and you can explain why a GitHub install needs a version bump but a folder install does not.

## 7 · Capstone: audit the research starter

`week29/mod-agentic-coding-fitness` is the v3.0.0 starter from this week's research: a `/fitness` progress pane, `/fitness-log`, `/fitness-grade`, a `log_lab` tool and auto-logging of passing `pytest` runs. Its author wrote it from the docs and said so: *"The starter `register.js` and tests were not executed against a live Claude Code."* Your job: make it pass test, validate **and** `tsc`, changing only what is actually wrong. Work on a copy; leave the original as it is.

```bash
# on: laptop
cp -R week29/mod-agentic-coding-fitness ~/mod-gym-lab/starter && find ~/mod-gym-lab/starter -name .DS_Store -delete
claude plugin validate ~/mod-gym-lab/starter
cd ~/mod-gym-lab/starter && claude plugin test .
```

**Expected output** (captured on this Mac; validate trimmed to the interesting lines)

```
  ❯ ./register.js hooks: session.start, tool.call{tool=?}, tool.call{tool=Bash}, turn.complete, prompt.submit, command.run{command=fitness}, …
  ❯ ./register.js gating hook without .catch: tool.call{tool=?}
…
✔ Validation passed

tests/fitness.test.ts:
(fail) /fitness-log records a lab and reports pass/fail [23.28ms]
  HooksError: agentic-coding-fitness: next() passed an argument with no { cwd }
…
 0 pass
 3 fail
```

Validate passes, yet it already shows a problem (`tool=?`), and every test fails. Walk it the way you would any untested code: one failure at a time, re-run after each change. What we found on this Mac, in order:

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `next() passed an argument with no { cwd }` | `$.session.start({})` | `$.session.start(START)` with `cwd`, `surface`, `isInteractive` |
| 2 | `no implementation for session.start` | no stub answers it | `on('session.start', ($, e) => ({ cwd: e.cwd }))` |
| 3 | `no implementation for turn.complete` | the auto-log test raises `turn.complete` with nothing beneath | `on('turn.complete', ($, e) => ({ text: e.answer }))` |
| 4 | 3 pass, but the store stub sees `[undefined]` | stubs written `(e) =>` (Rule 1) | `mock.store(on)`; every stub `($, e) =>` |
| 5 | `tool.call{tool=?}` in validate | matcher built as `'mcp__' + PLUGIN + '__log_lab'` | the literal `'mcp__agentic-coding-fitness__log_lab'` |
| 6 | pressing **Close** throws `ui.close: id is 1 to 64 of letters, digits, _ or - (host check)` | `$.ui.close(PANE)` passes a string | `$.ui.close({ id: PANE })` |
| 7 | `tsc` with `checkJs`: 43 errors | an untyped JS module; among them `'key' does not exist in type 'TextProps'`, a `Date - Date` subtraction and a `find(...)` destructured without a fallback | JSDoc types on `register`, the module state and the helpers; drop `key` from `Text`; `getTime()`; `?? ['', 'unknown']` |

After fix 3 the original three tests were green, but **green was not done**. Fixes 4 to 7 came from a second test file that asks questions the original tests never asked: does yesterday's progress load, can Claude actually call `log_lab`, does the pane draw, does **Close** work. That file found the Close bug, which would have hit every student who pressed **q**.

What the audit also tells you about the starter's design: the pane, the spinner suffix, the `pytest` auto-log and the `turn.complete` note all worked once the tests could reach them. Most of what was wrong was in the tests and in the types, which is what an untested starter usually looks like.

**Expected output** (captured on this Mac, after all seven fixes)

```
tests/audit.test.ts:
(pass) progress saved yesterday is loaded today [25.84ms]
(pass) Claude can log a lab with the log_lab tool [13.47ms]
(pass) the /fitness pane draws [16.13ms]

tests/fitness.test.ts:
(pass) /fitness-log records a lab and reports pass/fail [25.78ms]
(pass) a passing weekNN pytest run is auto-logged and announced after the turn [13.13ms]
(pass) /fitness-grade uses the model stub [11.72ms]

 6 pass
 0 fail
Ran 6 tests across 2 files. [0.19s]
```

```
  ❯ ./register.js hooks: session.start, tool.call{tool=mcp__agentic-coding-fitness__log_lab}, tool.call{tool=Bash}, …
✔ Validation passed
tsc exit 0
```

✓ Checkpoint: test, validate and `tsc` all pass on your copy, and you can name the one fix a student would have noticed first (the Close button).

## In the Mod Gym panel

Type `/mod-gym 8`, then press **c**. The gym checks `~/mod-gym-lab/hello-mod`:

| Check | Passes when |
|---|---|
| has a tests/*.test.ts | `hello-mod/tests/` holds a file ending `.test.ts` |
| claude plugin test passes | `claude plugin test` exits 0 |
| marketplace.json exists | `hello-mod/.claude-plugin/marketplace.json` exists |
| validate passes with the marketplace | `claude plugin validate` exits 0 |

Pass all eight lessons and the **Progress** tab (**3**) shows eight ✓.

## Exercise — the capstone, written up

Do section 7 yourself on `~/mod-gym-lab/starter`. For each fix, write one line: the symptom, the cause, the change. Then add the second test file that would have caught the Close bug before any student did.

<details><summary>Solution</summary>

**`tests/fitness.test.ts`**: correct stubs, full inputs.

```ts
import { expect, mock, test } from 'claude-code/testing'

// Runs with `claude plugin test` from the plugin directory: no session, sign-in, or network.
// Stubs answer in Claude Code's place. Register every stub before the first call on `$`.
// A stub is a hook like any other: ($, e, next). A one-argument stub receives `$`, not the event.

const START = { cwd: '/lab', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

test('/fitness-log records a lab and reports pass/fail', async ($, on) => {
  mock.store(on)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.invalidate', () => ({ value: undefined }))

  await $.session.start(START)
  const ok = await $.command.run(cmd('fitness-log', '05 agent-loops 0.86'))
  expect(ok.text).toContain('05/agent-loops → 0.86 (pass)')
  // … the low-score and usage cases as before
})
```

In the auto-log test add `on('turn.complete', ($, e) => ({ text: e.answer }))` and raise it with full fields: `$.turn.complete({ turnId: 't1', answer: 'ok', durationMs: 10, isAborted: false, reason: 'answer' })`. In the grader test replace the two store stubs with `mock.store(on)`.

**`tests/audit.test.ts`**: the questions the original tests never asked.

```ts
import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const START = { cwd: '/lab', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const

function engine(on: On, saved: Record<string, unknown> = {}) {
  mock.store(on, saved)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.invalidate', () => ({ value: undefined }))
}

test('progress saved yesterday is loaded today', async ($, on) => {
  engine(on, { progress: { labs: { '02/llm-fundamentals': { score: 1, at: 'x' } }, sessions: [] } })
  await $.session.start(START)
  expect((await $.command.run(cmd('fitness-next'))).text).toContain('week 03')
})

test('Claude can log a lab with the log_lab tool', async ($, on) => {
  engine(on)
  await $.session.start(START)
  const r = await $.tool.call({ tool: 'mcp__agentic-coding-fitness__log_lab', week: '05', skill: 'agent-loops', score: 0.9 })
  expect(String(r.result)).toContain('05/agent-loops')
})

test('the /fitness pane draws', async ($, on) => {
  engine(on)
  on('session.usage', () => ({ value: { context: { percent: 12 } } as never }))
  const closed: unknown[] = []
  on('ui.close', ($, e) => { closed.push(e); return { value: undefined } })
  await $.session.start(START)
  const ui = await $.ui.mount({
    plugin: 'agentic-coding-fitness', surface: 'terminal', component: 'Pane', requestId: 'fitness',
    props: { title: 'Agentic Coding Fitness', isFocused: true, bodyColumns: 80, placement: 'dock', scroll: { offset: 0, bodyRows: 40 }, view: {} },
  })
  expect(await ui.find({ type: 'Text', text: /W02/ })).toBeDefined()
  await ui.press({ key: 'close' })
  expect(closed).toEqual([expect.objectContaining({ id: 'fitness' })])
  await ui.unmount()
})
```

**`hooks/register.js`**: the whole diff (only `+`/`-` lines).

```diff
+/** @type {{ labs: Record<string, { score: number, at: string }>, sessions: string[] }} */
+/** @type {string | null} */
+/** @param {string[]} sessions */
-    const diff = Math.round((cursor - new Date(d + 'T00:00:00Z')) / 86400000)
+    const diff = Math.round((cursor.getTime() - new Date(d + 'T00:00:00Z').getTime()) / 86400000)
+/** @typedef {import('claude-code').EngineInterface} Engine */
+/** @param {Engine} $ */
+/** @param {Engine} $ @param {string} week @param {string} skill @param {number} score */
+/** @param {string} args */
+/** @param {import('claude-code').On} on */
-  on('tool.call', { tool: LOG_TOOL }, async ($, e) => {
+  on('tool.call', { tool: 'mcp__agentic-coding-fitness__log_lab' }, async ($, e) => {
-    const [, skill] = CURRICULUM.find(([cw]) => cw === w)
+    const [, skill] = CURRICULUM.find(([cw]) => cw === w) ?? ['', 'unknown']
-      return Text({ key: 'row-' + w, dimColor: !lab, children: [mark + ' W' + w + '  ' + s + score] })
+      return Text({ dimColor: !lab, children: [mark + ' W' + w + '  ' + s + score] })
-            Button({ key: 'close', label: 'Close', hotkey: 'q', onPress: () => $.ui.close(PANE) }),
+            Button({ key: 'close', label: 'Close', hotkey: 'q', onPress: () => $.ui.close({ id: PANE }) }),
```

For `tsc`, use the tsconfig from section 4 with `"allowJs": true, "checkJs": true` and `hooks` and `tests` included. The run ends `tsc exit 0`.

</details>

✓ Checkpoint: six green tests, `✔ Validation passed` with `tool=mcp__agentic-coding-fitness__log_lab`, and `tsc exit 0`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no implementation for X` | nothing beneath the plugin answers event X | add `on('X', ($, e) => …)` before the first `$` call |
| `on("X") after the test first called $` | a stub registered after `$.session.start` | move it up |
| A stub's `e.key` / `e.path` is `undefined` | the stub is written `(e) =>` | write `($, e) =>` |
| `expect(r.isError)` is `undefined` for a refused call | a plugin refusal is `{ deny }` | assert on `r.deny` |
| `tsc -p <mod>` cannot find `claude-code` | the mod has never been loaded, so `.claude-plugin/types/` is missing | load it once (`claude -p --plugin-dir <mod> "/reps"`), or point a tsconfig at `claude-code.d.ts` |
| Classmates do not get your fix | the installed copy is cached by `version` | bump `version`; they run `/plugin marketplace update`, `claude plugin update <mod>`, `/reload-plugins` |

## What's next

You finished Week 29. Keep Mod Gym installed: its **Events** tab is a live view of the hook pipeline in every session, and `/mod-gym status` shows your streak. Next stop: take the fixed starter, rename it, and make it your own class progress mod.
