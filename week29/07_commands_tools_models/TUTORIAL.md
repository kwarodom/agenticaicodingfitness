# ▶ Mod Gym Lesson 07 — Commands, tools and models

> Part of Week 29 · Claude Code mods. You learn mods by changing the coding agent you use every week. The **Mod Gym** panel (`/mod-gym`) walks you through eight lessons and checks your lab mod, `hello-mod`, as it grows. This lesson gives your mod things that can be *called*: a slash command you type, a tool Claude calls, and three ways out of the mod (a small cloud model, a local model on your own box, and a program on your machine).

**What you'll actually do**
- Register a slash command in `session.start` and answer it in `command.run` with `{ text }`. No Claude turn, no tokens.
- Register a tool, see Claude list it as `mcp__hello-mod__log_reps`, serve it with a `tool.call` hook, and watch Claude call it.
- Read two worked examples inside Mod Gym itself: the `gym_status` tool and the **Ask Claude** button (`$.prompt.fill`).
- Build `setpoint-coach`, a small hotel HVAC mod that asks a cloud model (`$.model.complete`), a local model (`$.http.fetch` to Ollama or a DGX Spark) and a program (`$.process.run`).
- Decide which of the five shapes fits a feature, and test all of them with no network.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** this laptop only (Ollama optional, for the sovereign step)

**Sources:** [Claude Code mods (Anthropic blog)](https://claude.com/blog/claude-code-mods) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api) · [React to events](https://code.claude.com/docs/en/plugins/mods/events) · [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) · the API declarations Claude Code 2.1.292 writes (`claude-code.d.ts`), which win wherever they and the docs differ

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lessons 01–06 | `hello-mod` has a band, a `reps` atom and a `types/index.d.ts` | this lesson reads and writes the `reps` atom |
| Claude Code 2.1.287+ | `claude --version` → `2.1.292 (Claude Code)` here | mods need it |
| Mod Gym installed | `/mod-gym` opens the panel | the checks at the end |
| Ollama (optional) | `curl -s localhost:11434/api/tags` lists `gemma4:12b` | only for `/spark-ask` in section 4 |

Your lab mod lives in `~/mod-gym-lab/hello-mod` (change it with `/mod-gym lab <path>`). The finished version of everything in this lesson is in `week29/reference/hello-mod`.

## 1 · A slash command: an answer with no Claude turn

A command has two halves. You **register** it once, in `session.start`, so it shows in the typeahead and `/help`. You **answer** it in a `command.run` hook whose matcher names it.

```ts
// hooks/register.tsx — inside register(on)
on('session.start', async ($, e, next) => {
  const saved = await $.store.get('reps')
  if (typeof saved === 'number') await update($, reps, () => saved)
  await $.command.register({ name: 'reps', description: 'Show the reps hello-mod has logged' })
  // … the tool from section 2 goes here too
  return next(e)
})

on('command.run', { command: 'reps' }, async $ => ({ text: (await read($, reps)) + ' reps logged' }))
```

Three things to notice:

- `session.start` is awaited before the first prompt, so a command registered there exists by turn one.
- The `command.run` hook returns `{ text }` and never calls `next`. It **answers** for Claude Code. The model is not asked anything, so the command costs no tokens.
- Want arguments? Add `argumentHint: '<room>'` to the registration and read `e.args` (the raw text after the command name) in the hook.

Run it headless with the mod loaded from its folder:

```bash
# on: laptop
claude -p --plugin-dir ~/mod-gym-lab/hello-mod "/reps"
```

**Expected output** (captured on this Mac, Claude Code 2.1.292)

```
hello-mod: 0 reps logged
```

Claude Code puts the plugin's name in front of a command's output (`hello-mod: …`), so leave it out of `text`, or it prints twice.

The first `hello-mod:` is Claude Code naming the plugin that answered. The second is the reference mod's own text. Once you know the engine adds the name, you can drop it from your own text.

`claude plugin validate` reports the command as one the module answers itself:

```
  ❯ ./register.tsx answers its own command: command.run{command=reps}
```

✓ Checkpoint: `/reps` prints a count, and you can say why it costs no tokens.

## 2 · A tool Claude can call

A command is for **you**. A tool is for **Claude**: it appears in the model's tool list, and the model decides when to call it. Register it the same way, then serve it with a `tool.call` hook.

```ts
// in session.start, after the command
await $.tool.register({
  name: 'log_reps',
  description: 'Record reps the student finished. Call it when the student says they did reps.',
  inputSchema: { type: 'object', properties: { n: { type: 'number' } }, required: ['n'] },
})

// serve the tool Claude sees as mcp__hello-mod__log_reps
on('tool.call', { tool: 'mcp__hello-mod__log_reps' }, async ($, e) => {
  const n = Number(e.n)
  if (!Number.isFinite(n) || n <= 0) return { result: 'n must be a positive number' }
  await update($, reps, r => r + n)
  return { result: 'Logged ' + n + ' reps. Total: ' + (await read($, reps)) }
})
```

- The model sees the tool as `mcp__<plugin>__<name>`. Your matcher must use that full name.
- Write the matcher as a **string literal**. `claude plugin validate` reads your source without running it; a name built with `+` shows up as `tool=?` (you meet one in Lesson 08's capstone).
- The `description` is a prompt. Say *when* to call the tool, not just what it does.
- Check the input. The schema is a request to the model, not a guarantee.
- A call that no hook answers fails with an error saying so.

Ask Claude in plain words, then read the count back with the command. In `-p` mode nobody can approve a tool, so allow this one tool by name:

```bash
# on: laptop
claude -p "I just did 10 squats. Log them with your tool, then tell me the total in one short line." \
  --plugin-dir ~/mod-gym-lab/hello-mod --allowedTools=mcp__hello-mod__log_reps
claude -p --plugin-dir ~/mod-gym-lab/hello-mod "/reps"
```

**Expected output** (captured on this Mac; the first line is the model's own words and will differ)

```
10 reps logged — total: 10. Great job! 💪
hello-mod: 10 reps logged
```

The second session read `10` back, because the `state.set` hook from Lesson 06 copies every change of `reps` into `$.store`.

✓ Checkpoint: Claude called `mcp__hello-mod__log_reps`, and a new session's `/reps` shows the new total.

## 3 · Two worked examples inside Mod Gym

Open `week29/mod-gym/hooks/register.tsx`. The panel uses both shapes.

**The `gym_status` tool.** Mod Gym registers a tool with an empty input schema. Its description tells Claude when to call it and how to coach:

```ts
await $.tool.register({
  name: 'gym_status',
  description:
    "Where the student is in Mod Gym (Week 29: Claude Code mods): current lesson, its tutorial path, the student's lab folder, last check results and passed lessons. Call it before coaching the student on a Mod Gym lesson; then guide with questions rather than writing their lab code.",
  inputSchema: { type: 'object', properties: {} },
})
```

The hook answers with JSON text: lesson, lab folder, last check results, a coaching rule. It also has a `.catch` that answers `'Mod Gym status is unavailable right now.'` if building the status throws. A tool should always give the model *some* answer.

```ts
const TOOL = 'mcp__mod-gym__gym_status'
on('tool.call', { tool: TOOL }, async $ => {
  // … build `status` from the atoms
  return { result: JSON.stringify(status, null, 2) }
}).catch(() => ({ result: 'Mod Gym status is unavailable right now.' }))
```

`TOOL` is a `const` holding one string literal, so validate can still read it: `tool.call{tool=mcp__mod-gym__gym_status}`.

**The Ask Claude button.** Press **a** in the panel. Mod Gym does *not* submit a prompt for you. It fills your prompt box and lets you edit before you press Enter:

```ts
async function askClaude($: $): Promise<void> {
  const lesson = lessonAt(await read($, lessonAtom))
  const lab = await labDir($)
  const text =
    '[Mod Gym · lesson ' + lesson.id + ': ' + lesson.title + ' · lab folder ' + lab + ' · tutorial ' + lesson.tutorial + ']\n' +
    lesson.ask
  await $.prompt.fill({ text, mode: 'replace' })
  $.ui.toast('Question is in your prompt: edit it or press Enter.')
}
```

`$.prompt.submit` would start a turn by itself. `$.prompt.fill` keeps a person in the loop, which is the right default for anything that spends tokens.

✓ Checkpoint: you can say which of the two the model triggers (the tool) and which the person triggers (the button).

## 4 · Reaching outside: a cloud model, a local model, a program

A hooks module has no Node, no `fetch` of its own and no shell. Everything outside goes through `$`, which is also what makes a mod auditable: `claude plugin validate` lists every call. Make a second lab mod, `setpoint-coach`, next to `hello-mod`:

```text
setpoint-coach/.claude-plugin/plugin.json   { "name": "setpoint-coach", "version": "0.1.0", "description": "Lesson 07 demo: a cloud model, a local model and a program, each behind a slash command." }
setpoint-coach/hooks/hooks.json             { "modules": ["./register.ts"] }
setpoint-coach/hooks/register.ts
```

```ts
import type { Register } from 'claude-code'

const SYSTEM = 'You explain hotel HVAC setpoints to a duty manager. Two sentences, no jargon.'
const SPARK = 'http://localhost:11434/v1/chat/completions' // Ollama here; a DGX Spark on the LAN in class

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'hvac-explain', description: 'Ask a small cloud model about a setpoint', argumentHint: '<question>' })
    await $.command.register({ name: 'spark-ask', description: 'Ask the local model (sovereign tier)', argumentHint: '<question>' })
    await $.command.register({ name: 'git-where', description: 'Show the branch this session is on' })
    return next(e)
  })

  // A side question to a small cloud model: no history, no tools, always a result
  on('command.run', { command: 'hvac-explain' }, async ($, e) => {
    if (!e.args.trim()) return { text: 'Usage: /hvac-explain <question>' }
    const r = await $.model.complete({ model: 'haiku', system: SYSTEM, prompt: e.args, maxTokens: 120, timeoutMs: 20_000 })
    return { text: r.isAnswered ? r.text.trim() : 'No answer (' + r.reason + ')' }
  })

  // The same question to a model on your own box: the data never leaves the building
  on('command.run', { command: 'spark-ask' }, async ($, e) => {
    if (!e.args.trim()) return { text: 'Usage: /spark-ask <question>' }
    const r = await $.http.fetch(SPARK, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma4:12b',
        reasoning_effort: 'none',
        max_tokens: 120,
        messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: e.args }],
      }),
    })
    if (!r.ok) return { text: 'Local model failed: HTTP ' + r.status }
    const reply = JSON.parse(r.text) as { choices: { message: { content: string } }[] }
    return { text: (reply.choices[0]?.message.content ?? '').trim() }
  })

  // A program, run by argv: no shell, so no quoting bugs and no injected `; rm`
  on('command.run', { command: 'git-where' }, async $ => {
    const r = await $.process.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], { cwd: await $.session.cwd(), timeoutMs: 10_000 })
    return { text: r.exitCode === 0 ? 'On branch ' + r.stdout.trim() : 'Not a git repo (' + r.stderr.trim() + ')' }
  })
}
```

What each call promises:

- **`$.model.complete`** sends one prompt to a model on the session's own client, with no history and no tools. It never throws over what the provider did: it resolves `isAnswered: true` with `text`, or `isAnswered: false` with a `reason` (`api-error`, `empty-reply`, `aborted` when `timeoutMs` ran out). Always check `isAnswered`.
- **`$.http.fetch`** resolves `{ status, ok, headers, text }`. Point it at an OpenAI-compatible endpoint on Ollama or a DGX Spark and the question never leaves the building: the sovereign tier of Alto Copilot. (`gemma4` is a thinking model; `reasoning_effort: 'none'` keeps it short.)
- **`$.process.run(argv)`** runs a program with an argument list, not a shell string. `; rm -rf ~` inside an argument is just text. It resolves `{ exitCode, stdout, stderr, … }`.

Validate it, and read the `calls:` line the way a reviewer would:

```bash
# on: laptop
claude plugin validate ~/mod-gym-lab/setpoint-coach
```

**Expected output** (captured on this Mac, the hooks part)

```
  ❯ ./register.ts hooks: session.start, command.run{command=hvac-explain}, command.run{command=spark-ask}, command.run{command=git-where}
  ❯ ./register.ts answers its own command: command.run{command=hvac-explain}
  ❯ ./register.ts answers its own command: command.run{command=spark-ask}
  ❯ ./register.ts answers its own command: command.run{command=git-where}
  ❯ ./register.ts calls: $.command.register, $.http.fetch, $.model.complete, $.process.run, $.session.cwd

✔ Validation passed with warnings
```

(The warning is the missing `author` in `plugin.json`.) Now run all three:

```bash
# on: laptop — from inside this repo, then from a folder that is not a git repo
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/git-where"
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/hvac-explain Why does the BMS hold guest rooms at 24 °C when they are unoccupied?"
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/spark-ask Why does the BMS hold guest rooms at 24 °C when they are unoccupied?"
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/spark-ask"
```

**Expected output** (captured on this Mac; model answers will differ; wall time 7.1 s for haiku and 12.5 s for gemma4:12b, each including Claude Code's own start-up)

```
setpoint-coach: On branch master
setpoint-coach: Not a git repo (fatal: not a git repository (or any of the parent directories): .git)
setpoint-coach: The BMS keeps guest rooms at about 24 °C to maintain a comfortable, stable temperature so the room doesn’t feel too hot or too cold when the guest returns. It’s also a way to protect the building and fixtures from large temperature swings while the room is empty.
setpoint-coach: The system maintains a consistent temperature to prevent the rooms from becoming too hot or too cold for guests. This ensures the space remains comfortable and stable when someone checks back into the room.
setpoint-coach: Usage: /spark-ask <question>
```

Read the two answers as a hotel engineer would: **neither says "to save energy"**, the main reason for an unoccupied setback. The models did what the system prompt asked (two sentences, no jargon) and nothing told them the hotel's goal. The prompt is part of your mod's code: test it like code.

✓ Checkpoint: validate lists `$.model.complete`, `$.http.fetch` and `$.process.run`, and you can say which one keeps guest data inside the hotel.

## 5 · Which shape fits?

| You want | Use | Who starts it | Costs | Hotel HVAC copilot example |
|---|---|---|---|---|
| A fact or action at your fingertips | slash command (`$.command.register` + `command.run`) | you | nothing | `/setpoint 1203` reads a room from the BMS |
| Claude to use your data mid-task | tool (`$.tool.register` + `tool.call`) | the model | the turn's tokens | `room_setpoint` while Claude drafts a comfort-complaint reply |
| A cheap side judgment | `$.model.complete({ model: 'haiku' })` | your hook | a small call | classify a guest complaint as "too hot" / "too cold" / "noise" |
| Data that must stay on site | `$.http.fetch` to a local model | your hook | your hardware | explain chiller alarms on a DGX Spark in the plant room |
| An existing program's answer | `$.process.run([...])` | your hook | a process | `git`, `pytest`, a BMS export CLI |

Rule of thumb: if a person decides when, make it a command. If the model decides when, make it a tool. If a hook needs it, call it directly, and put the call behind a timeout.

## 6 · Test it with no network

`claude plugin test` runs `tests/*.test.ts` with no session, no sign-in and no network. Each outside call is a stub hook beneath your plugin. Lesson 08 covers the kit in depth; here is the shape for the three calls (`tests/setpoint-coach.test.ts`):

```ts
import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const START = { cwd: '/hotel', surface: 'terminal', isInteractive: true } as const
const cmd = (command: string, args = '') =>
  ({ command, args, origin: { kind: 'composer' }, presentation: { isFullscreen: true, columns: 120 } }) as const
const USAGE = { input_tokens: 30, output_tokens: 20, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }

function engine(on: On) {
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
}

test('/hvac-explain shows the answer, or why there is none', async ($, on) => {
  engine(on)
  let calls = 0
  on('model.complete', ($, e) => {
    calls += 1
    expect(e.model).toBe('haiku')
    return calls === 1
      ? { value: { isAnswered: true, text: ' 24 °C saves energy while the room is empty. ', usage: USAGE } }
      : { value: { isAnswered: false, reason: 'aborted', usage: USAGE } }
  })
  await $.session.start(START)
  expect((await $.command.run(cmd('hvac-explain', 'why 24?'))).text).toBe('24 °C saves energy while the room is empty.')
  expect((await $.command.run(cmd('hvac-explain', 'why 24?'))).text).toBe('No answer (aborted)')
})

test('/spark-ask posts to the local model and reads choices[0]', async ($, on) => {
  engine(on)
  on('http.fetch', ($, e) => {
    expect(e.url).toContain('/v1/chat/completions')
    expect(JSON.parse(e.init?.body ?? '{}').model).toBe('gemma4:12b')
    return { value: { ok: true, status: 200, headers: {}, text: JSON.stringify({ choices: [{ message: { content: 'Local answer.' } }] }) } }
  })
  await $.session.start(START)
  expect((await $.command.run(cmd('spark-ask', 'why 24?'))).text).toBe('Local answer.')
})

test('/git-where runs git by argv, in the session folder', async ($, on) => {
  engine(on)
  on('session.cwd', () => ({ value: '/hotel' }))
  on('process.run', ($, e) => {
    expect(e.argv).toEqual(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    expect(e.init?.cwd).toBe('/hotel')
    return { value: { exitCode: 0, stdout: 'main\n', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  await $.session.start(START)
  expect((await $.command.run(cmd('git-where'))).text).toBe('On branch main')
})
```

```bash
# on: laptop
claude plugin test ~/mod-gym-lab/setpoint-coach
```

**Expected output** (captured on this Mac)

```
tests/setpoint-coach.test.ts:
(pass) /hvac-explain shows the answer, or why there is none [19.28ms]
(pass) /spark-ask posts to the local model and reads choices[0] [8.82ms]
(pass) /git-where runs git by argv, in the session folder [7.99ms]

 3 pass
 0 fail
Ran 3 tests across 1 file. [0.17s]
```

The second case in the first test is the one that matters: it proves the command says *why* there is no answer instead of printing `undefined`.

✓ Checkpoint: three green tests, and none of them touched the network.

## In the Mod Gym panel

Type `/mod-gym 7` (or press **n** until you reach lesson 07), then press **c**. The gym checks `~/mod-gym-lab/hello-mod`:

| Check | Passes when |
|---|---|
| registers a slash command | `register.tsx` contains `$.command.register(` |
| registers a tool for Claude | `register.tsx` contains `$.tool.register(` |
| serves mcp__hello-mod__… | `register.tsx` contains `mcp__hello-mod__` |
| validate lists command.run | `claude plugin validate` exits 0 and its output contains `command.run` |

Press **a** to put this lesson's question in your prompt.

## Exercise — one function, two doors

Give `setpoint-coach` a room setpoint lookup that both **you** and **Claude** can use: a `/setpoint <room>` command and a `room_setpoint` tool, answering from one table (a stand-in for the BMS) through one function. An unknown room must get a clear answer, not a crash. Add a test that calls both doors.

<details><summary>Solution</summary>

Add the table and the function at the top of `register.ts`:

```ts
const SETPOINTS: Record<string, number> = { '1101': 24, '1203': 22, 'lobby': 23 } // a stand-in for the BMS

function setpointOf(room: string): string {
  const c = SETPOINTS[room.trim().toLowerCase()]
  return c === undefined ? 'No room "' + room + '" in this hotel.' : 'Room ' + room + ' is set to ' + c + ' °C.'
}
```

Register both in `session.start`:

```ts
await $.command.register({ name: 'setpoint', description: 'Show a room setpoint', argumentHint: '<room>' })
await $.tool.register({
  name: 'room_setpoint',
  description: 'Read the current temperature setpoint of one hotel room or space, e.g. "1203" or "lobby".',
  inputSchema: { type: 'object', properties: { room: { type: 'string' } }, required: ['room'] },
})
```

Serve both:

```ts
// Exercise: one function, two doors: you type /setpoint, Claude calls room_setpoint
on('command.run', { command: 'setpoint' }, async ($, e) =>
  ({ text: e.args.trim() ? setpointOf(e.args) : 'Usage: /setpoint <room>' }))

on('tool.call', { tool: 'mcp__setpoint-coach__room_setpoint' }, async ($, e) => ({ result: setpointOf(String(e.room ?? '')) }))
```

The test:

```ts
test('/setpoint and room_setpoint answer from the same table', async ($, on) => {
  engine(on)
  on('tool.register', ($, e) => ({ value: { tool: e.name } }))
  await $.session.start(START)
  expect((await $.command.run(cmd('setpoint', '1203'))).text).toBe('Room 1203 is set to 22 °C.')
  const r = await $.tool.call({ tool: 'mcp__setpoint-coach__room_setpoint', room: 'lobby' })
  expect(r.result).toBe('Room lobby is set to 23 °C.')
  expect((await $.tool.call({ tool: 'mcp__setpoint-coach__room_setpoint', room: '9999' })).result).toContain('No room')
})
```

**Expected output** (captured on this Mac: `claude plugin test`, then both doors for real)

```
(pass) /setpoint and room_setpoint answer from the same table [9.08ms]

 4 pass
 0 fail
```

```bash
claude -p "What is room 1203 set to? Use your tool. One line." --plugin-dir ~/mod-gym-lab/setpoint-coach --allowedTools=mcp__setpoint-coach__room_setpoint
claude -p --plugin-dir ~/mod-gym-lab/setpoint-coach "/setpoint 1203"
```

```
Room 1203 is set to 22 °C.
setpoint-coach: Room 1203 is set to 22 °C.
```

The first line is Claude's reply after calling the tool; the second is the command, with no turn at all.

</details>

✓ Checkpoint: both doors answer from the same function, and the test covers an unknown room.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/reps` is not in the typeahead | the command is registered outside `session.start`, or `session.start` threw before it | register in `session.start`; run `claude --debug` and look for a skipped hook line |
| Claude says it has no `log_reps` tool | the mod did not load, or `$.tool.register` threw | `claude plugin validate`; check the dim reload line in the transcript |
| The tool call fails saying no hook answered | the matcher is not the full `mcp__<plugin>__<name>` | match `mcp__hello-mod__log_reps` exactly, as a literal |
| In `-p`, Claude asks for permission and stops | nobody can approve a tool in headless mode | pass `--allowedTools=mcp__hello-mod__log_reps` |
| `/hvac-explain` prints `No answer (aborted)` | `timeoutMs` ran out | raise it, or keep the prompt shorter |
| `/spark-ask` prints `Local model failed: HTTP 404` | the model is not pulled | `ollama pull gemma4:12b`, or change `model` |

## What's next

[Lesson 08 — Test, validate, ship](../08_test_and_ship/TUTORIAL.md): turn these tests into a habit, type-check the mod, publish it so a classmate can `/plugin install` it, and audit a starter mod that was written without ever being run.
