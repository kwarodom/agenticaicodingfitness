# ▶ Mod Gym Lesson 02 — Your first mod: three files

> Part of Week 29 · Claude Code mods. You learn to change how Claude Code itself looks and behaves, from inside the terminal you code in every day. The tutorials are the long form; **🏋 Mod Gym** (`/mod-gym`) is the short form inside Claude Code, with checks on your lab mod. Through the lessons you grow one mod, `hello-mod`, in your lab folder (default `~/mod-gym-lab`); the finished version is in `week29/reference/hello-mod`.

**What you'll actually do**
- Write the smallest real mod: three files, no build step, no Node, no bundler.
- Rewrite part of Claude Code's own screen: add your text to the spinner.
- Validate the mod, load it with `claude --plugin-dir`, and change it while it runs.
- Find the API types Claude Code writes beside your mod, and type-check the mod with `tsc`.
- Read the three validator errors beginners hit most.

**Time** ~40 min · **Difficulty** beginner · **Hardware** this laptop only (Node's `npx` for the optional type-check)

**Sources:** [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) · the API declarations Claude Code 2.1.292 writes (`.claude-plugin/types/claude-code/index.d.ts`)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Lesson 01 | Mod Gym is installed and `/mod-gym` opens | the panel checks this lesson's work |
| Claude Code 2.1.287+ | `claude --version` | mods |
| Your lab folder | `/mod-gym status` shows `lab …` | the checks look for `<lab>/hello-mod` |
| Node (optional) | `npx --version` | only for the `tsc` type-check in section 6 |

All commands below use `LAB=~/mod-gym-lab`. If you changed the lab with `/mod-gym lab <path>`, use that path.

## 1 · The three files

```text
hello-mod/
  .claude-plugin/plugin.json   the manifest: who the plugin is
  hooks/hooks.json             the "modules" key: this is what makes it a mod
  hooks/register.ts            the hooks module: your code
```

**`.claude-plugin/plugin.json`**

```json
{ "name": "hello-mod", "version": "0.1.0", "description": "My first mod" }
```

`name` becomes the plugin's identity everywhere: in `/plugin`, in tool names (`mcp__hello-mod__…`, Lesson 07) and in state keys (Lesson 06). The validator refuses names that pass as Anthropic's own, such as anything starting with `claude-`, `anthropic-` or `cc-plugin-`.

**`hooks/hooks.json`**

```json
{ "modules": ["./register.ts"] }
```

One path, relative to `hooks.json`. The same file may also hold classic settings hooks under a `hooks` key; a plugin can be both.

**`hooks/register.ts`**

```ts
import type { Register } from 'claude-code'

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )
}
```

Line by line:

- `import type { Register } from 'claude-code'` is for the type-checker only; at run time the import is empty.
- `register` receives `on`. Each `on(event, matcher?, hook)` adds one hook.
- `'ui.render'` fires whenever Claude Code draws a component. The matcher `{ component: 'Spinner' }` narrows it to the spinner, the "Thinking…" row you see while Claude works.
- The spinner's props are `word`, `message`, `suffix` and `mode` (`'requesting' | 'responding' | 'thinking' | 'tool-input' | 'tool-use'`). We pass `next` a copy of the event with a new `suffix`. That is the **rewrite** pattern: Claude Code still draws its own spinner, from props we changed.

Create the files by hand, or let Mod Gym do the typing: open `/mod-gym 2` and press **f** (**Copy starter**). It copies this script for your lab folder; paste it into a shell.

```bash
# on: laptop
LAB=~/mod-gym-lab
mkdir -p "$LAB/hello-mod/.claude-plugin" "$LAB/hello-mod/hooks"
cat > "$LAB/hello-mod/.claude-plugin/plugin.json" <<'EOF'
{ "name": "hello-mod", "version": "0.1.0", "description": "My first mod" }
EOF
cat > "$LAB/hello-mod/hooks/hooks.json" <<'EOF'
{ "modules": ["./register.ts"] }
EOF
cat > "$LAB/hello-mod/hooks/register.ts" <<'EOF'
import type { Register } from 'claude-code'

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · lifting…' } }),
  )
}
EOF
claude plugin validate "$LAB/hello-mod"
```

## 2 · Validate

**Expected output** of the last line (captured on this Mac; your path differs)

```
Validating plugin manifest: …/hello-mod/.claude-plugin/plugin.json

⚠ Found 1 warning:

  ❯ author: No author information provided. Consider adding author details for plugin attribution

Validating hooks: …/hello-mod/hooks/hooks.json

  ❯ ./register.ts hooks: ui.render{component=Spinner}
  ❯ ./register.ts calls: nothing on $

✔ Validation passed with warnings
```

The `hooks:` line is what the engine will register. `calls: nothing on $` is the audit from Lesson 01: this mod reaches nothing outside itself. The author warning is optional; add `"author": { "name": "…" }` to silence it.

✓ Checkpoint: `✔ Validation passed` (with or without warnings).

## 3 · The three mistakes the validator catches

The validator reads your source the way the engine will. It wants event names as **string literals**, because it reads them without running your code. Here is what it said for three broken copies of `hello-mod` (captured on this Mac, trimmed):

A typo in the event name (`'ui.rendr'`):

```
✘ Found 1 error:
  ❯ modules../register.ts: … compiled line 2 `on("ui.rendr", { component: "Spinner" }, …`: "ui.rendr" is not an event; $ is always spelled $.noun.event(...) at the call site, on is always on("<event>", hook), …
✘ Validation failed
```

The event name in a variable (`const EVENT = 'ui.render'; on(EVENT, …)`):

```
✘ Found 1 error:
  ❯ modules../register.ts: … compiled line 3 `on(EVENT, { component: "Spinner" }, …`: the event name passed to on() is not a string literal; …
✘ Validation failed
```

`hooks.json` with `"module"` instead of `"modules"`:

```
✘ Found 1 error:
  ❯ root: hooks.json must have `hooks` (the hook matchers) or `modules` (hooks modules), or both
✘ Validation failed
```

Also refused at load: `import()` anywhere in the module. A hooks module is an ES module that imports other plugin files with `import` declarations only.

## 4 · Load it and watch the spinner

Mod Gym is a plugin you installed. While you develop your own mod, you load it from its folder instead, for one session:

```bash
# on: laptop (a second terminal)
claude --plugin-dir ~/mod-gym-lab/hello-mod
```

Ask Claude anything that takes a few seconds ("list the files here and explain them"). The spinner row ends in `· lifting…`.

A quick non-interactive smoke test also works; the mod loads, and Claude answers as usual:

```bash
# on: laptop
claude -p --plugin-dir ~/mod-gym-lab/hello-mod --model haiku "Reply with the single word ok"
```

**Expected output** (captured on this Mac)

```
ok
```

If a module fails to load in a `-p` run, Claude Code names the plugin and the reason once on stderr.

**Hot reload.** In an interactive session started with `--plugin-dir`, the folder is watched. Save `register.ts` and the module reloads: `register` runs again in a fresh environment, and the transcript prints a dim line naming the plugin. Change the suffix to `' · deadlifting…'`, save, send another prompt, and watch it change without restarting. Two things reset on every reload: your module's own variables and its timers (Lesson 06 shows where to keep values that must survive).

✓ Checkpoint: you saw your suffix on the spinner.

## 5 · Where the types are

The first time Claude Code loads your mod from a folder you own, it writes the API declarations beside it, and a `tsconfig.json` that points at them:

```bash
# on: laptop
find ~/mod-gym-lab/hello-mod -type f | sort
```

**Expected output** (captured on this Mac after one `claude -p --plugin-dir` run, paths shortened)

```
hello-mod/.claude-plugin/plugin.json
hello-mod/.claude-plugin/types/.gitignore
hello-mod/.claude-plugin/types/claude-code-mcp/index.d.ts
hello-mod/.claude-plugin/types/claude-code-tools/index.d.ts
hello-mod/.claude-plugin/types/claude-code/index.d.ts
hello-mod/.claude-plugin/types/tsconfig.json
hello-mod/hooks/hooks.json
hello-mod/hooks/register.ts
hello-mod/tsconfig.json
```

| File | What it is |
|---|---|
| `types/claude-code/index.d.ts` | the whole mods API for **your** Claude Code build, every event and every `$` method with a comment and an example. Its first line names the version (`// Written by Claude Code 2.1.292.`). When a doc page and this file disagree, this file wins |
| `types/claude-code-tools/index.d.ts` | the built-in tools on this machine, so `e.tool === 'Bash'` narrows `e` to Bash's input |
| `types/claude-code-mcp/index.d.ts` | the MCP tools that were connected when the mod last loaded |
| `types/.gitignore` | `*`: the folder is regenerated, never committed |
| `tsconfig.json` (mod root) | `{ "extends": "./.claude-plugin/types/tsconfig.json" }` |

Grep the API file for the name at hand instead of reading all of it:

```bash
# on: laptop
grep -n "Spinner: {" -A 25 ~/mod-gym-lab/hello-mod/.claude-plugin/types/claude-code/index.d.ts
```

## 6 · Type-check it

```bash
# on: laptop
npx -y -p typescript@5.6 tsc -p ~/mod-gym-lab/hello-mod
echo "tsc exit $?"
```

**Expected output** (captured on this Mac)

```
tsc exit 0
```

Now make a mistake the validator cannot see. Change the suffix to a number, `suffix: 42`, and run `tsc` again:

**Expected output** (captured on this Mac, trimmed)

```
hello-mod/hooks/register.ts(5,39): error TS2769: No overload matches this call.
  …
    Type 'number' is not assignable to type 'string'.
tsc exit 2
```

Put the string back. Validate and `tsc` catch different things: validate checks what the engine will refuse to load; `tsc` checks that your values have the shapes the engine expects.

✓ Checkpoint: `tsc exit 0` on your mod.

## In the Mod Gym panel

Run `/mod-gym 2` and press **c**. Lesson 02 checks four things in `<lab>/hello-mod`.

**Expected output** (`/mod-gym check`, captured on this Mac against the reference mod)

```
Lesson 02 · Your first mod: three files · 4/4 checks
✓ plugin.json exists (hello-mod/.claude-plugin/plugin.json)
✓ hooks.json names a module (hello-mod/hooks/hooks.json)
✓ register.ts hooks the Spinner (hello-mod/hooks/register.tsx)
✓ claude plugin validate passes (exit 0)
```

On your lab the third line names `register.ts`; the reference is already at Lesson 08, where the file is `register.tsx`. The check accepts either.

Stuck? Press **a**: Mod Gym puts a coaching question into your prompt, and Claude reads your lesson and lab folder through the `gym_status` tool.

## Try it yourself — a spinner that knows what Claude is doing

The spinner's `mode` prop says what Claude is doing right now. Make the suffix follow it: `requesting` → `warming up`, `thinking` → `planning the set`, `responding` → `lifting`, `tool-input` → `loading the bar`, `tool-use` → `spotting`. Keep one hook. Validate and type-check.

<details><summary>Solution</summary>

```ts
import type { Register } from 'claude-code'

const REPS: Record<string, string> = {
  requesting: 'warming up',
  thinking: 'planning the set',
  responding: 'lifting',
  'tool-input': 'loading the bar',
  'tool-use': 'spotting',
}

export const register: Register = on => {
  on('ui.render', { component: 'Spinner' }, ($, e, next) =>
    next({ ...e, props: { ...e.props, suffix: ' · ' + (REPS[e.props.mode] ?? 'lifting') + '…' } }),
  )
}
```

**Expected output** (captured on this Mac)

```
  ❯ ./register.ts hooks: ui.render{component=Spinner}
  ❯ ./register.ts calls: nothing on $
✔ Validation passed with warnings
tsc exit 0
```

`e.props.mode` is typed, so `tsc` would flag a misspelt prop name. The `?? 'lifting'` fallback keeps the spinner sensible if a later Claude Code adds a mode.

</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `hooks.json must have hooks … or modules` | the key is spelled `module`, or the file is empty | `{ "modules": ["./register.ts"] }` |
| `… is not an event` | a typo in the event name | grep the types file for `'ui.render'` and copy the name |
| `… is not a string literal` | the event name or matcher is in a variable | write it inline: `on('ui.render', { component: 'Spinner' }, …)` |
| No suffix on the spinner | the session was not started with `--plugin-dir`, or the module failed to load | look for a dim line naming `hello-mod` in the transcript; run `claude --debug --plugin-dir …` |
| No `.claude-plugin/types` folder | the mod has not been loaded yet | run `claude -p --plugin-dir <dir> --model haiku "Reply with the single word ok"` once |
| `tsc` cannot find `claude-code` | no `tsconfig.json` yet (same cause) | load the mod once, then rerun `tsc -p <dir>` |

## Next

[Lesson 03 — The hook pipeline: observe, rewrite, answer](../03_hook_pipeline/TUTORIAL.md): add hooks on tool calls and prompts, and watch your own hooks next to Mod Gym's on the Events tab.
