# ▶ Mod Gym Lesson 01 — What a mod is

> Part of Week 29 · Claude Code mods. You learn to change how Claude Code itself looks and behaves, from inside the terminal you code in every day. The course has two halves that teach the same eight lessons: these tutorials (the long form) and **🏋 Mod Gym**, a panel inside Claude Code (the short form, with checks). Mod Gym is itself a mod, so everything it does on screen is something you will learn to build. Through the lessons you grow one small lab mod, `hello-mod`; the finished version is in `week29_claude_code_mods/reference/hello-mod`.

**What you'll actually do**
- Place mods on the map of Claude Code extension points: skills, settings hooks, MCP servers, LSP servers, output styles, the status line, and mods.
- Read the one idea every mod is built on: a hook `($, e, next)` that can **observe**, **rewrite** or **answer** an event.
- Audit a mod before you trust it, with the `hooks:` and `calls:` lines of `claude plugin validate`.
- Install Mod Gym and take the tour: the panel, its three tabs, the live Events tape, the band above the prompt, and the tool Claude uses to coach you.

**Time** ~30 min · **Difficulty** beginner · **Hardware** this laptop only

**Sources:** [Anthropic blog — Claude Code mods](https://claude.com/blog/claude-code-mods) · [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) · [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) · [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) · [Plugin components](https://code.claude.com/docs/en/plugins/components) · [Status line](https://code.claude.com/docs/en/statusline) · [Host a marketplace](https://code.claude.com/docs/en/plugins/host-marketplace) · the API declarations Claude Code 2.1.292 writes (`.claude-plugin/types/claude-code/index.d.ts`, the authority whenever the docs and the code differ)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Claude Code 2.1.287 or newer | `claude --version` | mods arrived in 2.1.287 (terminal) |
| This repo | `ls week29_claude_code_mods/mod-gym` | Mod Gym is installed from this folder |
| Week 15 (skills and plugins) | you know what a `SKILL.md` is | this lesson compares mods to skills |

```bash
# on: laptop
claude --version
```

**Expected output** (captured on this Mac)

```
2.1.292 (Claude Code)
```

✓ Checkpoint: your version is 2.1.287 or newer.

## 1 · Seven ways to extend Claude Code

A plugin is a folder with a `.claude-plugin/plugin.json` manifest. It can carry any mix of these pieces:

| Piece | File | Runs as | Can draw UI? | Class example |
|---|---|---|---|---|
| Skill | `skills/<name>/SKILL.md` | Markdown that Claude reads when it fits | no | the 20 `agentic-coding-fitness` lab skills |
| Settings hook | `hooks/hooks.json` → `hooks` | a shell command, HTTP call or prompt on a lifecycle event | no | append a CSV row on `SessionEnd` |
| MCP server | `.mcp.json` | a separate process that offers tools | no | a class roster API |
| LSP server | `.lsp.json` | a language server for code intelligence | no | Python diagnostics for the labs |
| Output style | `output-styles/<name>.md` | a system-prompt voice | no | a "fitness coach" tone |
| Status line | `statusLine` setting | a script that prints one line | one line | context % and cost |
| **Mod** | `hooks/hooks.json` → `modules` | **functions inside Claude Code** | **yes** | Mod Gym, this course's panel |

The last row is new. Anthropic introduced mods on 1 October 2026 as "small TypeScript functions that change how Claude Code works. A mod can rewrite a prompt, add new UI, replace a built-in feature, or add entirely new functionality." A mod is a plugin whose `hooks/hooks.json` has a `modules` key. That key, and nothing else, is what makes a plugin a mod.

What only a mod can do:

- draw its own panes and bands, with buttons, tabs and text fields;
- redraw Claude Code's own rows, such as the spinner;
- hold a tool call while it asks you a question;
- run a `/command` with no Claude turn at all;
- share values between its hooks, and keep them across reloads.

✓ Checkpoint: you can say which piece you would use for "a checklist Claude reads", "log every session to a CSV" and "a panel with buttons".

## 2 · The one idea: `($, e, next)`

Every mod hook has the same shape:

```ts
on('tool.call', ($, e, next) => next(e))
```

- `e` is the event's input, a plain frozen value (here: which tool, with which arguments).
- `next(e)` runs the plugins beneath you and then Claude Code's own behaviour, and resolves to the event's result.
- `$` is everything outside your module: the screen (`$.ui`), the store (`$.store`), files (`$.fs`), programs (`$.process`), models (`$.model`) and more. A mod has no Node and no DOM; it reaches the world only through `$`.

What you do around `next` decides what the hook is:

| You write | Pattern |
|---|---|
| `return next(e)` | **observe** |
| `return next({ ...e, x })` | **rewrite** what the rest of the chain sees |
| `const r = await next(e); …; return r` | **wrap**: act on the result |
| `return { … }` without calling `next` | **answer** in Claude Code's place |

Lesson 03 is all about these four. For now, notice that a mod is middleware: the same idea as the agent loops and tool guards you built in Weeks 4–5, now wrapped around the coding agent you use every day.

## 3 · Trust: read a mod before you load it

A mod is **not sandboxed**. It runs with your permissions: it can read your files and secrets, approve tool calls and spend your usage. So the first habit is the audit. `claude plugin validate <dir>` reads the manifest and the module's source the way the engine will, and prints what the module hooks and what it calls.

Audit Mod Gym before you install it:

```bash
# on: laptop (repo root)
claude plugin validate week29_claude_code_mods/mod-gym
```

**Expected output** (captured on this Mac, trimmed)

```
Validating hooks: …/week29_claude_code_mods/mod-gym/hooks/hooks.json

  ❯ ./register.tsx hooks: session.start, prompt.submit, tool.call, turn.complete, tool.call{tool=mcp__mod-gym__gym_status}, command.run{command=mod-gym}, ui.render{component=AbovePrompt}, ui.render{component=Pane, requestId=mod-gym}
  ❯ ./register.tsx answers its own command: command.run{command=mod-gym}
  ❯ ./register.tsx gating hook without .catch: prompt.submit
  ❯ ./register.tsx gating hook without .catch: tool.call
  ❯ ./register.tsx gating hook with .catch: tool.call{tool=mcp__mod-gym__gym_status}
  ❯ ./register.tsx calls: $.clock.now, $.command.register, $.env.get (via home), $.fs.exists (via runOne), $.fs.list (via runOne), $.fs.read (via runOne), $.process.run (via runOne), $.prompt.fill (via askClaude), $.session.version (via runOne), $.state.get, $.state.set, $.store.get, $.store.set, $.tool.register, $.ui.ask, $.ui.close, $.ui.copy, $.ui.open, $.ui.resolve, $.ui.status (via refreshStatus), $.ui.toast
  ❯ ./register.tsx env writes: nothing
  ❯ ./register.tsx env reads: HOME
  …
✔ Validation passed
```

How to read it:

- **hooks:** every event the mod listens to. `tool.call` with no matcher means it sees *every* tool call (Mod Gym only times them for the Events tape).
- **calls:** every `$` method it can reach, and through which of its functions. Ask of each one: do I expect this? Mod Gym reads files and runs `$.process.run` (only to run `claude plugin validate` and `claude plugin test` on your lab mod), and reads one environment variable, `HOME`. It makes no `$.http` or `$.model` calls.
- **gating hook:** a hook at a place where its answer could refuse something. A gating hook that throws is skipped, so for a guard you want `with .catch` (Lesson 04).

✓ Checkpoint: you can point to the line that tells you whether a mod can run programs on your machine.

## 4 · Built-ins worth knowing

| Built-in | What it gives you |
|---|---|
| `/plugin-authoring` | the built-in skill that teaches Claude to write mods; Claude writes them into `~/.claude/dev-mods/<session-id>/` and offers hot reloading |
| `/plugin` | browse, install, enable and disable plugins; the Installed tab counts active mods |
| `/reload-plugins` | re-reads installed plugins, including ones installed from a local folder |
| `claude plugin validate <dir>` | the audit and the static checks |
| `claude plugin test <dir>` | runs a mod's `*.test.ts` with no session, sign-in or network (Lesson 08) |

## 5 · Install Mod Gym

This repo's `week29_claude_code_mods/` folder is a plugin marketplace (`week29_claude_code_mods/.claude-plugin/marketplace.json`, named `fitness-week29`). Add it, then install Mod Gym at the user scope:

```bash
# on: laptop (repo root)
claude plugin marketplace add ./week29_claude_code_mods
claude plugin install mod-gym@fitness-week29 --scope user
claude plugin list
```

**Expected output** of the last command (captured on this Mac, the Mod Gym entry)

```
  ❯ mod-gym@fitness-week29
    Version: 0.1.0
    Read from: /Users/altodev/Desktop/agenticaicodingfitness/week29_claude_code_mods/mod-gym
    Scope: user
    Status: ✔ enabled
```

`Read from:` names the repo folder itself. A plugin installed from a folder marketplace is read from that folder, so after a `git pull` you only run `/reload-plugins` in a session; no reinstall, no version bump.

Once the class repo on GitHub carries Mod Gym, students who already have the course marketplace update it instead, at the Claude Code prompt:

```text
/plugin marketplace update agentic-coding-fitness
/plugin install mod-gym@agentic-coding-fitness
```

Pick one source, not both: two copies of the same plugin name would both try to load.

Start a new Claude Code session (or run `/reload-plugins` in an open one) so the mod loads.

✓ Checkpoint: `claude plugin list` shows `mod-gym` with `Status: ✔ enabled`.

## 6 · The tour

Type `/mod-gym` at the prompt. The panel opens with the keyboard on it.

```text
🏋 Mod Gym Week 29 · Claude Code mods · 0/8 lessons passed
[ Lesson ] [ Events ] [ Progress ]
Lesson 01 of 08 · What a mod is
…lesson text…
Checks
· Claude Code is 2.1.287 or newer
Long form: week29_claude_code_mods/01_what_is_a_mod/TUTORIAL.md · lab: ~/mod-gym-lab/hello-mod
[ Prev ] [ Next ] [ Check ] [ Ask Claude ] [ Close ]
```

(The layout above is a sketch; your terminal's colours and widths differ.)

| Key | What it does |
|---|---|
| `1` `2` `3` | the **Lesson**, **Events** and **Progress** tabs |
| `p` / `n` | previous / next lesson |
| `c` | **Check**: runs this lesson's checks on your lab mod and shows ✓ / ✗ with the reason |
| `a` | **Ask Claude**: puts a coaching question for this lesson into your prompt; edit it or press Enter |
| `f` | **Copy starter** (Lesson 02 only): copies a shell script that creates `hello-mod` |
| `x` | clear the Events tape (Events tab) |
| `r` | reset your progress (Progress tab, asks first) |
| `q` or Esc | close the panel |

**The Events tab** is Mod Gym's own hooks, live. Send any prompt, and rows appear:

| Mark | Pattern | Example row |
|---|---|---|
| `◦` | observe: `return next(e)` | `prompt.submit` with the first words of your prompt |
| `⟳` | wrap: `await next(e)`, then act | `tool.call` with the tool, its argument, the outcome (`ok`, `error`, `denied`) and how long it took |
| `◆` | answer: no `next` | (Mod Gym's own `gym_status` tool) |

**The band** is the one line above your prompt: `🏋 Mod Gym L01 What a mod is · 0/8 passed [Open] [Hide]`. Mod Gym draws it *beneath* whatever other plugins put in the band, instead of replacing it. Hide it with its button or `/mod-gym band off`; bring it back with `/mod-gym band on`.

**The status line** shows `Mod Gym L01 · 0/8`.

**The `gym_status` tool.** Mod Gym registers a tool Claude sees as `mcp__mod-gym__gym_status`. It returns your lesson, its tutorial path, your lab folder and your last check results, with the note "ask one question at a time, let the student type the code". So when you press **Ask Claude**, Claude can look up where you are and coach instead of writing your lab for you.

**Your lab folder.** Checks look for your mod at `<lab>/hello-mod`. The default lab is `~/mod-gym-lab`. Change it with:

```text
/mod-gym lab ~/code/my-mods
```

Other commands: `/mod-gym next`, `/mod-gym prev`, `/mod-gym 3` (jump to lesson 3), `/mod-gym check`, `/mod-gym status`, `/mod-gym reset`.

✓ Checkpoint: the Events tab shows at least one `⟳ tool.call` row from your session.

## In the Mod Gym panel

Run `/mod-gym 1`, then press **c**. Lesson 01 has one check.

**Expected output** (`/mod-gym check`, captured on this Mac)

```
Lesson 01 · What a mod is · 1/1 checks
✓ Claude Code is 2.1.287 or newer (running 2.1.292)
```

A full pass marks the lesson ✓ on the Progress tab and shows a toast: `Mod Gym: lesson 01 passed 💪`.

## Try it yourself — audit a stranger's mod

`week29_claude_code_mods/mod-agentic-coding-fitness/` is a progress-tracking mod drafted from the research notes for this week. It has never been run. Do **not** install it. Audit it:

```bash
# on: laptop (repo root)
claude plugin validate week29_claude_code_mods/mod-agentic-coding-fitness
```

Answer three questions from the output alone:

1. Which hook's matcher could the validator not read, and why does that matter for an audit?
2. Which call spends your Claude usage?
3. Which hooks could refuse something, and which of them would let it through if they threw?

<details><summary>Solution</summary>

**Expected output** (captured on this Mac, trimmed)

```
  ❯ ./register.js hooks: session.start, tool.call{tool=?}, tool.call{tool=Bash}, turn.complete, prompt.submit, command.run{command=fitness}, command.run{command=fitness-log}, command.run{command=fitness-next}, command.run{command=fitness-grade}, ui.render{component=Spinner}, ui.render{component=Pane}
  …
  ❯ ./register.js gating hook without .catch: tool.call{tool=?}
  ❯ ./register.js gating hook without .catch: tool.call{tool=Bash}
  ❯ ./register.js gating hook without .catch: prompt.submit
  ❯ ./register.js calls: $.command.register, $.model.complete, $.session.usage, $.store.get, $.store.set (via save), $.tool.register, $.ui.close, $.ui.invalidate, $.ui.open, $.ui.resolve, $.ui.status

✔ Validation passed
```

1. `tool.call{tool=?}`: the matcher is the variable `LOG_TOOL`, not a string literal, so the audit cannot tell you which tool it answers. You have to read the source. Prefer mods whose matchers are literals.
2. `$.model.complete` (the `/fitness-grade` command asks Haiku to grade a reflection).
3. The three `gating hook without .catch` lines. None of them is a guard that denies anything, so failing open is acceptable here; for a real guard it would not be (Lesson 04).

"Validation passed" means the engine would load it. It does not mean the mod is safe or correct.

</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/mod-gym` is not a command | the session started before the install | run `/reload-plugins` or start a new session |
| `claude plugin install` says the plugin is not found | the marketplace was not added from the repo root | `claude plugin marketplace add ./week29_claude_code_mods` from the repo root, then install again |
| Two `mod-gym` entries in `claude plugin list` | installed from both `fitness-week29` and `agentic-coding-fitness` | uninstall one: `claude plugin uninstall mod-gym@fitness-week29` |
| The band does not show | it is hidden, or another plugin above it draws the band without passing it on | `/mod-gym band on`; the status line and `/mod-gym` still work |
| The panel opens inline below the transcript, not as a sidebar | the terminal is not in the fullscreen layout | expected: only the fullscreen layout docks panes beside the transcript; the panel works the same inline |

## Next

[Lesson 02 — Your first mod: three files](../02_first_mod/TUTORIAL.md): write `hello-mod`, validate it, and see your own text on Claude Code's spinner.
