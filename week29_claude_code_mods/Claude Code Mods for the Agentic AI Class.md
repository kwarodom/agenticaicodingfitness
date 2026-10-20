# Claude Code Mods for the Agentic AI Coding Fitness Class

Research date: 9 October 2026 (Bangkok). Scope: which Claude Code "mods", plugins and built-in features help the class (a) track progress, (b) keep learning Claude Code techniques, (c) stay current on coding agents, and (d) how to grow the existing `agentic-coding-fitness` plugin (v2.2.0, 20 skills) into v3.0.0 with a real mod.

A note on the Skool file you linked: its signed URL expired on 9 Oct 00:56 UTC and the host now returns AccessDenied, so it is not folded in. If you upload it directly, I will reconcile this guide against it.

---

## 1. What "mods" means in Claude Code today

"Mods" is no longer slang for plugins. On 1 October 2026 Anthropic introduced mods: "small TypeScript functions that change how Claude Code works. A mod can rewrite a prompt, add new UI, replace a built-in feature, or add entirely new functionality", shipped inside plugins and installable like any plugin ([Anthropic blog](https://claude.com/blog/claude-code-mods)). The docs define a mod as "a plugin that changes how Claude Code looks and behaves", made of JavaScript or TypeScript event handlers that can watch, change, or take over an event such as a tool call, a submitted prompt, or a part of the interface being drawn ([Mods overview](https://code.claude.com/docs/en/plugins/mods/overview)).

Why this matters for the class: everything you have been teaching about agent loops, tool guards, memory and evaluation can now be practised by modifying the coding agent the students use every week.

Key facts to put on slide 1:

| Fact | Detail | Source |
|---|---|---|
| Minimum version | Terminal: Claude Code v2.1.287 or later; Desktop app from v2.1.286. Mods are on by default. | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| Minimal mod | Three files: `.claude-plugin/plugin.json`, `hooks/hooks.json` (with a `modules` key pointing at your code), and `hooks/register.js` exporting `register(on)`. No Node, bundler or build step. | [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) |
| Hook signature | Every hook receives `($, e, next)`: the mods API, the frozen event, and the next handler (middleware style). It can observe (`next(e)`), rewrite (`next({...e, ...})`), or answer (return a result without calling `next`). | [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) |
| What only a mod can do | Draw panes/bands with tabs, buttons and text fields; redraw Claude Code's own rows and spinner; hold a tool call while asking the user; run a `/command` with no Claude turn; share variables between hooks. | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| Mods vs settings hooks vs skills vs MCP | Mod = functions in-process (JS/TS, can draw); settings hook = shell/HTTP/prompt on a lifecycle event (cannot draw); skill = SKILL.md Claude reads; MCP = external tools. A plugin can hold all four. | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| Trust model | Not sandboxed; runs with your permissions, can read secrets, approve tool calls, spend usage. `claude plugin validate ./mod` prints `hooks:` and `calls:` lines so you can audit a mod before loading it. | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| Dev loop | `claude --plugin-dir ./mod` hot-reloads on save; `/reload-plugins`; `claude plugin validate`; `claude plugin test` runs `.test.ts` files with no session or network. | [Create a mod](https://code.claude.com/docs/en/plugins/mods/create), [Test a mod](https://code.claude.com/docs/en/plugins/mods/test) |
| Ask Claude to write one | Built-in `plugin-authoring` skill (`/plugin-authoring`); Claude writes into `~/.claude/dev-mods/<session-id>/` and offers hot reloading. | [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) |

Events you will use most (full list in the [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference)): `session.start`, `tool.call` (return `next(e)`, `{ deny }` or `{ result }`), `tool.check`, `prompt.submit` (rewrite text or add `context`), `command.run`, `turn.start`/`turn.step`/`turn.complete` (`e.answer`, `e.durationMs`, `e.usage`), `session.measure`, `agent.spawn`, `ui.render` (render sites `Pane`, `AbovePrompt`, `Spinner`, `ToolUse`, `AskUserQuestion`, …). API namespaces: `$.ui`, `$.command`, `$.tool`, `$.agent`, `$.model` (`complete`, `fork`, `classify`), `$.prompt`, `$.session` (`messages()`, `usage()`), `$.fs`, `$.process`, `$.http`, `$.store` (machine-wide JSON store, 4 MiB), `$.state` (reactive, survives reload), `$.clock`, `$.mcp`, `$.audio` ([Mods reference](https://code.claude.com/docs/en/plugins/mods/reference), [Use the mods API](https://code.claude.com/docs/en/plugins/mods/api)).

---

## 2. The full extension landscape (one table for the class)

| Extension point | File / command | Good for | Source |
|---|---|---|---|
| Skills | `skills/<name>/SKILL.md`; frontmatter `name`, `description`, `disable-model-invocation`, `user-invocable`, `allowed-tools`, `context: fork`, `agent`, `model`, `hooks`, `arguments`, plus `when_to_use`, `argument-hint`, `paths`… `/plugin:skill` namespacing. | Reusable lesson content, drills, checklists | [Skills](https://code.claude.com/docs/en/skills), [Components](https://code.claude.com/docs/en/plugins/components) |
| Commands | `commands/*.md`, "the older form of skills. Use `skills/` for new plugins." | Legacy only | [Create a plugin](https://code.claude.com/docs/en/plugins/create) |
| Subagents | `agents/*.md`; frontmatter `name`, `description`, `model`, `effort`, `maxTurns`, `tools`, `disallowedTools`, `skills`, `memory`, `background`, `isolation: worktree`…; invoked as `@agent-plugin:name` | Coach, grader, reviewer personas | [Components](https://code.claude.com/docs/en/plugins/components), [Subagents](https://code.claude.com/docs/en/sub-agents) |
| Settings hooks | `hooks/hooks.json` → `hooks` key; events `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, `SessionEnd`, `SubagentStart/Stop`, `PreCompact`, `TaskCompleted`, `FileChanged`…; handlers `command`, HTTP, MCP tool, prompt, agent | Logging, blocking, context injection via scripts | [Hooks reference](https://code.claude.com/docs/en/hooks) |
| Mods | `hooks/hooks.json` → `modules: ["./register.js"]` | UI, event rewriting, in-process commands/tools | [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference) |
| MCP servers | `.mcp.json` at plugin root; tool names `mcp__plugin_<plugin>_<server>__<tool>` | External data (class roster, Alto docs) | [Components](https://code.claude.com/docs/en/plugins/components) |
| LSP servers | `.lsp.json`; `extensionToLanguage` | Code intelligence for Python/TS labs | [Code intelligence](https://code.claude.com/docs/en/plugins/code-intelligence) |
| Output styles | `output-styles/<name>.md`; `/output-style` | A "fitness-coach" voice | [Components](https://code.claude.com/docs/en/plugins/components), [Output styles](https://code.claude.com/docs/en/output-styles) |
| Status line | `statusLine` setting; script receives JSON (`model.display_name`, `context_window.used_percentage`, cost…) on stdin; `/statusline <description>` generates one | Cheap always-on progress bar | [Status line](https://code.claude.com/docs/en/statusline) |
| Plugin evals | `evals/<case>/prompt.md` + `graders/*.md`; `claude plugin eval`, `claude plugin eval init`; compares against a no-plugin baseline | Grading your own plugin like a model | [Plugin evals](https://code.claude.com/docs/en/plugin-evals), [What's new W37](https://code.claude.com/docs/en/whats-new) |
| Manifest extras | `plugin.json`: `displayName`, `dependencies`, `userConfig` (prompted values), `types` (for `$.state`), `workflows`, `experimental.monitors/themes/evals` | Versioning, student config | [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference) |
| Marketplace | `.claude-plugin/marketplace.json` in a GitHub repo; `/plugin marketplace add owner/repo`; each version is cached, so bump `version` to ship edits | Distribution to the class | [Host a marketplace](https://code.claude.com/docs/en/plugins/host-marketplace) |

Name rule worth knowing: `claude plugin validate` rejects plugin names that pass as Anthropic's own, including the literal name `claude-mods` and anything starting with `claude-`, `anthropic-` or `cc-plugin-` ([Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference)). Your `agentic-coding-fitness` name is fine.

---

## 3. Built-in features the class can use tonight (no install)

| Feature | What it gives students | Source |
|---|---|---|
| `/powerup` | Built-in interactive lessons with animated demos, added in v2.1.90 (1 April 2026). The exact lesson list is not documented by Anthropic. | [What's new](https://code.claude.com/docs/en/whats-new), [claudefa.st](https://claudefa.st/blog/guide/mechanics/claude-powerup) |
| `/insights` | HTML report of the last 30 days of local sessions: usage patterns, friction points, context-window trends, prompting style; saved under `~/.claude/insights/`; unavailable in cloud sessions. | [Commands](https://code.claude.com/docs/en/commands), [ClaudeLab guide](https://claudelab.net/en/articles/claude-code/claude-code-insights-command-guide) |
| `/usage` (`/cost`, `/stats`) | Session cost, plan limits and a breakdown by skill, subagent, plugin and MCP server (since W21, May 2026); prompt-cache statistics since v2.1.251. | [Costs](https://code.claude.com/docs/en/costs), [What's new](https://code.claude.com/docs/en/whats-new) |
| `/skill-doctor` | Per-skill context cost and usage frequency (v2.1.252+). Useful to show students why 26 skills must stay lean. | [Commands](https://code.claude.com/docs/en/commands) |
| `/team-onboarding` | Generates a replayable onboarding guide from 30 days of usage; shareable link. Ideal for a "how I use Claude Code" show-and-tell. | [Commands](https://code.claude.com/docs/en/commands) |
| `/recap`, `/context`, `/release-notes` | One-line session summary; context-usage visualiser; changelog picker. | [Commands](https://code.claude.com/docs/en/commands) |
| `/plugin-authoring` | Built-in skill (`cc-plugin-plugin-authoring`) that teaches Claude to write mods. | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| `cc-plugin-you-should-know` | Built-in mod (disabled by default) that runs a side agent watching long tasks and posts notes above the prompt. Enable with `/plugin enable cc-plugin-you-should-know@builtin`. | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| Built-in mod source | `mods/` in the Claude Code repo: `diff`, `agents-md`, `sec-default`, `telemetry`, each a complete plugin with tests — reading material for week "self-evolving agents". | [Mods overview](https://code.claude.com/docs/en/plugins/mods/overview) |
| Sample mods | `claude-code-playground/claude-code/mods`: `token-weather` (context forecast above the prompt), `blast-radius` (holds `rm -rf`/force-push and shows the blast radius with Proceed/Cancel), `replay-theater` (`/replay` steps through last-turn edits). `git clone … && claude --plugin-dir ./token-weather`. | [Playground README](https://github.com/anthropics/claude-code-playground/tree/main/claude-code/mods) |

---

## 4. Curated mods and plugins by class goal

### 4a. Progress tracking

| Mod / plugin | What it does | Fit for the class | Source |
|---|---|---|---|
| Your own `agentic-coding-fitness` mod (Section 6) | `/fitness` pane with the 26-lab checklist, streak, auto-logged `pytest` passes, `/fitness-grade` | Core deliverable; the data stays in `$.store` on the student's machine | this guide |
| `token-weather` (sample mod) | Live context-window forecast above the prompt | Shows students context budgeting during long labs | [Playground](https://github.com/anthropics/claude-code-playground/tree/main/claude-code/mods) |
| ccusage | Reads local usage logs (no upload) and estimates USD spend from token counts with cache accounting; supports Claude Code, Codex, Gemini CLI, Copilot CLI and more | Weekly "what did this lab cost" ritual; `/usage` covers the current session only | [ccusage](https://ccusage.com/) |
| claude-mem | SessionStart/UserPromptSubmit/PreToolUse/PostToolUse/Stop hooks write observations to SQLite (`~/.claude-mem/claude-mem.db`), summarise them with the Agent SDK and re-inject them next session | Live example of the week-6 memory lesson; students can read its hooks.json | [claude-mem docs](https://docs.claude-mem.ai/architecture/overview) |
| claude-teacher-plugin | Always-on Socratic teaching; `/init-edu`, `/quiz-me` (spaced repetition, ≥80 % promotes, <50 % demotes), `/research`, `/compare`, `/demo` (animated HTML); "teacher must NEVER write code" rule | Closest existing analogue to a class-progress plugin; borrow the quiz/misconception model | [GitHub](https://github.com/yarikleto/claude-teacher-plugin) |
| now-next-methodology, claude-recap, claude-reflect | `NOW.md`/`NEXT.md` task files with `/next`; per-topic Markdown recaps via hooks; captures corrections into `CLAUDE.md` with review | Lightweight habits for students who do not want a database | [awesome-claude-code-plugins](https://github.com/ccplugins/awesome-claude-code-plugins) |

### 4b. Keep learning Claude Code techniques

| Mod / plugin | What it does | Source |
|---|---|---|
| `/powerup` | Interactive in-terminal lessons (see Section 3) | [What's new](https://code.claude.com/docs/en/whats-new) |
| learning-output-style plugin (Anthropic) | SessionStart hook that makes Claude ask the student to write 5–10 lines at decision points and prints `★ Insight` boxes; warns about extra token cost | [README](https://github.com/anthropics/claude-code/blob/main/plugins/learning-output-style/README.md) |
| superpowers (obra) | Complete methodology: brainstorm → design approval → plan → red/green TDD → subagent execution; installs from the official marketplace (`/plugin install superpowers@claude-plugins-official`) or `obra/superpowers-marketplace`; also installs on Codex, Gemini CLI, Cursor, Copilot CLI, OpenCode… | [GitHub](https://github.com/obra/superpowers) |
| Plugin Developer Toolkit / official marketplace | Official marketplace is `claude-plugins-official`; `/plugin install <name>@claude-plugins-official`; `/plugins/example-plugin` is the reference implementation | [claude-plugins-official](https://github.com/anthropics/claude-plugins-official) |
| Power-user tips, Academy | Anthropic's power-user article and Claude Academy subagents course for homework reading | [Power-user tips](https://support.claude.com/en/articles/14554000-claude-code-power-user-tips), [Academy](https://academy.claude.com/courses/claude-code-101/subagents) |

### 4c. Learn new things about coding agents

| Resource | Why | Source |
|---|---|---|
| What's new digest (weekly) | Dated feature table: `claude plugin eval` (W37), `/skill-doctor` (W36), fork mode default (W33), cross-session messaging (W32), background subagents by default (W27), sub-agents spawning sub-agents capped at 5 levels (W24), dynamic workflows orchestrating hundreds of subagents (W22), `/goal` (W20), conditional `if` hooks (W13) | [What's new](https://code.claude.com/docs/en/whats-new) |
| Mods API `.d.ts` | The TypeScript declarations are the ground truth per version; Claude Code writes them into `.claude-plugin/types/` of any `--plugin-dir` mod | [Create a mod](https://code.claude.com/docs/en/plugins/mods/create) |
| Awesome lists | `ccplugins/awesome-claude-code-plugins` (agent-triforce, artel, pro-workflow, nexus-agents, fractal…) for a monthly "plugin show-and-tell" | [awesome list](https://github.com/ccplugins/awesome-claude-code-plugins) |
| Cross-harness plugins | superpowers' install matrix shows the same skill set running on nine other coding agents — a natural week-10 discussion on portability of skills vs. MCP vs. mods | [GitHub](https://github.com/obra/superpowers) |

---

## 5. Gap analysis of the current plugin (v2.2.0)

What the repo has today (verified via the GitHub API on 9 Oct 2026): `.claude-plugin/marketplace.json` (marketplace `agentic-coding-fitness`, v2.2.0), `plugins/agentic-coding-fitness/.claude-plugin/plugin.json` (MIT, homepage lu.ma), `.claude/settings.json` enabling it, 20 `skills/*/SKILL.md` using `name`/`description`/`when_to_use` and a "Guided lab" kata each, `CHANGELOG.md` up to 2.1.0, `week15_production_graphrag/PLUGIN_TUTORIAL.md` (still says 9 skills).

| Gap | Impact | Fix in v3.0.0 |
|---|---|---|
| No `hooks/`, `agents/`, `.mcp.json`, output styles, statusline, evals | The plugin teaches agents but is not itself agentic | Add the mod, two agents, one eval (Section 6) |
| No progress tracking | Students cannot see where they are across 26 labs | `$.store`-backed `/fitness` pane, auto-logging |
| Weeks 19, 21, 23–26 and robotics exist in the repo but not as skills | Half the second semester is invisible to Claude | Add 7 skills (template provided) |
| `when_to_use` only | Fine — it is a documented optional field ([Skills](https://code.claude.com/docs/en/skills)) — but labs should set `disable-model-invocation: true` where you want manual `/plugin:skill` invocation | Review frontmatter per skill |
| Install docs mention `/plugin marketplace update` + reinstall | Still valid; add `claude plugin validate` and the version-pinning rule: "Claude Code caches an installed plugin by version, so your edits don't reach the installed copy until you increment" `version` ([Create a mod](https://code.claude.com/docs/en/plugins/mods/create)) | Update README + `week15_production_graphrag/PLUGIN_TUTORIAL.md` |
| No tests or evals | No way to prove the plugin helps | `claude plugin test` for the mod, `claude plugin eval` for the coach |

---

## 6. Step-by-step: build the v3.0.0 plugin with a mod

A ready-to-run starter kit is attached as `agentic-coding-fitness-v3-starter.zip` (11 files). Unzip it over `plugins/agentic-coding-fitness/` in your repo, keep your 20 existing skills, and follow the steps. The JavaScript parses (`node --check`), but I could not run Claude Code in this environment, so steps 3–4 are the ones that verify it against your installed version.

### Step 0 — Prerequisites (5 min)

```bash
claude --version          # need 2.1.287 or later for mods
git clone https://github.com/kwarodom/agenticaicodingfitness && cd agenticaicodingfitness
claude plugin validate ./plugins/agentic-coding-fitness   # baseline: should pass today
```

### Step 1 — Turn the plugin into a mod (10 min)

Add `hooks/hooks.json` with a `modules` key; "having it is what makes the plugin a mod" ([Create a mod](https://code.claude.com/docs/en/plugins/mods/create)). The same file may also carry classic settings hooks under `hooks` ([Mods reference](https://code.claude.com/docs/en/plugins/mods/reference)):

```json
{
  "description": "Agentic Coding Fitness: progress-tracking mod plus one classic settings hook",
  "modules": ["./register.js"],
  "hooks": {
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "\"${CLAUDE_PLUGIN_ROOT}/scripts/session-end.sh\"" }] }]
  }
}
```

The `SessionEnd` shell hook appends a CSV row per session to `~/.agentic-fitness/sessions.csv` — a deliberate side-by-side so students see a settings hook (shell, stdin JSON) next to a mod hook (in-process JS). Quote `${CLAUDE_PLUGIN_ROOT}` because a `command` without `args` runs through a shell ([Components](https://code.claude.com/docs/en/plugins/components)).

Bump `plugin.json` to `"version": "3.0.0"` and add `displayName`, `repository`, `keywords` (all optional fields per the [Manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference)).

### Step 2 — Write `hooks/register.js` (the core lesson, 40 min)

The starter module (280 lines, fully commented) registers these hooks:

| Hook | Pattern taught | What it does |
|---|---|---|
| `session.start` | Observe | Loads `progress` from `$.store`, records today's date for the streak, registers 4 commands and 1 tool, sets `$.ui.status(...)` |
| `tool.call` `{ tool: 'mcp__agentic-coding-fitness__log_lab' }` | Answer | Claude (or the grader agent) logs a lab; the full tool name is `mcp__<plugin>__<name>` ([Use the mods API](https://code.claude.com/docs/en/plugins/mods/api)) |
| `tool.call` `{ tool: 'Bash' }` | Wrap | Lets the command run (`await next(e)`), then if it was `pytest` on a `weekNN` path and the output says `N passed` with no failures, auto-logs the lab at 1.0 |
| `turn.complete` | Rewrite result | Returns `{ text: 'Lab auto-logged: …' }` under Claude's answer ([React to events](https://code.claude.com/docs/en/plugins/mods/events)) |
| `prompt.submit` | Add context | First prompt of a session gets `context: ['Agentic Coding Fitness progress: W05 · 7/26 labs · streak 3 …']` so the coach knows where the student is |
| `command.run` `{ command: 'fitness' }` | Open UI | `$.ui.open({ id: 'fitness', title, focus: true, closeOnEscape: true })` |
| `command.run` `fitness-log`, `fitness-next`, `fitness-grade` | Commands without a Claude turn | Manual log, next lab, and a `$.model.complete({ model: 'haiku', … })` rubric grader that checks `r.isAnswered` |
| `ui.render` `{ component: 'Spinner' }` | Redraw Claude's UI | Adds `· W05 · 7/26 labs · streak 3…` as the spinner suffix (the pattern from the docs' first-mod) |
| `ui.render` `{ component: 'Pane' }` | Draw your own UI | `$.ui.resolve(e)` → `Box`/`Text`/`Button`; one row per lab (`✓`, `○`, `·`), context percentage from `$.session.usage()`, `Reset labs` (hotkey `r`) and `Close` (`q`) buttons |

The CURRICULUM constant maps weeks to your skill directories (02–18 to the existing 20 skills, plus 19, 21, 23–26, robotics and the two cross-cutting skills). Edit it to match your week table.

Design notes that are easy to get wrong:

- Module variables reset on every hot reload; `$.store` is a JSON file under `~/.claude/plugins/store/` shared by every session on the machine (4 MiB cap); `$.state` survives reloads and redraws automatically but resets on `/clear`, `/resume`, `/branch` ([Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface), [Mods reference](https://code.claude.com/docs/en/plugins/mods/reference)). The starter uses `$.store` for durability; Exercise 2 migrates the pane state to `$.state`.
- `$.ui.open` boolean fields (`focus`, `closeOnEscape`) accept only `true`; passing `false` throws ([Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface)).
- Write event names and `$.store` keys as string literals; `claude plugin validate` does static analysis and refuses variables, dynamic `import()`, `require`, or a shadowed `on` ([Create a mod](https://code.claude.com/docs/en/plugins/mods/create)).
- The hooks module has no Node APIs, no `setTimeout`; use `$.clock.every/after`, `$.fs`, `$.process.run([...])` (no shell, 30 s default timeout) ([Use the mods API](https://code.claude.com/docs/en/plugins/mods/api)).

### Step 3 — Load, validate, iterate (10 min)

```bash
claude --plugin-dir ./plugins/agentic-coding-fitness
```

Then: `/fitness`, `/fitness-log 05 agent-loops 0.86`, `/fitness-next`, run `uv run pytest week05 -q` and watch the "Lab auto-logged" line. Edit `register.js` while the session runs; the transcript prints a reload line listing the hooks ([Create a mod](https://code.claude.com/docs/en/plugins/mods/create)). Check `/plugin` → Installed shows `1 mod active · agentic-coding-fitness`.

```bash
claude plugin validate ./plugins/agentic-coding-fitness
# expect lines like:
#  ❯ ./register.js hooks: session.start, tool.call{tool=mcp__agentic-coding-fitness__log_lab}, tool.call{tool=Bash}, turn.complete, prompt.submit, command.run{command=fitness}, ... ui.render{component=Spinner}, ui.render{component=Pane}
#  ❯ ./register.js calls: $.store.get, $.store.set, $.command.register, $.tool.register, $.ui.status, $.ui.open, $.ui.resolve, $.ui.invalidate, $.ui.close, $.session.usage, $.model.complete
```

After the first load, open `.claude-plugin/types/claude-code/index.d.ts` — it is "the fullest reference for your build, with a comment and an example for every mods API method" ([Create a mod](https://code.claude.com/docs/en/plugins/mods/create)). If a field name in the starter differs from your `.d.ts` (most likely candidates: the Bash tool result shape in `tool.call`, and `turn.complete` fields), the `.d.ts` wins.

### Step 4 — Test without a session (10 min)

`tests/fitness.test.ts` has three tests using `claude-code/testing`: manual logging and validation messages, the auto-log path (stubs `tool.call` to return `7 passed`), and the grader (stubs `model.complete` with `{ value: { isAnswered: true, text: 'PASS…' } }`). Stubs must be registered before the first call on `$`, and a mods-API stub returns `{ value }` while an event stub returns the event's result ([Test a mod](https://code.claude.com/docs/en/plugins/mods/test)).

```bash
cd plugins/agentic-coding-fitness && claude plugin test
```

If a stub is reported as unnecessary for your version, delete it; if a call is reported as unstubbed, add `on('<namespace.method>', () => ({ value: undefined }))`.

### Step 5 — Agents, skills, eval (20 min)

- `agents/coach.md` (Sonnet, read-only tools, Socratic, never writes lab code, AltoTech examples) and `agents/grader.md` (Haiku, runs `uv run pytest`, scores the rubric, calls `mcp__agentic-coding-fitness__log_lab`). Invoke as `@agent-agentic-coding-fitness:coach` ([Components](https://code.claude.com/docs/en/plugins/components)).
- `skills/fitness-whats-new/SKILL.md`: manual skill (`disable-model-invocation: true`, `allowed-tools: WebFetch`) that produces the 5-minute weekly digest from the What's new page and flags mods-API changes affecting your mod.
- `skills/week19-dgx-spark-sovereign-stack/SKILL.md`: template for the six missing weeks plus robotics; copy and fill from `week19_sovereign_ai_dgx/`, `week21_physical_ai_digital_twins/`, `week23_open_superintelligence_stack/`–`week26_nemoclaw_claws/`, `week_agentic_robotic/`.
- `evals/coach-explains-mods/`: one case with a rubric grader. Run `claude plugin eval init` first to let it draft cases/graders in the exact format for your version, then `claude plugin eval` to compare against the no-plugin baseline ([Plugin evals](https://code.claude.com/docs/en/plugin-evals)). The grader frontmatter in the starter (`type: rubric`) is a placeholder until `eval init` confirms the accepted grader types.

### Step 6 — Ship it (10 min)

1. `CHANGELOG.md`: add `3.0.0 — 2026-10-xx: progress-tracking mod (/fitness, /fitness-log, /fitness-grade, /fitness-next, log_lab tool), coach and grader agents, weeks 19–26 + robotics skills, plugin tests and evals`.
2. `.claude-plugin/marketplace.json`: set the plugin entry's `version` to `3.0.0`; the README should state the Claude Code version you tested with, since "the events and methods can change between releases" ([Create a mod](https://code.claude.com/docs/en/plugins/mods/create)).
3. Students update with `/plugin marketplace update agentic-coding-fitness`, then reinstall `agentic-coding-fitness@agentic-coding-fitness`, then `/reload-plugins` (or restart) — reloading warns if MCP tools would change ([Commands](https://code.claude.com/docs/en/commands)).
4. Update `week15_production_graphrag/PLUGIN_TUTORIAL.md` (currently says 9 skills) and the README skill count (26+).

### AltoTech tie-ins for the labs

- Blast-radius for Alto Reef: copy the `blast-radius` sample's pattern into a `tool.call { tool: 'Bash' }` hook that holds any `openshell`/policy-mutation command and shows the affected sandbox before Proceed/Cancel — the same per-session mutation gate the Reef runner enforces, now at the agent level.
- Sovereign tier: `$.model.complete` in `/fitness-grade` is a cloud call; Exercise 4 swaps it for `$.http.fetch` against a local NIM endpoint on DGX Spark so the grader never leaves the building.
- Hotel/HVAC: the `prompt.submit` context hook is the same "inject operating context" pattern Alto Copilot uses for a property's setpoints and alarms.

---

## 7. Exercises (with solutions)

**Exercise 1 — Deny a dangerous command (10 min).** Add a hook that refuses `git push --force` in any lab repo.

Solution:
```js
on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
  if (/git push .*--force/.test(e.command || '')) {
    return { deny: 'Force pushes are blocked in class repos. Push to a new branch.' }
  }
  return next(e)
})
```
A denied call does not run and shows no permission prompt; Claude receives the deny text as the tool result ([React to events](https://code.claude.com/docs/en/plugins/mods/events)).

**Exercise 2 — Move the pane state to `$.state` (20 min).** Why: a module variable resets on every save during development.

Solution: create `types/index.d.ts`
```ts
declare module 'claude-code' {
  interface PluginState { 'agentic-coding-fitness': { tab: 'labs' | 'stats' } }
}
```
add `"types": "./types/index.d.ts"` to `plugin.json`, then in `register.js`:
```js
import { atom, read, update } from 'claude-code'
const tab = atom({ plugin: 'agentic-coding-fitness', key: 'tab' }, 'labs')
// in ui.render:  const current = await read($, tab)
// in a Button:   onPress: () => update($, tab, () => 'stats')
```
Because the render hook read the atom, Claude Code redraws without `$.ui.invalidate` ([Draw in the interface](https://code.claude.com/docs/en/plugins/mods/interface)).

**Exercise 3 — Weekly reminder without a Claude turn (10 min).** Show "Class tonight 18:00 @ Rust Bar" on Tuesdays.

Solution:
```js
on('session.start', async ($, e, next) => {
  if (new Date().getDay() === 2) $.ui.toast('Agentic Coding Fitness tonight 18:00 @ Rust Bar')
  $.clock.every(30 * 60_000, () => $.ui.status(summaryLine()))
  return next(e)
})
```
`$.ui.toast` shows a transient notification; `$.clock.every` replaces `setInterval` and keeps running between turns ([Use the mods API](https://code.claude.com/docs/en/plugins/mods/api)).

**Exercise 4 — Sovereign grader (30 min).** Replace `$.model.complete` with a local endpoint.

Solution sketch:
```js
const r = await $.http.fetch('http://spark.local:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ model: 'local', messages: [{ role: 'system', content: RUBRIC }, { role: 'user', content: e.args }], max_tokens: 80 }),
})
if (!r.ok) return { text: 'Local grader failed: ' + r.status }
return { text: JSON.parse(r.text).choices[0].message.content.trim() }
```
`$.http.fetch` resolves to `{ status, ok, headers, text }` ([Use the mods API](https://code.claude.com/docs/en/plugins/mods/api)). Test it with `on('http.fetch', () => ({ value: { ok: true, status: 200, headers: {}, text: '…' } }))`.

**Exercise 5 — Audit before install (5 min).** Clone `claude-code-playground`, run `claude plugin validate ./claude-code/mods/blast-radius`, and list its `hooks:` and `calls:` lines. Discussion: which call would you refuse in a managed classroom? (Answer: anything under `$.process` or `$.http` you did not expect; organisations enforce this with a policy mod in the `prepend` tier — see [Manage mods](https://code.claude.com/docs/en/plugins/mods/admin).)

---

## 8. Suggested 3-week mini-module for the class

| Week | Topic | Lab | Checks |
|---|---|---|---|
| A | Extension landscape; `/powerup`, `/insights`, `/usage`, `/skill-doctor`; install `token-weather` | Students run `claude plugin validate` on a sample mod and explain every `calls:` line | Short written audit |
| B | Build the mod (Steps 1–4) | Starter kit → their own CURRICULUM edits → `claude plugin test` green | 3 tests pass; `/fitness` pane shows ≥1 ✓ |
| C | Agents + evals + ship (Steps 5–6) | Add coach/grader; `claude plugin eval init`; bump to 3.0.0; marketplace update | Eval beats baseline; a classmate installs from the marketplace |

---

## 9. What is verified vs. not

Verified against official docs or primary pages on 9 Oct 2026: everything cited to code.claude.com, claude.com/blog, github.com/anthropics, and the plugin repos above. Repo state verified via the GitHub API.

Not verified / treat with care:
- The starter `register.js` and tests were not executed against a live Claude Code (no CLI in this environment). Expect small field-name fixes after reading your generated `.d.ts`.
- The eval grader type `rubric` is a guess pending `claude plugin eval init`.
- `/powerup` lesson count and topics are not documented by Anthropic; third-party lists are speculative ([claudefa.st](https://claudefa.st/blog/guide/mechanics/claude-powerup)).
- ccusage subcommands (`daily`, `blocks`, `statusline`) were not confirmed from its homepage, only its purpose and sources ([ccusage](https://ccusage.com/)).
- Popularity figures (e.g. claude-mem star counts) come from secondary articles and were not checked.
- The Skool attachment could not be read (expired signed URL).
