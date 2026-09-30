# Alto Reef — Web Lab Runner for the NemoClaw on DGX Spark Tutorial

## UI specification, architecture, and frontier-model build brief

Prepared for Warodom Khamphanchai / AltoTech — 30 September 2026
Companion to "NemoClaw on DGX Spark — Beginner to Expert Tutorial with NAT"

---

## 0. What this document is for

The tutorial teaches every lab through the terminal. This document specifies a browser-based **lab runner** that wraps those same labs in a visual interface modelled on the NVIDIA booth demo you photographed at AI Day Singapore (23 September 2026): an isometric "reef" of sandboxes, a claw-profile builder, a sandbox manager, a per-sandbox Workbench with run/policy/chat/console tabs, and the embedded OpenClaw chat. It is written so that you can hand sections of it directly to frontier coding models (Claude Opus 5.5, GPT-6 Astra, Gemini 3.8 Flash) and get a working, honest runner — one that drives the real `nemoclaw`, `openshell` and `nat` commands on your Spark and never fakes state.

Two things to keep straight:

- **What is observed vs inferred.** Section 1 is an inventory of what the ten screenshots actually show. Everything after that is our own design, extended to cover tutorial Parts 3–6 (NAT, tracing, benchmarking, hardening), which the booth demo did not. The booth demo ("NemoClaw Reef") does not appear in NVIDIA's public community catalog at time of writing, so treat it as a design reference, not a codebase to fork ([NemoClaw Community Example Catalog](https://nvidia.github.io/nemoclaw-community/)).
- **Where the real UIs already exist.** OpenClaw ships its own browser Control UI (Vite + Lit) from the Gateway on port 18789, speaking directly to the Gateway WebSocket with a token or password supplied at handshake; it has agents, sessions, team mode, chat, terminal and browser panels ([OpenClaw Control UI docs](https://docs.openclaw.ai/web/control-ui)). NAT ships a FastAPI server with REST, streaming and OpenAI-compatible endpoints ([NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html)). Phoenix has its own trace UI on 6006 ([NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)). Alto Reef **orchestrates and embeds** these; it does not re-implement them.

---

## 1. Screenshot inventory (observed)

| # | Screen | What is visible |
|---|---|---|
| A | **Reef map** (`http://localhost:4454`) | Isometric sandy island in blue water. Each sandbox is a glass-walled enclosure with a coloured ring; claw avatars (pink/red crabs, a green lobster) sit inside; a central rock platform shows the NVIDIA logo, a named claw ("Reefus") and a hovering visitor claw ("Josh"). A top-right toolbar has small buttons (Reset booth, Kiosk, Build a Claw, language EN, etc.), a status banner ("Reef is partially active — LLM unreachable… running on fallback narration"), a bottom prompt bar ("Ask the team…"), and a right-hand **Comms stream** panel with tabs (Chat, Activity, History) listing timestamped messages per claw (System, ERIF, Captain Claw, You). |
| B | **Build a Claw** modal ("New sandbox profile") | Fields: Name (with Randomize); Runtime — "Lobster / OPENCLAW" (selected) or "Crab / HERMES" (badge "not configured"); Archetype grid — Researcher (e.g. Clawdia), Analyst (e.g. Shelldon), Critic (e.g. Coraline), Planner (e.g. Reefus), Writer (e.g. Pearl), Coder (e.g. Snips), Lead (e.g. Captain Claw), plus Reset; Shell colour swatches + custom hex; Accessories — Headwear, Eyewear; a right-hand live preview card (animated claw, archetype label, "Preview" one-liner: "Precise, data-driven. Won't claim without evidence.", Traits chips: DATA ANALYSIS, FILE IO; "OpenClaw skills (2)": MODEL-USAGE, MCPORTER); footer "Visitor profiles are saved as exportable OpenClaw agent packages."; buttons Cancel / Build Lobster. |
| C | **NemoClaw Sandboxes** panel ("Build teams in sandbox containers — click a claw, then click a sandbox; drag also works") | Header: Refresh, Reset Booth, collapse. Status tiles: GATEWAY "Healthy", INFERENCE "nvidia/Qwen3.6-35B-A3B-NVFP4". Left column **Sandbox profiles** (+New; search; filters ALL / LOBS / CRABS / FREE): cards with avatar letter, name, location ("in Quill Hollow", "in Workbench"), skill chips (SUMMARIZE, SESSION-LOGS, WEB RESEARCH, FILE IO; MODEL-USAGE, MCPORTER, DATA ANALYSIS; ORACLE, WEB RESEARCH, FACT CHECK; TASKFLOW, PLANNING, TEAM COORDINATION…). Right column **Workspaces** (+Sandbox): cards "Coral Cove" (model, "npm, pypi, huggingface +3", chips LIVE / GATEWAY OK / MODEL OK / POLICIES 6, "4 claws", MONITOR, member chips with REMOVE), "The Bridge" ("Run succeeded: 1/1 agents finished", 2 claws), "Quill Hollow", "Workbench" (OPEN MONITOR; **Quick squads**: Research Squad = Coraline + Pearl · brave; Code Squad = Snips + Shelldon · github+npm; Synthesis Squad = Captain Claw + Reefus; a task textarea; "Run Team — 2 assigned"; last result "Run failed: 0/2 agents succeeded" with run id `nemoclaw-snips-workbench-80f7bc61`). Bottom: STATUS ("Run failed: 0/2 agents succeeded", FINISHED), ASK "off", SECURITY "full"; **Live policies** chips BREW, GITHUB, HUGGINGFACE, LOCAL-INFERENCE, NPM, OPENCLAW-PRICING, PYPI, "23 more available — toggle in the Task Monitor", OPEN TASK MONITOR. |
| D | **Workbench** modal ("NemoClaw sandbox") | Title with pencil (rename); chips `nemoclaw-snips-workbench`, `sandbox_bench`, LIVE; FAILED badge; CLEAR FILES; close. Tabs: **Run + Outputs**, **Policies**, **Chat (50)**, **Console (668)**. **Quick starts** row: Relay Check (FAST), Build Web App (CODE), Policy Walk (SAFETY), Local Model (LOCAL), Skills Audit (SKILLS), Team Plan (PLAN), RETRY LAST. "Run a task in this sandbox" textarea + RUN. Chips: ASSIGNED 2, EXECUTABLE 2, NPM, PYPI, HUGGINGFACE, BREW, OPENCLAW-PRICING. **Stage crew**: per-claw cards (Sheldon — OPENCLAW · ANALYST · NO ACCESSORIES; Snips — OPENCLAW · CODER), "2 executable / 2 assigned", each with RUNS badge, soft tools (data analysis, file io / code authoring, code execution) and OpenClaw skills (model-usage, mcporter / coding-agent, github), counters READY 1 / SETUP 1 / FAILED 2. **Run timeline** cards: Profile setup DONE ("2 profiles prepared"); Skill readiness PARTIAL ("1 ready skill reported; 2 profile issues"); Sheldon FAILED EXEC_FAILED; Snips FAILED EXEC_FAILED; Result FAILED ("Run failed: 0/2 agents succeeded"); Diagnostics DONE. **Failure diagnostics** (TIMEOUT chips) and **Per-agent results** with raw stderr (`[UNDICI-EHPA] Warning: EnvHttpProxyAgent is experimental…`, `GatewayClientRequestError: FailoverError: LLM request timed out.`). |
| E | **OpenClaw chat** (`http://127.0.0.1:18790/chat?session=agent%3Amain%3A…`) | Breadcrumb OpenClaw › main › Chat. Message list with collapsible "Activity: 2 tools" → Tool Call → `Exec run python3 /sandbox/.openclaw/skills/franka-brain/scripts/franka_brain_client.py …` with TOOL INPUT JSON (`--bridge-url http://host.openshell.internal:8227 --runtime-parent … --json '{"operations":[{"op":"pick_place","object":"blue_cube","target":"blue_bin"}, …]}'`) and TOOL OUTPUT JSON (`{"method":"execute","ok":true,…}`). Composer "Message Isaac", "New messages" pill, footer status `inference/nvidia/Qwen3.6-35B-A3B-NVFP4 · Off` and "0%". Left half of the monitor: Isaac Sim driving a Franka arm — the claw's tool calls actuate a robot through a host-side bridge. |

Design language observed: deep navy surfaces (`#0b1a2b`-ish) with slightly lighter card panels, thin cyan/teal outlines, rounded 8–10 px corners, small uppercase letter-spaced section labels, colour-coded pill chips (teal for OK/LIVE, amber for PARTIAL, magenta/red for FAILED, blue for informational), monospace for ids and logs, and a playful isometric 3D map as the "home" view.

---

## 2. Product concept

**Alto Reef** is a single-page web app served from the Spark that turns the tutorial's labs into runnable, observable, gradeable tasks.

- **Reef map** is the home: every OpenShell sandbox is an enclosure; every claw profile is an avatar; NAT services, vLLM, Phoenix and the OTel collector are fixed landmarks (a "lighthouse" for the gateway, a "reactor rock" for vLLM).
- **Sandbox Manager** is the operator panel: create sandboxes from blueprints, assign profiles, see policy chips, inference model and gateway health.
- **Build a Claw** produces a profile: harness (OpenClaw "Lobster", Hermes "Crab", Deep Agents "Octo", NAT workflow "Nautilus"), archetype (system prompt + skill set + policy tier), colour and accessories, exported as an OpenClaw agent package / Hermes profile / NAT `workflow.yml`.
- **Workbench** is the per-sandbox cockpit with tabs **Run + Outputs**, **Policies**, **Chat**, **Console**, and — our additions — **Traces** and **Bench**.
- **Lab Mode** overlays the tutorial: each Part is a chapter, each Lab is a Quick Start with pre-filled commands, expected evidence and an auto-check. Progress is stored per learner.

Everything the runner shows must come from a real command or API on the Spark. When something is unreachable the UI must say so (the booth demo's "LLM unreachable — running on fallback narration" banner is the right instinct) rather than render stale or invented state.

---

## 3. Screen specifications

### 3.1 Reef map (route `/`)

- Canvas: isometric island rendered with SVG or a lightweight 2.5D canvas (no game engine). Phase 1 may use a flat card grid with the same semantics; the isometric skin is Phase 3.
- Entities: `Sandbox` enclosure (ring colour = health: teal running, amber degraded, red error, grey stopped), `Profile` avatar (colour = shell colour; badge = harness), `Landmark` (gateway, vLLM, NAT server, NAT MCP, Phoenix, OTel).
- Interactions: click avatar → profile card; click enclosure → Workbench; drag avatar onto enclosure → `POST /api/sandboxes/{id}/members`; hover landmark → health tooltip from `/api/health`.
- Overlays: status banner (gateway/inference health), Comms stream (right; tabs Chat / Activity / History; one line per run event or chat message across all sandboxes), Ask-the-team bar (fan a prompt to a selected squad).
- Toolbar: Refresh, Reset booth (destroys and recreates lab sandboxes — confirmation modal, lists exactly which `openshell sandbox delete` calls will run), Kiosk mode (hides destructive controls), Build a Claw, Language (EN/TH), Lab Mode toggle.

### 3.2 Sandbox Manager (route `/manager`, also a slide-in from the map)

Left column **Profiles** — list with search and filters ALL / LOBSTERS / CRABS / OCTO / NAUTILUS / FREE (unassigned). Card = avatar, name, harness, archetype, current sandbox, skill chips, policy-tier chip.

Right column **Workspaces** — one card per sandbox: name, model handle (from `openshell inference get` or per-sandbox provider), preset summary ("npm, pypi, huggingface +3" from `nemoclaw <s> policy list`), status chips LIVE / GATEWAY OK / MODEL OK / POLICIES n, member chips with REMOVE, last run summary, MONITOR (opens Workbench), + SANDBOX (opens blueprint picker: `openclaw`, `hermes`, `deepagents`, `base`, `alto-ops-nat`).

Footer — STATUS (last run), ASK mode (off / operator approval required), SECURITY (tier: restricted / balanced / open / personal), **Live policies** chips (from the sandbox's effective policy; click → Task Monitor with the full preset table and toggles that call `nemoclaw <s> policy add|remove <preset> --yes` after a confirm step that shows the `--dry-run` diff).

### 3.3 Build a Claw (modal)

Fields and behaviour:

| Field | Options | Effect on the generated artefact |
|---|---|---|
| Name | free text, Randomize | agent id, avatar label |
| Runtime | Lobster = OpenClaw, Crab = Hermes, Octo = Deep Agents, Nautilus = NAT workflow | which blueprint/onboarding path and which package format is exported |
| Archetype | Researcher, Analyst, Critic, Planner, Writer, Coder, Lead, **Ops Engineer** (Alto) | system prompt template, default skills, default policy presets (Researcher: brave/tavily; Coder: github, npm, pypi; Ops Engineer: alto-bms, local-inference only) |
| Shell colour | swatches + hex | avatar colour, ring colour when assigned |
| Accessories | headwear, eyewear | cosmetic; stored in profile metadata |
| Skills | multi-select from installed OpenClaw skills / Hermes plugins / NAT functions | added to package manifest |
| Policy tier | restricted / balanced / open (personal hidden in Kiosk mode) | `NEMOCLAW_POLICY_TIER` for onboarding or preset set for existing sandbox |

Preview card: avatar, one-line persona, trait chips, skills list, and — our addition — a **Policy preview** showing the endpoints the profile will be allowed to reach. Footer: "Profiles are saved as exportable packages" → download `.zip` (OpenClaw agent package), `.hermes-profile.yaml`, or NAT `workflow.yml` + `register.py` stub.

Runtime "not configured" badge appears when the harness's blueprint is not installed (probe: `nemohermes --version`, `nemo-deepagents --version`, `nat --version`).

### 3.4 Workbench (modal or route `/sandbox/{id}`)

Header: editable name, chips (sandbox id, blueprint, LIVE/STOPPED), last-run badge (RUNNING / DONE / FAILED), CLEAR FILES (empties `/sandbox/out` via `openshell sandbox exec`), Close.

**Tab: Run + Outputs**

- Quick starts row — in Lab Mode these are the tutorial labs for the current Part (see Section 6); outside Lab Mode: Relay Check (FAST — prove inference is local), Build Web App (CODE), Policy Walk (SAFETY — trigger a deny and show it), Local Model (LOCAL — `curl https://inference.local/v1/models`), Skills Audit (SKILLS), Team Plan (PLAN), Retry Last.
- Task textarea + RUN; assignment chips (ASSIGNED n / EXECUTABLE n) and live policy chips.
- Stage crew — per-profile cards with harness, archetype, soft tools, skills, counters READY / SETUP / FAILED.
- Run timeline — ordered step cards: Profile setup, Skill readiness, one per agent, Result, Diagnostics; each with status chip and one-line summary; click expands raw logs.
- Failure diagnostics — categorised chips (TIMEOUT, POLICY_DENY, TOOL_ERROR, EXEC_FAILED, LLM_UNREACHABLE) with the raw stderr preserved verbatim, plus a "What to check" hint drawn from the tutorial's troubleshooting table.
- Per-agent results — final text, files produced (download via `openshell sandbox download`), token/latency summary when available.

**Tab: Policies**

- Effective policy viewer (`openshell policy get <s> --full`) rendered as a table: entry name, host:port, protocol, enforcement, access/rules count, binaries.
- Revision list (`openshell policy list <s>`), diff between revisions.
- Preset toggles with dry-run preview (`nemoclaw <s> policy add <preset> --dry-run`) and apply.
- Decision stream — live allow / deny / inspect_for_inference lines tailed from `openshell logs <s> --tail --source sandbox`, filterable; a deny row offers "Propose allow rule" which drafts an `--add-allow` argument for review (never auto-applies).
- Policy editor — YAML with schema validation against the OpenShell policy schema and a "Set policy" action that runs `openshell policy set <s> --policy tmp.yaml --wait`.

**Tab: Chat**

- OpenClaw sandboxes: embed the Gateway Control UI in an iframe via the forwarded port (`openshell forward start --background 18789 <s>` → `http://127.0.0.1:<port>/chat?session=agent:main:main`), with the Gateway token supplied by the runner's config rather than typed by the learner ([OpenClaw Control UI docs](https://docs.openclaw.ai/web/control-ui)).
- Hermes sandboxes: native chat against the Hermes API on the forwarded 8642 port using its OpenAI-compatible route.
- NAT sandboxes: native chat against `POST /v1/chat/completions` on the forwarded NAT port, with `/v1/workflow/full?filter_steps=LLM_END,TOOL_END` used to render tool-call cards like screenshot E ([NAT API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html)).
- Footer status line mirrors screenshot E: `inference/<model handle> · <provider>` from `openshell inference get`.

**Tab: Console**

- Read-only tail of `openshell logs <s>`, `nemoclaw <s> logs --follow`, and the sandbox main-process output; counter badge shows unread lines.
- Command palette limited to an allow-list (`status`, `policy list`, `policy get`, `inference get`, `sandbox exec -- ls /sandbox/out`); no free-form shell (see 4.4).

**Tab: Traces (new)**

- Embedded Phoenix project view (iframe to `http://<spark>:6006/projects/<project>`) or a native list built from the OTel file exporter output when Phoenix is down.
- Correlation panel for Lab 4.5: for a selected run, show LLM span count (Phoenix), `inspect_for_inference` count (OpenShell decision stream) and tool spans, side by side, flagging mismatches.

**Tab: Bench (new)**

- Engine sweep runner: launches `vllm bench serve` (or Ollama `--verbose` loop) with chosen concurrencies; table + line chart of TTFT, per-request tok/s, aggregate tok/s.
- NAT eval runner: pick `eval_config.yml`, `max_concurrency`; renders `inference_optimization.json` percentiles, `accuracy_output.json` mean, and the `standardized_data_all.csv` token histogram.
- Sizing runner: wraps `nat sizing calc`; shows the p95 table and GPU estimate with the docs' "rough, not for production" caveat printed verbatim ([NAT sizing calculator](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/sizing-calc.html)).
- Sandbox-tax comparison: two eval runs (host vs sandbox) with the delta highlighted.
- Every number carries engine version, model handle, quantisation and OpenShell version captured at run time.

### 3.5 Task Monitor (drawer)

Full preset catalogue (all presets from `nemoclaw-blueprint/policies/presets/`) with risk notes from the security guide, current on/off state per sandbox, and a change log of every policy mutation the runner performed (who, when, command, exit code).

### 3.6 Lab Mode overlay

- Left rail: Parts 0–6 → Labs. Each lab card: goal, prerequisites (auto-checked: gateway up, vLLM healthy, NAT installed), the exact commands the runner will execute (visible before running — learners must be able to copy them), expected evidence, and an **auto-check** (Section 6).
- Exercises panel: shows the exercise text; "Reveal solution" is gated behind an attempt (textarea or a run).
- Progress: per learner (name + cookie-free local profile stored server-side), exportable as JSON.

---

## 4. Architecture

### 4.1 Topology on one DGX Spark

```
Browser ──http(s)──▶ Alto Reef frontend (static, :4454)
                     │  REST + SSE/WebSocket
                     ▼
              Reef Runner API (FastAPI, :4455, host-side, authenticated)
              ├─ CLI adapters: nemoclaw / nemohermes / nemo-deepagents / openshell / nat
              ├─ Port forward manager: openshell forward start --background <port> <sandbox>
              ├─ HTTP adapters: NAT (:8001), Phoenix (:6006), OTel collector (:4318), vLLM (:8000)
              ├─ Job queue + SQLite (runs, steps, profiles, sandboxes, lab progress)
              └─ Event bus → SSE stream /api/events
                     │
        ┌────────────┼──────────────────────────────┐
        ▼            ▼                              ▼
 OpenShell gateway  OpenShell sandboxes            vLLM / Ollama (host, 0.0.0.0:8000)
 (:8080, docker)    openclaw / hermes / nat        Phoenix (:6006)  OTel (:4318)
                    inference.local → provider
```

Port plan: 4454 UI, 4455 runner API, 8080 OpenShell gateway (auto 8990–9005), 18789 OpenClaw Control UI (forwarded per sandbox to an ephemeral local port), 8642 Hermes API, 8000 vLLM, 8001 NAT REST, 9901 NAT MCP, 6006 Phoenix, 4318 OTel. Sources for the fixed ports: OpenClaw Control UI on 18789 ([OpenClaw Control UI docs](https://docs.openclaw.ai/web/control-ui)); NAT MCP default 9901 ([NAT MCP server](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/mcp-server.html)); Phoenix 6006/4317 and OTel 4318 ([NAT observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html)); vLLM 8000 and the `0.0.0.0` binding requirement ([OpenShell playbook](https://build.nvidia.com/playbooks/openshell/instructions)).

### 4.2 Frontend

- React 18 + Vite + TypeScript + Tailwind + shadcn/ui (matches the template your build tooling already knows). Lit is acceptable if a model prefers parity with the OpenClaw Control UI, but do not mix.
- State: TanStack Query for REST, a small event store for SSE; no localStorage (kiosk browsers wipe it; the runner owns state).
- Routing: hash router (`/#/`, `/#/manager`, `/#/sandbox/:id`, `/#/lab/:part/:lab`).
- Charts: Recharts for Bench; SVG for the reef map; Monaco (lazy) for the YAML editor.
- i18n: EN + TH string tables from day one (Alto customers are Thai-speaking).

### 4.3 Runner API (Python 3.12, FastAPI)

- **Adapters** call the CLIs with `subprocess` and structured parsing; prefer JSON flags where they exist and fall back to regex with unit-tested fixtures. Every adapter method returns `{ok, stdout, stderr, exit_code, parsed, command}` so the UI can always show the exact command that ran.
- **Jobs**: long operations (onboard, rebuild, eval, sizing, bench) run in a worker with per-step events; cancellable; logs streamed.
- **Port-forward manager**: ensures one `openshell forward` per sandbox/port, records the local port, tears down on sandbox delete.
- **Health**: `/api/health` aggregates `openshell status`, `openshell inference get`, `curl /health` on vLLM, `/v1/models` on NAT, Phoenix root, OTel `:4318`.
- **Lab checks**: pure functions over adapter results (Section 6).
- **Persistence**: SQLite via SQLModel; tables `profiles`, `sandboxes`, `members`, `runs`, `run_steps`, `policy_changes`, `bench_results`, `lab_progress`, `learners`.

### 4.4 Security model for the runner itself

The runner is the most privileged process on the Spark — it can create and delete sandboxes and edit policies — so it must be treated as an operator console, not a demo page:

- Bind API to localhost by default; optional LAN exposure only behind the runner's own bearer token; kiosk mode disables all mutating endpoints.
- **No free-form shell.** Every endpoint maps to a fixed argv template; user input is passed as discrete arguments, never interpolated into a shell string. Sandbox names validated against `^[a-z0-9-]{1,40}$`.
- Policy mutations require a two-step flow: `--dry-run` result displayed → explicit confirm → apply. The runner logs each mutation.
- Never store provider API keys in the runner DB; credentials live in OpenShell providers/credential handles, matching the NemoClaw guidance that raw secrets stay outside the sandbox and the policy ([NemoClaw security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices)).
- OpenClaw Gateway token is read from the runner's config file and injected into the iframe URL/handshake server-side; the learner never sees it ([OpenClaw Control UI docs](https://docs.openclaw.ai/web/control-ui)).
- The runner runs as an unprivileged user in the `docker` group; it does not need root.

### 4.5 Honesty rules (the "no fallback narration" rule)

- If a probe fails, the corresponding chip turns red and the last successful timestamp is shown; no cached value is presented as live.
- Runs that fail keep the raw stderr (as the booth Workbench does) and never summarise a failure as success.
- Every metric shows its provenance (command, version, timestamp).
- Mock mode exists for frontend development only, is watermarked "MOCK DATA" across the viewport, and cannot be enabled when the API detects a real gateway.

---

## 5. Data model

```ts
type Harness = "openclaw" | "hermes" | "deepagents" | "nat";
type Archetype = "researcher"|"analyst"|"critic"|"planner"|"writer"|"coder"|"lead"|"ops_engineer";
type Tier = "restricted"|"balanced"|"open"|"personal";

interface Profile { id: string; name: string; harness: Harness; archetype: Archetype;
  color: string; accessories: {headwear?: string; eyewear?: string};
  skills: string[]; tier: Tier; systemPrompt: string; sandboxId?: string; createdAt: string; }

interface Sandbox { id: string; name: string; blueprint: "openclaw"|"hermes"|"deepagents"|"base"|"alto-ops-nat"|string;
  status: "running"|"degraded"|"error"|"stopped"|"unknown"; modelHandle?: string; provider?: string;
  presets: string[]; policyRevision?: number; forwards: {remotePort: number; localPort: number}[];
  members: string[]; lastRunId?: string; }

interface Run { id: string; sandboxId: string; kind: "task"|"quickstart"|"lab"|"eval"|"bench"|"sizing"|"policy";
  input: string; profileIds: string[]; status: "queued"|"running"|"done"|"failed"|"cancelled";
  startedAt: string; endedAt?: string; summary?: string; diagnostics: Diagnostic[]; }

interface RunStep { id: string; runId: string; order: number; title: string;
  status: "pending"|"running"|"done"|"partial"|"failed"; summary?: string; command?: string;
  stdout?: string; stderr?: string; exitCode?: number; startedAt?: string; endedAt?: string; }

interface Diagnostic { category: "TIMEOUT"|"POLICY_DENY"|"TOOL_ERROR"|"EXEC_FAILED"|"LLM_UNREACHABLE"|"UNKNOWN";
  message: string; hint?: string; }

interface PolicyChange { id: string; sandboxId: string; kind: "preset_add"|"preset_remove"|"set"|"update";
  dryRun: string; command: string; exitCode: number; at: string; by: string; }

interface BenchResult { id: string; kind: "engine"|"nat_eval"|"sizing"|"sandbox_tax"; modelHandle: string;
  engine: string; engineVersion: string; quant?: string; openshellVersion?: string;
  concurrency?: number; metrics: Record<string, number>; artefactPaths: string[]; at: string; }

interface LabProgress { learnerId: string; labId: string; status: "not_started"|"in_progress"|"passed"|"failed";
  evidence?: Record<string, unknown>; attempts: number; updatedAt: string; }
```

---

## 6. Lab-to-runner mapping (the contract with the tutorial)

Each tutorial lab becomes a Quick Start with an id, the commands the runner executes, and an auto-check. The runner must display the commands before running them.

| Lab id | Tutorial lab | Runner action (argv templates) | Auto-check passes when |
|---|---|---|---|
| L1.1 | Verify the Spark | `nvidia-smi`, `uname -r`, `docker --version`, `free -g` | kernel ≥ 6.2, docker present, GB10 detected |
| L1.2 | One-command install | opens a terminal pane streaming the installer; runner does not run it silently | `nemoclaw --version` succeeds |
| L1.3 | Scripted install variants | `NEMOCLAW_PROVIDER=<x> nemoclaw onboard …` chosen via form | sandbox appears in `openshell sandbox list` |
| L1.4 | Lifecycle | `nemoclaw <s> status|restart|stop|start` buttons | status transitions observed |
| L1.5 | First conversation, local proof | Chat tab + `openshell inference get` + `openshell sandbox exec -n <s> -- curl -s https://inference.local/v1/models` | provider is local (`ollama`/`local-vllm`) and models list returned |
| L1.6 | Telegram channel | guided form; runner only shows the command | learner confirms a message round-trip; screenshot upload |
| L1.7 | Hermes / Deep Agents | `nemohermes onboard`, `nemo-deepagents onboard` | both sandboxes present |
| L2.1 | Read the policy | `nemoclaw <s> policy get`, `openshell policy get <s> --base/--full` | Policies tab renders ≥ 6 baseline entries |
| L2.2 | Policy anatomy | YAML editor with schema hints | learner's YAML validates |
| L2.3 | Iterate loop | Decision stream + `policy update --add-endpoint … --dry-run` → apply | a former deny becomes allow after revision increments |
| L2.4 | TUI approval | embedded terminal running `openshell term` | new revision in `policy list` |
| L2.5 | Presets & posture | Task Monitor toggles | preset chips match `policy list` |
| L2.6 | Snapshot / rebuild | `nemoclaw <s> snapshot create --name …`, `rebuild` | snapshot listed; sandbox back to running |
| L2.7 | Raw OpenShell + vLLM | wizard: start vLLM container → `openshell provider create` → `inference set` → `sandbox create --from openclaw` | `inference get` shows `local-vllm`; Chat tab works |
| L3.1–3.3 | NAT install, vLLM, hello workflow | `uv venv`, `uv pip install …`, `nat run --config_file … --input …` | `nat --version` ok; run returns non-empty answer |
| L3.4 | Custom tool | file editor for `chiller_tool.py` + `uv pip install -e` + `nat run` | tool span appears in `/v1/workflow/full` |
| L3.5 | Serve | `nat serve --port 8001`; Chat tab (NAT mode) | `/v1/chat/completions` 200 |
| L3.6 | MCP both ways | `nat mcp serve --port 9901`; `nat mcp client tool list --url …` | tool list contains `chiller_kpi` |
| L3.7 | Code execution sandbox | start local sandbox script; run a code task | `code_execution` tool output rendered |
| L3.8 | NAT inside OpenShell | build image, `openshell sandbox create --from ./ --policy … --forward 8001 …` | sandbox running; NAT `/v1/models` via forward; `inspect_for_inference` seen |
| L4.1–4.2 | Phoenix / OTel | start containers; edit `general.telemetry`; run | Traces tab shows a trace for the run id |
| L4.3 | Hermes → Langfuse | guided credential-handle flow (runner never sees keys) | learner confirms trace in Langfuse |
| L4.4 | Policy plane | Decision stream + Console | learner tags one allow, one deny, one inspect |
| L4.5 | Correlate | Correlation panel | LLM span count equals `inspect_for_inference` count |
| L5.1 | Engine sweep | Bench tab engine runner | table with ≥ 3 concurrencies |
| L5.2 | NAT eval + profiler | Bench tab eval runner | `inference_optimization.json` parsed |
| L5.3 | Sizing | Bench tab sizing runner | p95 table with ≥ 10 concurrency rows |
| L5.4 | Sandbox tax | two evals, delta view | both runs present; delta computed |
| L5.5 | Harness benchmark | p50/p95 loop against Hermes/OpenClaw API | percentile table rendered |
| L6.1 | Production policy | editor → `policy set … --wait`; deny test | `write_setpoint` call denied (403 in decision stream) |
| L6.2 | Custom blueprint | `nemoclaw onboard --from Dockerfile` | sandbox from custom image running |
| L6.3 | Remote/external gateway | forms for `gateway start --remote`, `blueprint-runner plan` | plan output shown; status ok |
| Capstone | Alto Ops Claw v1 | all of the above; export report | report PDF/MD generated from stored runs |

---

## 7. Design system

- **Surfaces:** `--bg #0a1729`, `--panel #10213a`, `--panel-2 #16304f`, `--border rgba(120,200,255,.18)`.
- **Accent:** teal `#3fd0c9` (OK/LIVE/primary buttons), blue `#4aa3ff` (info), amber `#f2b64c` (PARTIAL/degraded), magenta `#ff5c8a` (FAILED), NVIDIA green `#76b900` only for the inference/provider chip.
- **Type:** Inter for UI, JetBrains Mono for ids, commands, logs; 11 px uppercase letter-spaced section labels; 13–14 px body; 16 px card titles.
- **Components:** Chip (status/tag), Tile (stat), Card (profile / workspace / step), Modal, Drawer, Tabs, Timeline, LogPane (virtualised), YamlEditor, ReefCanvas.
- **Motion:** 150 ms ease for chip state changes; avatars idle-bob only on the map; no motion in log panes.
- **Accessibility:** all chips carry text, not colour alone; keyboard access for every drag interaction (select profile → select sandbox → Enter).
- **Language:** EN/TH toggle in toolbar; Thai strings reviewed by a native speaker before customer demos.

---

## 8. Building it with frontier models

### 8.1 Division of labour

| Role | Model | Why |
|---|---|---|
| Architect + backend adapters + security review | Claude Opus 5.5 | long-context reasoning over CLI docs and the policy schema; careful with argv safety and fail-closed logic |
| Frontend (React + Tailwind + reef canvas) | GPT-6 Astra | strong UI generation from spec and screenshots; iterate on layout fidelity |
| Fixtures, parsers, tests, i18n tables, fast iterations | Gemini 3.8 Flash | cheap and fast for high-volume, low-ambiguity work: CLI output fixtures, regex parsers, unit tests, Thai string tables |
| Cross-check | any second model | ask a different model to review each milestone's diff against the honesty and security rules |

Give every model the same **Shared brief** (8.2) plus its **Milestone prompt** (8.4). Attach this document, the tutorial, and the screenshots. Do not let a model invent CLI flags — instruct it to mark any flag it is unsure of with `// VERIFY` and to read the docs URLs listed in Section 9.

### 8.2 Shared brief (paste as the system/project prompt)

```
You are building "Alto Reef", a web lab runner for NVIDIA NemoClaw / OpenShell / NeMo Agent Toolkit on a single DGX Spark.
Ground truth: alto_reef_web_runner_spec.md (this file) and nemoclaw_dgx_spark_nat_tutorial.md. Read both fully first.
Hard rules:
1. Never fabricate state. Every value shown in the UI comes from a real command/API result stored with its command, timestamp and exit code. Unreachable = red chip + last-seen time.
2. No free-form shell. Adapters use fixed argv templates; user input is passed as separate argv elements; names validated by regex.
3. Policy mutations are two-step: dry-run → confirm → apply, all logged.
4. Secrets never enter the runner DB or the browser; OpenShell providers/credential handles hold them.
5. If you are unsure a CLI flag or endpoint exists, write it with a `// VERIFY` comment and cite the doc URL you would check. Do not guess silently.
6. Frontend: React 18 + Vite + TS + Tailwind + shadcn/ui, hash routing, TanStack Query, no localStorage. Backend: Python 3.12 FastAPI + SQLModel + SQLite, SSE for events.
7. Ship tests: adapter parsers against fixtures; API contract tests; one Playwright smoke test per screen.
8. Mock mode is for frontend dev only and must be watermarked "MOCK DATA".
Deliver code as a single repository `alto-reef/` with `frontend/`, `runner/`, `fixtures/`, `docs/`, and a Makefile: `make dev`, `make test`, `make build`, `make serve` (serves frontend on :4454, runner on :4455).
```

### 8.3 Repository layout to request

```
alto-reef/
  frontend/            # Vite React app
    src/pages/{Reef,Manager,Sandbox,Lab}.tsx
    src/components/{Chip,Tile,ProfileCard,WorkspaceCard,BuildClawModal,Workbench/*,ReefCanvas,LogPane,YamlEditor}.tsx
    src/api/{client.ts,events.ts,types.ts}
    src/i18n/{en.json,th.json}
  runner/
    app/main.py                      # FastAPI app, auth, SSE
    app/adapters/{openshell.py,nemoclaw.py,hermes.py,deepagents.py,nat.py,vllm.py,phoenix.py,otel.py}
    app/jobs/{queue.py,onboard.py,run_task.py,eval.py,bench.py,sizing.py,policy.py}
    app/labs/{catalog.py,checks.py}  # lab ids from Section 6
    app/models.py app/db.py app/events.py app/security.py
    tests/{test_parsers.py,test_api.py}
  fixtures/            # real captured CLI outputs (redacted) used by parser tests
  docs/                # this spec + ADRs
  Makefile  README.md
```

### 8.4 Milestone prompts

**M0 — Fixtures (Gemini 3.8 Flash, run on the Spark first).** "Write `scripts/capture_fixtures.sh` that runs these read-only commands and stores stdout/stderr/exit code under `fixtures/<cmd>.txt`: `openshell status`, `openshell sandbox list`, `openshell inference get`, `openshell provider list`, `openshell policy list <s>`, `openshell policy get <s> --full`, `openshell logs <s> --tail --source sandbox` (10 s), `nemoclaw <s> status`, `nemoclaw <s> policy list`, `nemoclaw <s> policy get`, `nat --version`, `nat info components -t tracing`, `curl -s localhost:8000/v1/models`. Redact tokens. Then write pytest parsers for each fixture."

**M1 — Runner skeleton (Opus 5.5).** "Implement `runner/` per Sections 4–5: FastAPI with bearer auth, SQLModel models, adapters with argv templates and parsers from `fixtures/`, `/api/health`, `/api/sandboxes` (list/get/create/delete/members), `/api/profiles` CRUD + export, `/api/runs` (create task run, list, get, cancel) with a job queue emitting SSE on `/api/events`, `/api/sandboxes/{id}/policy` (get full, revisions, dry-run, apply, decision stream), `/api/forwards`. Include `security.py` with name validation and the two-step mutation flow. Provide OpenAPI and contract tests."

**M2 — Manager + Workbench UI (Astra 6).** "Build the Sandbox Manager and Workbench screens exactly per Sections 3.2 and 3.4 and the screenshots (attach B, C, D). Use the design tokens in Section 7. Wire to the M1 API via TanStack Query and SSE. Implement Run + Outputs (quick starts, stage crew, run timeline, failure diagnostics with raw stderr), Policies (table, revisions, dry-run/apply, decision stream), Console (virtualised log pane), Chat (iframe for OpenClaw via forwarded port; native chat for NAT/Hermes using OpenAI-compatible endpoints). Playwright smoke tests for each tab."

**M3 — Build a Claw + exports (Astra 6 + Opus 5.5).** "Implement the Build a Claw modal (Section 3.3) with live preview and policy preview. Backend export endpoints produce: OpenClaw agent package zip (agent config + system prompt + skills manifest), Hermes profile YAML, NAT `workflow.yml` + `register.py` stub. Mark any package-format assumption with `// VERIFY` and link the harness docs."

**M4 — Lab Mode + checks (Opus 5.5 for checks, Gemini 3.8 Flash for catalog JSON).** "Encode Section 6 as `labs/catalog.py` (id, part, title, goal, prerequisites, argv templates, expected evidence, exercise ids) and `labs/checks.py` pure functions over adapter results. Frontend: Lab rail, lab card with visible commands, Run, evidence panel, gated solutions, progress export."

**M5 — Traces + Bench (Opus 5.5).** "Implement Traces tab (Phoenix iframe + native fallback from OTel file exporter; correlation panel comparing LLM spans with `inspect_for_inference` counts) and Bench tab (engine sweep via `vllm bench serve` inside the vLLM container, NAT eval/profiler runner parsing `inference_optimization.json`, `accuracy_output.json`, `standardized_data_all.csv`; sizing runner; sandbox-tax delta). Every result row stores engine, version, quant, OpenShell version."

**M6 — Reef map + kiosk polish (Astra 6).** "Implement `ReefCanvas`: SVG isometric island, enclosures per sandbox with health rings, avatars per profile, landmarks per service, drag-to-assign with keyboard fallback, Comms stream drawer, status banner, Ask-the-team bar, kiosk mode. EN/TH toggle."

**Review prompt (any second model, every milestone).** "Review this diff against Section 4.4 (security) and 4.5 (honesty). List every place where (a) user input could reach a shell, (b) a value is shown without provenance, (c) a failure could be rendered as success, (d) a CLI flag is unverified. Output as a checklist with file:line."

### 8.5 Acceptance criteria per milestone

- M1: `make test` green; `curl -H 'Authorization: Bearer …' :4455/api/health` returns real probe results; creating a sandbox via API is visible in `openshell sandbox list`.
- M2: A failed run shows raw stderr and a diagnostic category; a policy deny appears in the decision stream within 2 s; the OpenClaw iframe loads without the learner typing a token.
- M3: Exported OpenClaw package onboards into a fresh sandbox and the chat answers.
- M4: Completing L1.5 through the UI turns the lab card green only when `inference get` reports a local provider.
- M5: Bench results reproduce Part 5's method; Correlation panel shows equal LLM-span and inspect counts for a one-tool NAT run.
- M6: Kiosk mode hides Reset booth and all mutations; the app survives a 6-hour booth session without a reload.

---

## 9. Verification sources for the models

Hand these URLs to the models for anything marked `// VERIFY`:

- NemoClaw: [repository](https://github.com/NVIDIA/NemoClaw), [how it works](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/about/how-it-works), [architecture](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/architecture), [network policies reference](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/reference/network-policies), [security best practices](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/security/best-practices), [sub-agent setup](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/configure-agents/set-up-sub-agent), [community catalog](https://nvidia.github.io/nemoclaw-community/).
- OpenShell: [repository](https://github.com/NVIDIA/openshell), [policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema), [sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies), [manage sandboxes](https://docs.nvidia.com/openshell/sandboxes/manage-sandboxes), [manage providers](https://docs.nvidia.com/openshell/sandboxes/manage-providers), [security best practices](https://docs.nvidia.com/openshell/security/best-practices), [playbook](https://build.nvidia.com/playbooks/openshell/instructions).
- OpenClaw: [Control UI](https://docs.openclaw.ai/web/control-ui), [docs home](https://docs.openclaw.ai/).
- NAT: [docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/), [API server endpoints](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/rest-api/api-server-endpoints.html), [MCP server](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/mcp-server.html), [MCP client](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html), [observe](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/observe/observe.html), [profiler](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/profiler.html), [evaluate](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/evaluate.html), [sizing calculator](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/sizing-calc.html), [CLI](https://docs.nvidia.com/nemo/agent-toolkit/latest/reference/cli.html).
- Spark: [vLLM instructions](https://build.nvidia.com/spark/vllm/instructions), [Ollama instructions](https://build.nvidia.com/spark/ollama/instructions), [NemoClaw on Spark](https://build.nvidia.com/spark/nemoclaw/overview).

---

## 10. Open items

- The booth demo's exact package format for "exportable OpenClaw agent packages" is not documented publicly; M3 defines our own and must be validated against OpenClaw's agent configuration docs.
- The OpenClaw Gateway WebSocket RPC surface is documented separately from the Control UI page; if we later want native chat instead of the iframe, that reference must be read first ([OpenClaw Control UI docs](https://docs.openclaw.ai/web/control-ui)).
- The Isaac Sim / Franka bridge seen in screenshot E (`host.openshell.internal:8227`) shows a host-side tool bridge pattern; an Alto equivalent (a BMS bridge) belongs in a later phase and needs its own policy entry and threat review.
- Whether `openshell` and `nemoclaw` expose stable JSON output flags should be checked at M0; if not, parsers must be fixture-tested and pinned to versions.
