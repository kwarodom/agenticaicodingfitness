# Week 23 — Subagent Team for the Stack Navigator

The **Stack Navigator** (`00_stack_navigator/`, port **8112**) — an interactive hub web app that
walks the whole NVIDIA AI & agent development stack, probes a live **DGX Spark** (or
**build.nvidia.com** cloud NIMs), and links into the 12 per-topic apps — was researched, built,
and QA'd by a team of six specialized subagents orchestrated as a 3-phase workflow:

```
Phase 1 · RESEARCH (parallel)          Phase 2 · BUILD (parallel)       Phase 3 · QA
┌──────────────────────────┐
│ NVIDIA AI Stack Specialist│──┐
├──────────────────────────┤  │       ┌───────────────┐
│ DGX Spark Specialist      │──┼──────▶│ Backend Agent  │──┐   ┌──────────────────┐
├──────────────────────────┤  │       ├───────────────┤  ├──▶│ QA & Integration  │
│ Research / Audit Agent    │──┘       │ Frontend Agent │──┘   └──────────────────┘
└──────────────────────────┘          └───────────────┘
   write research/*.{json,md}          disjoint file ownership   tests, fixes, README wiring
```

Orchestration principles used:
- **Research before build** — three specialists write their findings to
  `00_stack_navigator/research/` so builders work from files, not hearsay.
- **Contract-first parallel build** — backend and frontend build simultaneously against a fixed
  API contract (endpoints, JSON shapes, NDJSON streaming protocol) with strict file ownership
  (backend owns `*.py` + JSON; frontend owns `static/guide.html`) so they can't conflict.
- **QA owns everything at the end** — one agent launches the server, tests every endpoint in SIM
  and REAL mode, fixes bugs directly, and integrates the app into the week23 README.

---

## 1 · NVIDIA AI Stack Specialist

| | |
|---|---|
| **Role** | Domain expert for the full NVIDIA stack: Nemotron 3, NIM, Dynamo, TensorRT-LLM, Agent Skills, NeMo Agent Toolkit, AI-Q, NemoClaw, OpenShell, Guardrails, NeMo Relay/Phoenix, NeMo Gym/RL, Data Flywheel. |
| **Can do** | Read the 12 existing apps + `README.md`/`SLIDES.md`, synthesize with model knowledge, produce structured teaching content (what/why, demo prompts, SIM answers, links). |
| **Responsibility** | Author `research/stack_content.json` — the single content source the app serves at `/api/stack`: every stack layer/node, its DGX-Spark commands, its build.nvidia.com path, the guided-tour order. |
| **Has done** | Wrote `research/stack_content.json`: 5 bottom-up layers (hardware → runtime → model → harness → flywheel), **16 nodes**, 16-step guided-tour `journey[]` following the 01→12 learning path. All 12 apps mapped (05 shared by NAT + AI-Q; 07 by Guardrails + OpenShell). Content grounded in the READMEs, `SLIDES.md`, each app's chapter STEPS, and demo sources; web-verified `nvidia/nemotron-3-super-120b-a12b`, `-ultra-550b-a55b`, `-nano-omni-30b-a3b-reasoning` on build.nvidia.com. Honest runnable flags: `true` / `"partial"` / cloud-only per node. |
| **Needs later** | NIM image tags for Nano/Super and their aarch64/GB10 profiles unverified; `github.com/NVIDIA/NemoClaw` URL unverified; pip/module names for NAT (`nvidia-nat`) and Dynamo taken from knowledge, not tested. Sim-answer numbers are the course's illustrative figures, not benchmarks. Refresh when NVIDIA ships new components. |

## 2 · DGX Spark Specialist

| | |
|---|---|
| **Role** | Hardware/runnability expert for the DGX Spark (GB10 Grace Blackwell, 128 GB unified memory, ARM64) and the repo's `DGX_CONN local|tunnel|cloud` conventions. |
| **Can do** | Fit-math (which Nemotron tiers fit 1 Spark / 2 Sparks over QSFP 200 GbE / cloud-only), exact install/run commands per layer (Ollama, vLLM, TensorRT-LLM, NIM containers with ARM64 caveats, Dynamo, …), endpoint-probing specs. |
| **Responsibility** | Author `research/dgx_spark_runbook.md` — the definitive "can I run this on my Spark, and how?" guide, plus the build.nvidia.com on-ramp (nvapi- key, `https://integrate.api.nvidia.com/v1`). |
| **Has done** | Wrote the 4-section runbook: (1) GB10 capability matrix (128 GB unified LPDDR5X @ 273 GB/s, ~1 PFLOP sparse FP4, aarch64) + GB-per-param fit rules → **Nano = 1 Spark, Super = 1 Spark at Q4/NVFP4, Ultra = cloud-only** (357 GB NVFP4 exceeds even 2-Spark 256 GB); (2) per-layer install/run commands, ports, and probe endpoints for 11 layers with a SPARK-1/SPARK-2/CLOUD/SIM verdict table (NAT moved to :8001, Guardrails to :8500 to dodge the contested :8000); (3) nvapi- key flow + curl example + ~40 RPM free-tier note; (4) exact probe order (`DGX_BASE_URL` → `DGX_CONN` → localhost:11434/v1), 2 s timeouts, auth-header rules. Grounded in the repo's own on-Spark demos + web verification of NVIDIA's dgx-spark-playbooks. |
| **Needs later** | Items marked `[UNCERTAIN]` in the doc: NAT/Guardrails default ports, AI-Q compose ports, Dynamo module names (drift between releases). Deck-only products (NeMo Relay, OpenShell, NemoClaw) marked SIM — not verifiably installable. Validate all commands on the physical Spark. |

## 3 · Research / Audit Agent

| | |
|---|---|
| **Role** | Codebase archaeologist: extracts the week23 "house style" and audits existing content for inconsistencies. |
| **Can do** | Deep-read `tutorial_server.py` / `config.py` / `guide.html` of existing apps; document CSS tokens, FastAPI patterns, port auto-pick, SIM/REAL badge UX; cross-check README tables against actual code. |
| **Responsibility** | Author `research/house_style.md` (build spec so the new app is indistinguishable in style) and `research/audit_findings.md` (README-vs-code mismatches, missing per-app READMEs — report only, no fixes). |
| **Has done** | Read app 01 in full (server 343 + config 326 + guide.html 768 lines) and app 02's `view.py`/`sim.py`; documented the exact CSS `:root` tokens (dark olive + NVIDIA green `#76b900`), FastAPI/streaming patterns, connection-panel UX, the inline-SVG viz toolkit, and requirements. Audit found: all 12 ports match the README ✓, but 8 apps printed a wrong `DGX_GUIDE_PORT` busy-port hint, 10–12 apps carry stale week19 docstrings, 5 apps lack READMEs, app 01 drifts on naming (`ntsim`/`ntview`) and requirements, and `week23/reference/` is cited but gitignored. |
| **Needs later** | Unfixed findings held for human judgment (see QA row): stale week19 docstrings, 5 missing per-app READMEs, app-01 naming drift, and adding app 00 to the 12 apps' TUTS prev/next cycle (requires touching all 12 `guide.html` files). |

## 4 · Backend Agent

| | |
|---|---|
| **Role** | FastAPI server engineer for the hub app. |
| **Can do** | Implement the API contract: `GET /api/stack`, `GET/POST /api/status·probe` (2 s-timeout endpoint probing, never hangs), `GET /api/apps` (TCP liveness of ports 8100–8111), `POST /api/run` (NDJSON streaming — real OpenAI-compatible calls in REAL mode, token-paced simulator in SIM mode). |
| **Responsibility** | `tutorial_server.py`, `config.py` (adapted from app 01's `DGX_CONN` switch), `sim.py`, `stack_content.json` (runtime copy), `requirements.txt`, app `README.md`. Owns all `.py` files; must never touch `static/guide.html`. |
| **Has done** | Shipped all 6 files. `config.py` keeps the house `DGX_CONN` switch with probe timeouts tightened 3–4 s → 2 s per the runbook. `/api/run` streams NDJSON in both modes: REAL merges `delta.content` + `delta.reasoning` from SSE `/chat/completions` (360 s wall cap, run-lock against concurrent runs); SIM paces each node's `sim_answer` at plausible GB10 tok/s. `/api/probe` never hangs (~2.2 s shared deadline, Bearer/basic/x-api-key auth). `/api/apps` folds 16 nodes into 12 app cards with concurrent 0.3 s TCP liveness checks. Self-verified live on :8199 in SIM and REAL (real streamed inference against local Ollama), then killed all servers. |
| **Needs later** | `NT_GUIDE_PORT` env var collides with app 01's (exporting it globally moves both apps) — rename to `NAV_GUIDE_PORT`. In SIM mode the meta line can report a live-resolved model name (inherited house config behavior). `dgxsh.py` / DGX console deliberately omitted — add if the hub later wants the SSH overlay. ~~No connect endpoint~~ **fixed post-QA**: `/api/run` originally ignored the UI-selected connection (runs always hit the launch-time endpoint — a cloud run against local Ollama 404'd); it now accepts `conn`/`base_url`/`api_key` from the 🔌 panel and normalizes bare URLs to `/v1`. Remaining: the run-lock is held until a stream fully finishes server-side, even if the browser disconnects — a second Run during that window is rejected. |

## 5 · Frontend Agent

| | |
|---|---|
| **Role** | Single-file UI engineer (sovereign style: inline CSS/JS/SVG, zero external libraries or CDNs). |
| **Can do** | Animated clickable inline-SVG stack diagram, sticky connection panel with cloud presets (Ollama / tunnel / build.nvidia.com / HF router) + localStorage persistence, guided tour over `journey[]`, per-node run panels rendering live NDJSON token streams, 12-app status strip. |
| **Responsibility** | Exactly one file: `static/guide.html`, coded strictly against the API contract (backend may not exist yet while it works). |
| **Has done** | Shipped `static/guide.html` (~680 lines, zero external resources). Fully **data-driven** hero SVG built from `/api/stack` — 5 layer bands, 16 clickable node boxes with runnable-status dots, animated upward token pulses; content changes in the JSON flow through with no HTML edits. Sticky connection bar with 4 presets (Ollama / DGX tunnel / build.nvidia.com / HF router), REAL/SIM pill, model picker, localStorage persistence. 16-stop guided tour with arrow-key nav. Per-node panels: DGX-Spark vs cloud tabs with copy-button commands + generated curl, editable demo prompt, live NDJSON token streaming with tok/s footer. 12-app status strip auto-refreshing every 15 s. Boot retries `/api/stack` so the page is never blank. |
| **Needs later** | Verified via node DOM-stub, not a real browser — one human look at http://127.0.0.1:8112 for SVG label fit at narrow widths and sticky-bar stacking. API keys persist in localStorage (fine for a local dev tool; strip if unacceptable). Screen-reader labels TBD. Hub intentionally not added to the 12 apps' prev/next TUTS cycle. |

## 6 · QA & Integration Agent

| | |
|---|---|
| **Role** | Test engineer with fix authority over the whole app. |
| **Can do** | Launch the server (SIM via `DGX_MODE=sim`, REAL against local Ollama if up), curl every endpoint incl. streaming (`curl -N`), JS syntax check via node, sovereignty grep (no external `<script src>`/fetch), port-collision and content cross-checks — then fix and re-test until green. |
| **Responsibility** | End-to-end verification; kill all test servers; wire app 00 into `week23/README.md` as the recommended entry point; apply only trivially-safe doc fixes from the audit findings. |
| **Has done** | **All green.** SIM launch on :8112: every endpoint passed (stack integrity, 12 app cards, dead-URL probe < 1 s, 4 node runs streaming meta→tokens→done, unknown-node error line, run-lock, path-traversal blocked). REAL smoke on :8113 against local Ollama streamed 46 tokens incl. reasoning channel. **Found + fixed 1 real bug**: the frontend ignored `channel:"reasoning"` tokens, burying answers in thinking text — now rendered as dim `〔thinking〕` spans. Live-probed build.nvidia.com `/models` (102 models), which **validated the research agents' uncertain IDs** (`nemotron-3-nano-30b-a3b`, `-super-120b-a12b`, `-ultra-550b-a55b`, nemoguard) for free. Integrated "Phase 0 · The hub" into `week23/README.md` + Quick start; fixed the wrong port-busy hint in 8 apps and the `week23/reference/` note. Killed all servers, cleaned scratch files. |
| **Needs later** | REAL-mode test against the actual DGX Spark (`DGX_CONN=tunnel`) and a live nvapi- key run; real-browser visual pass. Skipped-by-design audit items: stale week19 docstrings (10–12 apps), 5 missing per-app READMEs, app-01 naming drift, TUTS cycle integration. NemoClaw GitHub URL still unverified. |

---

## Mission 2 — the hands-on tutorial track (workflow `week23-hands-on-tutorials`, 25 agents)

After the hub shipped, the user asked for real do-it-yourself tutorials, not just explainers.
A second workflow pipelined all 12 folders through **one author + one verifier each**, then a
**consistency editor** — grounded in Mission 1's `research/dgx_spark_runbook.md` and
`stack_content.json`.

| Agent role | What it did |
|---|---|
| **Tutorial authors ×12** | Wrote `TUTORIAL.md` per folder (uniform template: Path A 🖥️ DGX Spark / B ☁️ build.nvidia.com / C 💻 laptop; numbered steps with copy-paste commands + expected output + ✓ checkpoints; exercises with hidden solutions; troubleshooting incl. the 404/401/`/v1` pitfalls) and **36 lab scripts** (`labs/`, 3 per folder) that do real inference through each folder's `config.py` connection switch, degrade gracefully with no endpoint, and self-terminate < 60 s. Runbook `[UNCERTAIN]` items stay hedged ("verify with `GET /v1/models` first"). |
| **Lab verifiers ×12** | Actually executed every lab against live Ollama (real inference) AND endpoint-less; fixed real bugs — e.g. 01: wrong Super fit-verdict, reasoning eating token budgets; 07: bare `except` swallowing timeouts → false-positive guardrail blocks + reasoning leaking as answers; 08: stacked cold-load timeouts → added whole-lab deadline; verified OTLP export against a live Phoenix. |
| **Consistency editor ×1** | All 12 TUTORIALs exist, `## Next` chain 01→…→12→README verified, zero structure drift, all 36 labs cross-referenced with no orphans; added the hands-on track to `week23/README.md` (How-to-use + Quick start). |

Run notes: the first run hit the org spend limit at 22/25 agents; resumed with
`resumeFromRunId` — cached agents replayed free, only the 3 failed ones re-ran.
**Needs later:** labs were REAL-verified on gemma4 (a thinking model), not actual Nemotron —
field names for the reasoning channel may differ on real NIM endpoints (tutorials hedge this);
Ultra-on-2-Sparks tension between the app demos and the runbook fit-math (357 GB > 256 GB) was
resolved in favor of the runbook — the capstone app chapter still shows the 2-Spark framing.

---

## Mission 3 — the Lab Runner web app (workflow `week23-lab-runner`, 3 agents)

The hands-on track (Mission 2) was markdown-only; the user wanted to drive it step-by-step in
the browser like the past weeks' tutorial apps. A lean 3-agent workflow (research reused from
Mission 1) built **`00_lab_runner/` (port 8113, env `LAB_GUIDE_PORT`)**.

| Agent role | What it did |
|---|---|
| **Backend** | FastAPI server that parses all 12 `TUTORIAL.md` at startup (fence-aware H2 scanner; re-reads on mtime change so edits show on reload) into `/api/course` (141 sections, 36 labs), `/api/source` (strict allowlisted lab source), and `POST /api/run` — spawns the real `labs/*.py` with the house `__EXIT__`-sentinel streaming pattern, whitelisting only the 4 `DGX_*` env keys, 150 s cap, run-lock. |
| **Frontend** | Single-file `guide.html`: course-tree sidebar with per-folder progress rings, continuous prev/next across folder boundaries, bespoke markdown renderer (✓-checkpoint lines become persisted checkboxes; Expected-output blocks auto-dimmed; copy buttons), Path A/B/C switch that injects the matching `DGX_*` env into each run, streamed terminal per lab with house colorize. |
| **QA** | All contract endpoints tested live incl. a REAL lab run over Ollama (genuine inference, `__EXIT__ 0`), traversal attacks 404'd, env-whitelist and run-lock proven, JS node-checked + DOM-stub render test against real tutorial content; integrated port 8113 into `week23/README.md`. |

**Needs later:** real-browser visual pass (all rendering verified via node DOM-stub); `###`
headings render one level demoted (currently unexercised); no per-checkpoint sync across
browsers (localStorage only).

---

## Mission 4 — Lab Runner visualization upgrade (workflow `week23-lab-runner-viz`, 7 agents + MCP)

UX/UI pass on the Lab Runner driven by two MCP servers used in the main loop: **Mobbin** design
research (Codecademy lesson affordances; LangSmith/Modal dark metric-card chart grids; Cloudflare
sparklines) set the design contract, and **Higgsfield** generated the 21:9 header banner
(`media/lab-runner-hero.jpg`, dark circuit-stack, green token streams).

| Agent role | What it did |
|---|---|
| **Diagram authors ×4** | Wrote `diagrams.json` in all 12 folders (3 each): an architecture diagram (lanes/nodes/animated dataflow edges), a runtime sequence diagram, and 1–3 charts — every number grounded in the folder's own TUTORIAL/labs/runbook figures with sourced captions (e.g. 09's $0.2381 vs $1.2445 vs $1.80 per M-token; 01's NVFP4 fit-math; 07's measured red-team block rates). |
| **Backend** | `/api/course` gains `hero` + `diagrams` per folder and injects a synthetic "📊 Visualize it" section before Labs; `/media/` route (allowlisted, traversal-safe); diagrams re-read on mtime change. |
| **Frontend** | Generic inline-SVG toolkit, zero per-folder hardcoding: `drawArch` (lanes, topology-derived columns, SMIL glowing-dot dataflow pulses), `drawSeq` (lifelines, solid calls / dashed returns / note pills, staggered fade-in), `drawChart` (grow-in bars/hbars, draw-in lines, sweep-in donuts via IntersectionObserver) — all gated on `prefers-reduced-motion`; hero cards per tutorial intro + header banner. |
| **QA** | All 12 specs referentially validated (fixed one rounding slip in 06's fit-math values), synthetic-section positioning verified, mtime hot-reload proven live, render smoke via node DOM-stub on all 12, `/api/run` + `/api/shell` regression green. |

**Needs later:** diagrams are per-folder static specs — a future agent could generate charts live from actual lab-run output.

---

## Post-mission solo improvements (main-loop, not workflows)

Incremental Lab Runner upgrades made directly after the Spark came online:

1. **Per-run model picker** — `DGX_MODEL` added to the run/terminal env whitelist; a path-bar dropdown fed live from the Spark's `/v1/models` sends the chosen model with every ▶ Run and terminal command (persisted). Lets a lab run against `nemotron-3-super:120b` vs `nemotron-3-nano:4b` without leaving the page.
2. **Configurable lab timeout** — `RUN_TIMEOUT` now reads `LAB_RUN_TIMEOUT` (default 150s); `connect-remote.env` sets 420s for slower tunneled big-model runs. Status API + timed-out message report the active cap.
3. **Real-browser visual pass** (gstack `browse`) — screenshotted every view; fixed a real bug (Dynamo's architecture diagram collapsed to 27px because feedback edges inflated the canvas to 13k units — replaced with a topo-order layout that drops back-edges; all 12 now sane), a doubled 📊 sidebar icon, and a stale "local Ollama" path-C hint.
4. **Run-history sparklines** — server records each lab run (ts, model, exit code, seconds, parsed tok/s) to gitignored `.run_history.json` (last 50/lab) via `/api/history`; each lab card shows a 20-run sparkline (dot color = exit status, height = duration, hover = full detail) + a "✓ Ns · N tok/s" last-run label, refreshed after each run. Verified live against the Spark.
5. **Server-side checkpoint storage** — ticks persist to gitignored `progress.json` via `GET/POST /api/progress` (key-validated, lock-serialized write). Boot merges server ∪ localStorage (one-time migration of pre-existing local ticks up to the server), then writes through on every tick with localStorage kept as offline fallback. Verified: a tick survives a full localStorage wipe + reload (restored checked + green from the server), and browser ticks/unticks round-trip to the file both ways.

**Still open (from the improvement list):** terminal has no stdin (interactive commands hang to the cap); decide whether `connect-remote.env` is committed or gitignored (contains the tailnet hostname).

---

## How to re-run or extend the team

The workflow script is persisted by Claude Code under the session directory (see the Workflow
tool result for the exact path) and can be resumed/edited — completed agents return cached
results. Typical future missions:

- **Content refresh** — re-run only the two specialists (1–2) when NVIDIA updates the stack.
- **New chapter** — Stack Specialist drafts the node → Backend adds it to `stack_content.json` →
  Frontend needs no change (UI renders from `/api/stack`) → QA re-tests.
- **Real-hardware validation** — a future *DGX Spark field-test agent* running with the Spark
  reachable (`DGX_CONN=local|tunnel`) to confirm every command in the runbook.
