# ▣ Week 23 — Slide-Source Pack (throw this at ChatGPT to generate styled decks)

<img src="media/hero-banner.jpg" alt="The Open Superintelligence Stack — Agent = Model + Harness" width="100%">

This file is a **slide-source pack** for the 12 Week 23 tutorials. It is designed to be pasted
into ChatGPT (or any LLM) to generate one styled slide deck per tutorial.

## How to use

1. **Per deck (recommended):** paste §1 THE PROMPT + §2 DESIGN SYSTEM + §3 DECK TEMPLATE +
   *one* deck brief from §4 into a fresh chat. One deck per chat keeps quality high.
2. **Pick an output format** (edit the first line of THE PROMPT):
   - `reveal.js single-file HTML` — **default.** Closest match to the tutorials' dark NVIDIA
     look, animated, opens in any browser, no dependencies.
   - `PPTX via python-pptx` — when you need to edit in PowerPoint/Keynote/Google Slides
     (ask ChatGPT to write + run the script and give you the .pptx).
   - `Marp Markdown` — when you want the deck itself to live in git as text.
3. Deck briefs give **content and numbers**; the design system gives **style**. Tell ChatGPT
   both are binding. Anything not in the brief must not be invented — that's rule #1 below.

---

## §1 · THE PROMPT (paste first, edit the [bracketed] bits)

> Create a **[reveal.js single-file HTML]** slide deck from the DECK BRIEF below, following
> the DESIGN SYSTEM and SLIDE TEMPLATE exactly.
> Rules:
> 1. **Do not invent facts, numbers, product names, or benchmarks.** Use only what the brief
>    contains. If a slide feels thin, tighten it — don't pad it.
> 2. Keep the brief's honest/caveat notes — they go on the slide, styled as `honest-note`,
>    never deleted or softened.
> 3. Every slide: one idea, ≤ 5 bullets, ≤ 12 words per bullet. Put longer prose in
>    speaker notes.
> 4. Diagrams: rebuild the ASCII sketches as simple styled boxes/arrows (HTML/CSS or SVG) in
>    the deck's own theme — no external images, no stock art, no emoji soup.
> 5. Add speaker notes to every slide (2–4 sentences, conversational, from the brief's
>    "notes" lines).
> 6. Output one complete, self-contained file I can save and open.

---

## §2 · DESIGN SYSTEM (the tutorials' own look — keep decks consistent with the apps)

**Theme: "sovereign terminal" — dark, technical, NVIDIA-green.**

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0c0f0a` | slide background |
| `--panel` | `#13160f` | cards, diagram boxes |
| `--line` | `#2c3322` | borders, dividers |
| `--txt` | `#e8efe0` | body text |
| `--dim` | `#8f9a82` | captions, speaker cues |
| `--accent` | `#76b900` | NVIDIA green — titles, key numbers, highlights |
| `--cyan` | `#56d4dd` | secondary accent — dataflow, "laptop/client" |
| `--amber` | `#d29922` | warnings, costs, honest-notes |
| `--red` | `#f85149` | dangers, failure modes, "don't" |
| `--violet` | `#bc8cff` | advanced/expert tags |

- **Type:** system sans (SF/Segoe/Roboto); titles 650 weight; `ui-monospace` for code,
  commands, model names, numbers. Generous line-height (1.5).
- **Slide furniture:** top-left kicker `WEEK 23 · APP NN/12 · <PHASE>`; bottom-right page
  number; a 3px `--accent`→`--cyan` gradient bar under every slide title.
- **Level chips** on chapter slides: BEGINNER (green tint) · INTERMEDIATE (cyan tint) ·
  ADVANCED (violet tint) — small uppercase pills.
- **`honest-note` component:** amber left-border card, prefix "⚠ honestly:". One per deck
  minimum — this course's voice is *numbers over adjectives, caveats kept*.
- **Diagram style:** rounded `--panel` boxes with 1px `--line` borders; labeled arrows;
  green = the component being taught, cyan = data in motion, red = the guard/gate.
- **Voice:** direct, plain English, second person, no marketing superlatives. "$0 · on your
  DGX" framing wherever inference cost appears.

---

## §3 · SLIDE TEMPLATE (12–14 slides per deck)

1. **Title** — app name, one-line promise, kicker, port (`http://127.0.0.1:PORT`).
2. **Where this sits** — the Week 23 stack diagram with THIS app's layer lit green
   (stack: MODEL → RUNTIME → HARNESS → FLYWHEEL; see each brief's "layer").
3. **The big idea** — the one sentence the audience must remember, full-slide size.
4–9. **One slide per chapter** (use the brief's chapter bullets + diagram + numbers; chip =
   the chapter's level; title = the chapter title verbatim).
10. **The numbers** — the brief's key metrics as a bar/table visual.
11. **⚠ Honestly** — the brief's caveats, amber-styled.
12. **Takeaway + what's next** — takeaway line verbatim + "next: App NN".
13. *(optional)* **Try it** — the run command + what to click first.

---

## §4 · DECK BRIEFS (one per tutorial — paste ONE per chat)

Common context for every brief: Week 23 = "The Open Superintelligence Stack (NVIDIA)".
Through-line: **Agent = Model + Harness** — the model reasons; the harness (context,
orchestration, tools, memory, security, observability, self-improvement) makes it an agent.
Every app is dual-mode (REAL endpoint or faithful SIM), always $0 cloud cost, launched as
`.venv/bin/python week23/<folder>/tutorial_server.py`.

**Optional title-slide cover art** — every deck has a matching card in [`media/`](media/)
(`media/01-nemotron-models.jpg` … `media/12-capstone-smart-hotel.jpg`). Don't paste these into
the LLM chat (rule #4 above bans external images *inside* the generated slides); if your output
format supports it, drop the card in manually as the title-slide background afterwards. Samples:

<p>
  <img src="media/01-nemotron-models.jpg" alt="Deck 01 cover — Nemotron 3 model family" width="240">
  <img src="media/07-guardrails-openshell.jpg" alt="Deck 07 cover — OpenShell guardrails" width="240">
  <img src="media/12-capstone-smart-hotel.jpg" alt="Deck 12 cover — sovereign smart hotel capstone" width="240">
</p>

---

### DECK 01 · Nemotron Open Models — the MODEL layer
- port 8100 · phase "The Model" · layer MODEL · prereq none · next App 02
- Story: everything else this week is harness *around* this — the open model family.
- Ch 1 · Agent = Model + Harness [beginner]: the week's equation; model reasons, harness acts.
- Ch 2 · The Nemotron 3 family — pick by task [beginner]: Nano 30B-A3B (edge/routing) ·
  Super 120B-A12B (workhorse) · Ultra 550B-A55B (frontier, 2 Sparks) + RAG/Speech/Safety
  variants; open weights, RL-post-trained.
- Ch 3 · Mamba-Transformer MoE + 1M context [intermediate]: hybrid architecture — why MoE
  activates a fraction of params/token; 1M-token context for long-running agents.
- Ch 4 · Reasoning (RLM) — think, then answer [intermediate]: reasoning mode on/off per call;
  when to pay the thinking tokens.
- Ch 5 · Tool-calling — a sovereign sub-agent [intermediate]: native function calling; the
  demo drives a real tool loop on-box.
- Ch 6 · Run it — 1 Spark & 2 Sparks [advanced]: DGX Spark (GB10, 128 GB unified, ~200B params
  quantized); two Sparks over QSFP 200GbE → ~405B; Station GB300 (784 GB) → 670B-class.
- Numbers slide: the family table (params, active params, target hardware).
- Honest note: "open weights ≠ open data/recipe; cloud Nemotron endpoints are usage-billed —
  the $0 claim is only true on your own hardware."
- Takeaway: "Pick the smallest Nemotron that passes your eval — the harness does the rest."

---

### DECK 02 · NIM Microservices — serve it sovereignly
- port 8101 · phase "Serve" · layer RUNTIME · prereq App 01 · next App 03
- Story: one signed container = model + optimized engine + OpenAI API. Deployment, solved.
- Ch 2 · Deploy a NIM (one command) [beginner]: `docker run` → serving; signed, versioned.
- Ch 3 · NIM vs raw vLLM vs Ollama [intermediate]: the trade table — convenience/support vs
  control/simplicity; all three speak OpenAI API.
- Ch 4 · Call a NIM — same OpenAI API [beginner]: swap `base_url`, nothing else changes
  (the Week 18/19 lesson, now with NIM).
- Ch 5 · The catalog + your own custom NIM [advanced]: build.nvidia.com catalog; packaging a
  fine-tuned model as your own NIM.
- Numbers: deploy-time comparison (one command vs assemble-it-yourself stack).
- Honest note: "NIM adds licensing (NVIDIA AI Enterprise) in production — raw vLLM/Ollama
  stay free; you're buying the signed, supported path, not capability you can't self-build."
- Takeaway: "A NIM is the deployment unit of the stack: pull, run, call — same API everywhere."

---

### DECK 03 · Dynamo — serve at scale
- port 8102 · phase "Serve" · layer RUNTIME · prereq App 02 · next App 04
- Story: one NIM serves a model; Dynamo serves an *always-on agent fleet* economically.
- Ch 2 · What is Dynamo? [beginner]: four pieces — disaggregated prefill/decode, KV-cache-aware
  routing, SLO Planner, NIXL KV transfer.
- Ch 3 · Disaggregated + cache-aware [intermediate]: prefill is compute-bound, decode is
  memory-bound → separate right-sized pools; route to the worker that already cached the
  prefix. Diagram: request → router → (cold: prefill pool → decode pool / warm: decode pool).
- Ch 4 · SLO Planner — hold latency [advanced]: declare TTFT/ITL targets; pools autoscale
  under a load ramp; naive worker's TTFT balloons, Dynamo's stays flat.
- Ch 5 · Token economics [advanced]: stacking wins — ~1.8× (cache routing) → ~3.1×
  (+disaggregation) → ~4.4× throughput (+planner) at ~0.24× cost/token vs naive (sim figures).
- Numbers: that 1.0→4.4× / 1.0→0.24× stacked-bars slide.
- Honest note: "sim-derived illustrative numbers — real gains depend on prefix-sharing rate
  and workload shape; a single-model, low-QPS shop doesn't need Dynamo."
- Takeaway: "Disaggregate + cache-route + autoscale ≈ 3× throughput at ⅓ the cost/token —
  the difference between an always-on agent you can and can't afford."

---

### DECK 04 · Agent Skills — connect frontier agents to your business
- port 8103 · phase "Build" · layer HARNESS · prereq App 01 · next App 05
- Story: portable, framework-agnostic capabilities — write once, load into Claude/GPT/Nemotron.
- Ch 2 · The skills catalog [beginner]: `github.com/NVIDIA/skills` — AI-Q research, NeMo
  Retriever, RL & Gym, Evaluator, Curator, cuOpt, cuDF, VSS video, Voice…
- Ch 3 · Load a skill into a frontier agent [intermediate]: the SKILL.md pattern (Week 7!);
  the same skill file powering different frontier models.
- Ch 4 · Connect to your business [intermediate]: skills wrap YOUR systems (telemetry, CMMS,
  docs) — the demo wires a hotel's tools in.
- Ch 5 · Skills + MCP + A2A [advanced]: the interop triangle — skills package know-how, MCP
  serves tools, A2A connects agents (Week 17 callback).
- Honest note: "young catalog — expect gaps; the durable idea is the packaging pattern, not
  any single skill."
- Takeaway: "Skills are the reuse layer: your business connects once, every agent benefits."

---

### DECK 05 · AI-Q Research Lab — orchestrate deep work
- port 8104 · phase "Build" · layer HARNESS · prereq App 04 · next App 06
- Story: "a research lab for any domain" — router, planner, and researcher sub-agents.
- Ch 2 · Intent Router — route or escalate [beginner]: Nano triages; cheap requests stay
  cheap; hard ones escalate to Super.
- Ch 3 · Deep Agent — orchestrate & plan [intermediate]: plan → delegate → synthesize.
- Ch 4 · Researcher sub-agents fan out [intermediate]: parallel workers, each with a slice.
- Ch 5 · Tools & data via NeMo Agent Toolkit [advanced]: NAT (Week 16) supplies the tool
  registry & builder.
- Numbers: the appendix claim set — accuracy/customization/observability + ~50% cost via
  routing (label as NVIDIA's blueprint claims).
- Honest note: "blueprint ≠ product — you assemble and eval it; the 50% figure is workload-
  dependent routing math, not a guarantee."
- Takeaway: "Route small, escalate rarely, fan out in parallel — that's how deep work gets
  cheap."

---

### DECK 06 · NemoClaw — build specialized agents
- port 8105 · phase "Build" · layer HARNESS · prereq Apps 04+05 · next App 07
- Story: specialist = base model + persona + skills + tools + signed policy. Then a fleet.
- Ch 2 · Define a specialized agent [beginner]: the NemoClaw agent spec.
- Ch 3 · Equip it — attach skills & tools [intermediate]: composition over fine-tuning first.
- Ch 4 · Run it safely in OpenShell [intermediate]: every action through the signed-policy
  gate (preview of App 07).
- Ch 5 · A fleet of specialists [advanced]: Energy · Maintenance · Guest agents cooperating
  (the capstone's cast, introduced).
- Honest note: "specialization by composition beats fine-tuning until you have eval-proven
  gaps — weights are the last resort (App 11 covers when)."
- Takeaway: "Ship specialists, not one do-everything agent — and give each a signed policy."

---

### DECK 07 · Guardrails + OpenShell — run it safely
- port 8106 · phase "Safety" · layer HARNESS (the gate) · prereq App 06 · next App 08
- Story: a long-running autonomous agent needs rails, a sandbox, and an egress allowlist.
- Ch 2 · Why guard a long-running agent? [beginner]: threat model — prompt injection, tool
  misuse, data exfiltration, runaway loops.
- Ch 3 · NeMo Guardrails — author & test rails [intermediate]: input/output/dialog rails as
  code; test them like unit tests.
- Ch 4 · OpenShell — sandbox + allowlist + signed policy [advanced]: actions execute in a
  sandbox; egress allowlisted; policies signed so agents can't self-modify them.
- Ch 5 · Privacy router — keep data sovereign [advanced]: route by sensitivity — PII stays
  on-box, generic goes wherever is cheapest.
- Honest note: "rails reduce, not eliminate, risk — pair with the autonomy ladder and a human
  kill switch; guardrails you never test are decoration."
- Takeaway: "Autonomy is earned: sandbox + allowlist + signed policy before the first real
  action."

---

### DECK 08 · NeMo Relay — observe, learn, optimize
- port 8107 · phase "Observe" · layer HARNESS (telemetry) · prereq App 05 · next App 09
- Story: you can't improve what you can't see — every call traced, then right-sized.
- Ch 2 · Observe — capture every tool & LLM call [beginner]: spans for every step.
- Ch 3 · Agent Insights with Phoenix — read the trace [intermediate]: span tree, latency,
  cost per turn; find the slow/expensive step in seconds.
- Ch 4 · Optimize — router right-sizes the model [intermediate]: send cheap requests to Nano,
  hard ones to Super — cost drops at equal outcome.
- Ch 5 · Learn — export telemetry & close the loop [advanced]: OTel export (Phoenix/Datadog/
  LangSmith); traces become App 11's training data.
- Honest note: "tracing adds overhead and stores prompts — scrub PII before export; the
  router needs an eval to prove 'equal outcome', not vibes."
- Takeaway: "Trace everything, route by difficulty — observability is where self-improvement
  starts."

---

### DECK 09 · Inference Economics — tokens, goodput, megawatts
- port 8108 · phase "Measure" · layer FLYWHEEL (the meter) · prereq Apps 03+08 · next App 10
- Story: the unit economics that decide whether an always-on agent is viable.
- Ch 1 · Tokens — the unit of AI work [beginner]: every agent action is tokens in/out.
- Ch 2 · Cost per million tokens [beginner]: compute it for your own box vs cloud pricing.
- Ch 3 · Throughput — per GPU and per Megawatt [intermediate]: tokens/s/GPU and tokens/s/MW —
  the datacenter-scale lens NVIDIA prices in.
- Ch 4 · From tokens to goodput [advanced]: cost per *successful task* — retries, judge fails
  and dead-ends make cheap tokens expensive.
- Ch 5 · Evaluate — score task success [advanced]: LLM-judge + golden set (Week 10/15
  callback) turn "goodput" into a measurable number.
- Honest note: "$/M-token comparisons across providers hide context, caching and quality —
  goodput (cost per successful task) is the only honest metric."
- Takeaway: "Optimize cost per successful task, not cost per token."

---

### DECK 10 · NeMo Gym + RL — verifiable rewards
- port 8109 · phase "Improve" · layer FLYWHEEL · prereq App 09 · next App 11
- Story: how Nemotron itself was forged — RL against rewards you can verify, not vibes.
- Ch 2 · What is verifiable-reward RL? [beginner]: reward = objective check (tests pass,
  answer matches, constraint holds) — no reward model to game.
- Ch 3 · Define an environment [intermediate]: task + verifier = a Gym env for agents.
- Ch 4 · GRPO training loop on the DGX [advanced]: sample groups, score, update — the
  post-training loop, on-box.
- Ch 5 · Multi-environment RL + evaluate [advanced]: train across envs; eval on held-out
  tasks to catch overfitting to one verifier.
- Honest note: "verifiable rewards only cover verifiable tasks — subjective quality still
  needs judges/humans; reward hacking shifts to verifier bugs."
- Takeaway: "If you can verify it, you can train on it — RL turns checks into capability."

---

### DECK 11 · Data Flywheel — production logs → better, cheaper model
- port 8110 · phase "Improve" · layer FLYWHEEL · prereq App 10 · next App 12
- Story: the loop that compounds — yesterday's operations become tomorrow's cheaper model.
- Ch 2 · The flywheel loop [beginner]: logs → Curator → Customizer → Evaluator → promote →
  repeat.
- Ch 3 · Curate — logs into training data [intermediate]: dedupe, filter, format the traces
  App 08 exported.
- Ch 4 · Customize — distill teacher→student [advanced]: LoRA/SFT/DPO/GRPO — distill Super's
  behavior into Nano for your domain.
- Ch 5 · Evaluate + promote [advanced]: LLM-judge + golden set gate the swap; only promote on
  proven parity (the Week 15 CI-gate pattern).
- Honest note: "flywheels amplify whatever you feed them — curate garbage, distill garbage;
  the Evaluator gate is the whole game."
- Takeaway: "Log → curate → distill → gate → promote: the agent that runs also trains its
  cheaper successor."

---

### DECK 12 · Capstone — the Sovereign Autonomous Hotel
- port 8111 · phase "Capstone" · layer ALL · prereq all · next Week 21 (the twin gets a body)
- Story: AltoTech Grand Bangkok run by a self-improving agent fleet — every app, one system.
- Ch 2 · Morning ops brief — AI-Q Deep Agent [beginner]: router + planner fan specialists
  across the hotel (Apps 05/06).
- Ch 3 · CRITICAL alarm, room 1203 — safe triage [intermediate]: telemetry → SOP RAG →
  policy-gated CRITICAL work order (Apps 04/07).
- Ch 4 · VIP request — guardrails stop unsafe action [intermediate]: signed policy DENIES an
  autonomous setpoint change on a VIP room → routed to a human (App 07).
- Ch 5 · Observe & optimize [advanced]: Relay spans + Phoenix + router economics on the
  morning's work (Apps 08/09).
- Ch 6 · Self-improve [advanced]: verifiable rewards score the morning; clean traces feed the
  flywheel; distill toward Nano (Apps 10/11).
- Numbers: the fleet's morning scorecard (actions taken, policy denies, cost per task).
- Honest note: "the hotel is simulated — the harness is real; swapping SIM for a live BMS is
  an integration project, not a slide."
- Takeaway: "Agent = Model + Harness, proven end-to-end — and Week 21 gives this fleet a
  digital-twin body."

---

*Generated from the Week 23 tutorial apps (chapter titles verbatim from each
`tutorial_server.py`; numbers from each app's sim/README). If a tutorial changes, regenerate
the affected deck brief.*
