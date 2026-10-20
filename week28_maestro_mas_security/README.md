# Week 28 · Securing multi-agent systems with MAESTRO

Threat-model, attack and defend a **multi-agent system** with the Cloud Security Alliance's **MAESTRO** framework.
You have built multi-agent systems since Week 9 (CrewAI and LangGraph crews, the smart-hotel crew with shared Neo4j
memory, Google ADK and A2A, the AltoTech Grand capstone, the NAT supervisor). This week asks what those labs never
did: *whose words is this agent acting on?* Every lab drives one small, deliberately weak system, the
**Hotel Ops Ensemble**: an orchestrator, three specialist agents and an outside vendor agent, running a simulated
hotel. Each attack is short and toy-like, and each one exists to motivate a control: why it works, where it stops
working, and what it costs in utility. Ten modules, each with labs, an exercise and a Thai translation, served by the
**🎼 Maestro Lab Runner**.

**Why MAESTRO.** Classic threat models (STRIDE and friends) ask what one component can do. A multi-agent system
fails *between* components: an instruction travels from a guest to the chiller through three agents, a poisoned
memory note is obeyed in another guest's session, a look-alike Agent Card receives the orders. MAESTRO splits an
agentic system into seven layers, lists threats per layer **and across layers**, and gives a six-step method you
can turn into code and a CI gate.

## Start

```bash
# every time (from the repo root)
.venv/bin/python week28_maestro_mas_security/00_maestro_lab_runner/tutorial_server.py      # → http://127.0.0.1:8128
```

The runner picks a free port if 8128 is taken. In its header:

- **🧠 SIM / 🦙 LIVE** — the agents' brain. **SIM** (default) is a deterministic, offline, instant stand-in for a
  gullible LLM. **LIVE** uses a real local model on Ollama (`gemma4:12b` by default). The browser can set only
  `MAS_BRAIN` and `MAS_MODEL`.
- **🗺 MAESTRO map** — the seven layers, their threats, the cross-layer threats, the mitigations, the six steps and
  the OWASP agentic list, parsed from the saved CSA page.

**Prerequisites.** The repo's `.venv` (Python 3.13) with `cryptography` and `pyyaml`, both already installed.
Optional: [Ollama](https://ollama.com) with `gemma4:12b` pulled, for the LIVE brain. Every lab runs without it.
Labs cap LIVE at 6 model calls each; `nemotron-3-nano` is about 150 s per call on an M4 Max, so it is not for labs.

## Modules

| # | module | MAESTRO focus | OWASP (ASI) | you build |
|---|---|---|---|---|
| 01 | [Why multi-agent systems need their own threat model](01_why_mas_security/TUTORIAL.md) | all | ASI01, ASI08 | meet the Hotel Ops Ensemble; watch one request cross trust boundaries; find the boundaries |
| 02 | [MAESTRO: seven layers, cross-layer threats, six steps](02_maestro_seven_layers/TUTORIAL.md) | L1–L7 | map | decompose the ensemble into layers; map OWASP ASI01–10 onto layers; the 8 patterns |
| 03 | [Threat-model the ensemble: decompose, score, prioritise, gate](03_threat_model_as_code/TUTORIAL.md) | method | — | a YAML threat model, a risk heatmap, a markdown report, threat diffing, a CI gate |
| 04 | [Goal hijack: injections that travel between agents](04_goal_hijack_injection/TUTORIAL.md) | L1 · L7 | ASI01 | direct vs indirect injection; spotlighting (static vs nonce); structured hand-offs |
| 05 | [Tool misuse, confused deputies and least agency](05_tool_misuse_privilege/TUTORIAL.md) | L3 · L6 | ASI02, ASI03, ASI09 | least agency, parameter bounds, provenance-based authorisation, HITL that works |
| 06 | [Inter-agent trust: Agent Cards, signatures and replay](06_inter_agent_trust_a2a/TUTORIAL.md) | L3 · L7 | ASI07, ASI04 | signed + pinned Agent Cards; signed, fresh, single-use A2A messages |
| 07 | [Memory and RAG poisoning, and the cascades they cause](07_memory_rag_poisoning/TUTORIAL.md) | L2 | ASI06, ASI08 | memory write gates with provenance; a signed SOP manifest; tracing a cascade |
| 08 | [Detect, contain, prove: audit chains, budgets, kill switches](08_observability_response/TUTORIAL.md) | L4 · L5 · L6 | ASI08, ASI10 | a hash-chained audit log + witness; hop/action budgets; "a prompt is a claim, not a control" |
| 09 | [Red-team the ensemble: security vs utility](09_red_team_scorecard/TUTORIAL.md) | all | all | the full scenario suite, a defense × attack matrix, minimal defense sets, LIVE spot checks |
| 10 | [Capstone: harden the Hotel Ops Ensemble and ship the MAESTRO report](10_capstone_secure_ensemble/TUTORIAL.md) | all | all | a defense set within a budget, the threat model updated, a CI gate, a report for the GM |

## The Hotel Ops Ensemble

```text
guest / staff ──► orchestrator ──► concierge     (guest info, email, SOP search, web pages, memory)
                               ├─► energy        (room setpoints, the chiller — the dangerous tools)
                               └─► maintenance   (work orders, SOPs, parts orders over A2A)
                                                     └─► vendor agent (external, found by its Agent Card)
```

Every control is a switch on `Defenses`, all off by default: `spotlight` · `forward_summary` · `tool_allowlist` ·
`param_bounds` · `principal_authz` · `hitl` · `egress_allowlist` · `signed_cards` · `signed_messages` ·
`memory_gate` · `sop_integrity` · `audit_chain` · `circuit_breaker`. The red-team suite has 12 attacks (A01–A12) and
8 benign tasks (B01–B08); a defense is only good if it stops attacks **without** breaking the hotel.

## What is where

```text
week28_maestro_mas_security/
  00_maestro_lab_runner/   tutorial_server.py (FastAPI, :8128) + static/guide.html — course, ▶ labs, ⌨ terminal,
                           🧠 SIM / 🦙 LIVE switch, 🗺 MAESTRO map (/api/maestro)
  common/maskit.py         output helpers + the two brains (SIM stand-in, Ollama LIVE)
  common/ensemble.py       the Hotel Ops Ensemble: world, agents, tools, PEP, A2A, audit log, Defenses
  common/redteam.py        the scenario suite (A01–A12 attacks, B01–B08 benign), run_suite, scorecard
  common/maestro.py        MAESTRO as data (parsed from the CSA page) + threat model as code: risk, heatmap,
                           validate, report_md, diff, gate
  common/sources/          text snapshots of every cited page (what the reference audit checks against)
  common/audit_references.py · check_translation.py   maintainer checks
  NN_*/                    TUTORIAL.md (+ .th.md), labs/, exercises/ (+ solutions/), diagrams.json, data/
```

## Sources

Saved as text snapshots in `common/sources/`; tutorials quote only from these.

- Ken Huang, *Agentic AI Threat Modeling Framework: MAESTRO*, CSA, 6 Feb 2025 — https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro
- Ken Huang & Idan Habler, *Threat Modeling Google's A2A Protocol with the MAESTRO Framework*, CSA, 30 Apr 2025 — https://cloudsecurityalliance.org/blog/2025/04/30/threat-modeling-google-s-a2a-protocol-with-the-maestro-framework
- CSA, *Threat Modeling OpenAI's Responses API with the MAESTRO Framework*, 24 Mar 2025 — https://cloudsecurityalliance.org/blog/2025/03/24/threat-modeling-openai-s-responses-api-with-the-maestro-framework
- Steven Leath & Ken Huang, *Applying MAESTRO to Real-World Agentic AI Threat Models: From Framework to CI/CD Pipeline*, CSA, 11 Feb 2026 — https://cloudsecurityalliance.org/blog/2026/02/11/applying-maestro-to-real-world-agentic-ai-threat-models-from-framework-to-ci-cd-pipeline
- Ken Huang, *MAESTRO Analysis of OpenAI and Anthropic Agent Hacking Incidents*, CSA, 13 Aug 2026 — https://cloudsecurityalliance.org/articles/maestro-analysis-of-openai-and-anthropic-agent-hacking-incidents
- OWASP, *Top 10 for Agentic Applications for 2026* — https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
  (summaries: Giskard — https://www.giskard.ai/knowledge/owasp-top-10-for-agentic-application-2026 · Modulos — https://docs.modulos.ai/frameworks/owasp-top-10-agentic)

## Honesty rules (short version)

- **SIM is a teaching model.** It obeys every instruction it reads, whoever wrote it: the weakness prompt
  injection exploits, made deterministic. SIM numbers are never a measurement of a real model's injection rate.
- **Everything is simulated.** A fake hotel, fictional guests, `.example` domains, no network, no credentials.
  Attack text stays short and toy-like.
- **Labelled output.** `(captured on this Mac, SIM brain)` and `(captured on this Mac, LIVE brain gemma4:12b)` are
  real runs; LIVE output can differ on a rerun. `(REFERENCE — quoted from …)` is verbatim from a saved source.
  `(EXAMPLE — …)` is an illustrative shape.
- **What is the course's own.** MAESTRO's layer, threat and mitigation names come verbatim from the CSA page. The
  OWASP→layer mapping, the scenario tags, the 1–5 risk matrix and its bands, and the capstone's price list and
  budget are the course's design, and are labelled so wherever they appear.

## Maintainer checks

```bash
# on: laptop
.venv/bin/python week28_maestro_mas_security/common/audit_references.py          # every quote verbatim in common/sources/ → 0 not found
.venv/bin/python week28_maestro_mas_security/common/check_translation.py         # every TUTORIAL.th.md matches its TUTORIAL.md → all ✓
.venv/bin/python week28_maestro_mas_security/common/redteam.py                   # naive 12/12 harm · every defense on 0/12, 8/8
```

Adding or changing a module: read [AUTHORING.md](AUTHORING.md) first.
