# Week 23 content audit — findings (report only, nothing fixed)

Audit date: 2026-08-03. Scope: `week23/README.md` table vs each app's actual code; missing
per-app READMEs; inconsistencies. All paths relative to `week23/`.

## 1. Port table vs reality — ✅ all 12 match

`README.md:32–67` claims ports 8100–8111. Verified against each `tutorial_server.py` default:

| App | env var | default | source |
|---|---|---|---|
| 01_nemotron_models | `NT_GUIDE_PORT` | 8100 | 01_nemotron_models/tutorial_server.py:44 |
| 02_nim_microservices | `NIM_GUIDE_PORT` | 8101 | 02_nim_microservices/tutorial_server.py:44 |
| 03_dynamo_serving | `DYNAMO_GUIDE_PORT` | 8102 | 03_dynamo_serving/tutorial_server.py:44 |
| 04_agent_skills | `SKILLS_GUIDE_PORT` | 8103 | 04_agent_skills/tutorial_server.py:44 |
| 05_aiq_research_lab | `AIQ_GUIDE_PORT` | 8104 | 05_aiq_research_lab/tutorial_server.py:48 |
| 06_nemoclaw | `CLAW_GUIDE_PORT` | 8105 | 06_nemoclaw/tutorial_server.py:44 |
| 07_guardrails_openshell | `GUARD_GUIDE_PORT` | 8106 | 07_guardrails_openshell/tutorial_server.py:44 |
| 08_nemo_relay | `RELAY_GUIDE_PORT` | 8107 | 08_nemo_relay/tutorial_server.py:44 |
| 09_inference_economics | `ECON_GUIDE_PORT` | 8108 | 09_inference_economics/tutorial_server.py:48 |
| 10_nemo_gym_rl | `GYM_GUIDE_PORT` | 8109 | 10_nemo_gym_rl/tutorial_server.py:44 |
| 11_data_flywheel | `FLYWHEEL_GUIDE_PORT` | 8110 | 11_data_flywheel/tutorial_server.py:44 |
| 12_capstone_smart_hotel | `HOTEL_GUIDE_PORT` | 8111 | 12_capstone_smart_hotel/tutorial_server.py:44 |

The `TUTS` navigator array (ports 8100–8111, 12 titles) is consistent across all 12
`static/guide.html` files (e.g. 01_nemotron_models/static/guide.html:669–675) and matches the
README table. The per-app `<PREFIX>_GUIDE_PORT` env vars are NOT documented in `README.md`
(minor doc gap).

## 2. Wrong env-var name in the "port busy" banner hint — 8 apps

The startup banner tells the user to "set DGX_GUIDE_PORT", but no app reads that variable —
each reads its own `<PREFIX>_GUIDE_PORT`:

- 01_nemotron_models/tutorial_server.py:340 — says `DGX_GUIDE_PORT`, actual is `NT_GUIDE_PORT` (line 44)
- 02_nim_microservices/tutorial_server.py:330 — actual `NIM_GUIDE_PORT`
- 03_dynamo_serving/tutorial_server.py:325 — actual `DYNAMO_GUIDE_PORT`
- 04_agent_skills/tutorial_server.py:337 — actual `SKILLS_GUIDE_PORT`
- 06_nemoclaw/tutorial_server.py:339 — actual `CLAW_GUIDE_PORT`
- 07_guardrails_openshell/tutorial_server.py:337 — actual `GUARD_GUIDE_PORT`
- 08_nemo_relay/tutorial_server.py:343 — actual `RELAY_GUIDE_PORT`
- 11_data_flywheel/tutorial_server.py:339 — actual `FLYWHEEL_GUIDE_PORT`

Correct in 05 (:344, AIQ_GUIDE_PORT), 09 (:343, ECON_GUIDE_PORT), 10 (:338, GYM_GUIDE_PORT),
12 (:349, HOTEL_GUIDE_PORT).

## 3. Stale "Week 19 / sovereign_dgx / port 8092" copy-paste leftovers

The apps were forked from `week19/sovereign_dgx` and 10 of 12 still carry its launch
instructions in the module docstring — the printed launch path and port are wrong for Week 23:

- `tutorial_server.py` lines 15–18 ("Launch (auto-picks a free port if 8092 is taken):
  `.venv/bin/python week19/sovereign_dgx/tutorial_server.py` → http://127.0.0.1:8092") in:
  01 (:15–18), 02 (:15–18), 03 (:15–18), 04 (:15–18), 06 (:15–18), 07 (:15–18), 08 (:15–18),
  10 (:15–18), 11 (:15–18), 12 (:15–18). Only 05 (:19–22, correct AIQ_GUIDE_PORT=8104 line)
  and 09 (:19–22, ECON_GUIDE_PORT=8108) were updated.
- Stale path comment `# …/week19/sovereign_dgx` on `PKG = …` in every `tutorial_server.py`
  (line 37; 05 and 09 at line 41) — all 12 apps.
- Every `config.py` docstring still opens with "Shared configuration for the **Week 19**
  Sovereign AI on DGX demos" (config.py:2 in all 12 apps) and has the stale
  `# …/week19/sovereign_dgx` comment at config.py:204 (all 12).
- Generic docstring title: 10 of 12 `tutorial_server.py` line 2 still read
  "Interactive, explainable tutorial for **Sovereign AI on a DGX**" instead of naming the app
  (only 05 "NVIDIA AI-Q Open Agent Blueprint" and 09 "AI Performance & Evaluation" are
  customized).
- `week23/.gitignore:1` comment says "Week 19 — generated / runtime artifacts".

## 4. Missing per-app README.md — 5 of 12 apps

Present: 01, 02, 03, 07, 10, 11, 12. Missing:

- `04_agent_skills/` — no README.md
- `05_aiq_research_lab/` — no README.md
- `06_nemoclaw/` — no README.md
- `08_nemo_relay/` — no README.md
- `09_inference_economics/` — no README.md

The root `README.md:22–24` says each folder is "a standalone interactive app" but does not
promise per-app READMEs; still, the inconsistency is visible when browsing.

## 5. Inconsistent README heading styles (where READMEs exist)

- 01_nemotron_models/README.md:1 — `# ▣ Week 23 · App 1 — Nemotron Open Models`
- 02_nim_microservices/README.md:1 — `# App 2 · NIM Microservices — …` (no ▣, no "Week 23")
- 03/07/10/11 follow 02's `# App N · …` style
- 12_capstone_smart_hotel/README.md:1 — `# ▣ Capstone — Sovereign Autonomous Hotel Operations`
  (no app number)

## 6. requirements.txt drift — app 01 differs from 02–12

- 02–12 (identical): `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `openai>=1.30`.
- 01_nemotron_models/requirements.txt:7–9: `openai>=1.40`, `fastapi>=0.110`, `uvicorn>=0.29`
  (no `[standard]` extra, higher openai floor, plus a 6-line comment header the others lack).

## 7. File-naming outlier in app 01

01_nemotron_models uses `ntsim.py` / `ntview.py` (import at
01_nemotron_models/tutorial_server.py:144 `import ntsim as dgxsim`), while apps 02–12 use
`sim.py` / `view.py` (e.g. 02_nim_microservices/tutorial_server.py:134 `import sim as dgxsim`).
02_nim_microservices/view.py:4–5 documents the convention: "Each Week 23 app ships this file
plus a `sim.py`" — app 01 predates/violates it. Purely cosmetic, but relevant if any tooling
globs for `view.py`/`sim.py`.

## 8. README references a folder that does not exist locally

`README.md:143` cites the source deck "photographed, in `week23/reference/`", but
`week23/reference/` is absent and explicitly gitignored (`week23/.gitignore:10` —
"reference only, not for distribution"). Intentional, but a fresh clone follows a dead
reference; a parenthetical "(not distributed)" would help.

## 9. Impact of adding 00_stack_navigator (for the team, not a defect)

- The prev/next navigator `TUTS` array is hard-coded in all 12 `static/guide.html` files
  (e.g. 01_nemotron_models/static/guide.html:669–675) and displays "app NN/12". If the hub
  app should appear in the cycle, all 12 files (plus the count label logic) need editing;
  if it runs on a port outside 8100–8111 the navigator in the hub's own guide will simply
  hide itself (guide.html:681 behavior).
- Root `README.md:22` states "The folders are numbered 01…12" and `README.md:3` says
  "Twelve interactive… web apps" — both sentences become stale once `00_stack_navigator`
  lands and should be revisited by whoever edits the README.

## 10. Verified-consistent (no action)

- FastAPI `title=` matches each guide's `<title>` and the README table naming in all 12 apps
  (e.g. 09: "AI Performance & Evaluation" in tutorial_server.py:128 and static/guide.html:6,
  matching README.md:56).
- `dgxsh.py` is byte-identical (same md5) across all 12 apps.
- Cross-reference numbering ("App 5", "App 7", phases) in README.md:32–69 matches folder
  numbering; the ASCII stack diagram (README.md:80–95) references the right app numbers.
