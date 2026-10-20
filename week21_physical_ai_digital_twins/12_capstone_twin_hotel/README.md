# ◳ 12 · Capstone — AltoTech Grand Bangkok, the living twin

Week 21 · App 12 of 12 · Phase 6: **the CAPSTONE** · port **8211**

The Week 23 capstone gave the hotel an agent **fleet**; this capstone gives the fleet a
**body**. One running system combines every app of Week 21 on one 30-floor Bangkok hotel:

> **Digital Twin = SCENE (OpenUSD) + STATE (live data) + SIMULATION (physics) + AGENTS (control & copilot)**

```
        ┌─ AGENTS ────────────────────────────────────────────────┐
        │ learning operator (autonomy ladder · guardrails · G36    │  Apps 09–10
        │ fallback) · staff copilot (graph/TSDB/CMMS/docs · 2-opt   │  App 11
        │ dispatch) · flywheel (episodes → facts → skills)          │  W18/W20
        ├─ SIMULATION ────────────────────────────────────────────┤
        │ 5R1C zones · ASHRAE G14 calibration gate · millisecond    │  Apps 06–08
        │ surrogate with envelope guard · forecast-driven pre-cool  │
        ├─ STATE ─────────────────────────────────────────────────┤
        │ mini bus → binding service → SESSION-layer opinions ·     │  App 05
        │ graph + TSDB + scene joined on IFC GlobalId               │
        ├─ SCENE ─────────────────────────────────────────────────┤
        │ discipline layers · payload per floor (30) · 4,200 chairs │  Apps 02–04
        │ as 6 prototypes · COBie attrs · twin-readiness validator  │
        └──────────────────────────────────────────────────────────┘
          twin/  = world · scene · telemetry · energy · controls · flywheel · copilot · runtime
```

## Chapters

| Ch | Demo | Level | The story |
|----|------|-------|-----------|
| 1 | — (concept) | beginner | The map: four layers, one building, nothing screenshot-faked. |
| 2 | `step01_assemble_twin.py` | beginner | SCENE from BIM: layers, payloads, instancing, GlobalIds — then the twin-readiness validator (its 2 WARNs are missing *data*, not geometry). |
| 3 | `step02_go_live.py` | intermediate | STATE: telemetry → session-layer opinions; the canonical graph+TSDB+scene query; floor-5 heat-map; room **1203** raises its hand again (Week 23 lore). |
| 4 | `step03_whatif_console.py` | intermediate | SIMULATION: pass the G14 gate, sweep, fit a guarded surrogate, price tomorrow's pre-cool in baht and comfort. |
| 5 | `step04_learning_operator.py` | advanced | AGENTS-control: up the autonomy ladder; override → consolidated fact; frozen sensor → watchdog → Guideline 36 fallback; LLM postmortem (REAL if connected). |
| 6 | `step05_a_day_in_the_life.py` | advanced | The finale: 06:00 morning brief · 09:30 VIP pre-conditioning inside guardrails · 14:00 chiller alarm end-to-end (detect → diagnose → dispatch → verify → remember). |
| — | Appendix (concept) | all | The whole course in one building + which app taught each layer. |

## Quick start

```bash
uv pip install -r week21/12_capstone_twin_hotel/requirements.txt
.venv/bin/python week21/12_capstone_twin_hotel/tutorial_server.py   # → http://127.0.0.1:8211
```

Everything runs in **SIM** with no GPU, $0 — deterministic, so the numbers are reproducible.
An Ollama / vLLM / NIM / DGX endpoint (🔌 Connection panel or `DGX_BASE_URL`) makes the
agent-reasoning moments (morning brief, postmortem, copilot answers) genuinely model-written.

Run any chapter standalone: `.venv/bin/python week21/12_capstone_twin_hotel/demos/step05_a_day_in_the_life.py`.

## Take it to your building — the variant briefs

| Variant | What changes first |
|---|---|
| **Hospital** | Redundancy + IAQ + compliance dominate: N+1 plant in the graph, air-changes/pressure cascades as watchdogs, every action auditable. |
| **Office** | Tenant comfort & leases: comfort SLAs per tenancy in the reward, after-hours billing from the schedule twin. |
| **Factory** | The Mega blueprint's home turf: robot fleets join the twin; production takt, not comfort, is the KPI. |
| **Smart-city district** | CityLearn-style demand response across buildings: storage dispatch, district peaks, grid signals. |

## Continuity

The graphs are **Week 14/15**, the memory loop is **Week 18**, the fleet and the room-1203
incident are **Week 23** ([`week23/12_capstone_smart_hotel/`](../../week23/12_capstone_smart_hotel/)),
and every mechanism here was taught step-by-step in **Apps 01–11** of this week.

## Sources

- [Assembling Digital Twins With Omniverse and OpenUSD](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html) (the NVIDIA learning path Week 21 transposes)
- [openusd.org](https://openusd.org) · [Brick Schema](https://brickschema.org) · [EnergyPlus](https://energyplus.net) · ASHRAE Guidelines 14 & 36
- [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) · [cuOpt](https://github.com/NVIDIA/cuopt) · [Sinergym](https://github.com/ugr-sail/sinergym) · [BOPTEST](https://ibpsa.github.io/project1-boptest/)
