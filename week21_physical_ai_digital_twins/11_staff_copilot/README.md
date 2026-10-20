# App 11 · The staff copilot — a twin that boosts the humans

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app is the *human* half of
the AGENTS layer: the same twin the agent fleet controls (App 10) also makes every
technician, engineer and duty manager faster — an **LLM copilot** whose tools are the
twin's four stores (semantics graph · time-series historian · CMMS · O&M-doc RAG),
**work orders with spatial context**, **cuOpt** technician dispatch, and
**Metropolis/VSS** vision analytics feeding occupancy back to the energy model.

Runs **dual-mode**: Ch 2's final synthesis goes **REAL** against an Ollama / vLLM / NIM
endpoint (🔌 Connection panel), or **SIM** with no GPU (`sim.py` — a mini building twin,
pure stdlib). Ch 3–5 run on the built-in twin stores everywhere. Cloud cost $0.

## Run the web tutorial

```bash
cd week21/11_staff_copilot
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8210
```

## Run the chapters standalone

```bash
python demos/step01_copilot_tools.py    # 1 question → 4 tool calls → grounded answer
python demos/step02_work_orders.py      # NL complaint → spatial work order + batched PMs
python demos/step03_cuopt_dispatch.py   # 8 WOs × 3 techs: naive vs optimized routes
python demos/step04_metropolis_vss.py   # occupancy → queue/after-hours → energy what-if
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | Copilot over the twin | beginner |
| 2 | Work-order intelligence | intermediate |
| 3 | cuOpt dispatch | advanced |
| 4 | Eyes on the building — Metropolis & VSS | advanced |

## REAL-mode upgrades

- **Ch 2** — any OpenAI-compatible endpoint (`ollama serve`, or point 🔌 Connection at a DGX).
- **Ch 4** — the demo's greedy + 2-opt is an honest stand-in; the real solver is
  `pip install cuopt-server cuopt-sh-client` or the cuOpt NIM container (GPU required).

## Where this sits in Week 21

SCENE (Apps 2–4) → STATE (App 5, the stores this copilot queries) → SIMULATION
(Apps 6–8, receives the occupancy corrections) → AGENTS (App 10 controls · **App 11
(this) boosts the humans**) → capstone App 12. Ch 2 is the Week 15 GraphRAG pattern
grounded in a building.

## Sources

- [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) — GPU-accelerated VRP/TSP + LP/MILP,
  open-sourced Apache-2.0 at GTC March 2025, also served as a NIM microservice
- [Metropolis / VSS blueprint](https://www.nvidia.com/en-us/autonomous-machines/intelligent-video-analytics/) —
  DeepStream · TAO · Metropolis microservices · Video Search & Summarization
- Semantics: [Brick Schema](https://brickschema.org) · [RealEstateCore](https://www.realestatecore.io) · ASHRAE 223P
