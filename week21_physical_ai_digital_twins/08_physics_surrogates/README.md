# App 8 · Physics-ML surrogates — millisecond what-if

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app finishes the
**SIMULATION** layer: a twin viewport needs answers in **milliseconds** while a
facilities manager drags a slider, but CFD takes minutes–hours and an annual
EnergyPlus run takes minutes. The pattern — **SOLVER for truth, SURROGATE for
interaction** — is shipping today in data-center twins (Cadence Reality DC, the
Omniverse AI-factory Blueprint with Schneider Electric / ETAP / Vertiv). Here you
train a tiny surrogate **live** (pure Python gradient descent, no numpy), benchmark
it, break it outside its training envelope, and install the fallback-to-physics guard.

Runs **dual-mode**: only Ch 5 makes an LLM call — **REAL** against an Ollama / DGX /
cloud endpoint, else a faithful **SIM** (`sim.py`). Ch 2–4 are pure stdlib, offline, $0.

## Run the web tutorial

```bash
cd week21/08_physics_surrogates
pip install -r requirements.txt
python tutorial_server.py            # → http://127.0.0.1:8207  (env: TWIN_GUIDE_PORT)
```

## Run the chapters standalone

```bash
python demos/step01_why_surrogates.py     # the latency ladder + the data-center proof
python demos/step02_physicsnemo.py        # PhysicsNeMo (ex-Modulus): the surrogate factory
python demos/step03_train_surrogate.py    # train live, benchmark, then BREAK it + the guard
python demos/step04_earth2_forecast.py    # Earth-2 forecast in → pre-cool decision table
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | Why surrogates — the latency ladder | beginner |
| 2 | PhysicsNeMo — the surrogate factory | intermediate |
| 3 | Train a surrogate live — then break it | advanced |
| 4 | Earth-2 forecasts — pre-cool before the spike | advanced |

## REAL-mode upgrades

- **Ch 5 LLM call**: run `ollama serve` (or point 🔌 Connection / `DGX_BASE_URL` at any
  OpenAI-compatible endpoint) — the recommendation is generated for real, $0.
- **Ch 3**: `pip install nvidia-physicsnemo` flips its MODE line to REAL (the taught
  workflow is identical; actual FNO training wants a CUDA GPU).

## The honest trade

A surrogate is a **cache of physics — and caches go stale**: it is only trustworthy
inside its training envelope, it extrapolates confidently past real limits (like a
chiller's capacity), and it must be retrained when the building changes. Know the
envelope; fall back to physics outside it. Prereq: App 7 (calibration — a surrogate
of an uncalibrated model is a fast wrong answer). Feeds App 9 (RL) and App 12 (capstone).

## Sources

- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (Apache-2.0; renamed from Modulus at GTC 2025)
- [earth2studio](https://github.com/NVIDIA/earth2studio) — the Earth-2 AI weather/climate stack in Python
- [Omniverse DSX Blueprint (AI-factory digital twins)](https://blogs.nvidia.com/blog/omniverse-dsx-blueprint/)
- Week 21 overview and the full twin stack: [`week21/README.md`](../README.md)
