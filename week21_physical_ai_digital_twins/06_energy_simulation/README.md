# App 06 · Energy simulation — giving the twin physics

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app is the **SIMULATION**
layer (Week 21 · Phase 4): the physics that lets a twin answer *what-if* instead of only
reporting what is. It teaches **EnergyPlus** from the file formats up, builds a genuine
**ISO 13790-style 5R1C reduced-order model** live in ~60 lines, drives the real engine
from Python with **pyenergyplus**, and closes with the **tool ladder** — Spawn of
EnergyPlus, Modelica Buildings, CDL/Guideline 36, TRNSYS, IES-VE — and when each wins.

Runs **dual-mode**: Ch 2–4 are pure-stdlib physics (offline, $0); Ch 5 makes one short
LLM call — **REAL** against an Ollama / vLLM / NIM / DGX endpoint, or a faithful **SIM**.

## Run the web tutorial

```bash
cd week21/06_energy_simulation
pip install -r requirements.txt
python tutorial_server.py            # → http://127.0.0.1:8205  (TWIN_GUIDE_PORT to change)
```

Open the URL, read Ch 1, then click **Run** per chapter. Use **🔌 Connection** for Ch 5's
LLM call, or leave it in SIM.

## Run the chapters standalone

```bash
python demos/step01_energyplus_anatomy.py   # IDF/epJSON + EPW → engine → SQL/ABUPS
python demos/step02_5r1c_live.py            # build 5R1C live; a full year in milliseconds
python demos/step03_pyenergyplus.py         # callbacks, handles, and the two gotchas
python demos/step04_tool_ladder.py          # Spawn · Modelica · CDL/G36 · TRNSYS · IES-VE
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | Energy simulation — giving the twin physics (concept) | beginner |
| 2 | EnergyPlus anatomy — inputs, engine, outputs | beginner |
| 3 | Build a 5R1C reduced-order model — live | intermediate |
| 4 | pyenergyplus — Python inside the engine | advanced |
| 5 | The tool ladder — when each simulator wins | advanced |

## REAL-mode upgrades

- `pip install eppy` + an EnergyPlus install ([energyplus.net](https://energyplus.net)) —
  Ch 2 parses/patches real IDF files; `pyenergyplus` ships inside the E+ install dir (Ch 4).
- An Ollama/DGX endpoint (or 🔌 Connection) lights up Ch 5's LLM call — still $0 cloud.

## Where this sits in Week 21

Prereq: **App 05** (live twin binding). Feeds **App 07** (ASHRAE G14 calibration + what-if),
**App 08** (PhysicsNeMo surrogates) and **App 09** (RL building controls). The honest rule
of the whole phase: an uncalibrated model is just a guess with more decimal places.

## Sources

- [EnergyPlus](https://energyplus.net) · [github.com/NREL/EnergyPlus](https://github.com/NREL/EnergyPlus)
- [OpenStudio](https://github.com/NREL/OpenStudio)
- [Spawn of EnergyPlus](https://lbl-srg.github.io/soep/) · [Modelica Buildings Library](https://simulationresearch.lbl.gov/modelica/)
- ASHRAE Guideline 14 (calibration) · Guideline 36 (high-performance sequences) · 231P (CDL)
