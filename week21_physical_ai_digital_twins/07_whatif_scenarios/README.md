# App 7 · What-if engineering — scenarios you can trust

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app is the trust layer of
**SIMULATION**: App 6 gave the twin an energy model; App 7 turns it into a what-if
machine whose answers deserve belief — parametric **sweeps** (eppy / OpenStudio
Measures / jEPlus / besos), the **ASHRAE Guideline 14 calibration gate** (NMBE +
CV(RMSE) against real meters), **FMI/FMU co-simulation** (models from different tools,
one `do_step()` loop), and **Alfalfa** — NREL's virtual building behind a BMS-shaped
REST API where agents practice before touching the real BMS.

> An uncalibrated twin's what-if answer is an opinion, not a prediction — put a G14
> gate in the pipeline.

Runs **dual-mode**: Ch 2–4 are pure stdlib (offline, $0; `eppy`/`fmpy` optional for
REAL flavor). Ch 5 makes one short LLM call — **REAL** against an Ollama / vLLM / DGX
endpoint, else a faithful **SIM** (`sim.py`).

## Run the web tutorial

```bash
cd week21/07_whatif_scenarios
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8206  (env TWIN_GUIDE_PORT)
```

Open **http://localhost:8206**. Use **🔌 Connection** for Ch 5's LLM call, or leave it
in SIM — every chapter still works, $0.

## Run the chapters standalone

```bash
python demos/step01_parametric_sweeps.py    # COP × setpoint × film — 18 ranked scenarios
python demos/step02_calibration_gate.py     # ASHRAE G14: NMBE + CV(RMSE), pass/fail gate
python demos/step03_fmi_cosim.py            # FMI 2.0 CS wire trace: do_step() between FMUs
python demos/step04_alfalfa_whatif.py       # pre-cool what-if via a BMS-shaped API + LLM planner
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 2 | Parametric sweeps | beginner |
| 3 | The calibration gate (ASHRAE Guideline 14) | intermediate |
| 4 | FMI/FMU co-simulation | advanced |
| 5 | Alfalfa — what-if behind a BMS API | advanced |

## REAL-mode upgrades

- `pip install eppy` — pythonic IDF read/modify/write; the Ch 2 sweep pattern on real models.
- `pip install fmpy` — load and step real `.fmu` files with exactly the Ch 4 verbs.
- `openstudio run -w workflow.osw` — apply shareable Measures from the BCL (bcl.nrel.gov).
- An Ollama / DGX endpoint (🔌 Connection) makes Ch 5's what-if planner call real.

## Where this sits in Week 21

Prereq: **App 06** (the energy model). Feeds **App 08** (surrogates train on these
parametric runs) and **App 12** (the capstone's what-if console). Apps 09–10's RL
environments (Sinergym, BOPTEST) are the gym-shaped siblings of Alfalfa.

## Sources

- Alfalfa — [github.com/NREL/alfalfa](https://github.com/NREL/alfalfa)
- FMI standard — [fmi-standard.org](https://fmi-standard.org) (Modelica Association)
- OpenStudio — [github.com/NREL/OpenStudio](https://github.com/NREL/OpenStudio) · BCL [bcl.nrel.gov](https://bcl.nrel.gov)
- EnergyPlus — [energyplus.net](https://energyplus.net)
- ASHRAE Guideline 14 (measurement of energy, demand and water savings) · IPMVP Option D · FEMP M&V
