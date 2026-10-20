# App 9 · RL on the building — gyms over the simulation

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app starts the **AGENTS**
layer: wrap the building simulation (Apps 6–8) in **gym environments** so a control
policy can fail a thousand times for free — **Sinergym** (one building's HVAC over
EnergyPlus), **CityLearn** (district demand response, the MARL testbed), and
**BOPTEST** (fixed scenarios + fixed KPIs = fair benchmarking over REST).

Runs **dual-mode**: everything works in **SIM** on a built-in pure-stdlib RC hotel
zone (`sim.py` — same MDP shape as Sinergym, $0, no GPU); optional REAL upgrades below.

## Run the web tutorial

```bash
cd week21/09_rl_building_controls
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8208
```

Open **http://localhost:8208**. Use **🔌 Connection** for Ch 5's live MPC-vs-RL
verdict (Ollama / DGX / cloud), or leave it in SIM.

## Run the chapters standalone

```bash
python demos/step01_sinergym_mdp.py        # the MDP: obs / action / reward, stepped live
python demos/step02_learn_policy.py        # baseline vs learned policy — learning curve + scorecard
python demos/step03_citylearn_district.py  # 3-building district: batteries flatten the peak
python demos/step04_boptest_kpis.py        # BOPTEST REST wire-trace + standard KPI table
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | Building control as an MDP — Sinergym | beginner |
| 2 | Learn a policy LIVE — baseline vs search | intermediate |
| 3 | CityLearn — district demand response | advanced |
| 4 | BOPTEST — fair benchmarking over REST | advanced |

## REAL-mode upgrades (all optional, all CPU-friendly)

- `pip install sinergym` — the actual EnergyPlus gym (or its Docker image, E+ preinstalled)
- `pip install CityLearn` — the actual district env + Challenge datasets
- `git clone https://github.com/ibpsa/project1-boptest` + `docker compose up` — the real benchmark service

## The honest calibration

Published RL results typically show **5–20% HVAC savings vs rule-based controllers**
at equal-or-better comfort — and the margin **shrinks** against a well-tuned ASHRAE
Guideline 36 baseline. On BOPTEST, **well-tuned MPC generally still beats RL**; RL
earns its place when models are unavailable or buildings drift. And a policy that
wins in the gym is **not yet safe on the building** — that's App 10.

## Sources

- [Sinergym](https://github.com/ugr-sail/sinergym) (Univ. of Granada SAIL)
- [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) (UT Austin IEL)
- [BOPTEST](https://ibpsa.github.io/project1-boptest/) · [ibpsa/project1-boptest](https://github.com/ibpsa/project1-boptest) (IBPSA Project 1)

## Where this sits in Week 21

Apps 2–4 **SCENE** · App 5 **STATE** · Apps 6–8 **SIMULATION** · **App 9 (this) gyms + RL** ·
App 10 **safe self-evolving ops** · App 11 **staff copilot** · App 12 **capstone twin**.
