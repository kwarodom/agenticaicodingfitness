# ◳ Week 21 — Physical AI & Digital Twins for Buildings (Omniverse · OpenUSD · the NVIDIA stack)

Thirteen **interactive, explainable web apps** (Week 19/20 style: dual-mode SIM/REAL, animated
inline-SVG architecture diagrams, runnable demos, one chapter at a time) that teach how to build
**digital twins of real buildings** — hotel, hospital, office, factory, data center, smart city —
on **NVIDIA Omniverse + OpenUSD + the Physical AI stack**, and then wire the Week 18/20
**self-evolving agent fleet** into the twin so it *operates* the building.

> **The through-line of the whole week:
> Digital Twin = SCENE (OpenUSD) + STATE (live data) + SIMULATION (physics) + AGENTS (control & copilot).**
> A 3D model is not a twin. A BIM file is not a twin. A twin is a living system: geometry that
> mirrors the building (scene), telemetry bound to that geometry (state), physics that can answer
> *what-if* (simulation), and agents that learn from it and act on it (agents). Week 21 builds all
> four layers, in order.

> **Dual-mode, always $0.** Every app auto-detects a live endpoint (Ollama / vLLM / NIM / a DGX via
> `DGX_CONN`) for its LLM-agent chapters and runs **REAL**, else a faithful **SIM**. Physics/USD
> chapters likewise: if an optional library (`usd-core`, `eppy`, …) is installed they use it for
> real, else a faithful built-in simulator teaches the same concept. Real commands are always
> shown; cloud cost $0.

---

## How to use this folder — the learning path

**The folders are numbered `01…13` in the order you should learn them.** Each is a standalone
interactive app (its own `config.py`, engine, `tutorial_server.py`, `static/guide.html`, `demos/`) —
launch it, read the concept, view the demo source, and click **Run**. Budget ~20–40 min each.
Cross-references inside the apps ("… App 7") use these same numbers.

### Phase 1 · Physical AI — the map

| # | Folder | Port | What you learn |
|---|--------|------|----------------|
| 01 | [`01_physical_ai_landscape`](01_physical_ai_landscape/) | 8200 | What **Physical AI** is; the **three-computer model** (DGX trains · OVX/Omniverse simulates · Jetson/AGX deploys); Omniverse as **libraries + microservices**, not an app; **Cosmos** world foundation models; the **Mega** and **DSX** Omniverse Blueprints; where *buildings* fit in NVIDIA's factory/data-center-first story. |

### Phase 2 · The SCENE — OpenUSD & BIM

| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 02 | [`02_openusd_foundations`](02_openusd_foundations/) | 8201 | **OpenUSD** from zero: stage · prim · attribute · relationship; **layers & composition** (sublayer, reference, payload, variant); non-destructive *opinions*; `kind`/`purpose`; units & up-axis. Runs REAL with `pip install usd-core` (no GPU!). | 01 |
| 03 | [`03_bim_to_usd`](03_bim_to_usd/) | 8202 | **BIM → twin**: LOD 100–500 (Level of *Development*, not detail — and why "LOD 500" ≠ prettiest mesh); the twin-readiness bar (*as-built LOD 300 geometry + LOD 500-grade / COBie data*); IFC spatial tree → USD hierarchy; **GUID preservation** as the join key; the post-connector era (IFC-first pipelines, AOUSD). | 02 |
| 04 | [`04_scene_assembly`](04_scene_assembly/) | 8203 | **Assemble the building stage** (mirrors NVIDIA's *Assembling Digital Twins* learning path): asset libraries & SimReady metadata, centralized materials, discipline layers (site/arch/MEP/furniture/sensors), **payloads per floor**, **instancing** for the 4,000 chairs, asset validation, navigation waypoints. | 02, 03 |

### Phase 3 · The STATE — spatial-temporal management

| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 05 | [`05_live_twin_binding`](05_live_twin_binding/) | 8204 | Make the scene **live**: BACnet/Modbus → gateway → MQTT/Kafka → twin service → **USD attributes on a session/live layer** (Fabric for high-frequency); the three-store architecture — **USD (geometry) + graph (Brick/RealEstateCore semantics) + TSDB (telemetry)** joined on GlobalId; spatial-temporal queries ("all VAVs fed by AHU-3 on floor 5, last 24 h"). Continues Week 14/15 graphs. | 04 |

### Phase 4 · The SIMULATION — energy & what-if

| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 06 | [`06_energy_simulation`](06_energy_simulation/) | 8205 | **EnergyPlus** fundamentals: IDF/epJSON, EPW weather, zones → surfaces → HVAC, run an annual sim, read the outputs; build a **5R1C reduced-order model** live in the demo; **Spawn of EnergyPlus / Modelica** for controls-realistic dynamics; when each tool wins. | 05 |
| 07 | [`07_whatif_scenarios`](07_whatif_scenarios/) | 8206 | **What-if engineering**: parametric sweeps (eppy/Measures); the **ASHRAE Guideline 14 calibration gate** (NMBE ≤ ±10 %, CV(RMSE) ≤ 30 % hourly) — a twin's answers are only credible after calibration; **FMI/FMU co-simulation**; **Alfalfa** — a simulated building behind a BMS-shaped REST API. | 06 |
| 08 | [`08_physics_surrogates`](08_physics_surrogates/) | 8207 | **PhysicsNeMo** (ex-Modulus) surrogates: train on parametric sim/CFD data → **millisecond what-if** in the twin viewport (the data-center-twin pattern: Cadence/Schneider/ETAP); **Earth-2** forecast boundary conditions; surrogate vs full physics — the honest trade. | 07 |

### Phase 5 · The AGENTS — learn to control

| # | Folder | Port | What you learn | Prereq |
|---|--------|------|----------------|--------|
| 09 | [`09_rl_building_controls`](09_rl_building_controls/) | 8208 | **RL on the building**: gym environments over the sim — **Sinergym** (E+ HVAC), **CityLearn** (district/DR, multi-agent), **BOPTEST** (Modelica benchmark + fixed KPIs); obs/action/reward design; PPO/SAC vs rule-based vs **Guideline 36** — and the honest result that tuned MPC still often wins. | 06 |
| 10 | [`10_self_evolving_ops`](10_self_evolving_ops/) | 8209 | The **self-evolving operator**: Week 18 tripartite memory + Week 23 data flywheel applied to building controls — episodes → consolidation → skills; learning from operator overrides; the **safe-autonomy ladder** (offline → shadow → advisory → supervised autonomy) with hard guardrails (action clamps, rate limits, watchdogs, fallback to G36). | 09 |
| 11 | [`11_staff_copilot`](11_staff_copilot/) | 8210 | **Boost the humans**: a copilot over the twin (graph query + time-series + CMMS + O&M-doc RAG — the Week 15 GraphRAG pattern, grounded in a building); **cuOpt** technician dispatch/routing; **Metropolis/VSS** vision analytics (occupancy, safety, NL video Q&A) and simulating camera placement in the twin first. | 05 |

### Phase 6 · Capstone — the living building

| # | Folder | Port | What you build | Prereq |
|---|--------|------|----------------|--------|
| 12 | [`12_capstone_twin_hotel`](12_capstone_twin_hotel/) | 8211 | **Capstone I · AltoTech Grand Bangkok — the twin**: one running system combining every app above. BIM→USD scene of the hotel, live telemetry bound to prims, calibrated energy model + surrogate, what-if console, RL-tuned HVAC under the safe-autonomy ladder, self-evolving fleet, staff copilot — the Week 23 capstone's agent fleet now has a *body*. Variant briefs: hospital, office, factory, smart-city district. | all |
| 13 | [`13_capstone_smart_city`](13_capstone_smart_city/) | 8212 | **Capstone II · the Sovereign Smart City (Krung Alto)** — the same four layers at city scale, grounded in NVIDIA's real [Smart City Blueprint (VSS 3.0)](https://docs.nvidia.com/vss/latest/smartcity-docs/Quickstart-Guide.html): a **twin of twins** (App 12's hotel *referenced* into the city stage), RTVI CV + **VLM alert verification** over 24 camera streams (documented per-GPU budgets), corridor-reroute & district-peak what-ifs, and a self-evolving fleet whose autonomy is earned **per action type** — all sovereign: video never leaves city-owned GPUs. | all |

**Fastest useful path (≈2 hrs):** 01 → 02 → 03 → 05 → 07 → 12. Add 04/06/08/09/10/11 for the full stack.

---

## The twin stack (the mental model)

```
                 ┌──────────────────────────────────────────────────┐
  Digital Twin = │  AGENTS      copilot · RL controller · fleet      │ ← 09,10,11 (+W18/20)
  Scene + State  │              (shadow → advisory → autonomy)       │
  + Simulation   ├──────────────────────────────────────────────────┤
  + Agents       │  SIMULATION  EnergyPlus/Spawn · FMU · surrogate   │ ← 06,07,08
                 │              (calibrated, then fast)              │
                 ├──────────────────────────────────────────────────┤
                 │  STATE       BMS/IoT → MQTT → live layer · graph  │ ← 05
                 │              (Brick/REC) · TSDB — join: GlobalId  │
                 ├──────────────────────────────────────────────────┤
                 │  SCENE       OpenUSD stage ← BIM (IFC, LOD 200+)  │ ← 02,03,04
                 │              layers · payloads · instancing       │
                 └──────────────────────────────────────────────────┘
   NVIDIA rails: Omniverse Kit/USD · Cosmos · PhysicsNeMo · Earth-2 · Metropolis ·
                 cuOpt · NIM/NeMo agents ·· three computers: DGX train / OVX sim / AGX act
```

---

## Quick start (any app)

```bash
uv pip install -r week21/01_physical_ai_landscape/requirements.txt
.venv/bin/python week21/01_physical_ai_landscape/tutorial_server.py   # → http://127.0.0.1:8200
```

Open the URL, use the **🔌 Connection** panel for LLM chapters (local / tunnel / cloud), and run the
chapters. With nothing installed/reachable everything runs in **SIM** — every concept still works, $0.
Optional REAL upgrades per app: `pip install usd-core` (02–05), `pip install eppy` (06–07); an
Ollama/DGX endpoint lights up the agent chapters (10–12).

---

## Where this sits in the course

```
W14/15 knowledge graphs · W18 sovereign edge + self-evolving · W19 DGX · W20 open superintelligence stack
                                        │
   WEEK 21: give the agents a BODY — a digital twin of the building they run.
   Learn the map (01), build the SCENE from BIM in OpenUSD (02–04), make it LIVE
   (05), teach it PHYSICS so it can answer what-if (06–08), let agents LEARN
   controls safely (09–10), boost the humans (11), then run the whole living
   building in the capstone (12).
```

---

## Sources

- NVIDIA learning path: [Assembling Digital Twins With Omniverse and OpenUSD](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html)
  (Sections: scene assembly · managing assets · optimization & data integration)
- [Omniverse — develop Physical AI applications](https://www.nvidia.com/en-us/omniverse/) · [Cosmos world foundation models](https://www.nvidia.com/en-us/ai/cosmos/) · [GTC 2026: virtual worlds powering Physical AI](https://blogs.nvidia.com/blog/gtc-2026-virtual-worlds-physical-ai/)
- Blueprints: [Omniverse DSX (AI-factory digital twins)](https://blogs.nvidia.com/blog/omniverse-dsx-blueprint/) · [Mega (industrial twins for physical-AI training)](https://blogs.nvidia.com/blog/mega-omniverse-blueprint-industrial-digital-twins/)
- OpenUSD: [openusd.org](https://openusd.org) · [AOUSD](https://aousd.org) · NVIDIA *Learn OpenUSD* DLI path · [NVIDIA-Omniverse/iot-samples](https://github.com/NVIDIA-Omniverse/iot-samples)
- BIM: [BIMForum LOD Specification](https://bimforum.org/resources/lod/) · [IfcOpenShell](https://ifcopenshell.org) · buildingSMART × AOUSD IFC↔USD collaboration
- Energy & controls: [EnergyPlus](https://energyplus.net) · [OpenStudio](https://github.com/NREL/OpenStudio) · [Spawn](https://lbl-srg.github.io/soep/) · [Modelica Buildings](https://simulationresearch.lbl.gov/modelica/) · [Alfalfa](https://github.com/NREL/alfalfa) · [Sinergym](https://github.com/ugr-sail/sinergym) · [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) · [BOPTEST](https://ibpsa.github.io/project1-boptest/) · ASHRAE Guideline 14 / Guideline 36 / Standard 223P
- Semantics: [Brick Schema](https://brickschema.org) · [Project Haystack](https://project-haystack.org) · [RealEstateCore](https://www.realestatecore.io)
- NVIDIA AI: [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) · [Earth-2 / earth2studio](https://github.com/NVIDIA/earth2studio) · [cuOpt](https://github.com/NVIDIA/cuopt) · [Metropolis / VSS blueprint](https://www.nvidia.com/en-us/autonomous-machines/intelligent-video-analytics/) · [Smart City Blueprint quickstart](https://docs.nvidia.com/vss/latest/smartcity-docs/Quickstart-Guide.html) (Capstone II)

**Honesty notes baked into the apps:** Omniverse's proven wins are factories, warehouses and
data centers — building-*operations* twins (hotel/hospital/office) on Omniverse are genuinely
frontier, which is exactly why this week teaches the durable format-level skills (USD, IFC, Brick,
FMI) rather than any single vendor app; NVIDIA deprecated the Launcher-era connectors (Oct 2025) in
favor of native OpenUSD interchange; tuned MPC still often beats RL on BOPTEST; LOD 500 is a
field-verification claim, not "more polygons".
