# ◳ 13 · Capstone II — the Sovereign Smart City (Krung Alto)

Week 21 · App 13 · Phase 6: **the CAPSTONE, scaled up** · port **8212**

Capstone I gave one building a living twin. Capstone II scales the same four layers to a
**city** — and adds the thing a city needs that a building doesn't: **eyes**. Grounded in
NVIDIA's real **Smart City Blueprint** ([VSS 3.0 quickstart](https://docs.nvidia.com/vss/latest/smartcity-docs/Quickstart-Guide.html)),
run **sovereignly** — video, models and decisions never leave GPUs the city owns.

> **A city twin is a twin of twins.** Buildings compose in as *references* to their own
> stages — Capstone I's Grand Bangkok hotel is one prim here, keeping its own layers,
> telemetry and validator. Fix the hotel once; the city sees it.

```
   24 cams ─► RTVI CV (RTDETR·GDINO) ─► Kafka ─► VLM verify ─► verified alerts only
                    the blueprint's eyes (STATE)                        │
   city SCENE (USD): districts payloaded · hotel twin referenced ◄──────┤
   SIMULATION: corridor congestion · district energy dispatch           │
   AGENTS: VSS agent · operator fleet · 🛡 rung-per-action gate · flywheel
   SOVEREIGN: all of it on city-owned GPUs — $0/token, no data leaves town
```

## Chapters

| Ch | Demo | Level | The story |
|----|------|-------|-----------|
| 1 | — (concept) | beginner | A twin of twins; why cities make sovereignty non-negotiable. |
| 2 | `step01_assemble_city_twin.py` | beginner | 4 districts as payloads, 2,900 streetlights → 3 prototypes, the App 12 hotel **referenced** in, readiness validator. |
| 3 | `step02_vss_camera_network.py` | intermediate | The blueprint pipeline: per-GPU stream budgets (RTDETR 30 vs GDINO 6–12), Kafka wire trace, the VLM rejecting 2 of 5 raw events, NL Q&A via the VSS agent. |
| 4 | `step03_district_whatif.py` | intermediate | Price a corridor reroute before retiming a signal (v/c 0.89 < jam); flatten the 35 MW district peak CityLearn-style — hospital hard-excluded. |
| 5 | `step04_sovereign_self_evolving.py` | advanced | The sovereignty audit (6 data classes; local / local_shared / remote-NIM honestly labeled) + autonomy **per action type** (alerts rung 4 · retiming rung 3 · substation switching rung 1 forever) + the flywheel. |
| 6 | `step05_a_night_in_the_city.py` | advanced | Three acts: the 23:15 sideswipe end-to-end, the 01:00 substation trip ridden through by the DR fleet (hotel included), the 06:00 brief + consolidation. |
| — | Appendix (concept) | all | Room → building → district → city: the same four layers at every scale. |

## Quick start

```bash
uv pip install -r week21/13_capstone_smart_city/requirements.txt
.venv/bin/python week21/13_capstone_smart_city/tutorial_server.py   # → http://127.0.0.1:8212
```

Everything runs in **SIM**, deterministic, $0. An Ollama / vLLM / NIM / DGX endpoint
(🔌 Connection or `DGX_BASE_URL`) makes the VSS-agent answers, postmortems and the morning
brief genuinely model-written.

## Run the real blueprint

The numbers in Ch 3 are the documented ones — the actual deployment is a real afternoon:
NGC login → download compose + models (RTDETR/GDINO) → set the hardware profile
(H100 / L40S / RTX PRO 6000) → `docker compose up` → UIs at `:7777` (chat), `:3002` (map),
APIs incl. **VST-MCP `:8001`** and **VA-MCP `:9901`** (MCP — Week 7 pays off), LLM/VLM NIMs
at `:30081`/`:30082`. Supported models include Nemotron and Qwen3-VL / Cosmos VLMs.

## Sources

- [Smart City Blueprint — VSS Quickstart](https://docs.nvidia.com/vss/latest/smartcity-docs/Quickstart-Guide.html) (RTVI CV, Alert Verification, VSS agents, stream budgets, endpoints)
- [Metropolis / VSS](https://www.nvidia.com/en-us/autonomous-machines/intelligent-video-analytics/) · [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) (the district-dispatch pattern, App 9)
- Capstone I: [`../12_capstone_twin_hotel/`](../12_capstone_twin_hotel/) — the nested twin · Weeks 18/20 — the self-evolving, sovereign lineage
