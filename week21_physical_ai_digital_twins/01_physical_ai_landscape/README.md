# App 01 · Physical AI — the map (the NVIDIA digital-twin stack)

**Physical AI is AI that perceives, reasons and ACTS in the physical world** — and a
building is a physical-AI system as much as a robot is. This app is the **map** the rest
of Week 21 builds on: the **three-computer model** (a DGX **trains** the models,
OVX/Omniverse **simulates** the world, Jetson/AGX **deploys** at the edge), Omniverse's
2026 shape (**libraries + microservices** — Kit SDK, OpenUSD, Kit App Streaming, USD
NIMs; Launcher EOL and first-party connectors deprecated Oct 2025 in favor of native
OpenUSD interchange), **Cosmos** world foundation models (generative worlds vs
Omniverse's deterministic physics), and the **Mega** and **DSX** Omniverse Blueprints.

Deployments are tiered **honestly**: factories (BMW, Foxconn), warehouses, retail
(Lowe's) and data centers are PROVEN; hotels/hospitals/offices are FRONTIER — which is
exactly why Week 21 bets on durable formats (USD, IFC, Brick, FMI), not one vendor app.

Runs **dual-mode**: only Ch 5 makes an LLM call — **REAL** against an Ollama / vLLM /
NIM / DGX endpoint, else a faithful **SIM**. Ch 2–4 are pure stdlib. Either way, $0.

## Run the web tutorial

```bash
uv pip install -r week21/01_physical_ai_landscape/requirements.txt
.venv/bin/python week21/01_physical_ai_landscape/tutorial_server.py   # → http://127.0.0.1:8200
```

Open **http://127.0.0.1:8200**. Use **🔌 Connection** to point Ch 5 at your endpoint,
or leave it in SIM. Set `TWIN_GUIDE_PORT` to change the port.

## Run the chapters standalone

```bash
cd week21/01_physical_ai_landscape
../../.venv/bin/python demos/step01_three_computers.py        # DGX train · OVX sim · AGX act
../../.venv/bin/python demos/step02_omniverse_2026.py         # libraries + microservices
../../.venv/bin/python demos/step03_cosmos_wfm.py             # deterministic vs generative
../../.venv/bin/python demos/step04_blueprints_deployments.py # Mega · DSX · who runs it
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 2 | The three-computer model | beginner |
| 3 | Omniverse's 2026 shape | beginner |
| 4 | Cosmos world foundation models | intermediate |
| 5 | Blueprints & real deployments | advanced |

## Where this sits in Week 21

Phase 1 · the map. It feeds every twin layer that follows: **SCENE** (Apps 02–04) ·
**STATE** (App 05) · **SIMULATION** (Apps 06–08) · **AGENTS** (Apps 09–11) · capstone
(App 12). Builds on Week 18 (sovereign edge), Week 19 (DGX) and Week 23 (agent stack).

## Sources

- [Omniverse — develop Physical AI applications](https://www.nvidia.com/en-us/omniverse/) · [GTC 2026: virtual worlds powering Physical AI](https://blogs.nvidia.com/blog/gtc-2026-virtual-worlds-physical-ai/)
- [Cosmos world foundation models](https://www.nvidia.com/en-us/ai/cosmos/)
- Blueprints: [Mega (industrial twins)](https://blogs.nvidia.com/blog/mega-omniverse-blueprint-industrial-digital-twins/) · [DSX (AI-factory twins)](https://blogs.nvidia.com/blog/omniverse-dsx-blueprint/)
- [openusd.org](https://openusd.org) · [AOUSD](https://aousd.org)
