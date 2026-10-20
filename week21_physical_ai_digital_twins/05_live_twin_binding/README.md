# App 05 · The living twin — binding real-time data to the scene

**A 3D model is not a twin.** It becomes one the moment live telemetry is BOUND
to the geometry. This app is the whole of Week 21's **STATE** phase: a reading
born on a BACnet field bus ends its journey as a USD attribute on the right
prim — via a gateway, MQTT, and a binding service doing the ID join — while a
**Brick semantics graph** holds meaning and a **TSDB** holds history, all three
stores joined on the **IFC GlobalId** that App 03 preserved.

Runs **dual-mode**: only Ch 5 makes an LLM call — **REAL** against an Ollama /
vLLM / NIM / DGX endpoint, or **SIM** with a canned query plan ($0). Ch 2–4 are
pure stdlib and run offline; Ch 3 upgrades to REAL OpenUSD with `usd-core`.

## Run the web tutorial

```bash
cd week21/05_live_twin_binding
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8204  (env TWIN_GUIDE_PORT)
```

Open **http://localhost:8204**. Use **🔌 Connection** to point Ch 5 at your
endpoint, or leave it in SIM.

## Run the chapters standalone

```bash
python demos/step01_protocol_chain.py     # BACnet/Modbus → gateway → MQTT → binding → prim
python demos/step02_live_layers.py        # session layer + Fabric — authored files untouched
python demos/step03_three_stores.py       # USD + Brick graph + TSDB, joined on the GlobalId
python demos/step04_data_to_viewport.py   # heat map · alert prim · LLM query plan
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | The protocol chain — field bus to prim | beginner |
| 2 | Live layers & Fabric | intermediate |
| 3 | The three-store architecture | advanced |
| 4 | From data to the viewport | advanced |

## REAL-mode upgrades

- `pip install usd-core` — Ch 3's session layer becomes a real `Usd.Stage` (no GPU).
- An Ollama/DGX endpoint (🔌 Connection) — Ch 5's NL → query-plan call runs for real.
- NVIDIA's reference IoT→USD connector: `git clone https://github.com/NVIDIA-Omniverse/iot-samples`.

## Where this sits in Week 21

Prereq: **App 04** (an assembled scene). Feeds **App 10** (self-evolving ops),
**App 11** (staff copilot — it runs exactly Ch 4's three-store query) and the
capstone **App 12**. The graph store continues Week 14/15's Neo4j/GraphRAG work.

## Sources

- [NVIDIA-Omniverse/iot-samples](https://github.com/NVIDIA-Omniverse/iot-samples) — the reference MQTT→USD connector pattern
- [Brick Schema](https://brickschema.org) · [Project Haystack](https://project-haystack.org) · [RealEstateCore](https://www.realestatecore.io) (+ WillowInc/opendigitaltwins-building) · ASHRAE Standard 223P
