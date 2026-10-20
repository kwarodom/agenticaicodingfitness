# App 02 · OpenUSD foundations — the scene language of the digital twin

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app is the start of the
**SCENE** layer (Week 21 · Phase 2): **OpenUSD** from zero — stage · prim · attribute ·
relationship, then the composition arcs that make building-scale stages possible
(sublayers, references, payloads, variants, instancing), taught on BIM-shaped examples
(`/Site/Building/Storey_03/Rooms/Room_301`, `ifc:GlobalId`, discipline layers).

Runs **dual-mode** — and here REAL means **`pip install usd-core`** (pip OpenUSD, the
`pxr` modules — ~30 MB, no GPU, no Omniverse), not an LLM endpoint. Without it, a
faithful mini-USD teaching engine (`sim.py`) mimics the same API shape. Either way $0.

## Run the web tutorial

```bash
cd week21/02_openusd_foundations
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8201
```

Open **http://localhost:8201** and run the chapters. Each demo prints its own MODE
line (REAL if `pxr` imports, else SIM) and writes its stages to `.sandbox/`.

## Run the chapters standalone

```bash
python demos/step01_stage_prims.py            # stage · prim · attribute · relationship
python demos/step02_layers_composition.py     # sublayers · opinion strength · session layer
python demos/step03_refs_payloads_variants.py # reference · payload · variants · instancing
python demos/step04_mini_hotel.py             # compose a mini hotel; flatten vs live
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | Stage · prim · attribute · relationship (+ metersPerUnit/upAxis gotchas) | beginner |
| 2 | Layers & composition — how BIM disciplines federate | intermediate |
| 3 | References, payloads, variants, instancing | intermediate |
| 4 | Compose a mini hotel — flatten vs live composition | advanced |

## Go REAL

```bash
pip install usd-core                 # then re-run any chapter
cat .sandbox/usd_step04/hotel.usda   # genuine USD, plain text
```

## Where this sits in Week 21

App 01 the Physical AI map · **App 02 (this) the USD scene language** · App 03 BIM→USD
(LOD, GUID preservation) · App 04 scene assembly at scale · App 05 live state on USD
attributes · Apps 06–08 simulation · Apps 09–11 agents · App 12 the capstone twin hotel.

## Sources

- [openusd.org](https://openusd.org) — the OpenUSD project and docs
- [AOUSD](https://aousd.org) — the Alliance for OpenUSD (the open-standard body)
- NVIDIA *Learn OpenUSD* DLI learning path (free, no GPU needed)
- NVIDIA learning path: [Assembling Digital Twins With Omniverse and OpenUSD](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html)
