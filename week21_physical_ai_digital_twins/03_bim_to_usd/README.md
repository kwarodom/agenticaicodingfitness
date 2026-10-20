# App 3 · BIM → USD — from model to twin-ready scene

**A BIM file is not a twin.** This app is the **SCENE conversion** step of Week 21:
getting a building model into **OpenUSD** without losing the two things a twin lives
on — element identity (the **IFC GlobalId**, the join key App 5 binds telemetry to)
and handover data (**COBie**). Geometry is the easy part; every pipeline keeps it.

The twin-readiness bar to drill: **as-built LOD 300 geometry + LOD-500-grade / COBie
data** — LOD is Level of *Development* (reliability), not detail, and LOD 500 is a
field-verification attestation, not more polygons.

Runs **dual-mode**: Ch 3 goes **REAL** against a NIM / Ollama / vLLM endpoint (else a
faithful SIM), Ch 5 goes **REAL** with `pip install usd-core` (no GPU) — else it emits
faithful USD-flavored text. Either way, $0.

## Run the web tutorial

```bash
cd week21/03_bim_to_usd
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8202
```

Open **http://localhost:8202**. Use **🔌 Connection** to point at your endpoint for
Ch 3, or leave it in SIM. Everything else is pure stdlib, offline.

## Run the chapters standalone

```bash
python demos/step01_lod_levels.py       # LOD 100–500, properly (Development, not detail)
python demos/step02_what_survives.py    # ALWAYS/USUALLY/NEVER survival table + the GUID rule
python demos/step03_pipelines.py        # 5 real 2026 paths, ranked + preservation scorecard
python demos/step04_mini_converter.py   # mini IFC→USD converter + twin-readiness lint
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | BIM LOD, properly | beginner |
| 2 | What survives BIM → USD | intermediate |
| 3 | Pipelines in 2026 — the post-connector era | intermediate |
| 4 | A mini IFC → USD converter, live | advanced |

## REAL-mode upgrades

- `pip install usd-core` — Ch 5 writes a genuine USD stage in memory (no GPU needed).
- `pip install ifcopenshell` — the real toolkit behind pipeline path 1 (IFC4-first).
- An Ollama / DGX endpoint (🔌 Connection) — Ch 3's LLM call runs for real.

## Where this sits in Week 21

Phase 2 · the **SCENE**. Prereq: App 02 (OpenUSD foundations). Feeds App 04 (scene
assembly) and App 05 (live telemetry bound to the GlobalIds this app preserved).

## Sources

- [BIMForum LOD Specification](https://bimforum.org/resources/lod/) · [IfcOpenShell](https://ifcopenshell.org) · [AOUSD](https://aousd.org)
- NVIDIA learning path: [Assembling Digital Twins With Omniverse and OpenUSD](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html)
