# App 4 · Scene assembly — assembling the building twin scene

**Digital Twin = SCENE + STATE + SIMULATION + AGENTS.** This app finishes the **SCENE**
layer. It is a direct transposition of NVIDIA's learning path
**[Assembling Digital Twins With Omniverse and OpenUSD](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html)**
(sections: getting started · managing given assets · scene optimization & data
integration) — with their **factory** swapped for the **AltoTech Grand Bangkok hotel**.
Same moves: asset library & SimReady metadata → validator gate → one central materials
library → precision placement + custom attributes → discipline layers → payloads per
floor → instancing → asset inventory export → navigation waypoints.

Everything runs **offline, stdlib-only, $0** on a built-in mini-stage (`sim.py`); each
demo prints the "in USD Composer / your Kit app you would…" equivalent, and the Ch 5
export is a real `.usda` that opens in usdview / USD Composer.

## Run the web tutorial

```bash
cd week21/04_scene_assembly
pip install -r requirements.txt
python tutorial_server.py            # → http://127.0.0.1:8203  (env: TWIN_GUIDE_PORT)
```

Open the URL and run the chapters. Use **🔌 Connection** for Ch 5's twin-copilot
question (local Ollama / tunneled DGX / cloud), or leave it in SIM.

## Run the chapters standalone

```bash
python demos/step01_asset_hygiene.py         # project layout + the validator gate
python demos/step02_materials_placement.py   # central materials + precision placement
python demos/step03_optimize_scale.py        # instancing · payloads · inventory CSV
python demos/step04_assemble_hotel.py        # Grand Bangkok end-to-end + checklist
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 2 | Project structure & asset hygiene | beginner |
| 3 | Materials & precision placement | intermediate |
| 4 | Optimize for scale | intermediate |
| 5 | Assemble Grand Bangkok end-to-end | advanced |

## REAL-mode upgrades (optional, still $0)

- `pip install usd-core` — the exported stage is re-opened with the real **pxr** API
  (no GPU needed); open `.sandbox/grand_bangkok.usda` in usdview / USD Composer.
- An OpenAI-compatible endpoint (Ollama / vLLM / NIM / a DGX via `DGX_CONN`) makes the
  Ch 5 copilot answer for real.

## Where this sits in Week 21

Prereqs: **App 2** (OpenUSD grammar) · **App 3** (BIM → USD). This app composes those
assets into ONE validated, optimized, identified stage; **App 5** binds live BACnet
telemetry onto its GlobalIds (STATE), and the **App 12** capstone runs the living hotel.

## Sources

- NVIDIA learning path: [Assembling Digital Twins With Omniverse and OpenUSD](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html)
- [openusd.org](https://openusd.org) · [AOUSD](https://aousd.org) · [Omniverse](https://www.nvidia.com/en-us/omniverse/)
