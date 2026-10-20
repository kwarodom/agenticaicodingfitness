# ▣ Lab Runner — Week 23's hands-on track as a step-by-step web app

The DO-IT side of Week 23, in the browser. Lab Runner parses the 12 `TUTORIAL.md`
files (`01_nemotron_models` … `12_capstone_smart_hotel`), presents each one as a
step-by-step course, and runs every folder's `labs/*.py` scripts **server-side on
this machine**, streaming their output live to the page — REAL against your
DGX/Ollama endpoint via the shared `config.py` connection switch, or SIM with no
GPU ($0 either way).

## Launch

```bash
.venv/bin/python week23/00_lab_runner/tutorial_server.py
# → http://127.0.0.1:8113   (auto-picks a free port; override with LAB_GUIDE_PORT)
```

## How it relates to the rest of Week 23

- **The 12 `TUTORIAL.md` files** — Lab Runner is a web runner for exactly that
  track: same text (rendered verbatim, section by section), same `labs/` scripts,
  same connection contract (`DGX_CONN` / `DGX_BASE_URL` / `DGX_API_KEY` /
  `DGX_MODE`). Everything here you can also do by hand in a terminal.
- **`00_stack_navigator`** — the concept hub (what each layer of the Open
  Superintelligence Stack *is* and how the pieces fit). Lab Runner is its
  hands-on twin: the place where you actually type/run.
- Each tutorial also has its own companion explainer app on ports 8100–8111;
  Lab Runner links to them but replaces none of them.
