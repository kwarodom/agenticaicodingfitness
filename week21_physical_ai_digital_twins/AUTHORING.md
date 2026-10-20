# Week 21 module authoring spec (internal — delete before merge if desired)

Every module `week21/NN_name/` is a standalone interactive tutorial app that must match the
Week 23 template **exactly in structure and style**. Template to mirror:
`week23/03_dynamo_serving/` (read all of its files first). The capstone additionally mirrors
`week23/12_capstone_smart_hotel/` (which adds a `hotel/` package and a 5th demo).

## Files per module

| File | How to produce it |
|---|---|
| `config.py` | **Copy verbatim** from `week23/03_dynamo_serving/config.py`. Do not edit (it's the shared DGX connection switch; the docstring mentioning week19 is fine — it's shared infra). |
| `view.py` | **Copy verbatim** from `week23/03_dynamo_serving/view.py`. |
| `sim.py` | Module-specific simulator. MUST export `installed_models() -> list[str]`, `tok_s(model) -> float`, `stream_generate(prompt, model)` (canned streamed answer relevant to the module) — plus whatever tables/curves/engines your demos need. Pure stdlib. |
| `demos/step01..04.py` (05 for capstone) | 50–130 lines each, the pattern of `week23/03_dynamo_serving/demos/step02_disaggregated.py`: docstring header (`PART n · Title [LEVEL]`), `sys.path.insert` to parent, `import sim, view`, `view.banner(...)`, `view.mode_line()` **only when the demo actually uses the LLM endpoint** (physics/USD demos should print their own MODE line: REAL if optional lib importable, else SIM), ASCII diagram, printed tables/bars with real numbers, plainly-worded takeaways, forward pointer to the next chapter. Demos MUST run offline, exit 0, in <20 s, with only stdlib — optional libs (`usd-core`→`from pxr import Usd`, `eppy`, `openai`) behind try/except with a faithful fallback. When a real-world command exists (`pip install usd-core`, `ollama run …`, `docker run boptest …`, `openstudio run -w …`), print it in a "run it for real" block. |
| `tutorial_server.py` | Copy from template, then edit: title strings, `GUIDE_PORT` env var name `TWIN_GUIDE_PORT` + module default port, and the `STEPS` list — intro concept chapter (long desc: what this tutorial is, bullet per chapter, "why it matters", "where it fits" with App numbers + Week 21 phase, "how to run"), 4 run chapters (id `step01..04`, `group`, `level` beginner→advanced, 2–3-sentence desc), outro concept chapter ("Appendix · where this sits in the twin stack" naming SCENE/STATE/SIMULATION/AGENTS layer + App cross-refs). |
| `static/guide.html` | Copy from template, then edit ONLY content: `<title>`, header `<h1>`/sub, the help-overlay text, and the `FLOW`/`STACK` architecture-diagram definitions (search for `archDiagram(` usage near the bottom script) so the intro shows THIS module's architecture (boxes/arrows renamed; keep the SVG toolkit untouched). Keep the NVIDIA-green theme and all CSS/JS as is. |
| `requirements.txt` | `fastapi`, `uvicorn`, `openai`, + commented optional extras (e.g. `# usd-core  # optional — REAL USD mode`). |
| `README.md` | ~40–60 lines: what the app teaches, chapter list, quick start, REAL-mode upgrades, sources (real URLs from week21/README.md). |

## Style rules (non-negotiable)

- Voice: the Week 23 voice — direct, plain-English explanations of every term, honest about
  maturity/marketing, numbers over adjectives, "Takeaway:" lines, cost/$0 framing.
- Levels: beginner → intermediate → advanced across the 4 chapters.
- Cross-reference other Week 21 apps as "App N" and earlier weeks as "Week N".
- Facts must come from the module brief you were given — do not invent product names, versions,
  or benchmark numbers beyond it. Where the brief flags something unverified, either omit it or
  present it with the same caveat.
- Every claim about NVIDIA products uses the 2026 names: Omniverse = libraries/SDK + Kit App
  Streaming (Launcher/connectors deprecated Oct 2025), Cosmos WFMs, Mega & DSX Blueprints,
  PhysicsNeMo (ex-Modulus), Earth-2, Metropolis/VSS, cuOpt (Apache-2.0), NIM/NeMo.

## Verify before you finish

Run each demo and the server import with the repo venv (fall back to `python3` if absent):

```bash
cd week21/NN_name
for d in demos/step0*.py; do ../../.venv/bin/python "$d" || echo "FAIL $d"; done
../../.venv/bin/python - <<'EOF'
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("ts", pathlib.Path("tutorial_server.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("server import OK; steps:", len(m.STEPS))
EOF
```

All demos must exit 0 in SIM mode. Do not launch the server (ports); import is enough.
