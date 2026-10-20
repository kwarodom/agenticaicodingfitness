# 05 · AI-Q Research Lab — router → deep agent → researcher fan-out

App 05 of Week 23 (The Open Superintelligence Stack). Teaches the **NVIDIA AI-Q Open Agent Blueprint**: an Intent Router (Nemotron Nano) triages every query shallow-vs-deep, a Deep Agent plans and decomposes the hard ones, parallel Researcher sub-agents (Nemotron Super) fan out through the NeMo Agent Toolkit's tool bus, and findings synthesize into a cited answer. Runs REAL against any OpenAI-compatible Nemotron endpoint (DGX Spark, build.nvidia.com, local Ollama) or in SIM with no GPU — either way the same architecture, $0 by default.

Launch the interactive explainer:

```bash
.venv/bin/python week23/05_aiq_research_lab/tutorial_server.py   # → http://127.0.0.1:8104
```

Then do it by hand: **[TUTORIAL.md](TUTORIAL.md)** — the hands-on lab (curl the router yourself, measure routing economics, fan out real researchers, close a tool loop, install the NeMo Agent Toolkit). Lab scripts live in `labs/`.
