# ▣ Week 23 · App 1 — Nemotron Open Models

The **MODEL** layer of "Agent = Model + Harness": NVIDIA's open **Nemotron 3** family
— *built for long-running, self-evolving agents* — explored as a runnable, dual-mode
web tutorial (Week 19 style).

```
Nano 30B-A3B · Super 120B-A12B · Ultra 550B-A55B  (+ RAG · Speech · Safety)
hybrid Mamba-Transformer MoE · 1M-token context · RL-post-trained reasoning + tool-calling
```

## Run
```bash
uv pip install -r week23/01_nemotron_models/requirements.txt
.venv/bin/python week23/01_nemotron_models/tutorial_server.py      # → http://127.0.0.1:8100
```
**🔌 Connection** panel: point at a Nemotron endpoint (Ollama `nemotron-3-*`, a NIM,
or a DGX via tunnel). No endpoint → **SIM** (no GPU). Pick a model in the dropdown.

## Chapters
1. *(concept)* Agent = Model + Harness — why Nemotron for sovereign agents
2. The family — Nano/Super/Ultra + RAG/Speech/Safety; fit on 1 vs 2 Sparks
3. Hybrid Mamba-Transformer MoE + 1M context — why it's efficient for long runs
4. Reasoning (RLM) — watch REASON → ANSWER, live or simulated
5. Tool-calling — Nemotron as a sovereign sub-agent
6. Run it on the DGX — Nano/Super on 1 Spark; Ultra across 2 Sparks (QSFP 200GbE)

## Layout
`config.py` (connection switch) · `ntsim.py` (family registry + sim) · `ntview.py`
(reason engine, real+sim) · `tutorial_server.py` (:8100) · `static/guide.html` · `demos/`.

Part of **Week 23 — the Open Superintelligence Stack**; the harness apps (NIM, Dynamo,
Data Flywheel, NeMo Gym, OpenShell) build on this model layer.
