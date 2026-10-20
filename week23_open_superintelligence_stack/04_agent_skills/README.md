# 04 · NVIDIA Agent Skills — connect frontier agents to your business

Week 23 · The Open Superintelligence Stack, app 04 of 12. Teaches NVIDIA Agent Skills (github.com/NVIDIA/skills): portable, framework-agnostic capability packages — a `SKILL.md` of instructions plus tools — that any frontier agent (Claude, GPT, Gemini, or an on-box Nemotron) discovers by metadata and loads on demand. You browse the catalog, watch progressive disclosure keep an agent's context small, connect a NeMo Retriever-style skill to your sovereign data, and see the same skill ride MCP (agent→tools) and A2A (agent→agent) unchanged — write once, load anywhere.

Launch the interactive explainer (REAL against your Spark/Ollama/build.nvidia.com endpoint, or SIM with no GPU):

```bash
.venv/bin/python week23/04_agent_skills/tutorial_server.py   # → http://127.0.0.1:8103
```

Then do it yourself: **[TUTORIAL.md](TUTORIAL.md)** is the hands-on side — author a skill, load it into a live model, and drive it over MCP + A2A via the scripts in `labs/`.
