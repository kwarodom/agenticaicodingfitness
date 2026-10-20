# ▣ Week 19 · App 4 — Self-Evolving Agent v2 (sovereign, on a DGX)

The Week 18 self-evolving agent, made **sovereign** — with a **switchable brain**
and a tripartite memory that lives **on the DGX**. The capstone of Week 19: a model
you run (App 1), tuned to your domain (App 2), observable (App 3), that **learns
over time** — and whose mind *and* memories never leave the building.

```
← RECALL (on-DGX memory) → ~ THINK (switchable brain) → → REMEMBER (episodic)
        → CONSOLIDATE → durable facts + self-written skills ↺ each run starts smarter
```

## The two big ideas

1. **Switchable brain.** The same agent + memory engine thinks with whatever you set:
   ```bash
   export BRAIN=local     # DGX / Ollama model (OpenAI-compatible) — sovereign default
   export BRAIN=claude    # Anthropic Claude  (needs ANTHROPIC_API_KEY)
   export BRAIN=sim       # scripted stub — no model, $0, fully offline
   export BRAIN=auto      # local if up, else claude, else sim (default)
   ```
   The memory engine never changes — it's brain-agnostic. And it makes the
   sovereignty cost explicit: only `local` keeps your prompts on the box.

2. **Sovereign memory.** A cloud agent's memory sits in a vendor's database. Here
   it's files on the DGX:
   - **Episodic** → `.memory/episodes.jsonl` (raw log of what happened)
   - **Semantic** → `.memory/MEMORY.md` (consolidated durable facts)
   - **Procedural** → `.memory/skills/` (reusable skills the agent writes itself)

---

## Quick start

```bash
uv pip install -r week19/self_evolving_agent_v2/requirements.txt

# pick a brain (or rely on auto):
ollama run qwen3.6:35b-a3b-q8_0          # then BRAIN=local (sovereign)
# export ANTHROPIC_API_KEY=...           # then BRAIN=claude
# (nothing running → BRAIN=sim, $0)

.venv/bin/python week19/self_evolving_agent_v2/tutorial_server.py
# → open http://127.0.0.1:8095   (run Ch 3→4→5→6 in order to watch it evolve)

# or a single demo
BRAIN=local .venv/bin/python week19/self_evolving_agent_v2/demos/step03_consolidate.py
```

The 🧹 button (or `POST /api/cleanup`) wipes `.memory/` so you can watch the agent
grow from amnesiac to expert again.

---

## Layout

```
week19/self_evolving_agent_v2/
├── README.md · requirements.txt · config.py
├── brain.py            → the SWITCHABLE BRAIN (local DGX / Claude / sim)
├── memory.py           → tripartite memory on the DGX + consolidation + recall
├── sevview.py          → recall → think → remember framing
├── tutorial_server.py  → FastAPI control plane (cleanup = reset memory)
├── static/guide.html   → clickable UI with live brain + memory pills
└── demos/
    ├── step01_brain_switch.py     Ch 2 · the switchable brain
    ├── step02_episodic.py         Ch 3 · episodic memory (log what happened)
    ├── step03_consolidate.py      Ch 4 · the 'sleep' loop → facts + a skill
    ├── step04_skills.py           Ch 5 · the skill the agent wrote for itself
    ├── step05_evolve.py           Ch 6 · prove it evolved — cold vs warm
    └── step06_sovereign_audit.py  Ch 7 · brain + memory both on the DGX
```

---

## The 7 chapters

| Ch | Demo | The one thing to notice |
|----|------|--------------------------|
| 1  | *(concept)* | Switchable brain + sovereign memory = a self-evolving agent you own. |
| 2  | `step01` | One env var swaps the brain; the **memory engine is identical**. |
| 3  | `step02` | Every run appends to the **episodic** log on the DGX. |
| 4  | `step03` | **Consolidation** distills episodes → durable facts + a reusable skill. |
| 5  | `step04` | The agent **wrote its own skill** — files you can review + version. |
| 6  | `step05` | **Warm beats cold**: recall makes the next run answer with house policy. |
| 7  | `step06` | Audit: **brain + memory both on the DGX** → fully sovereign. |

---

## Where this sits in Week 19

```
App 1 sovereign_dgx → the model it thinks with runs on the DGX
App 2 dgx_finetune  → those weights can be tuned to your domain
App 3 dgx_observability → its agent loop is traceable (Phoenix + NAT)
App 4 (THIS) self_evolving_agent_v2 → it LEARNS, brain + memory on the DGX
```

This continues the Week 18 self-evolving agent (which also has garbage collection,
GEPA-style prompt evolution, and a live visualizer). v2's contribution: prove the
architecture is **brain-agnostic** (swap local↔Claude) and make the whole thing —
including the memory that is your most valuable asset — **sovereign on a DGX**.
