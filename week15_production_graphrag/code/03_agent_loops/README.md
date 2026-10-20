# 03 · Agent Loops (Weeks 4–5)

> Skill: `agentic-coding-fitness:agent-loops`
> **One idea:** an agent is a loop — REASON → ACT → OBSERVE → repeat — until it declares itself done.

Files:

```bash
python 01_weather_agent.py       # multi-step goal; read the 💭/🔧/📋 trace
python 02_code_review_agent.py   # the Agent reads, fixes & re-runs a buggy file
```

- `agent.py` — the reusable ~40-line `Agent` class (import it; don't run it directly).
- `02_...` writes a fresh `buggy_sample.py` each run, so you can re-run freely.

### Things to try
- In `01_...`, set `max_iterations=1` and rerun → it can't finish. That's the seatbelt. Restore to 10.
- Notice the only thing that changes between the two agents is the **tools + system-prompt playbook**. The loop is identical.

**Chatbot vs. agent:** a chatbot answers once; an agent keeps going until the goal is met. **Carry forward:** one agent is powerful — folder 06 runs *several* together.
