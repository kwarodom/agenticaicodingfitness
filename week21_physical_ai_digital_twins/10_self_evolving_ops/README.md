# ◳ 10 · The self-evolving operator — agents that learn your building, safely

Week 21 · App 10 of 12 · Phase 5: **the AGENTS** · port **8209**

An App-9 policy that aces the gym still meets a building that *drifts* — seasons turn,
coils foul, tenants change. This app turns the stateless controller into a **learning
operator**: Week 18's tripartite memory + Week 23's data flywheel, applied to building
controls — and then, the part that decides whether any of it ships, the **safe-autonomy
ladder** with hard guardrails.

## Chapters

| Ch | Demo | Level | What you learn |
|----|------|-------|----------------|
| 1 | — (concept) | beginner | Why fixed policies decay, and the map of this app. |
| 2 | `step01_tripartite_memory.py` | beginner | EPISODIC control episodes → a CONSOLIDATION pass distils SEMANTIC facts ("ballroom thermal lag ≈ 45 min") + a PROCEDURAL playbook ("morning-startup-monsoon-season"). |
| 3 | `step02_override_flywheel.py` | intermediate | Every operator override is a labeled preference (chosen ≻ rejected — DPO-shaped data). Curate → customize → evaluate → promote, and the compound-returns table: 4 corrections on run 1, 0 by run 5. |
| 4 | `step03_autonomy_ladder.py` | advanced | OFFLINE → SHADOW → ADVISORY → SUPERVISED AUTONOMY, with printed gate criteria per rung. Constrain the **action space**, not just the reward — clamps and masking beat penalty terms. Honest calibration: DeepMind's data-center cooling (~40 %, later safety-bounded); advisory/tightly-bounded is the deployed norm. |
| 5 | `step04_guardrailed_week.py` | advanced | 7 simulated days at rung 4: a frozen sensor trips the watchdog into the ASHRAE Guideline 36 fallback; an override flows into memory; the LLM writes the postmortem (REAL if an endpoint is up). |
| — | Appendix (concept) | all | Where this sits: the AGENTS layer of Twin = SCENE + STATE + SIMULATION + AGENTS. |

## Quick start

```bash
uv pip install -r week21/10_self_evolving_ops/requirements.txt
.venv/bin/python week21/10_self_evolving_ops/tutorial_server.py   # → http://127.0.0.1:8209
```

Every chapter runs in **SIM** with no GPU, $0. Run any demo standalone:
`.venv/bin/python week21/10_self_evolving_ops/demos/step01_tripartite_memory.py`.

**REAL-mode upgrades:** an Ollama / vLLM / NIM / DGX endpoint (🔌 Connection panel or
`DGX_BASE_URL`) makes Ch 5's postmortem a genuine agent-written note; `pip install sinergym`
to train the underlying policy against real EnergyPlus before rung 1.

## Where it fits

Prereq: **App 09** (the policy that climbs the ladder). Feeds: **App 12** (the capstone's
learning operator). Bridges: **Week 18** (tripartite memory, consolidation, compound
returns) and **Week 23 App 11** (the data flywheel) — same loops, now wearing a hard hat.

## Sources

- Week 18 `week18/` (self-evolving agents) · Week 23 [`week23/11_data_flywheel/`](../../week23/11_data_flywheel/)
- [Sinergym](https://github.com/ugr-sail/sinergym) · [BOPTEST](https://ibpsa.github.io/project1-boptest/) (the sim rungs)
- ASHRAE Guideline 36 (the high-performance sequences the fallback runs)
- DeepMind × Google data-center cooling (the canonical bounded-autonomy story)
