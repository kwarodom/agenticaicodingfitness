---
name: agent-drills
description: "The practice menu — the 'gym workouts' of the bootcamp. A catalog of repeatable drills (warm-up, skill, endurance, sparring, capstone) that train each agent skill, grounded in the 14 real hands-on exercises in week11_models_and_patterns/exercises (ex01–ex14, Beginner→Expert). Use when someone says 'I want to PRACTICE not read', 'give me an exercise/drill/kata', 'how do I get reps on tool use / agent loops / multi-agent', wants to self-grade at $0, or is building a study plan."
when_to_use: "Learner wants hands-on reps instead of theory — a drill to run, a practice menu to pick from, or a way to self-grade an exercise at zero cost."
---

# Agent Drills — The Practice Menu (Gym Workouts)

> **The one idea:** You get fit by *repping drills*, not by reading. Every other skill in this plugin ends with a `🧪 Guided lab` — this skill is the menu that ties them together and the format that lets you self-grade each rep at **$0** with no API key.

The bootcamp already ships a real gym: the **14 hands-on exercises** in `week11_models_and_patterns/exercises` (`ex01`–`ex14`) span Beginner → Expert and *are* drills. This skill is how you choose one, run it, and score yourself.

> 📁 Class repo: `week11_models_and_patterns/exercises/README.md` — the 14 exercises grouped into 5 difficulty tiers (Beginner → Expert) with framework + key concept per exercise.

---

## Part A — The five drill types

Like a training program, drills come in escalating intensity. Match the type to how much time and focus you have *right now*.

| Type | Time | Shape | Pass signal | Example |
|---|---|---|---|---|
| **Warm-up** | 5–10 min | One concept, **binary** pass/fail | A `test_*()` returns True | "Rewrite this vague tool docstring so an LLM can call it" |
| **Skill Drill** | 15–30 min | Build one pattern end-to-end, **runnable**, `MockLLM` = **$0** | Loop completes against a mock | "Build a ReAct agent from a skeleton" |
| **Endurance** | 45–60 min | Fuse **3+ patterns** into one system | Integration test passes | "Router → ReAct worker → Reflection reviewer pipeline" |
| **Sparring** | 45–60 min | Adversarial / paired — **break a peer's agent** | You find/fix the planted bugs | "Race to fix 8 bugs in a buggy agent; red-team an injection" |
| **Capstone** | 2–4 hrs | **Ship something real** end-to-end | Portfolio-ready deliverable runs | "Autonomous research agent with cited report" |

The progression mirrors a workout: warm-ups prime the pattern, skill drills build the muscle, endurance fuses muscles under load, sparring adds adversarial stress, capstones are the graduation lift. Don't skip warm-ups — they're focused, not trivial.

---

## Part B — The catalog

Each row is a drill, the sibling skill it trains, plus time + level. The `ex##` rows are **real, runnable files** in `week11_models_and_patterns/exercises`; the named rows are the flagship `🧪 Guided lab` katas authored into other skills.

| Drill | Trains skill | Type | Level | Where |
|---|---|---|---|---|
| Tool Definition Mastery | `tool-use` | Warm-up | Novice | lab in `tool-use` |
| Pattern Recognition | `models-and-patterns` | Warm-up | Novice | lab in `models-and-patterns` |
| Schema Validation Basics | `tool-use` | Warm-up | Novice | lab in `tool-use` |
| Build-a-ReAct-agent | `agent-loops` | Skill | Practitioner | lab in `agent-loops` |
| LLM-as-judge | `agent-evaluation` | Skill | Practitioner | lab in `agent-evaluation` |
| `interrupt()` + budget cap | `production-and-observability` | Skill | Practitioner | lab in `production-and-observability` |
| Chat-with-your-docs (RAG) | `rag-knowledge-agents` | Skill | Practitioner | lab in `rag-knowledge-agents` |
| **ex01** two-agent dialogue | `multi-agent-systems` | Warm-up | Beginner | `week11_models_and_patterns/exercises/ex01_two_agent_dialogue/` |
| **ex03** LangGraph router | `multi-agent-systems` | Skill | Beginner | `week11_models_and_patterns/exercises/ex03_langgraph_router/` |
| **ex06** parallel fact-checker | `multi-agent-systems` | Skill | Easy | `week11_models_and_patterns/exercises/ex06_parallel_fact_checker/` |
| **ex10** self-healing pipeline | `agent-loops` | Endurance | Advanced | `week11_models_and_patterns/exercises/ex10_langgraph_self_healing/` |
| **ex13** planner-executor-critic | `agent-loops` | Capstone | Expert | `week11_models_and_patterns/exercises/ex13_autonomous_research/` |
| **ex14** selector hotel-ops | `multi-agent-systems` | Capstone | Expert | `week11_models_and_patterns/exercises/ex14_hotel_ops_command_center/` |
| Memory-enabled agent | `agent-memory-graphs` | Skill | Practitioner | lab in `agent-memory-graphs` |
| Cypher graph memory | `knowledge-graph-mastery` | Skill | Specialist | lab in `knowledge-graph-mastery` |
| Build an MCP server | `mcp-and-skills` | Skill | Practitioner | lab in `mcp-and-skills` |
| First LLM call + streaming | `llm-fundamentals` | Warm-up | Novice | lab in `llm-fundamentals` |
| Agent Code Review (paired) | `agent-evaluation` | Sparring | Specialist | red-team a peer's agent |
| Debugging Competition (8 bugs) | `production-and-observability` | Sparring | Specialist | race to fix planted bugs |
| Trace + evaluate a run | `agent-evaluation` | Skill | Practitioner | lab in `agent-evaluation` |

> 📁 Class repo: `week11_models_and_patterns/exercises/ex10_langgraph_self_healing/ex10_langgraph_self_healing.py` — bounded-retry cycle + AI repair node; a perfect Endurance drill for the agent loop. `week11_models_and_patterns/exercises/ex13_autonomous_research/` ships a planner-executor-critic loop with native Claude tool calls (and `graph.png` / `DIAGRAMS.md`) — the Capstone of the set.

**Pick by what you want to train:** want `tool-use` reps? → Tool Definition Mastery. Want `agent-loops`? → Build-a-ReAct-agent → ex10. Want `multi-agent-systems`? → ex01 → ex03 → ex06 → ex14. Want `agent-evaluation`? → LLM-as-judge → trace a run.

---

## Part C — Difficulty progression

The same six **movement patterns** of agentic development, drilled at four escalating levels. Find your current level in a column; the cell tells you the next rep.

| Movement | Novice (Warm-up) | Practitioner (Skill) | Specialist (Endurance/Sparring) | Architect (Capstone) |
|---|---|---|---|---|
| **Prompting** | refine a vague prompt | reflection / planning prompts | prompt chains in a pipeline | multi-agent prompt orchestration |
| **Tool use** | fix a tool schema (ex none) | ReAct tools, MCP server | multi-tool pipelines | tool marketplace design |
| **Evaluation** | schema validation | unit tests, LLM-as-judge | quality gates / CI eval | full eval platform |
| **Orchestration** | spot the pattern (ex01) | single-pattern build (ex03) | multi-pattern fusion (ex10) | enterprise system (ex14) |
| **Memory** | first message history | memory-enabled agent | graph memory queries | durable multi-agent memory |
| **Debugging** | spot a bug in a snippet | 5-bug challenge | 8-bug sparring race | production incident sim |

**Weekly time-investment guide** (one focused, uninterrupted session per slot):

| Slot | Drills | Budget |
|---|---|---|
| Weekday mornings | 1 Warm-up (5–10 min) | ~1 hr/week |
| Weekday evenings | 1 Skill Drill (15–30 min) | ~3 hrs/week |
| Weekend | 1 Endurance **or** Sparring | ~1.5 hrs/week |
| Monthly | 1 Capstone (2–4 hrs) | ~3 hrs/month |

That cadence finishes the core library in ~6 weeks — a fitness *mesocycle*: progressive overload in 4-week blocks, then a lighter recovery week. Short on time? The minimum viable path: all warm-ups → 4 skill drills (ReAct, router, MCP, eval) → 1 endurance → 1 capstone ≈ 7.5 hrs, and you ship one portfolio piece.

---

## Part D — The drill template

Every drill self-grades at **$0**: a `MockLLM` stub (so it runs with no API key), a **weighted rubric**, and an **explicit pass threshold**. Copy this skeleton when you author or run a drill.

```python
# drill.py — runs at $0, no API key needed
class MockLLM:
    """Canned responses keyed on the prompt — lets you test the whole loop free."""
    def complete(self, prompt: str) -> str:
        if "plan" in prompt.lower():
            return "1. search 2. summarize 3. answer"
        if "TOOL:" in prompt:                 # the agent decided to act
            return "TOOL: calculator(2+2)"
        return "FINAL: 4"                     # the agent decided to stop

def run_agent(goal: str, llm) -> str:
    """The pattern under test — e.g. a tiny ReAct loop."""
    for _ in range(5):                        # always bound the loop
        out = llm.complete(goal)
        if out.startswith("FINAL:"):
            return out.removeprefix("FINAL:").strip()
    return "FAILED: no termination"

# ---- self-grading rubric (weights MUST sum to 100) ----
def grade(answer: str) -> dict:
    return {
        "terminates":      (25, not answer.startswith("FAILED")),   # loop is bounded
        "uses_tool":       (20, "4" in answer),                     # acted, not guessed
        "correct_output":  (30, answer == "4"),                     # right answer
        "bounded_steps":   (15, True),                              # range(5) guard exists
        "no_api_key":      (10, True),                              # ran on MockLLM
    }

if __name__ == "__main__":
    ans = run_agent("What is 2+2?", MockLLM())
    rows = grade(ans)
    score = sum(w for w, ok in rows.values() if ok)
    for name, (w, ok) in rows.items():
        print(f"[{'x' if ok else ' '}] {name} ({w} pts)")
    print(f"SCORE: {score}/100  →  {'PASS' if score >= 80 else 'RETRY'}")
```

Rules every drill follows:

1. **MockLLM keyed on prompt** — never call a real API to grade yourself. $0, deterministic, offline.
2. **Weighted rubric summing to 100** — weight the *failure points* heaviest (termination, correct routing, security), not cosmetics.
3. **One explicit pass threshold** — e.g. *Pass = 80/100* (or *Pass = 4/5 criteria*). No ambiguity about "done."
4. **Bound every loop** — a `range(N)` or `retry_count` guard. ex10's bounded-retry cycle is the canonical example.

Cross-links: `llm-fundamentals` · `tool-use` · `agent-loops` · `mcp-and-skills` · `rag-knowledge-agents` · `multi-agent-systems` · `agent-memory-graphs` · `knowledge-graph-mastery` · `models-and-patterns` · `production-and-observability` · `agent-evaluation`.

---

## 🧪 Guided lab (offer this): pick a drill, run it, self-grade

**Warm-up (5–10 min, binary).** From Part B, pick one drill by **skill + time + level** (e.g. "I want `agent-loops`, ~20 min, Practitioner → Build-a-ReAct-agent"). Write the one sentence: *"This drill trains `<skill>` at `<level>` and I pass when `<threshold>`."* Pass = you named the skill, the type, and a concrete pass threshold.

**Skill Drill (15–30 min, $0).** Open the real file `week11_models_and_patterns/exercises/ex03_langgraph_router/ex03_langgraph_router.py` (a router) **or** run the Part D template as-is. Then add ONE new branch/tool and re-grade with the rubric. It must run with **no API key** — swap the model call for `MockLLM` if needed. Print the `SCORE: n/100 → PASS/RETRY` line.

**Weighted evaluation criteria (100 pts):**
| Criterion | Pts | Pass when |
|---|---|---|
| Picked a drill by skill **and** time **and** level | 20 | named all three |
| Ran something at **$0** (MockLLM, no API key) | 25 | program runs offline |
| Rubric weights sum to **100** and weight the failure points | 20 | heaviest weight on the risky step |
| **Explicit pass threshold** printed | 20 | `PASS/RETRY` line emitted |
| Self-scored honestly + named your **next** drill | 15 | wrote the next rep |

**Pass threshold: 80/100 (4 of 5 criteria).** Below that, retry — that's the drill working as designed.

End on the meta-lesson: frameworks churn, but the *reps* build the judgment you keep. Do one warm-up a day; the catalog in Part B is your training plan.
