# 08 · Models & Patterns (Week 11)

> Skill: `agentic-coding-fitness:models-and-patterns`
> **One idea:** building agents well is mostly *judgment* — which model, which pattern, which framework for *this* problem.

```bash
python 01_reflection.py     # draft → critique → improve (the Reflection pattern)
python 02_model_chooser.py  # rules-of-thumb engine for picking a model + a decision flow
```

- `01_reflection.py` makes a real API call and shows an agent improving its own
  output — one of the cheapest ways to raise quality.
- `02_model_chooser.py` makes **no** API calls — it's a thinking tool. Edit the
  `SCENARIOS` list with your own use cases and see the reasoning.

### The patterns you've now built across this course
ReAct (03) · Tool Use (02) · Reflection (08) · Sequential / Swarm / Router (06)
· RAG (05) · GraphRAG + Event Sourcing (07). Frameworks and models change every
few months — **the patterns and the judgment are what you keep.**
