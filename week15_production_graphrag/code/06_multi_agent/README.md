# 06 · Multi-Agent Systems (Week 9)

> Skill: `agentic-coding-fitness:multi-agent-systems`
> **One idea:** a team of focused agents + a coordination pattern beats one do-everything agent.

The same idea, three patterns — build all three to make them muscle memory:

```bash
python 01_sequential.py      # assembly line: Researcher → Writer → Editor
python 02_parallel_swarm.py  # 5 specialists at once (asyncio) + speedup report
python 03_router.py          # classify a ticket → route to the right specialist
```

```
Sequential:  A → B → C
Parallel:    A,B,C all at once → aggregate
Router:      classify → (billing | technical | general)
```

These use the **raw Anthropic SDK** so they always run. The class repo shows
the framework versions: `week9/ex1_crewai_sequential.py` (CrewAI),
`week9/ex2_LangGraphSupportGraph.py` (LangGraph), `week9/ex3_ParallelSwarm.py`.

### Don't over-build
Multi-agent adds latency, cost, and coordination bugs. Reach for it only when
sub-tasks are genuinely distinct, parallelizable, or need different expertise.
Most problems are fine with one focused agent (folder 03).

**Carry forward:** give a shared memory to a team and the graph becomes their blackboard → folder 07.
