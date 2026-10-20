# GraphRAG Failure-Mode Taxonomy (kg_mastery.pdf §4.3)

When RAGAS scores are low, the score tells you *which dimension* is broken; this
taxonomy tells you *why* and *how to fix it*. Diagnose the lowest metric first,
map it to a failure mode below, then make one targeted change (see §4.4
improvement loop).

| Failure Mode | Root Cause | Symptom | Fix |
|---|---|---|---|
| **Cypher Syntax Error** | LLM emits invalid Cypher (bad label, missing clause, wrong function) | Query throws / returns nothing; high error rate in monitoring | Give the LLM the live schema; add few-shot Cypher examples; validate & retry generated queries |
| **Hallucinated Entities** | LLM invents nodes/relationships not in the graph | Answer cites rooms/devices/people that don't exist | Lower temperature; constrain generation to schema; ground the answer strictly in returned rows |
| **Wrong Relationship Direction** | Query traverses `(a)<-[:R]-(b)` instead of `(a)-[:R]->(b)` | Returns the wrong side of a relationship (e.g. who reports to whom inverted) | Document directionality in the schema prompt; add directional examples; test traversal both ways |
| **Missing Context** | Retrieval doesn't pull all nodes needed to answer | Low `context_recall`; answer omits known facts | Add graph traversal / multi-hop expansion; raise top-k; widen the Cypher pattern; hybrid (vector + graph) retrieval |
| **Over-Retrieval** | Too many irrelevant nodes/rows returned | Low `context_precision`; noisy, padded answers; higher latency/cost | Tighten the Cypher (filters, `LIMIT`, ranking); rerank retrieved context; remove redundant chunks |
| **Low Faithfulness** | Generator adds claims not supported by retrieved context | High `answer_relevancy` but low `faithfulness`; confident hallucination | Add an "answer only from the provided context" guardrail; cite sources; reduce temperature; verify claims against context |

## Metric -> likely failure mode (quick map)

- Low **faithfulness** -> Low Faithfulness, Hallucinated Entities
- Low **answer_relevancy** -> prompt/answer-shaping issue (model dodges the question)
- Low **context_precision** -> Over-Retrieval
- Low **context_recall** -> Missing Context, Wrong Relationship Direction
- Query errors / empty results in monitoring -> Cypher Syntax Error

## The Improvement Loop (§4.4)

```
        +-----------+
        |  DESIGN   |   pick schema + retrieval strategy for the question types
        +-----+-----+
              |
        +-----v-----+
        |   BUILD   |   loaders, Cypher templates, prompts
        +-----+-----+
              |
        +-----v-----+
   +--->| EVALUATE  |   run RAGAS over the test set -> 4 scores
   |    +-----+-----+
   |          |
   |    +-----v-----+
   |    | DIAGNOSE  |   lowest metric -> failure mode (table above)
   |    +-----+-----+
   |          |
   |    +-----v-----+
   |    |  IMPROVE  |   ONE targeted change for that failure mode
   |    +-----+-----+
   |          |
   +----------+         REPEAT: keep the change only if scores rise
```

Track the progression with `05_improvement_loop.py`, which logs each
iteration's four scores plus the action taken.
