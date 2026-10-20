---
name: rag-knowledge-agents
description: "Teach Retrieval-Augmented Generation (RAG) — how to ground an LLM in your own documents so it answers from facts instead of guessing, AND how to PROVE the RAG is any good with RAGAS metrics, a golden testset, and a CI/CD regression gate. Covers embeddings, vector search, chunking, the retrieve-then-generate loop, when RAG beats stuffing everything in the prompt, and how to score every change so quality can't silently regress. Use when someone asks 'how do I make the AI answer from my docs/PDFs?', 'how do I know my RAG is accurate / not hallucinating?', mentions embeddings/vector DBs/hallucination/RAGAS/faithfulness/eval, or is reviewing Week 8."
when_to_use: "Learner wants the model to answer from their own documents/knowledge base, asks about embeddings/vector search/RAG/hallucination, wants to measure or regression-test RAG quality (RAGAS, golden testset, CI gate), or is catching up on Week 8."
---

# RAG & Knowledge Agents — Grounding the Model in Your Facts (Week 8)

> **The one idea:** LLMs make things up when they don't know. **RAG** fixes this by *retrieving* the most relevant chunks of *your* documents and pasting them into the prompt before the model answers. The model stops guessing and starts quoting your sources.

```
Question → find the most relevant doc chunks → put them in the prompt → model answers from them
```

---

## Why not just paste the whole document?

- **Context limits & cost** — you can't fit a 500-page manual in every prompt, and you'd pay for all of it every time.
- **Noise** — burying the answer in 500 pages makes the model *worse*, not better.
- RAG sends only the handful of chunks that actually matter for *this* question.

---

## The four moving parts

### 1. Chunking
Split documents into bite-sized pieces (a few paragraphs each). You retrieve *chunks*, not whole files, so the model gets just the relevant slice.

### 2. Embeddings — turning meaning into numbers
An **embedding** is a vector (a list of numbers) that captures the *meaning* of a chunk. Texts about the same topic land near each other in vector space, even with different wording. "How do I reset my password?" sits near "account recovery steps" — no shared keywords needed.

```python
# Conceptual — embed each chunk once, store the vectors
chunks = split_into_chunks(documents)         # ["...", "...", ...]
vectors = embed(chunks)                        # each chunk → [0.12, -0.4, ...]
store(vectors, chunks)                         # in a vector DB (Chroma, FAISS, pgvector, ...)
```

### 3. Vector search (semantic search)
At question time, embed the *question* and find the chunks whose vectors are closest (cosine similarity). Those are your most-relevant passages.

```python
q_vector = embed([question])[0]
top_chunks = store.search(q_vector, k=4)       # 4 nearest chunks
```

### 4. Augmented generation
Paste the retrieved chunks into the prompt and ask the model to answer *using only those*.

```python
context = "\n\n".join(top_chunks)
prompt = f"""Answer using ONLY the context below. If it's not there, say you don't know.

Context:
{context}

Question: {question}"""

answer = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
).content[0].text
```

That instruction — *"use ONLY the context, say 'I don't know' otherwise"* — is what kills hallucination. The model is now a librarian reading from your shelf, not a know-it-all.

> 📁 Class repo: `week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf` — the full lab walkthrough.

---

## The whole RAG flow

```
INDEX (once):   docs → chunk → embed → store in vector DB
                                          │
QUERY (each Q): question → embed → search ┘→ top-k chunks → prompt → grounded answer
```

---

## RAG vs. the other "memory" you'll meet

- **RAG (Week 8)** = retrieve relevant *text/documents* by meaning. Great for manuals, wikis, PDFs, support docs.
- **Knowledge-graph memory (Week 14)** = store *structured facts and relationships* (this room → has → this sensor) and query them with Cypher. Great for connected, evolving state.
- **GraphRAG** combines them: retrieve over a graph instead of a flat pile of text. (See `agent-memory-graphs` and `knowledge-graph-mastery`.)

Different tools for different shapes of knowledge — knowing which to reach for is the skill.

---

## How do I know my RAG is any good?

Building the pipeline is *half* the job. The other half is **proving** it works — and catching the day a "harmless" prompt tweak quietly starts hallucinating. You can't eyeball this; you need a score. The course uses **RAGAS**: it grades a RAG answer on four independent dimensions, 0..1, using an LLM as the judge.

### Part A — The four RAGAS metrics (what each one catches)

You give RAGAS, per question: the `question`, the `ground_truth` (gold answer), the `contexts` your retriever actually returned, and the `answer` your pipeline generated. It scores:

| Metric | Asks | Catches | "Good" |
|---|---|---|---|
| **faithfulness** | Is every claim in the answer supported by the retrieved context? | **Hallucination** — making things up | > 0.90 |
| **answer_relevancy** | Does the answer actually address the question (not waffle)? | Off-topic / padded answers | > 0.85 |
| **context_precision** | Of what we retrieved, how much was relevant & ranked high? | **Over-retrieval** (noise) | > 0.80 |
| **context_recall** | Did retrieval pull *all* the context the gold answer needs? | **Misses** — retriever too weak | > 0.75 |

Reading the scores is diagnostic: **low faithfulness + high relevancy = the model is confidently inventing.** **Low context_recall = your retriever (vector search / Cypher) is the bottleneck — fix retrieval before you touch the prompt.**

```python
# Real shape from the repo: Claude is the judge, local MiniLM embeddings
# (so answer_relevancy doesn't need an OpenAI key).
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                     context_precision, context_recall],
                   llm=eval_llm, embeddings=eval_embeddings)
print(results.to_pandas()[["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"]])
```

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/01_ragas_eval.py` — runs all four metrics over two hotel examples and prints the per-question scores.

### Part B — The golden testset (your regression net)

You can't score a change without questions to score it *against*. A **golden testset** is a curated list of real questions, each with a known-good answer — the ground truth regression testing compares against.

- **Rule of 25–100:** write 25 to 100 questions users *actually ask*, including edge cases. Fewer and you're guessing; more and maintenance hurts.
- **Score EVERY change.** Re-run the whole suite on every prompt/retriever/model edit — same inputs, before and after.
- **Hand-writing is slow and biased** toward cases you already thought of. RAGAS can *read your source docs* and synthesize a diverse set for you, mixing **simple** lookups, **reasoning** (inference across facts), and **multi_context** (stitch multiple chunks — the case GraphRAG wins). The course weights them **40 / 40 / 20** to bias toward the harder, graph-favoring questions, then you **spot-check the CSV** and drop any leaking or unanswerable rows before trusting it.

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/02_testset_generation.py` — generates a synthetic testset with that 40/40/20 mix and writes `hotel_kg_testset.csv` (columns: `user_input`, `reference_contexts`, `reference`, `query_style`, `synthesizer_name`, …).

### Part C — The regression-guard: a CI/CD threshold gate

A score nobody acts on is decoration. The gate makes evaluation **block bad changes from shipping**: it runs your RAG function over the fixed testset, scores it, compares the mean of each metric to a floor, and **exits non-zero if any metric regresses** — failing the build.

```python
# THRESHOLDS = the floor you refuse to ship below (looser than "good" targets).
THRESHOLDS = {"faithfulness": 0.85, "answer_relevancy": 0.80,
              "context_precision": 0.75, "context_recall": 0.70}

scores = {m: float(rdf[m].mean()) for m in THRESHOLDS}
scores["passed"] = all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)
sys.exit(0 if scores["passed"] else 1)   # non-zero → the PR check goes red
```

Wire that one script into a GitHub Actions workflow (`on: pull_request`) and a prompt tweak that tanks faithfulness fails the PR instead of reaching production. The contract is deliberately tiny — `rag_function(question) -> {"answer", "contexts"}` — so you swap in your real pipeline and nothing else changes.

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part4_evaluation/03_cicd_gate.py` — the `THRESHOLDS` dict, the score → `sys.exit` gate, a dummy pipeline so it runs end-to-end, and the full `.github/workflows/graphrag_eval.yml` at the bottom.

**Want the general (not RAG-specific) version of all this?** See `agent-evaluation` — golden datasets, LLM-as-judge bias and calibration, and the multi-gate agent CI/CD pipeline apply to *any* agent, not just retrieval. For RAG over a **graph** (where context_recall lives or dies on your traversal), see `knowledge-graph-mastery`.

---

## When to use RAG (and when not to)
✅ Answers must come from *your* private/large/changing documents.
✅ You need citations / "where did this come from?"
✅ The corpus is too big for the context window.
❌ The question is general knowledge the model already has.
❌ Everything relevant easily fits in the prompt — then just paste it (simpler, no infra).

---

## 🧪 Guided lab (offer this)

Two halves: **build** a tiny "chat with your docs," then **prove it's good** with a $0 RAGAS-style gate (no API key, no vector DB).

### Warm-up (5–10 min, pass/fail)

Stand up a no-infra RAG by hand:

1. **Pick a source** — a README or a few markdown files they care about. **Chunk it** into ~500-word pieces; have them eyeball a chunk so "chunk" stops being abstract.
2. **Store + retrieve** with a plain Python list + cosine similarity (no DB yet). Embed a question, print the top 3 chunks, and confirm the right passages surfaced.
3. **Generate** with the *"use ONLY this context, else say I don't know"* prompt. Ask one question whose answer **is** in the docs and one whose answer **isn't**.

**Pass = ** the in-docs question is answered from a retrieved chunk AND the out-of-docs question gets "I don't know" (not a confident fabrication).

### Skill Drill (15–30 min, runnable, $0)

Score a RAG answer the way `03_cicd_gate.py` does — but with a **MockLLM judge** so it runs offline at no cost. The learner writes the loop, then makes the gate **fail** by feeding a hallucinated answer.

```python
"""Tiny RAGAS-style regression gate — no API key, no vector DB. Runs in <1s."""
import sys

# --- A 3-doc in-memory corpus (the "retriever" just substring-matches) -------
CORPUS = [
    "Room 305 air conditioner AC-305 triggered an HVAC fault alert on 2026-05-20.",
    "Technician Somchai resolved the air-conditioner alert in room 305.",
    "Room 204 device AC-204 triggered a HIGH temperature alert on 2026-05-18.",
]
def retrieve(q):
    kws = [w.lower() for w in q.split() if len(w) > 3]
    return [c for c in CORPUS if any(k in c.lower() for k in kws)] or CORPUS[:1]

# --- MockLLM judge: deterministic stand-in for the RAGAS LLM-as-judge --------
# Real RAGAS asks Claude these same questions; we approximate with set overlap
# so the lab costs $0 and is reproducible.
def _overlap(a, b):
    aw, bw = set(a.lower().split()), set(b.lower().split())
    return len(aw & bw) / max(len(aw), 1)

class MockJudge:
    def faithfulness(self, answer, contexts):      # every claim grounded?
        joined = " ".join(contexts)
        return _overlap(answer, joined)
    def answer_relevancy(self, answer, question):  # on-topic, not waffle?
        return _overlap(answer, question)
    def context_recall(self, contexts, ground_truth):  # did we pull enough?
        return _overlap(ground_truth, " ".join(contexts))

# --- The golden testset (would be 25–100 rows; 2 here) -----------------------
TESTSET = [
    {"question": "Who resolved the alert for room 305?",
     "ground_truth": "Technician Somchai resolved the alert in room 305.",
     "answer": "Technician Somchai resolved the air-conditioner alert in room 305."},
    {"question": "Which room had an HVAC fault on 2026-05-20?",
     "ground_truth": "Room 305 had an HVAC fault on 2026-05-20.",
     "answer": "Room 305 air conditioner AC-305 triggered an HVAC fault alert."},
]

THRESHOLDS = {"faithfulness": 0.70, "answer_relevancy": 0.20, "context_recall": 0.25}

def run_gate(testset):
    judge, totals = MockJudge(), {m: 0.0 for m in THRESHOLDS}
    for row in testset:
        ctx = retrieve(row["question"])
        totals["faithfulness"]    += judge.faithfulness(row["answer"], ctx)
        totals["answer_relevancy"]+= judge.answer_relevancy(row["answer"], row["question"])
        totals["context_recall"]  += judge.context_recall(ctx, row["ground_truth"])
    scores = {m: totals[m] / len(testset) for m in THRESHOLDS}
    scores["passed"] = all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)
    return scores

if __name__ == "__main__":
    s = run_gate(TESTSET)
    for m, floor in THRESHOLDS.items():
        tag = "PASS" if s[m] >= floor else "FAIL"
        print(f"  {m:<18} {s[m]:.2f}  (>= {floor})  [{tag}]")
    print("Overall:", "PASSED" if s["passed"] else "FAILED")
    sys.exit(0 if s["passed"] else 1)   # non-zero fails a CI build
```

Then have the learner:
- **Break faithfulness on purpose:** change one answer to `"Room 999 caught fire."` and re-run — watch faithfulness crater and the gate `sys.exit(1)`. This is the regression a real CI gate catches.
- **Diagnose:** drop a needed doc from `CORPUS` so `retrieve` misses it — watch **context_recall** (not faithfulness) fall, proving the *retriever* is the bottleneck.
- **Map it to reality:** the only swaps to reach `03_cicd_gate.py` are MockJudge → RAGAS+Claude and the inline list → `hotel_kg_testset.csv`.

**Weighted evaluation criteria:**

| # | Criterion | Weight |
|---|---|---|
| 1 | Gate runs, prints per-metric scores, and exits 0 on the clean testset | 25% |
| 2 | Injecting a hallucinated answer makes faithfulness fail and exit non-zero | 25% |
| 3 | Learner explains what each metric catches (faithfulness vs recall vs relevancy) | 20% |
| 4 | Dropping a corpus doc moves **context_recall**, and learner names the retriever as the cause | 20% |
| 5 | Learner states the 25–100 rule and why you score *every* change | 10% |

**Pass = 4/5 criteria** (criterion 2 — the gate actually blocks a regression — is required).

End by zooming out: "RAG retrieves *text*; `knowledge-graph-mastery` retrieves *structured* memory; and `agent-evaluation` applies this same golden-set + threshold-gate discipline to *any* agent, not just retrieval."
