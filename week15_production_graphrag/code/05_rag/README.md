# 05 · RAG — Knowledge Agents (Week 8)

> Skill: `agentic-coding-fitness:rag-knowledge-agents`
> **One idea:** retrieve the most relevant chunks of *your* docs, paste them into the prompt, and the model answers from facts instead of guessing.

```bash
python 01_chat_with_docs.py
```

It indexes the markdown in `sample_docs/`, then answers three questions — two
that ARE in the docs and one that **isn't** (password reset). Watch it correctly
say *"I don't know based on the provided documents."* That honesty comes from
the **"use ONLY the context"** instruction in the prompt — the anti-hallucination trick.

### About the "embeddings" here
To stay dependency-free, retrieval uses **TF-IDF + cosine similarity** in pure
Python. Real systems use a neural embedding model + a vector DB
(Chroma/FAISS/pgvector) so they match by *meaning*, not just shared words — but
the four-step flow (chunk → embed → retrieve → generate) is exactly the same.

### Things to try
- Drop your own `.md` into `sample_docs/` and ask about it.
- Lower `k` to 1 in `retrieve(...)`, or delete the "use ONLY the context" line in
  the prompt, and watch answer quality / hallucination change.

**Carry forward:** RAG retrieves *text*. Folder 07 gives agents *structured* memory in a knowledge graph (and GraphRAG = RAG over a graph).
