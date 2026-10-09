---
name: dgx-spark-sovereign-stack
description: Week 19 — run an agent stack on NVIDIA DGX Spark (NIM, NeMo Agent Toolkit, local models) for the sovereign tier of Alto Copilot. Use when the student works with DGX Spark, local inference, or sovereign deployment.
when_to_use: DGX Spark setup, local NIM endpoints, sovereign/edge agent deployment, NAT on Spark
---

# Week 19 — DGX Spark sovereign stack

(Template. Port the content of `week19/` here, keeping the structure of the other 20 skills.)

## Concepts
- Why a sovereign tier exists: hotel operators who cannot send telemetry off-site.
- NIM vs. vLLM vs. llama.cpp on a 128 GB unified-memory box.

## Guided lab (kata)
- Goal: swap the MockLLM in `week05` for a local NIM endpoint via an env var; all tests still pass at $0 with MockLLM.
- Rubric (0/1 each): config via env, graceful fallback, latency logged, README updated.
- Pass threshold: 0.8. Then `/fitness-log 19 dgx-spark-sovereign-stack <score>`.
