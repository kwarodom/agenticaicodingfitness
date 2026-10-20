# 09 · Inference Economics — AI Performance & Evaluation

The meter on the whole stack: tokens are the unit of AI work, so this app teaches cost per million tokens (open-on-your-DGX vs a hosted cloud API), throughput per GPU and per Megawatt (power, not chip count, is the ceiling at scale), and the crucial upgrade from raw tokens to **goodput** — cost per SUCCESSFUL task — which makes evaluation (LLM-judge + golden set) part of performance. REAL mode measures your live endpoint's tok/s and derives the dollars; SIM mode teaches every formula with no GPU at $0.

Launch:

```bash
.venv/bin/python week23/09_inference_economics/tutorial_server.py   # → http://127.0.0.1:8108
```

Hands-on side: **[TUTORIAL.md](TUTORIAL.md)** — measure your own tok/s, do the $/M-token math on your wall power, and run the three labs in `labs/` (streamed tok/s → $/Mtok, a live goodput bench, and an LLM-judge you score against ground truth).
