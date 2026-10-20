# 08 · NeMo Relay — observe, learn & optimize a long-running agent

Week 23 · The Open Superintelligence Stack · app 08 of 12. This app teaches the observe → learn → optimize layer that sits under a long-running agent: NeMo Relay captures every tool and LLM call as a telemetry span, Agent Insights (Arize Phoenix) renders them as a trace tree with per-span status, latency and cost, a Router/Gateway right-sizes the model per request (easy → Nano/Mini, hard → the big model), and OpenTelemetry export fans the same spans out to Phoenix / Datadog / LangSmith — feeding App 11's Data Flywheel. Note: NeMo Relay itself is early/limited availability, so the app teaches its flow in SIM against the real Phoenix + OTel substrate you can run today. Works in REAL mode against any OpenAI-compatible endpoint (local Ollama, a DGX Spark over a tunnel, or build.nvidia.com), or in SIM with no GPU at $0.

Launch the interactive explainer:

```bash
.venv/bin/python week23/08_nemo_relay/tutorial_server.py   # → http://127.0.0.1:8107
```

Then do it for real: **[TUTORIAL.md](TUTORIAL.md)** is the hands-on side — build a minimal relay, stand up a live Phoenix on :6006, write and measure a right-sizing router, and export a real OTel trace (labs in `labs/`).
