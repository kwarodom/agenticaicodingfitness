# App 3 · NVIDIA Dynamo — serving long-running agents at scale

**Agent = Model + Harness.** This app is the **serve-at-scale** layer: **NVIDIA
Dynamo**, the distributed inference framework that keeps a **long-running** agent
economical — disaggregated prefill/decode, KV-cache-aware routing, an SLO Planner, and
NIXL for fast cross-GPU KV transfer — across 1 or many DGX Sparks.

Runs **dual-mode**: **REAL** against a NIM / Ollama / vLLM endpoint, or **SIM** with no
GPU (`week23/03_dynamo_serving/sim.py`).

## Run the web tutorial

```bash
cd week23/dynamo_serving
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8102
```

Open **http://localhost:8102**. Use **🔌 Connection** to point at your DGX, or leave it
in SIM. Pick a model with the model picker.

## Run the chapters standalone

```bash
python demos/step01_what_is_dynamo.py       # the four pieces of Dynamo
python demos/step02_disaggregated.py        # disaggregated P/D + cache-aware routing
python demos/step03_slo_planner.py          # SLO Planner autoscaling under load
python demos/step04_token_economics.py      # cost per 1M tokens, per GPU, per MW
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | What is NVIDIA Dynamo? | beginner |
| 2 | Disaggregated + cache-aware | intermediate |
| 3 | SLO Planner — hold latency | advanced |
| 4 | Token economics | advanced |

## The scaling story

- **1 DGX Spark** — a single serving pool for a Nano/Super-class model.
- **2 DGX Sparks over QSFP 200GbE** — disaggregated prefill/decode across nodes, with
  **NIXL** moving KV-cache between them; the SLO Planner scales each pool independently.

## Where this sits in Week 23

App 1 **Nemotron** (model) · App 2 **NIM** (serve) · App 11 **Data Flywheel** (improve) ·
**App 3 (this) Dynamo** (scale) · App 10 **NeMo Gym** (RL) · App 7 **OpenShell** (guard).
