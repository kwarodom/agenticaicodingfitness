# App 10 · NeMo Gym + NeMo RL — agents that learn from outcomes

**Agent = Model + Harness.** This app is the **learning** layer: NVIDIA **NeMo Gym**
(verifiable-reward *environments* — a task + a programmatic *verifier* that returns a
reward) + **NeMo RL** (the trainer that runs **GRPO**, Group Relative Policy
Optimization, to post-train Nemotron models). Outcome-based rewards — did the code /
tool-call / math answer **pass**? — beat human preference for coding, math, and tool-use.

Runs **dual-mode**: **REAL** against a Nemotron / Ollama / vLLM OpenAI endpoint, or **SIM**
with no GPU (`week23/10_nemo_gym_rl/sim.py`).

## Run the web tutorial

```bash
cd week23/nemo_gym_rl
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8109
```

Open **http://localhost:8109**. Use **🔌 Connection** to point at your DGX
(`http://<dgx>:8000/v1`) or leave it in SIM. Pick a model with the model picker.

## Run the chapters standalone

```bash
python demos/step01_verifiable_reward.py   # what is verifiable-reward RL?
python demos/step02_define_environment.py  # define an environment (task + verifier)
python demos/step03_grpo_training.py       # GRPO training loop on the DGX (live/sim)
python demos/step04_multienv_evaluate.py   # multi-env RL + evaluate (live/sim)
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | What is verifiable-reward RL? | beginner |
| 2 | Define an environment (task + verifier) | intermediate |
| 3 | GRPO training loop on the DGX | advanced |
| 4 | Multi-environment RL + evaluate | advanced |

## The loop (GRPO)

`rollout` (sample a GROUP per prompt) → `reward` (verifier scores each: 1.0 pass / 0.0) →
`advantage` (reward − group mean, the group-relative baseline) → `policy update`. Repeat
over N rounds; mean reward climbs (e.g. 0.30 → 0.80). Runs on **1 DGX Spark** for a small
policy (Nemotron Nano), or **2 Sparks** over QSFP 200GbE (`cluster.tp=2`) for a larger one.

## Where this sits in Week 23

App 1 **Nemotron** (model) · App 2 **NIM** (serve) · App 11 **Data Flywheel** (improve) ·
App 3 **Dynamo** (scale) · **App 10 (this) NeMo Gym** (RL) · App 7 **OpenShell** (guard).

NeMo Gym / NeMo RL are open; production use runs on **NVIDIA AI Enterprise** (bundled with
DGX). Evaluation is the promotion gate, and the **Data Flywheel** (App 11) is the engine that
mines new tasks and keeps the policy improving.
