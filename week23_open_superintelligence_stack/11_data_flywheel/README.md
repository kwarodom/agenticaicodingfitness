# App 11 · NeMo Data Flywheel — self-evolving agents

**Agent = Model + Harness.** This app is the **self-evolving** layer: the NVIDIA
**NeMo Data Flywheel** turns production traffic into a model that gets **cheaper AND
better** over time — without your data leaving the DGX. NeMo Relay names the loop:
**observe → learn → optimize**.

```
① OBSERVE → ② CURATE → ③ CUSTOMIZE → ④ EVALUATE → (promote) → ①…
 prod logs   Curator    Customizer     Evaluator
             dedup/PII   LoRA/SFT/      LLM-judge A/B
             /label      DPO/GRPO       promote the winner
```

Runs **dual-mode**: **REAL** against a NIM / Ollama / vLLM endpoint, or **SIM** with no
GPU (`week23/11_data_flywheel/sim.py`).

## Run the web tutorial

```bash
cd week23/11_data_flywheel
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8110
```

Open **http://localhost:8110**. Use **🔌 Connection** to point at your DGX, or leave it
in SIM. Pick a model with the model picker.

## Run the chapters standalone

```bash
python demos/step01_the_loop.py             # the observe→curate→customize→evaluate loop
python demos/step02_curate.py               # NeMo Curator: logs → clean training data
python demos/step03_customize.py            # NeMo Customizer: distill teacher → student
python demos/step04_evaluate_promote.py     # NeMo Evaluator: A/B + promote the winner
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | What is the Data Flywheel? | beginner |
| 2 | The flywheel loop | beginner |
| 3 | Curate — logs into training data | intermediate |
| 4 | Customize — distill teacher→student | advanced |
| 5 | Evaluate + promote | advanced |

## Where this sits in Week 23

App 1 **Nemotron** (model) · App 2 **NIM** (serve) · **App 11 (this) Data Flywheel**
(improve) · App 3 **Dynamo** (scale) · App 10 **NeMo Gym** (RL) · App 7 **OpenShell** (guard).
