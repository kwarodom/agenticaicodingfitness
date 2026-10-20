# App 2 · NIM Microservices — sovereign inference in one container

**Agent = Model + Harness.** This app is the **runtime** layer: NVIDIA **NIM**
(NVIDIA Inference Microservice) — a model + an auto-selected optimized engine
(TensorRT-LLM / vLLM / SGLang) + an OpenAI-compatible API, all in **one signed
container** that runs on your DGX. One `docker run` → a sovereign production endpoint.

Runs **dual-mode**: **REAL** against a NIM / Ollama / vLLM OpenAI endpoint, or **SIM**
with no GPU (`week23/02_nim_microservices/sim.py`).

## Run the web tutorial

```bash
cd week23/nim_microservices
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8101
```

Open **http://localhost:8101**. Use **🔌 Connection** to point at your DGX
(`http://<dgx>:8000/v1`) or leave it in SIM. Pick a model with the model picker.

## Run the chapters standalone

```bash
python demos/step01_deploy_nim.py       # deploy a NIM (one command)
python demos/step02_nim_vs_diy.py       # NIM vs raw vLLM vs Ollama
python demos/step03_call_nim.py         # call it — same OpenAI API (live/sim)
python demos/step04_catalog_custom.py   # the catalog + wrap YOUR model as a NIM
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | What is a NIM? | beginner |
| 2 | Deploy a NIM (one container) | beginner |
| 3 | NIM vs raw vLLM vs Ollama | intermediate |
| 4 | Call a NIM — same OpenAI API | intermediate |
| 5 | The catalog + your own custom NIM | advanced |

## Where this sits in Week 23

App 1 **Nemotron** (model) · **App 2 (this) NIM** (serve) · App 11 **Data Flywheel**
(improve) · App 3 **Dynamo** (scale) · App 10 **NeMo Gym** (RL) · App 7 **OpenShell** (guard).

Get NIMs at **build.nvidia.com**. Production use needs an **NVIDIA AI Enterprise**
license (bundled with DGX); containers are free to pull for dev/eval.
