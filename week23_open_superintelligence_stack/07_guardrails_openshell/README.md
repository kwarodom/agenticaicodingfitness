# App 7 · NeMo Guardrails + OpenShell — securing sovereign agents

**Agent = Model + Harness.** This app is the **guardrail / runtime-security** layer:
how to SECURE a long-running sovereign agent that lives for days on your DGX with
tools, memory, and network. Three layers of defense:

- **NeMo Guardrails** — programmable rails around the LLM: **input** rails (jailbreak /
  prompt-injection), **topic/dialog** rails, **output** rails (no secrets/PII).
  Controls *what the model says*.
- **OpenShell Secure Runtime** — a hardened sandbox for agent tool-use with a network
  **egress allowlist** and a **signed, versioned policy** enforced by the **NemoClaw**
  gateway. Controls *what the tools can do*.
- **Privacy router** — routes PII/secrets to a **LOCAL sovereign NIM** (never leaves the
  perimeter); non-sensitive traffic may use a larger model. Controls *where data goes*.

Runs **dual-mode**: **REAL** against a local NIM / Ollama / vLLM OpenAI endpoint, or
**SIM** with no GPU (`week23/07_guardrails_openshell/sim.py`).

## Run the web tutorial

```bash
cd week23/07_guardrails_openshell
pip install -r requirements.txt
python tutorial_server.py            # → http://localhost:8106
```

Open **http://localhost:8106**. Use **🔌 Connection** to point at your DGX
(`http://<dgx>:8000/v1`) or leave it in SIM. Pick a model with the model picker.

## Run the chapters standalone

```bash
python demos/step01_threat_model.py      # why guard a long-running agent (threat model)
python demos/step02_author_rails.py      # NeMo Guardrails — author & test rails (ALLOW/BLOCK)
python demos/step03_secure_runtime.py    # OpenShell — sandbox + egress allowlist + signed policy
python demos/step04_privacy_router.py    # privacy router — keep PII on a LOCAL NIM
```

## Chapters

| # | Chapter | Level |
|---|---------|-------|
| 1 | Securing a sovereign agent | beginner |
| 2 | Why guard a long-running agent? (threat model) | beginner |
| 3 | NeMo Guardrails — author & test rails | intermediate |
| 4 | OpenShell — sandbox + allowlist + signed policy | advanced |
| 5 | Privacy router — keep data sovereign | advanced |

## Where this sits in Week 23

App 1 **Nemotron** (model) · App 2 **NIM** (serve) · App 11 **Data Flywheel** (improve) ·
App 3 **Dynamo** (scale) · App 10 **NeMo Gym** (RL) · **App 7 (this) OpenShell** (guard).

Get **NeMo Guardrails** + NIMs at **build.nvidia.com**. Production use needs an
**NVIDIA AI Enterprise** license (bundled with DGX); containers are free to pull for
dev/eval.
