# Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents

> 🇹🇭 **ภาษาไทย:** ทุกบทมีฉบับภาษาไทย (`TUTORIAL.th.md`) — กดปุ่ม **ไทย** ที่มุมขวาบนของ Spark Lab Runner เพื่อสลับภาษา โค้ดและคำสั่งเหมือนกันทุกประการ

A hands-on course built on NVIDIA's official **DGX Spark playbooks**
([build.nvidia.com/spark](https://build.nvidia.com/spark), source
[NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)). It covers as many of them
as one week can hold, in the Week 24 style: one continuous course in the browser, ✓ checkpoints,
▶ runnable labs, offline-checked exercises, and diagrams for every module.

The through-line: **one Spark, or two cabled together, gives you a private AI stack.** You fine-tune a
model, serve it several ways, put one gateway in front, and run agents on it inside a sandbox, with no
cloud involved.

> 🧑‍🏫 **Instructors: start with [`START_HERE.md`](START_HERE.md).** It gets both Sparks ready in one check and one command, and covers who runs which lab live.

## Launch the course (web app)

```bash
.venv/bin/python week25/00_spark_lab_runner/tutorial_server.py
# → http://127.0.0.1:8125   (auto-picks a free port; override with SPARK_GUIDE_PORT)
```

The **Spark Lab Runner** has the same shape as the Week 24 Jev Lab Runner:

- one course with a sidebar, ✓ checkpoints and saved progress, **EN / ไทย**
- **▶ Run** on every lab and exercise; the lab runs on your laptop and drives your Spark over SSH
- a **⚡ Live / 📄 Dry run** switch and live status chips: **Spark A**, **Spark B**, and each serving
  endpoint that answers (Ollama, vLLM, SGLang, TensorRT-LLM, llama.cpp, LM Studio, LiteLLM)
- **⚡ Ask the Spark** blocks: edit a chat request in the page and send it to an engine on your Spark,
  with TTFT and tok/s shown
- a **🖥 Spark setup** dialog: Spark hostnames plus `HF_TOKEN` / `NGC_API_KEY` / `NVIDIA_API_KEY`, saved
  server-side to `week25/.env.local` (gitignored, mode 0600) and never sent to the browser
- a **⌨ terminal** that runs each command on this Mac, on Spark A, or on Spark B. Code blocks start with
  `# on: spark` / `# on: spark-b` / `# on: laptop`, and ▶ run picks the matching target.

### LIVE, DRY, and the laptop stand-in

| Mode | What happens |
|---|---|
| **LIVE** | `SPARK_HOST` answers `ssh -o BatchMode=yes`, and labs run real commands on it, printed as `[ssh spark-a]` |
| **DRY** | Nothing runs. You see each command plus one of three clearly labelled outputs: **RECORDED** (captured from a real Spark with `SPARK_RECORD=1`), **REFERENCE** (quoted word-for-word from the NVIDIA playbook), or **EXAMPLE** (an illustrative shape, not a measurement) |
| **💻 laptop stand-in** | For HTTP labs only. When a Spark endpoint is down, the client-side ideas (streaming, tool calls, gateways, agents) run against Ollama on your laptop. The output is labelled `LAPTOP STAND-IN`, and its speed is never presented as a Spark number |

`week25/common/audit_references.py` checks that every REFERENCE in every lab really appears in a playbook.

## The modules

| # | Module | Official playbooks | Sparks |
|---|---|---|---|
| **Phase 0 · Your Spark(s)** | | | |
| 01 | [Meet your DGX Spark: connect, check, and budget memory](01_meet_your_spark/TUTORIAL.md) | connect-to-your-spark · tailscale · dgx-dashboard · vscode | 1 |
| 02 | [Two Sparks, one cluster: QSFP, 200 Gb/s, NCCL](02_two_sparks_nccl/TUTORIAL.md) | connect-two-sparks · nccl · connect-three-sparks · multi-sparks-through-switch | 2 |
| **Phase 1 · Serve** | | | |
| 03 | [Ollama + Open WebUI: your first model server](03_ollama_open_webui/TUTORIAL.md) | open-webui · ollama | 1 |
| 04 | [llama.cpp + LM Studio: GGUF and quantized models](04_llama_cpp_lm_studio/TUTORIAL.md) | llama-cpp · lm-studio | 1 |
| 05 | [vLLM: high-throughput serving, tool calling, two-Spark tensor parallel](05_vllm/TUTORIAL.md) | vllm | 1–2 |
| 06 | [SGLang, TensorRT-LLM, NIM and Nemotron: the engine bake-off](06_sglang_trtllm_nim/TUTORIAL.md) | sglang · trt-llm · nim-llm · nemotron | 1 |
| 07 | [NVFP4 quantization and speculative decoding](07_nvfp4_speculative/TUTORIAL.md) | nvfp4-quantization · speculative-decoding | 1 |
| 08 | [LiteLLM: one gateway for every engine and both Sparks](08_litellm_gateway/TUTORIAL.md) | *course-original* | 1–2 |
| **Phase 2 · Fine-tune** | | | |
| 09 | [Fine-tune with LLaMA Factory: LoRA, QLoRA, full](09_llama_factory/TUTORIAL.md) | llama-factory | 1 |
| 10 | [Unsloth: fast LoRA fine-tuning](10_unsloth/TUTORIAL.md) | unsloth | 1 |
| 11 | [PyTorch and NeMo AutoModel fine-tuning, on one and two Sparks](11_pytorch_nemo_two_sparks/TUTORIAL.md) | pytorch-fine-tune · nemo-fine-tune | 1–2 |
| 12 | [Fine-tune a vision-language model and FLUX.1](12_vlm_flux_finetune/TUTORIAL.md) | vlm-finetuning · flux-finetuning | 1 |
| 13 | [Close the loop: evaluate, merge, serve and route your fine-tune](13_finetune_to_serve/TUTORIAL.md) | *course-original* | 1 |
| **Phase 3 · Agents** | | | |
| 14 | [NeMo Agent Toolkit: agents on your Spark's models](14_nat_agents/TUTORIAL.md) | *course-original* (NAT 1.9) | 1 |
| 15 | [OpenShell: sandbox and govern AI agents](15_openshell_sandbox/TUTORIAL.md) | openshell | 1 |
| 16 | [NemoClaw: always-on sandboxed agents](16_nemoclaw/TUTORIAL.md) | nemoclaw · nemoclaw-applications | 1 |
| 17 | [OpenClaw and Hermes Agent with a local LLM](17_openclaw_hermes/TUTORIAL.md) | openclaw · hermes-agent | 1 |
| 18 | [Coding agents on local inference](18_coding_agents/TUTORIAL.md) | cli-coding-agent · local-coding-agent · vibe-coding | 1 |
| 19 | [Multi-agent chatbot, RAG and knowledge graphs](19_multi_agent_rag_kg/TUTORIAL.md) | multi-agent-chatbot · txt2kg · rag-ai-workbench | 1 |
| **Capstone + everything else** | | | |
| 20 | [Capstone: a sovereign hotel agent — fine-tune → serve → gateway → NAT agent in a sandbox](20_capstone_sovereign_agent/TUTORIAL.md) | all of the above | 1–2 |
| 21 | [Playbook atlas: every other Spark playbook](21_playbook_atlas/TUTORIAL.md) | comfyui · live-vlm-webui · isaac · jax · cuda-x · vss · … | 1 |

**21 modules · 294 sections · 76 labs · 21 exercises**, every one in English and Thai.

### The capstone in one picture

```text
 Spark A: OpenShell sandbox → NAT agent ──inference.local──► LiteLLM :4000 ──► vLLM Qwen3.6 (agent brain)
                               └─ route_guest_message tool ──►   (one key)   ──► Spark B: vLLM Qwen3-4B + LoRA hotel-ft
```

The hotel router you fine-tune in Module 09 is scored in Module 13 and served on Spark B. The Module 14
agent calls it as a tool through the Module 08 gateway, from inside a Module 15 sandbox that can reach
exactly two destinations. On a laptop the whole wiring runs against Ollama. It passed 3 of 4 acceptance
scenarios there, and the one failure was a finding from Module 13 again: a prompt-only model marks
normal Thai requests urgent. That is the fine-tune's job to fix.

## Setup

1. **Your Spark:** Module 01 walks through key-based SSH (`ssh -o BatchMode=yes spark-a true` must
   work), Tailscale for access from anywhere, and the `ssh -L` tunnel for localhost-only services.
2. **The runner:** open **🖥 Spark setup** and set `SPARK_HOST` (plus `SPARK_HOST2` for the two-Spark
   modules). Or copy `week25/.env.example` into `week25/.env.local`.
3. **Gated downloads:** log in **on the Spark** once, with `hf auth login` for Hugging Face and
   `docker login nvcr.io` for NGC. No lab puts a Hugging Face or NGC token on a command line. The one
   documented exception is the capstone's gateway key (Module 20 §6), which also names the safer alternative.

Local tool environments (all gitignored, created with `uv`):

| Path | What | Used by |
|---|---|---|
| repo `.venv` | Python 3.13, FastAPI, PyYAML | the runner and every lab |
| `week25/.venv-nat` | `nvidia-nat[langchain]` 1.9.0 (Python 3.12) | Modules 14, 20 |
| `week25/.venv-litellm` | `litellm[proxy]` 1.89.0 | Modules 08, 13, 20 |
| `week25/15_openshell_sandbox/.venv-openshell` | the OpenShell CLI (pip), used only to parse policies on the laptop | Module 15 |
| `dgx-spark-playbooks/` (repo root) | a clone of NVIDIA's playbooks | Module 21, `audit_references.py` |

```bash
# recreate them
uv venv -p 3.12 week25/.venv-nat && uv pip install -p week25/.venv-nat/bin/python "nvidia-nat[langchain]==1.9.0" greenlet "nvidia-nat-mcp~=1.9"
uv venv week25/.venv-litellm && uv pip install -p week25/.venv-litellm/bin/python "litellm[proxy]==1.89.0" openai pyyaml
uv pip install -p week25/.venv-nat/bin/python --no-deps -e week25/14_nat_agents/hotel_ops_nat \
  -e week25/20_capstone_sovereign_agent/hotel_capstone_nat          # the NAT plugins (Modules 14, 20)
git clone --depth 1 https://github.com/NVIDIA/dgx-spark-playbooks.git dgx-spark-playbooks
```

## Layout

```text
week25/
  00_spark_lab_runner/   the web app (tutorial_server.py + static/guide.html), port 8125
  common/sparkkit.py     ~600-line stdlib helper: sh() on Spark A/B or DRY, chat_any(), memory math
  common/audit_references.py   every REFERENCE must be verbatim in a playbook
  common/check_translation.py  every TUTORIAL.th.md matches its English twin section by section
  common/recorded/       real Spark transcripts replayed in DRY mode (SPARK_RECORD=1 to capture)
  AUTHORING.md           the module contract and honesty rules (read before adding a module)
  NN_module/
    TUTORIAL.md · TUTORIAL.th.md · diagrams.json
    labs/labNN_*.py · exercises/exNN_*.py · exercises/solutions/exNN_*.py
```

Run any lab without the web app:

```bash
.venv/bin/python week25/01_meet_your_spark/labs/lab02_memory_budget.py            # arithmetic, runs anywhere
SPARK_HOST=spark-a .venv/bin/python week25/01_meet_your_spark/labs/lab01_spark_doctor.py
SPARK_MODE=dry .venv/bin/python week25/05_vllm/labs/lab01_launch_builder.py       # $0, no Spark
```

Record real Spark output for DRY replays (maintainers):

```bash
SPARK_RECORD=1 SPARK_HOST=spark-a .venv/bin/python week25/NN_module/labs/labNN_x.py
```

## Status and honesty note

The course was written on 2026-09-29, while both of the author's Sparks were offline. Every
laptop-runnable lab was run on a Mac, and those runs include real NAT agents, a real LiteLLM gateway and
real evaluations. Everything that needs a Spark is DRY: each output is labelled REFERENCE (quoted
word-for-word from a playbook, checked by `audit_references.py`: 139 lines, 0 mismatches) or EXAMPLE (an
illustrative shape). Nothing presented as a Spark measurement was measured on anything else.

The first live pass should follow Module 20 section 7, and run each lab once with `SPARK_RECORD=1`, so
DRY mode replays real Spark transcripts from then on.

While building the course, the module authors found and flagged problems in the upstream playbooks. Each
is noted where it matters:
- The PyTorch fine-tune scripts disagree about `--output_dir` (Module 11).
- The NeMo container loses its checkpoints on exit (Module 11).
- The NCCL playbook tests all-gather, not all-reduce (Module 02).
- The OpenShell/NemoClaw policy docs contradict themselves on group-name underscores (Module 16).
- The Claude Code local-inference playbook lists only DGX Station (Module 18).
