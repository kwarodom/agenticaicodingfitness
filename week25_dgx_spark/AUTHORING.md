# Week 25 · authoring guide (for maintainers and agents adding modules)

Week 25 turns NVIDIA's official DGX Spark playbooks (<https://build.nvidia.com/spark>, source:
<https://github.com/NVIDIA/dgx-spark-playbooks>) into one hands-on course, in the style of Week 24.
**Module 01 (`01_meet_your_spark/`) is the reference implementation — copy its shape exactly.**

Local resources: the playbooks are cloned at the repo root in `dgx-spark-playbooks/` (gitignored; the
newest versions are `nvidia/playbook-<name>/README.md`, older twins without the prefix). NeMo Agent
Toolkit 1.9.0 is installed in `week25/.venv-nat/` (gitignored; `week25/.venv-nat/bin/nat`). PyYAML is in the repo `.venv`. The LiteLLM proxy (1.89, with the `[proxy]` extra) is in
`week25/.venv-litellm/` (gitignored; `week25/.venv-litellm/bin/litellm`) because the repo `.venv` copy lacks the proxy deps. This Mac runs Ollama at `http://localhost:11434` (laptop stand-in).
Install extra Python packages only into a week25-local venv (never globally, never with sudo, no
`curl | sh` installers).

## The module list (folder · title · main playbooks)

| # | folder | title | playbooks (`nvidia/playbook-<name>/README.md`) |
|---|---|---|---|
| 01 | `01_meet_your_spark` | Meet your DGX Spark: connect, check, and budget memory | connect-to-your-spark, tailscale, dgx-dashboard, vscode |
| 02 | `02_two_sparks_nccl` | Two Sparks, one cluster: QSFP, 200 Gb/s, NCCL | connect-two-sparks, nccl, connect-multiple-sparks, connect-three-sparks, multi-sparks-through-switch |
| 03 | `03_ollama_open_webui` | Ollama + Open WebUI: your first model server | open-webui, llms, (old dir `nvidia/ollama/`) |
| 04 | `04_llama_cpp_lm_studio` | llama.cpp + LM Studio: GGUF and quantized models | llama-cpp, lm-studio |
| 05 | `05_vllm` | vLLM: high-throughput serving, tool calling, two-Spark tensor parallel | vllm |
| 06 | `06_sglang_trtllm_nim` | SGLang, TensorRT-LLM, NIM and Nemotron: the engine bake-off | sglang, trt-llm, nim-llm, nemotron |
| 07 | `07_nvfp4_speculative` | NVFP4 quantization and speculative decoding | nvfp4-quantization, speculative-decoding |
| 08 | `08_litellm_gateway` | LiteLLM: one gateway for every engine and both Sparks | *course-original* (LiteLLM proxy) |
| 09 | `09_llama_factory` | Fine-tune with LLaMA Factory: LoRA, QLoRA, full | llama-factory |
| 10 | `10_unsloth` | Unsloth: fast LoRA fine-tuning | unsloth, fine-tuning |
| 11 | `11_pytorch_nemo_two_sparks` | PyTorch and NeMo AutoModel fine-tuning, on one and two Sparks | pytorch-fine-tune, nemo-fine-tune |
| 12 | `12_vlm_flux_finetune` | Fine-tune a vision-language model and FLUX.1 | vlm-finetuning, flux-finetuning |
| 13 | `13_finetune_to_serve` | Close the loop: evaluate, merge, serve and route your fine-tune | *course-original* (uses 05/08/09) |
| 14 | `14_nat_agents` | NeMo Agent Toolkit: agents on your Spark's models | *course-original* (NAT 1.9 docs) |
| 15 | `15_openshell_sandbox` | OpenShell: sandbox and govern AI agents | openshell |
| 16 | `16_nemoclaw` | NemoClaw: always-on sandboxed agents | nemoclaw, nemoclaw-applications |
| 17 | `17_openclaw_hermes` | OpenClaw and Hermes Agent with a local LLM | openclaw, hermes-agent |
| 18 | `18_coding_agents` | Coding agents on local inference | cli-coding-agent, local-coding-agent, vibe-coding |
| 19 | `19_multi_agent_rag_kg` | Multi-agent chatbot, RAG and knowledge graphs | multi-agent-chatbot, txt2kg, rag-ai-workbench |
| 20 | `20_capstone_sovereign_agent` | Capstone: fine-tune → serve → gateway → NAT agent in a sandbox | all of the above |
| 21 | `21_playbook_atlas` | Playbook atlas: every other Spark playbook | the rest |

Each module's `## Next` links to the next folder: `[Lab NN — title](../NN_folder/TUTORIAL.md)`.

## Files per module

```text
NN_folder/
  TUTORIAL.md            the course text (the runner splits it on "## " headings)
  diagrams.json          architecture + sequence + charts (schema: see 01's file)
  labs/labNN_<name>.py   2–4 runnable labs; docstring line 1 = "Lab NN-k · <title>."
  exercises/exNN_<name>.py              starter with 2–4 TODOs + an offline checker
  exercises/solutions/exNN_<name>.py    reference solution (import path uses parents[3])
```

## TUTORIAL.md contract (the parser depends on it)

- H1: `# ▶ Spark Lab NN — <title>`; then the `> Part of Week 25 …` blockquote (copy 01's).
- `**What you'll actually do**` bullets, then one meta line exactly:
  `**Time** ~NN min · **Difficulty** beginner|intermediate|advanced · **Hardware** <1 Spark | 2 Sparks | none>`
- `**Official playbooks covered:**` with build.nvidia.com links (`https://build.nvidia.com/spark/<slug>`).
- Sections, in order: `## 0 · Before you start`, `## 1 · …` … `## N · …`, `## Labs — run them here`,
  `## Try it yourself`, `## Troubleshooting`, `## Next`.
- Every numbered section ends with one `✓ Checkpoint: …` line.
- In "Labs — run them here", one line per lab: `**labs/labNN_x.py** — One-sentence title.`
- Bash blocks start with a target hint so the ⌨ terminal picks the right machine:
  `# on: laptop`, `# on: spark` (Spark A), or `# on: spark-b`.
- `**Expected output**` precedes an output block, followed by its provenance in parentheses when it is not
  obvious: `(captured on this Mac, DRY mode)`, `(REFERENCE — quoted from the playbook)`,
  `(EXAMPLE — illustrative shape, not a measurement)`. The runner hides the ▶ run button on those blocks.
- Optional inline live blocks: a fenced block with language `spark` holding JSON
  `{"target": "ollama|vllm|sglang|trtllm|llamacpp|lmstudio|nim|litellm", "which": "a|b", "model": "…",
  "messages": [...], "max_tokens": 256, "tools": [...]?}`. The runner sends it to that endpoint on the Spark
  (or the laptop stand-in, labelled). Use them in serving/agent modules.
- Use `<details><summary>Hint — …</summary> … </details>` in "Try it yourself".

## Honesty rules (non-negotiable)

1. **Never invent Spark output.** Nobody has run your module on a Spark yet. Output that only a Spark can
   produce is either quoted from the playbook (`sh(..., reference=...)` → labelled REFERENCE) or written as
   an illustrative shape (`sh(..., example=...)` → labelled EXAMPLE). Never put made-up throughput,
   loss curves or timings in a REFERENCE, and never present an EXAMPLE as measured.
2. **Everything that CAN run on the laptop, you run, and you paste the real output**: arithmetic labs,
   dataset builders, config generators and validators, LiteLLM/NAT/agent labs against the laptop Ollama
   (`http://localhost:11434/v1`, models `nemotron-3-nano:latest`, `gemma3:4b`, `gemma4:12b`), and every
   exercise checker. Run every lab and exercise (and solution) before you finish; every one must exit 0
   (a starter exercise exits 1 with ✕ TODO lines).
3. Commands, container tags, model ids, ports and flags are copied from the playbook, not from memory.
   If the course deviates (e.g. llama.cpp on :30080), say so in the text.
4. Label laptop results as `LAPTOP STAND-IN` (sparkkit does this). Never compare laptop tok/s with Spark tok/s.
5. Labs never do destructive or hard-to-undo things on the Spark without an explicit opt-in flag
   (e.g. `--yes` / env `SPARK_APPLY=1`): no `rm -rf`, no netplan writes, no `apt upgrade`, no reboot.
   Read-only checks, `docker run` of playbook containers, and downloads are fine but print what they do.
   Long jobs (training, big pulls) are started with `nohup … > ~/w25/logs/<name>.log 2>&1 < /dev/null &` (without `< /dev/null` the ssh session can hang) and a
   separate lab or command tails the log — the runner kills foreground labs after 900 s.
6. **Self-audit before you report:** `.venv/bin/python week25/common/audit_references.py <your_module>` must
   print `0 not found verbatim`. It checks every `reference=` string, `("ref", …)` tuple and `REF_*` constant
   word-for-word against the playbooks, plus TUTORIAL.md blocks marked `**Expected output** (REFERENCE …)`. A value you composed from a playbook's stated minimum ("driver
   580.95.05 or higher") is an EXAMPLE, not a REFERENCE.
7. Secrets: never print or interpolate tokens into a command line (`sh()` echoes commands). Gated
   downloads on the Spark rely on the Spark's own login: the tutorial tells learners to run
   `hf auth login` (Hugging Face) or `docker login nvcr.io` (NGC) once **on the Spark**. Labs may check
   `bool(cfg("HF_TOKEN"))` to print a hint, nothing more.

## sparkkit (week25/common/sparkkit.py) — read it, do not edit it

`banner(title, sub, status=True)` · `step(n, text)` · `ok/warn/note/result(msg)` · `table(rows, headers)` ·
`bar(v, vmax)` · `check(cond, good, bad)` · `sh(cmd, which="a"|"b", reference=, example=, timeout=)` →
`Result(code, out, source)` · `put(local, remote, which)` · `url(kind, which)` · `up(base)` · `models(base)` ·
`chat(base, model, messages, …)` · `chat_any(kind, model, messages, which=, laptop_model=, tools=, reference=)` ·
`show_chat(r)` · `weights_gb` · `kv_cache_gb` · `decode_ceiling_tok_s` · `SPEC` · `PORTS` · `cfg(name)` ·
`host(which)` · `where(which)` · `mode()` · `on_spark()` · `LAPTOP_OLLAMA` · `pick_laptop_model()`.
Output glyphs the runner colours: `━` banner · `▣` step · `✓` pass · `✕` fail · `⚠` warn · `◈` dry ·
`◆` metric · `═` result · `→` action · `│` table · `$` command · `~ REASON` · `· ANSWER`.
If you need a helper that is missing, write it inside your lab file.

## diagrams.json

`{"architecture": {title, caption, lanes:[{id,label}], nodes:[{id,label,lane,sub,accent}],
edges:[{from,to,label,animated?}]}, "sequence": {title, caption, actors:[{id,label}],
steps:[{from,to,label,kind:"call"|"return"|"note"}]}, "charts": [{title, type:"bars"|"hbars"|"line"|"donut",
unit, labels:[…], series:[{name, values:[…], color}], caption}]}` — accents/colors: green cyan amber violet red.
Chart numbers must come from a lab you ran, from arithmetic, or from the playbook, and the caption says which.

## Style

Plain English, short sentences, active voice, like Week 24 and Module 01. Explain *why* before *how*.
Tables for comparisons. No marketing adjectives. Keep each module 25–60 minutes of work.
Point to the playbook for long vendor steps instead of pasting 200 lines, but every command a learner
must type is in the tutorial.
