# ▣ Week 19 · App 2 — Fine-tuning on a DGX (real, executed end-to-end)

An **interactive web app that actually runs a fine-tune** — laptop steps execute
locally, DGX steps execute **on your DGX over SSH** and stream the real output back.
Framework: **Unsloth** (LoRA/QLoRA → GGUF → Ollama). Your data and weights never
leave hardware you control.

```
💻 dataset + script  →  🖥️ push (scp)  →  🖥️ container  →  🖥️ Unsloth train  →  🖥️ GGUF
   →  🖥️ ollama serve  →  💻 evaluate base vs tuned
```

## What runs where

| Step | Where | What actually happens |
|------|-------|------------------------|
| 1 · dataset + script | 💻 laptop | writes real `train/val.jsonl` + `train_hvac_unsloth.py` to `.sandbox/` |
| 2 · connect & push | 💻→🖥️ | tests SSH + GPU, `scp`s dataset + script to the DGX |
| 3 · prepare DGX | 🖥️ SSH | `docker pull` NVIDIA PyTorch container, report free VRAM |
| 4 · **train** | 🖥️ SSH | installs Unsloth, loads **your** model 4-bit, trains LoRA, exports Q4_K_M GGUF — **live loss streams back** |
| 5 · serve | 🖥️ SSH | `ollama create hvac-assistant` from the GGUF + start it |
| 6 · evaluate | 💻 laptop | asks held-out questions to **base vs your tuned** model over the connection |

---

## Quick start

```bash
uv pip install -r week19/dgx_finetune/requirements.txt
.venv/bin/python week19/dgx_finetune/tutorial_server.py      # → http://127.0.0.1:8093
```

Then in the web app:
1. **🖥️ DGX SSH** panel → enter your DGX **host / user / (key)**, the **HuggingFace
   model** to fine-tune, and an **HF token** if it's gated. (Key-based SSH only —
   test first with `ssh user@host nvidia-smi -L`.)
2. Run **STEP 1 → 6 in order.** The 🖥️ steps execute on the DGX and stream output.
3. After STEP 5, pick **`hvac-assistant`** in the model dropdown; STEP 6 compares it
   to the base model.

> **Prereqs on the DGX:** SSH access, Docker + NVIDIA Container Toolkit, a GPU with
> free VRAM (stop other models with `ollama stop <model>`), and Ollama (for STEP 5).
> Smaller models (1–3B) train in minutes; bigger models take longer and need more VRAM.

---

## Layout

```
week19/dgx_finetune/
├── README.md · requirements.txt · config.py   (SSH + connection state)
├── ftssh.py            → runs real commands on the DGX over ssh/scp, streams output
├── ftview.py           → framing + the eval engine (real over the connection)
├── tutorial_server.py  → FastAPI control plane (/api/ssh, long timeouts for training)
├── static/guide.html   → clickable UI + 🖥️ DGX SSH panel + 🔌 Connection panel
└── demos/
    ├── step01_dataset_prep.py    💻 dataset + Unsloth script
    ├── step02_connect_push.py    💻→🖥️ test SSH + scp to DGX
    ├── step03_prepare_dgx.py     🖥️ pull container + GPU check
    ├── step04_train.py           🖥️ REAL Unsloth fine-tune (your model)
    ├── step05_serve.py           🖥️ register + serve via Ollama
    └── step06_evaluate.py        💻 base vs tuned, real
```

---

## Knobs (🖥️ DGX SSH panel → saved via `POST /api/ssh`)

| Field | Env | Meaning |
|-------|-----|---------|
| Host / User / Port / Key | `FT_SSH_*` | how to reach the DGX (key-based) |
| Remote workdir | `FT_WORKDIR` | where files + outputs live (default `~/dgx_finetune_demo`) |
| HF model | `FT_HF_MODEL` | the model to fine-tune (e.g. `unsloth/Llama-3.2-3B-Instruct`) |
| HF token | `FT_HF_TOKEN` | only for gated weights |
| (advanced) | `FT_MAX_STEPS` | training steps (default 60) |

The **🔌 Connection** panel (separate) sets the *inference* endpoint STEP 6 evaluates
against — point it at the same DGX (tunnel/local) so it sees `hvac-assistant`.

---

## Where this sits in Week 19
App 1 run/serve · **App 2 (THIS) fine-tune for real** · App 3 observe (Phoenix+NAT) ·
App 4 self-evolving · App 5 LiteLLM gateway. The model you train here becomes a
first-class citizen on the DGX — selectable in every app's model dropdown.
