# Week 24 · Typed AI decisions with Jev (TypeSafe System One)

> 🇹🇭 **ภาษาไทย:** ทุกบทมีฉบับภาษาไทย (`TUTORIAL.th.md`) — กดปุ่ม **ไทย** ที่มุมขวาบนของ Jev Lab Runner เพื่อสลับภาษา โค้ดและคำสั่งเหมือนกันทุกประการ

Hands-on course: learn to use **Jev**, TypeSafe's System One model. Jev doesn't
generate text; it returns **typed judgments** (`choice`, `noul`, `score`) that
your code can consume directly. Every module teaches the same idea: **Jev
judges, and your code decides**. Actions stay behind deterministic policy and
human approval.

## Launch the course (web app)

```bash
.venv/bin/python week24_typed_decisions_jev/00_jev_lab_runner/tutorial_server.py
# → http://127.0.0.1:8124   (auto-picks a free port; override with JEV_GUIDE_PORT)
```

The **Jev Lab Runner** has the same shape as Week 23's Lab Runner, with these features:

- one continuous step-by-step course with a sidebar, ✓ checkpoints, and progress saved server-side
- **▶ Run** on every lab and exercise file, with output streamed live from this machine
- **⚡ Ask Jev** inline blocks: edit a real request in the page and see typed answers drawn as probability bars
- a **⚡ Live / 📄 Dry run** switch plus a model picker (`jev-1.13.0` pinned, `jev-1.14.0`, or the `jev-latest` / `jev-preview` aliases)
- **EN / ไทย**: every module has a Thai tutorial. Sections pair by position, so checkpoints, labs and progress are shared across languages
- an **LLM picker** for the model that *writes* while Jev *decides* (Module 11): Claude, ChatGPT, Gemini, DeepSeek, Kimi, GLM, OpenRouter, or local Ollama. Pick any model id, or list the provider's live models
- a **🔑 Keys** dialog: paste your own keys, which are saved server-side to `week24_typed_decisions_jev/.env.local` (gitignored, owner-only) and never shown again
- 📊 architecture, sequence and chart diagrams per module (`diagrams.json`), and a built-in ⌨ terminal

### New in October 2026: Jev 1.14, local System One models, and fine-tuning them (Modules 15–17)

TypeSafe released **Jev 1.14.0** on 2026-10-14 ([release post](https://typesafe.ai/blog/jev-1-14)), and
[systemonemodels.org](https://systemonemodels.org/) now lists 25+ decision models, nine of them open weights.
Three new modules cover them. Verified with this repo's keys on 2026-10-17:

| Module | What you build | Runs on |
|---|---|---|
| **15 · Jev sees images** | Photos go straight into Jev (`assets` + header `TypeSafe-Danger-Zone-Images: true`) in `state`, `instructions` or `criteria`. A **Box** question finds the puddle and code crops it. Then direct-vs-pipeline against Module 13, 1.13 vs 1.14 on Thai, and a 47k-token log through **long context** | TypeSafe API (`jev-1.14.0`; Box needs 1.14) |
| **16 · Local System One models** | **Laya** (Convai), classic **GLiNER** (spans → Jev decides) and **GLiNER2.5-Decide** (Fastino) run offline on your laptop, then a fair 18-ticket EN/TH bake-off against Jev: accuracy, speed, cost and calibration | `week24_typed_decisions_jev/.venv-local` + Jev API |
| **17 · Fine-tune a local System One model** | Fine-tune **Laya** (full, Apple GPU) and **GLiNER2.5-multi-Decide** (5 MB LoRA adapter vs full, CPU) on 456 EN/TH tickets; Mac or **DGX Spark** decided from measured memory and speed; criteria locked before training; graded on 66 held-out tickets (CIs, fixed/broke, calibration, forgetting) → SHIP/HOLD + model card. Measured: +23 to +30 points, all three on HOLD for one missed water-near-electricity ticket | `week24_typed_decisions_jev/.venv-local` (Mac), optional DGX Spark via ssh + NGC PyTorch |

Images, Box and long context are what TypeSafe calls **"danger zone"** features: dreamer-ready, not yet
vouched for in production. The course treats them that way. Every new lab keeps a person and a
`execute: False` policy behind the model.

Module 16 needs a second environment (torch + transformers stay out of the course `.venv`). Create it once:

```bash
uv venv --python 3.13 week24_typed_decisions_jev/.venv-local
uv pip install --python week24_typed_decisions_jev/.venv-local/bin/python laya "gliner2[local]" gliner tiktoken sentencepiece pillow
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab00_download_models.py   # ≈3 GB, once
```

Its labs re-run themselves inside `.venv-local` automatically. Without it, they replay the local
answers recorded on 2026-10-17. Module 17 uses the same environment. Its training runs (`--train`,
4–7 min each on an M4 Max) start from the ⌨ terminal, and every other lab reads your checkpoints or the recordings.

### Beyond text: multimodal and live (Modules 13–14)

In Modules 13–14 Jev reads **text**. Multimodal models turn pixels, sound and video into text; Jev turns
that text into fast, cheap, typed decisions; code decides. Verified with this repo's keys
on 2026-09-27:

| Capability | Models used in the labs | Helper |
|---|---|---|
| see (image → text) | `gemini-3.5-flash`, `gpt-5.4-mini`, `claude-sonnet-5`, local `gemma4` (Ollama) | `common/mediakit.see()` |
| hear (speech → text) | `gpt-4o-mini-transcribe`, `gemini-3.5-flash` | `mediakit.hear()` |
| speak (text → speech) | `gpt-4o-mini-tts`, `gemini-3.8-flash-tts` | `mediakit.speak()` |
| watch (video → text) | `gemini-3.8-flash` | `mediakit.watch()` |
| live (streaming voice) | `gemini-3.8-live`, `gpt-realtime-2.1-mini` | `common/livekit.live_turn()` |

All sample media in `13_multimodal_jev/media/` and `14_live_realtime_jev/media/` is
synthetic: images drawn with PIL or generated by an image model, voices from TTS, and a
CCTV clip rendered with ffmpeg. Nothing is a real site, guest or recording. The live labs
use the `websockets` package, which is already in the repo `.venv`.

### LIVE vs DRY

| Mode | What happens | Cost |
|---|---|---|
| **LIVE** | Real calls to `https://api.typesafe.ai/v1/systemone` | $0.042 per million input tokens, output free; a typical lab call is ≈ $0.00002 |
| **DRY** | No network. Answers recorded from real `jev-1.13.0` calls are replayed and labelled `RECORDED`. A request that was never recorded (e.g. one you edited) gets a neutral `PLACEHOLDER` (flat probabilities, noul = 0.5), which is **not a prediction** | $0 |

### API keys: bring your own

Nothing in this repo contains a key. After cloning, give the course keys one of two ways:

1. **Easiest:** start the runner, click **🔑 Keys**, and paste them. They are saved to
   `week24_typed_decisions_jev/.env.local` (gitignored, mode 0600).
2. Copy `week24_typed_decisions_jev/.env.example` into the repo-root `.env` and fill in the lines you need.

Lookup order is process env → `week24_typed_decisions_jev/.env.local` → repo-root `.env`. Only
`TYPESAFE_API_KEY` is needed for Modules 01–10, 12, 15 and 16 (Module 17 uses Jev only as a baseline and teacher, DRY works). Module 11 needs **any one**
LLM key, or none at all with local Ollama. Keys are injected only into lab
subprocesses and the built-in terminal. They are never sent to the browser,
printed, or written to recordings. Every state-changing endpoint rejects
cross-site requests (Host + Origin check).

| Role | Provider | Key variable | Default model (verified 2026-09-27) |
|---|---|---|---|
| decides | Jev (TypeSafe) | `TYPESAFE_API_KEY` | `jev-1.13.0` |
| writes | Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| writes | ChatGPT | `OPENAI_API_KEY` | `gpt-5.4-mini` |
| writes | Gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` |
| writes | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash` |
| writes | Kimi | `KIMI_API_KEY` | `kimi-k3` |
| writes | GLM | `ZAI_API_KEY` | `glm-5.3` |
| writes | OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-5.4-mini` |
| writes | Ollama (local) | none | `$OLLAMA_MODEL` or `gemma4:latest` |

## Layout

```text
week24_typed_decisions_jev/
  00_jev_lab_runner/   the web app (tutorial_server.py + static/guide.html)
  common/jevkit.py     ~450-line stdlib Jev client every lab uses (ask/show/choice/noul/score/box,
                       image() assets + danger-zone headers for Jev 1.14)
  common/llmkit.py     stdlib client for 8 generative-LLM providers (generate/list_models/providers)
  common/recorded_llm/ real LLM answers replayed in DRY mode
  common/check_translation.py   validates every TUTORIAL.th.md against its English twin
  common/recorded/     real jev-1.13.0 answers replayed in DRY mode
  common/record_inline.py   maintainer tool: record every ```jev block (needs the key)
  NN_module/
    TUTORIAL.md        the course text (rendered by the runner, readable on GitHub too)
    TUTORIAL.th.md     the same course in Thai (identical code blocks, sections and checkpoints)
    diagrams.json      visuals for the 📊 section
    labs/labNN_*.py    runnable, commented labs
    exercises/exNN_*.py        starter files with TODOs + a free offline checker
    exercises/solutions/…      reference solutions
  15_jev_sees_images/vision.py        image helpers for Module 15 (load · crop_to_box · pad)
  16_local_system_one_models/localkit.py   Laya / GLiNER loaders, Jev→GLiNER adapter, re-exec into .venv-local
  16_local_system_one_models/recorded/     local-model answers replayed when .venv-local is absent
  17_finetune_system_one/ftkit.py      data formats, engines, metrics (CIs, McNemar, ECE), ship_decision()
  17_finetune_system_one/datagen.py    72 EN/TH situations → the training pool; data/ holds splits + held-out
  17_finetune_system_one/.runs/        (gitignored) your checkpoints: Laya ~1.3 GB, GLiNER full ~1.1 GB, LoRA ~5 MB
  .venv-local/         (gitignored) torch + laya + gliner for Modules 16–17
  jev_lab/             the original research scripts, cleanly named and runnable:
    jev_lab.py              7-workflow reference lab (selftest: 46 offline checks)
    jev_laya_benchmark.py   shared Jev/Laya benchmark runner
    test_jev_laya_benchmark.py   32 offline tests
  JEV practical tutorials … .md   the source research write-up (unchanged)
```

Run any lab without the web app:

```bash
.venv/bin/python week24_typed_decisions_jev/01_hello_jev/labs/lab01_first_call_raw.py
JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/01_hello_jev/labs/lab02_read_the_answer.py   # $0 replay
```

Refresh the DRY recordings after changing a lab (maintainers only; this makes live calls):

```bash
JEV_RECORD=1 .venv/bin/python week24_typed_decisions_jev/NN_module/labs/labNN_x.py
JEV_RECORD=1 .venv/bin/python week24_typed_decisions_jev/common/record_inline.py
```

## Changes from the original research files

- `jev_lab/jev_lab.py` and `jev_lab/jev_laya_benchmark.py` default to the **TypeSafe
  direct** provider (`--provider typesafe` / `--jev-provider typesafe`) because this
  course ships a TypeSafe key. OpenRouter still works with `--provider openrouter`
  plus `OPENROUTER_API_KEY`. The two offline tests that patched `OPENROUTER_API_KEY`
  now patch `TYPESAFE_API_KEY`, and all 32 pass.
- The original files with spaces in their names are left untouched.
