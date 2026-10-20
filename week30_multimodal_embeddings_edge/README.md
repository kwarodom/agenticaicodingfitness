# Week 30 · Multimodal embeddings on local, sovereign and edge hardware

Turn text, photos, voicemails and CCTV into vectors **on your own machine** with Google DeepMind's
**EmbeddingGemma 2** (740M, Apache 2.0, released 6 October 2026). Then build search, private RAG and an agent
memory that never send a byte off the premises. The device ladder comes from Tina Huang's
[*Every Size Local AI In 24 Minutes*](https://youtu.be/rPGJhrunbxo), from 32 KB chips to 8× H100.
This week adds the model class her video skips: the embedding model, which fits several rungs lower than any chat model.

Running example: **Alto Edge Memory**, a hotel's private, offline memory of maintenance photos, CCTV, guest
voicemails (English and Thai) and operations notes. All data is **synthetic**.

## Start

```bash
# once — the Week 30 venv (transformers ≥ 5.19 knows embedding_gemma2; the shared .venv has 5.10)
uv venv --python 3.12 week30_multimodal_embeddings_edge/.venv-embed
uv pip install --python week30_multimodal_embeddings_edge/.venv-embed/bin/python 'sentence-transformers>=6.1.0' 'transformers>=5.19' torch torchvision torchcodec soundfile librosa pillow numpy fastapi uvicorn
# optional — Modules 05 and 07 use a local chat model
ollama pull gemma4:12b
# Module 08 (Mac) — MLX in its own venv (mlx-vlm pinned to the commit the mlx-community builds were converted with)
uv venv --python 3.12 week30_multimodal_embeddings_edge/.venv-mlx
uv pip install --python week30_multimodal_embeddings_edge/.venv-mlx/bin/python "git+https://github.com/Blaizzy/mlx-vlm.git@3d87e88402f307efbf68e568971aa887ee7d9ed0" "mlx>=0.32.3" "transformers>=5.18.0" torch torchvision torchcodec soundfile librosa pillow numpy fastapi uvicorn
uv pip install --python week30_multimodal_embeddings_edge/.venv-embed/bin/python onnxruntime

# every time
.venv/bin/python week30_multimodal_embeddings_edge/00_embed_lab_runner/tutorial_server.py      # → http://127.0.0.1:8130
```

The first lab downloads the model once (1.5 GB, no login). After that, every lab runs offline from the cache.
Video needs ffmpeg (`brew install ffmpeg`).

## Modules

| # | module | you build |
|---|---|---|
| 01 | [Local, sovereign, edge: the device ladder](01_local_sovereign_edge/TUTORIAL.md) | a fit calculator and a measured profile of your machine |
| 02 | [Meet EmbeddingGemma 2](02_first_embeddings/TUTORIAL.md) | first vectors, semantic search, task prefixes measured, selective encoder loading |
| 03 | [Smaller vectors, smaller models](03_matryoshka_and_quantization/TUTORIAL.md) | Matryoshka 768→128, int8 and binary vectors, GGUF and int8 weights |
| 04 | [One space for text, photos, voice and video](04_multimodal_search/TUTORIAL.md) | cross-modal and Thai⇄English search, interleaved incident records, the modality gap |
| 05 | [Sovereign RAG: nothing leaves the building](05_sovereign_rag/TUTORIAL.md) | an egress log, a one-file vector store, answers from a local Gemma 4 |
| 06 | [Benchmark like an edge engineer](06_edge_benchmarking/TUTORIAL.md) | p50/p95 per modality, the float16 test, bandwidth floor vs compute |
| 07 | [Capstone: Alto Edge Memory](07_capstone_alto_edge_memory/TUTORIAL.md) | an offline multimodal agent, graded on evidence, plus the grader's blind spot |
| 08 | [Your Mac as the edge box](08_mac_edge_box/TUTORIAL.md) | the model in MLX, ONNX Runtime and PyTorch on Apple silicon; a Mac mini as an always-on embedding server |

All eight modules have `TUTORIAL.md` and a Thai `TUTORIAL.th.md`, 2–3 runnable labs, an exercise with an offline
checker and a reference solution, and `diagrams.json` for the runner's 📊 page.

## What is where

```text
week30_multimodal_embeddings_edge/
  00_embed_lab_runner/   tutorial_server.py (FastAPI, :8130) + static/guide.html — course, ▶ labs, ⌨ terminal (this machine only)
  common/embedkit.py     lab helper: load(modalities) with selective encoders, offline-by-default, timers, metrics, Ollama
  common/egress.py       loopback-only network guard that logs every connection attempt (a teaching tripwire)
  common/vectorstore.py  one-file SQLite + numpy vector store for notes, images, audio, video and records
  common/mackit.py       Module 08: MLX and ONNX Runtime embedders, re-exec into .venv-mlx, power readings (no sudo)
  common/embed_service.py  always-on embedding HTTP service (MLX 4-bit) — loopback by default, token for the LAN
  common/data/           SYNTHETIC hotel knowledge base (18 notes, 18 questions, 2 in Thai)
  common/media/          SYNTHETIC photos, voicemails and a CCTV clip (from Week 24, Module 13)
  common/sources/        model card snapshot + the video transcript (what the course quotes)
  common/check_translation.py   maintainer check: TUTORIAL.th.md mirrors TUTORIAL.md
  RESEARCH.md            every fact and number with its source · AUTHORING.md · the rules for new modules
  NN_*/                  TUTORIAL.md (+ .th.md), labs/, exercises/, diagrams.json
```

## Honesty rules (short version)

- Every number a lab prints was measured on the machine running it. Tutorials show output **captured on this
  Mac** (Apple M4 Max, 36 GB). In Modules 01–07 they are LAPTOP STAND-IN numbers, never claimed for a Pi or a phone;
  Module 08 measures the Mac as the edge device itself.
- Third-party numbers (model-card benchmarks, Pixel RAM, vendor bandwidth) are quoted with their source.
  Projections for other devices are labelled **ESTIMATE**.
- The GGUF builds are **text-only**. On this Mac, Ollama 0.34.4 and llama.cpp 0.5.0 could not yet load
  `gemma-embedding2`; Lab 03-3 says so and tells you what to upgrade.
- Media and data are synthetic. Do not embed real guest photos, voices or CCTV without approval, even locally.

Maintainers: see [AUTHORING.md](AUTHORING.md).
