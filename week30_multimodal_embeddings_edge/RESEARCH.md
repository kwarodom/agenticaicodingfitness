# Week 30 research notes — EmbeddingGemma 2, local AI, sovereign AI, edge AI

Collected 2026-10-07, the day after EmbeddingGemma 2 launched. Every number here has a source next to it.
The course modules quote from this file; the maintainers' audit (`common/audit_references.py`) checks that every
`REFERENCE` block in a tutorial appears verbatim in this file or in the model card snapshot `common/sources/`.

## 1 · EmbeddingGemma 2 (Google DeepMind, released 2026-10-06)

Sources: [launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) ·
[model card](https://huggingface.co/google/embeddinggemma-2) (snapshot: `common/sources/embeddinggemma-2-model-card.md`) ·
[ggml-org GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) · [Unsloth run-locally guide](https://unsloth.ai/docs/models/embeddinggemma-2)

| fact | value | source |
|---|---|---|
| total parameters | 740M (text 270M = 130M transformer + 140M embedder; vision encoder 170M; audio encoder 300M) | model card |
| modalities | text (incl. code), images, video, audio, and interleaved combinations, in one 768-d space | model card |
| output dims (Matryoshka) | 768, 512, 256, 128 — "up to a **6x reduction** in vector storage costs" | model card |
| context | 8,192 tokens shared by all modalities | model card |
| token cost | text 1/subword · image 280 (default; 70–1120 budget) · video 140/frame at 1 fps · audio 25/s (mono 16 kHz) | model card |
| architecture | 24 layers, model dim 512, hidden 2048, sliding window 1024, vocab 262,144, mean pooling, projection 512→768 | model card |
| precision | bfloat16 or float32 — **never float16** (NaN / silent degradation) | model card |
| license | Apache 2.0, not gated on Hugging Face | model card + HF API (`gated: False`) |
| on-device RAM (quantized, Pixel 11 Pro) | "~191MB active RAM" text-only · "~567MB" full multimodal | launch blog |
| runtimes named | transformers, sentence-transformers (≥ 6.1.0), MLX, vLLM, llama.cpp, SGLang, Ollama, LM Studio, MediaPipe, LiteRT, transformers.js/WebGPU | launch blog + model card |
| GGUF | ggml-org: Q8_0 310 MB, BF16 558 MB; Unsloth UD-Q4_K_XL 176 MB — **text-only** (no mmproj) | GGUF cards |

Benchmarks (768d, full precision; model card):

| benchmark | metric | EmbeddingGemma 2 | EmbeddingGemma 1 |
|---|---|---|---|
| MTEB multilingual v2 | Mean(Task) | 61.36 | 61.15 |
| MTEB code v1 | NDCG@10 | 78.68 | 68.76 |
| MIEB lite | Mean(TaskType) | 64.64 | – |
| MMEB v2 Image | Hit@1 | 57.28 | – |
| MMEB v2 VisDoc | NDCG@5 | 67.84 | – |
| MMEB v2 Video | Hit@1 | 50.67 | – |
| MSEB retrieval | MRR@10 | 69.54 | – |
| MAEB | Mean(Task) | 49.39 | – |

Matryoshka truncation (model card):

| dims | ratio | MTEB multi | MTEB eng | MTEB code | MIEB | MMEB | MSEB | MAEB |
|---|---|---|---|---|---|---|---|---|
| 768 | 1:1 | 61.36 | 68.46 | 78.68 | 64.64 | 59.01 | 69.54 | 49.39 |
| 512 | 1:1.5 | 61.17 | 68.41 | 77.24 | 64.32 | 58.38 | 69.18 | 49.21 |
| 256 | 1:3 | 60.41 | 67.78 | 76.18 | 63.13 | 56.24 | 66.76 | 48.91 |
| 128 | 1:6 | 57.89 | 65.68 | 71.41 | 59.06 | 45.65 | 56.71 | 46.92 |

Task prefixes (text only; media gets no prefix) — model card:

| use | prompt_name | query | document |
|---|---|---|---|
| search | SearchQuery | `task: search result \| query: {query}` | `title: {title} \| text: {content}` |
| Q&A | QuestionAnswering | `task: question answering \| query: {question}` | `title: {title} \| text: {passage}` |
| fact check | FactChecking | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| code | CodeRetrieval | `task: code retrieval \| query: {query}` | `title: {title or filename} \| text: {code}` |
| classification / clustering / similarity | Classification / Clustering / SentenceSimilarity | `task: <name> \| query: {content}` (symmetric) | — |

`prompt_name="Document"` applies `title: none`.

Selective encoder loading (`config_kwargs`): text only `{"vision_config": None, "audio_config": None}` → 270M ·
text+image `{"audio_config": None}` → 440M · text+audio `{"vision_config": None}` → 570M · full `{}` → 740M.

### Verified on this Mac (Apple M4 Max, 36 GB, macOS) — 2026-10-07

- `week30_multimodal_embeddings_edge/.venv-embed`: torch 2.14.1, transformers 5.19.0, sentence-transformers 6.1.0 (the shared repo `.venv`
  has transformers 5.10.2, which does not know `embedding_gemma2`).
- Weights: `model.safetensors` 1,488,915,288 bytes. Text-only load = 271.0M parameters; full = 744.4M; device `mps:0`.
- Model-card quick start: query "What causes the northern lights?" vs its document → cosine **0.8717**;
  vs an unrelated chiller sentence → 0.5686.
- `truncate_dim=128, normalize_embeddings=True` → norm ≈ 1.00 (sentence-transformers re-normalises after the cut;
  bfloat16 on MPS gives 1.00x, not exactly 1). A MANUAL slice `v[:128]` has norm ≈ 0.60 and, used with a dot
  product, reshuffles 27 of 54 top-3 slots on the hotel set (Lab 03-1) — the model card's warning, reproduced.
- float16 (model card: "returns NaN or silently degraded embeddings"): 36 short texts → 0 NaN, but one 3,502-token
  document → a **NaN vector** on both CPU and MPS, with no error (Lab 06-2). bfloat16 and float32 → fine.
- Cross-modal: 7 text queries over 6 synthetic hotel images + 3 synthetic voicemails (week 24 media) → **7/7
  correct at rank 1**, including an English query finding the Thai voicemail.

## 2 · "Every Size Local AI In 24 Minutes" — Tina Huang, 2026-09-28

Source: <https://youtu.be/rPGJhrunbxo> (24 min 22 s). Notes from the auto-generated English transcript.

The device ladder she demonstrates:

| class | her device | memory she quotes | what ran |
|---|---|---|---|
| chips (no OS) | Arduino Uno R4 | 32 KB | no model — but it can call a bigger machine over Wi-Fi |
| chips | ESP32-S3 (~$20 with screen) | 8 MB | TinyStories 260K and 3M |
| mini computers | Raspberry Pi 5 | 8 GB | Whisper → Qwen3 1.7B → Piper voice loop; Moondream on webcam frames; no image generation (no GPU) |
| mini computers | iPhone 16 | 8 GB (apps get ~4–5 GB) | Gemma 4 via Google AI Edge Gallery, Stable Diffusion 1.5 via Draw Things, Whisper Large, YOLO11n via Ultralytics |
| personal computers | MacBook Pro | 32 GB | "pretty much any category": Qwen3 14B/32B, Qwen3 Coder 30B, Flux Dev |
| home servers | Mac Studio | 64 GB | 27–32B LLMs; Qwen 3.6 35B serving the Arduino over Wi-Fi |
| home servers | AMD Ryzen AI Halo | 128 GB advertised, ~96 GB usable | Llama 3.3 70B, GPT-OSS 120B; big capacity, "pretty small bandwidth" |
| GPUs (rented) | RTX 4090 | 24 GB | fast image/video generation; small capacity, large bandwidth |
| GPUs (rented) | 8× H100 | 640 GB | ultra-large models; capacity and bandwidth together |

Her teaching points:

- **Kitchen analogy:** storage = freezer, RAM = prep counter, memory bus = conveyor belt, CPU = head chef,
  GPU = brigade of line cooks, NPU = single-purpose machines bolted to the wall.
- **Capacity vs bandwidth:** same 8 GB RAM, but the iPhone is far faster than the Pi because its "conveyor
  belt" (memory bandwidth) is wider and it has GPU + NPU. LLM decoding is bandwidth-bound.
- **Fit rule of thumb:** usable RAM = RAM − 25 %; max parameters (billions) ≈ usable RAM (GB) ÷ 0.6.
  Her example: 32 GB → 24 GB usable → 24 / 0.6 = 40B. (0.6 GB per billion parameters ≈ a 4-bit quant plus
  overhead — our arithmetic, not hers.)
- Small devices can still use AI by calling a bigger local machine on the same network.

What the video does **not** cover and this week adds: embedding models (not generative) — tiny, fast,
and the backbone of local search, RAG and agent memory; they fit far lower on the ladder than chat models.

## 3 · Sovereign AI, local AI, edge AI — working definitions

- **Local AI:** the model runs on hardware in front of you (laptop, phone, home server).
- **Sovereign AI:** "building and running artificial intelligence on infrastructure you own or directly control,
  inside your own legal jurisdiction, using models and data that never leave your perimeter" — no phone-home,
  operator controls hardware, model version and control plane
  ([Sector88, 2026](https://www.sector88.co/blog/what-is-sovereign-ai)). Residency or on-prem alone is not
  sovereignty.
- **Edge AI:** inference at the point where data is created, often on constrained devices with intermittent
  connectivity — "sovereignty is a matter of physics" there.
- **Embeddings are sensitive data.** Embeddings "encode information about sensitive content; if embedding
  generation or vector storage happens outside the organization's control, it represents a meaningful data
  transfer risk" ([VDF AI, 2026](https://vdf.ai/blog/data-sovereignty-risks-2026-regulated-industries/)).
  Research shows text can be largely reconstructed from its embedding: Morris et al., *Text Embeddings Reveal
  (Almost) As Much As Text*, EMNLP 2023, [arXiv:2310.06816](https://arxiv.org/abs/2310.06816).
- Thailand's PDPA (B.E. 2562) governs personal data — guest voices, CCTV and photos are personal data; keeping
  the embedding step on premises is the simplest compliance story.

## 4 · Hardware facts used in the labs (vendor specs)

Peak memory bandwidth, per the vendor's published spec. Used only for the rough tier projection in Module 06,
labelled as an estimate.

| device | memory | peak bandwidth | source |
|---|---|---|---|
| Raspberry Pi 5 | 8 GB LPDDR4X-4267, 32-bit | ~17 GB/s (4267 MT/s × 4 B) | Raspberry Pi product brief (arithmetic) |
| Apple M4 Max (14-core CPU, 32-core GPU — **this Mac**, 36 GB) | unified | 410 GB/s | Apple MacBook Pro tech specs |
| AMD Ryzen AI Max+ 395 ("Halo") | 128 GB LPDDR5X-8000, 256-bit | 256 GB/s | AMD spec |
| NVIDIA RTX 4090 | 24 GB GDDR6X | 1,008 GB/s | NVIDIA spec |
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | NVIDIA spec |
| NVIDIA DGX Spark (GB10) | 128 GB LPDDR5x | 273 GB/s | NVIDIA spec (see week 25) |

## 5 · EmbeddingGemma 2 on a Mac (Module 08) — verified 2026-10-07 on an Apple M4 Max, 36 GB

Sources (snapshots in `common/sources/`): [mlx-community bf16](https://huggingface.co/mlx-community/embeddinggemma-2-bf16) ·
[mlx-community 4-bit](https://huggingface.co/mlx-community/embeddinggemma-2-4bit) ·
[onnx-community ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX). Also on the Hub: MLX 8/6/5-bit,
mxfp4/mxfp8/nvfp4 builds and LiteRT builds (`litert-community/…`).

| build | size (card) | card's minimum cosine vs FP32 |
|---|---|---|
| MLX bf16 (all encoders) | 1.489 GB | text 0.999937 · image 0.999954 · audio 0.999947 |
| MLX 4-bit (affine, group 64; vision/audio towers stay bf16) | 1.098 GB | text 0.980877 · image 0.990386 · audio 0.974435 |
| ONNX q4 (text 175 MB · vision 109 · audio 189) | 473 MB total | worst 0.975 (text 0.988) |
| ONNX fp16 | 1465 MB total | 0.9998 |

Measured here (Labs 08-1 to 08-3):
- One text query p50: MLX 4-bit 6.1 ms · MLX bf16 7.3 ms · ONNX q4 CPU 15.5 ms · ONNX fp16 CPU 17.2 ms · PyTorch MPS 22.5 ms.
- ONNX Runtime CoreML EP: 238 of 1,066 graph nodes supported (22 %), and it fails at run time — "Input (video_features)
  has a dynamic shape ({-1,512}) but the runtime shape ({0,512}) has zero elements. This is not supported by the CoreML EP."
- MLX multimodal (hotel media): 7/7 cross-modal questions in bf16 and 4-bit; 4-bit minimum cosine text 0.9865 · photo
  0.9831 · voice 0.9794; ≈ 150 ms per photo, 54–59 ms per voicemail (PyTorch MPS in Lab 04-1: 247 / 111 ms).
- MLX multimodal needs the model's `EmbeddingGemma2Processor`, which needs PyTorch installed (transformers 5.19).
- Local service (MLX 4-bit, loopback HTTP): search round trip p50 9.0 ms; 60 s sustained ≈ 409 → 383 notes/s (−6.3 %).
