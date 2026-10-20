# Week 30 · authoring guide (for maintainers and agents adding modules)

Week 30 teaches **multimodal embeddings on local, sovereign and edge hardware** with Google DeepMind's
**EmbeddingGemma 2** (released 2026-10-06). The facts come from [`RESEARCH.md`](RESEARCH.md) (model card, launch
blog, GGUF cards, Tina Huang's *Every Size Local AI In 24 Minutes*, sovereign-AI definitions, vendor specs).
The course is served by the **🧭 Embed Lab Runner** (`week30_multimodal_embeddings_edge/00_embed_lab_runner/`, port 8130), a local-only fork
of the Week 26 runner. **Module 02 (`02_first_embeddings/`) is the reference implementation — copy its shape.**

Running example: **Alto Edge Memory** — a hotel's private, offline memory of maintenance photos, CCTV, guest
voicemails (English and Thai) and ops notes, searchable by meaning, that never leaves the building.

## The module list

| # | folder | title | you build |
|---|---|---|---|
| 01 | `01_local_sovereign_edge` | Local, sovereign, edge: the device ladder | a fit calculator + a measured profile of your own machine |
| 02 | `02_first_embeddings` | Meet EmbeddingGemma 2: vectors, similarity, task prefixes | your first embeddings, ranked search, prefix A/B, selective loading |
| 03 | `03_matryoshka_and_quantization` | Smaller vectors, smaller models: Matryoshka, int8 and GGUF | dimension/quality/storage trade-offs; the text-only GGUF on Ollama |
| 04 | `04_multimodal_search` | One space for text, photos, voice and video | cross-modal and cross-lingual search, interleaved incident records |
| 05 | `05_sovereign_rag` | Sovereign RAG: nothing leaves the building | an egress guard, a local vector store, answers from local Gemma 4 |
| 06 | `06_edge_benchmarking` | Benchmark like an edge engineer | latency, throughput and memory per modality; a tier projection |
| 07 | `07_capstone_alto_edge_memory` | Capstone: Alto Edge Memory | an offline multimodal agent with a `search_memory` tool, graded on evidence |
| 08 | `08_mac_edge_box` | Your Mac as the edge box: MLX, ONNX and an always-on embedding server | runtimes measured on Apple silicon; a local embedding service + launchd |

Each module's `## Next` links to the next folder: `[Lab NN — title](../NN_folder/TUTORIAL.md)`.

## Local resources (already installed — do not pip install into the shared `.venv`)

| resource | path | what it is |
|---|---|---|
| embedding venv | `week30_multimodal_embeddings_edge/.venv-embed/` | Python 3.12 · torch 2.14.1 · transformers 5.19.0 · sentence-transformers 6.1.0 · soundfile · librosa. The shared repo `.venv` has transformers 5.10.2, which cannot load `embedding_gemma2`. |
| model | `google/embeddinggemma-2` in `~/.cache/huggingface` | 1.49 GB safetensors, Apache 2.0, not gated |
| media | `week30_multimodal_embeddings_edge/common/media/` | SYNTHETIC hotel media copied from week 24 module 13: 6 images, 1 CCTV mp4, 3 voicemails (mono 16 kHz; one in Thai) |
| sources | `week30_multimodal_embeddings_edge/common/sources/` | model-card snapshot + the video transcript (what `audit_references.py` checks against) |
| MLX venv | `week30_multimodal_embeddings_edge/.venv-mlx/` | mlx 0.32.3 + mlx-vlm @3d87e88 (pinned by the mlx-community cards) + torch (the processor needs it). Labs call `mackit.ensure_mlx()` to re-exec into it. |
| ONNX | `week30_multimodal_embeddings_edge/.models/embeddinggemma-2-ONNX/` (git-ignored) | q4 + fp16 text model; ONNX Runtime refuses the HF cache's symlinked blobs, so it lives here |
| laptop Ollama | `http://localhost:11434` | `gemma4:12b` (thinking model: give `max_tokens` ≥ 1024), `nemotron-3-nano:latest`; module 03 pulls `hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0` |

## Files per module

```text
NN_folder/
  TUTORIAL.md            course text (the runner splits it on "## " headings)
  TUTORIAL.th.md         Thai translation — structurally identical (check_translation.py)
  diagrams.json          architecture + sequence + charts (same schema as week 26)
  labs/labNN_k_<name>.py 2–3 runnable labs; docstring line 1 = "Lab NN-k · <title>."
  exercises/exNN_<name>.py              starter with 2–4 TODOs + an offline checker (exits 1 with ✕ TODO lines)
  exercises/solutions/exNN_<name>.py    reference solution (exits 0)
```

## TUTORIAL.md contract (the runner parses it)

- H1 `# ▶ Embed Lab NN — <title>`, then `> Part of Week 30 · …` blockquote.
- `**What you'll actually do**` bullets, then exactly one meta line:
  `**Time** ~NN min · **Difficulty** beginner|intermediate|advanced · **Hardware** <…>`.
- `**Sources:**` line with links (model card, blog, video, RESEARCH.md).
- Sections in order: `## 0 · Before you start`, `## 1 · …` … `## N · …`, `## Labs — run them here`,
  `## Try it yourself`, `## Troubleshooting`, `## Next`. Every numbered section ends with one `✓ Checkpoint: …` line.
- In "Labs — run them here", one line per lab: `**labs/labNN_k_x.py** — One-sentence title.`
- Bash blocks start with `# on: laptop`.
- `**Expected output**` precedes an output block with provenance in parentheses:
  `(captured on this Mac)` · `(REFERENCE — quoted from <source>)` · `(EXAMPLE — illustrative shape, not a measurement)`.

## Honesty rules

1. Every number a lab prints is measured on the machine running it. Tutorials paste **real** output captured on
   this Mac (M4 Max, 36 GB) and say so. Laptop numbers are a **LAPTOP STAND-IN** for edge devices — never claim
   them for a Pi or a phone.
2. Third-party numbers (benchmarks, Pixel RAM, vendor bandwidth) are quoted with their source from RESEARCH.md.
3. The tier projection in module 06 is an **ESTIMATE** (bandwidth scaling) and is labelled so.
4. The GGUF path is **text-only**; say so wherever it appears.
5. Media is synthetic. Never embed real guest photos, voices or CCTV without approval — even locally.
6. Labs run offline once the model is cached (`embedkit.load()` sets `HF_HUB_OFFLINE=1` when the cache exists).
7. The runner kills a lab after 900 s; keep laptop LLM calls ≤ 6 per lab.

## embedkit (week30_multimodal_embeddings_edge/common/embedkit.py)

Output helpers with the week-26 glyphs (`banner`, `step`, `ok`, `warn`, `note`, `result`, `metric`, `table`,
`bar`) plus: `load(modalities=("text",), dtype=None)` (cached SentenceTransformer with selective encoders) ·
`media(name)` · `MEDIA` · `cosine`, `top_k`, `renorm` · `timer()` · `rss_mb()` · `ollama_up()` ·
`ollama_chat(model, messages, tools=None, max_tokens=1024)` · `ollama_embed(model, texts)` · `RUNS(module)`.
Labs import it with `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))`.
Glyphs the runner colours: `━` banner · `▣` step · `✓` pass · `✕` fail · `⚠` warn · `◆` metric · `═` result ·
`→` action · `│` table · `$` command.

## Thai translation

Same rules as week 26: same `## ` sections in the same order, same `✓ Checkpoint:` count per section, fenced
code blocks byte-identical, same bold `**labs/…**` / `**exercises/…**` paths, literal `**Time** … · **Difficulty** … ·
**Hardware** …` labels and `**Expected output**` markers. Keep technical terms (embedding, vector, Matryoshka,
GGUF, RAG, edge, sovereign) in English where Thai engineers would. Natural Thai, not word-for-word.
