# ▶ Embed Lab 02 — Meet EmbeddingGemma 2: vectors, similarity, task prefixes

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Download EmbeddingGemma 2 once (1.5 GB). From then on, every lab runs offline from the local cache.
- Turn a sentence into 768 numbers, and see why the length of that vector is 1.
- Rank 18 hotel operations notes by **meaning**, with no keyword overlap needed.
- Measure what the model's **task prefixes** are worth on 18 real questions.
- Load only the encoders you need, and measure what each one costs in memory.

**Time** ~45 min · **Difficulty** beginner · **Hardware** any laptop · 1.5 GB disk, ~2 GB free RAM

**Sources:** [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2) (snapshot: `week30_multimodal_embeddings_edge/common/sources/embeddinggemma-2-model-card.md`) · [launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) · [sentence-transformers](https://sbert.net) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §1

## 0 · Before you start

The model is Apache 2.0 and **not gated**: no Hugging Face login is needed. Lab 02-1 downloads it the first time it runs. The checkpoint is `model.safetensors`, 1,488,915,288 bytes. After that, `embedkit` sets `HF_HUB_OFFLINE=1` as soon as it finds the model in your cache, so no lab contacts the network again.

```bash
# on: laptop
ls ~/.cache/huggingface/hub/ | grep embeddinggemma-2 || echo "not downloaded yet — Lab 02-1 fetches it"
```

The data this week is **synthetic**: `week30_multimodal_embeddings_edge/common/data/hotel_kb.json` holds 18 invented hotel operations notes (two of them in Thai) and 18 questions, each with its correct answer. `week30_multimodal_embeddings_edge/common/media/` holds synthetic photos, voicemails and a CCTV clip, made for Week 24.

✓ Checkpoint: you know where the model will be cached and that no account or token is needed.

## 1 · An embedding is a point in meaning-space

An embedding model reads an input and returns a fixed-length list of numbers, a **vector**. Inputs that mean similar things get vectors that point in similar directions. You compare two vectors with **cosine similarity**: 1 means the same direction, 0 means unrelated, −1 means opposite.

EmbeddingGemma 2's last module normalises every vector to length 1. That makes cosine similarity a plain dot product, which is why vector databases can rank millions of items quickly.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_1_first_vectors.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · load the text-only model (vision and audio encoders left out)
✓ loaded google/embeddinggemma-2 · encoders text · 271.0M params · mps · 1.0s

▣ STEP 2 · one sentence → one vector
│ property        value
│ ──────────────  ─────────────────────────────────────────────────
│ shape           (768,)
│ dtype           float32
│ first 6 values  [-0.0221  0.0083  0.0347 -0.0084  0.0117  0.014 ]
│ L2 norm         0.9996
✓ 768 numbers, length ≈ 1 — the model's last module normalises every vector

▣ STEP 3 · the model card's quick start — similarity is a dot product of unit vectors
│                               lovely  sunny  stadium
│ ────────────────────────────  ──────  ─────  ───────
│ The weather is lovely today.  1.000   0.851  0.672
│ It's so sunny outside!        0.851   1.000  0.681
│ He drove to the stadium.      0.672   0.681  1.000
✓ 'lovely' vs 'sunny' = 0.851 > 'lovely' vs 'stadium' = 0.672
```

The norm prints as 0.9996 rather than exactly 1 because the model runs in bfloat16 on the Apple GPU, which keeps about three significant digits.

✓ Checkpoint: you can say what shape the vector has (768) and why cosine similarity equals the dot product here.

## 2 · Your first search by meaning

Step 4 of the same lab embeds the 18 hotel notes and asks three questions that share almost no words with the right answers.

**Expected output** (captured on this Mac)

```
▣ STEP 4 · first semantic search over the hotel knowledge base (SYNTHETIC)
◆ embedded 18 documents in 0.24s
  'water on the floor under the air handler'
     1. 0.821  AHU condensate leak
     2. 0.659  Bathroom flooding
  'a guest wants more towels'
     1. 0.780  Towel and amenity requests
     2. 0.692  Guest room too hot
  'someone is stuck in the lift'
     1. 0.819  Lift entrapment
     2. 0.661  Pool pump trip
```

"Air handler" finds "AHU", and "lift" finds "Lift entrapment" even though the note says "elevator" nowhere. Keyword search finds words; embedding search finds meaning. Look at the gap between the first and second hit too (0.821 vs 0.659). A clear gap means a confident match.

✓ Checkpoint: you ran the lab and every question's top hit is the note you would have picked yourself.

## 3 · Task prefixes — measured, not assumed

EmbeddingGemma 2 was trained with a short instruction in front of every text. Queries get `task: search result | query: …`; documents get `title: … | text: …`. Media gets no prefix. The model card says leaving them out "still works but reduces precision". Lab 02-2 measures that claim on this week's data.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_2_task_prefixes.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · the prompt strings this checkpoint ships with (config_sentence_transformers.json)
│ prompt_name         prepended text
│ ──────────────────  ─────────────────────────────────────
│ SearchQuery         'task: search result | query: '
│ Document            'title: none | text: '
│ QuestionAnswering   'task: question answering | query: '
│ Classification      'task: classification | query: '
│ Clustering          'task: clustering | query: '
│ SentenceSimilarity  'task: sentence similarity | query: '

▣ STEP 2 · three ways to embed the same data
│ how                                   recall@1  MRR
│ ────────────────────────────────────  ────────  ─────  ────────────────────
│ A · no prefixes at all                0.944     0.972  ███████████████████░
│ B · prompts (SearchQuery / Document)  1.000     1.000  ████████████████████
│ C · prompts + real titles             1.000     1.000  ████████████████████

▣ STEP 3 · where they disagree
  guest says room 1203 is boiling hot                  A→✕ แอร์มีกลิ่นไหม้             C→✓ Guest room too hot
```

Without prefixes, one question in 18 went to the wrong note. With them, all 18 were right. Eighteen questions is a small test, and the model card's own benchmarks are the bigger evidence. The habit is cheap, though: a query prefix on every question and `title: … | text: …` on every document. In sentence-transformers that is `prompt_name="SearchQuery"` for queries. `prompt_name="Document"` writes `title: none`, so format real titles yourself.

✓ Checkpoint: you can write the exact query and document strings for a question-answering task from memory.

## 4 · Load only what you use

The 740M parameters are three parts you can pick: text (270M), vision (170M) and audio (300M). sentence-transformers leaves an encoder out when its config is `None`:

| active modalities | `config_kwargs` | size (model card) |
|---|---|---|
| text only | `{"vision_config": None, "audio_config": None}` | 270M |
| text + image | `{"audio_config": None}` | 440M |
| text + audio | `{"vision_config": None}` | 570M |
| everything | `{}` | 740M |

Lab 02-3 loads each configuration in a fresh process on the CPU in float32, as a small Linux edge box would, and measures the memory.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_3_selective_loading.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 5 · compare
│ configuration  params  weights fp32  RSS after encode  peak RSS  load
│ ─────────────  ──────  ────────────  ────────────────  ────────  ────  ────────────────────
│ text only      271M    1034 MB       1760 MB           2249 MB   2.9s  ██████████░░░░░░░░░░
│ text + image   439M    1674 MB       2405 MB           2929 MB   3.0s  █████████████░░░░░░░
│ text + audio   577M    2200 MB       1466 MB           3721 MB   3.3s  █████████████████░░░
│ all encoders   744M    2840 MB       1223 MB           4686 MB   3.6s  ████████████████████
✓ text vector with and without media encoders: cosine 1.000000 on the first 16 dims — identical
◆ size an edge box by the PEAK: macOS compresses idle pages, so RSS after encode can read below the weights
```

Two lessons. First, the peak memory more than doubles from text-only (2.2 GB) to everything (4.7 GB) in float32, so on a small box you load the text model and add an encoder only where photos or voice must be searchable. Second, the text vectors are **identical** either way, so you can index notes with the small model today and add photos later without re-embedding anything.

✓ Checkpoint: you can say which `config_kwargs` you would ship on a gateway that only searches maintenance notes.

## Labs — run them here

**labs/lab02_1_first_vectors.py** — Load the text model, look inside one vector, reproduce the model card's quick start, and search the hotel notes by meaning.

**labs/lab02_2_task_prefixes.py** — The same 18 questions embedded three ways, with recall@1 and MRR to show what task prefixes are worth.

**labs/lab02_3_selective_loading.py** — Each encoder configuration in its own process, with weights, resident memory and peak memory measured.

## Try it yourself

`exercises/ex02_cosine_and_rank.py` has four TODOs. Together they are the core of every search this week, written by hand in numpy:

1. `cosine(a, b)`
2. `rank(q, docs, k)`: the best k rows, highest first
3. `query_text(text, task)`: the model card's query prefix
4. `doc_text(title, text)`: the document format, with `title: none` when there is no title

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/exercises/ex02_cosine_and_rank.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ cosine: (3,4)·(4,3) = 0.96 · self = 1 · opposite = −1
✓ rank: best three documents in order [1, 2, 0]
✓ query_text: the model card's search and question-answering prefixes, exactly
✓ doc_text: 'title: … | text: …', and 'title: none' when there is no title
```

<details><summary>Hint — rank()</summary>

Score every row with your `cosine`, sort the `(index, score)` pairs by score from high to low, and keep the first k.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: … model type embedding_gemma2 …` | transformers older than 5.19 (the shared repo `.venv` has 5.10) | run the labs with `week30_multimodal_embeddings_edge/.venv-embed/bin/python` |
| the first run hangs at "loading" | it is downloading 1.5 GB | wait once; later runs load in about 2 s |
| `OfflineModeIsEnabled` | `HF_HUB_OFFLINE=1` is set but the model is not cached yet | `unset HF_HUB_OFFLINE` once, run Lab 02-1, then go offline again |
| vectors are NaN | something loaded the model in float16 | use bfloat16 or float32, never float16 (model card; Module 06 tests it) |

## Next

[Lab 03 — Smaller vectors, smaller models](../03_matryoshka_and_quantization/TUTORIAL.md): cut vectors to 128 numbers, store them in 1 byte or 1 bit, and shrink the model itself.
