# ▶ Embed Lab 05 — Sovereign RAG: nothing leaves the building

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Turn "it runs locally" into **evidence**: a log of every network connection the pipeline tried.
- Watch two deliberate leaks get blocked.
- Store a hotel's notes, photos, voicemails and CCTV in **one SQLite file** at 256 dimensions.
- Run retrieval-augmented generation end to end on your machine: EmbeddingGemma 2 finds the notes, and a local Gemma 4 answers in English and Thai, citing them.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** any laptop · Ollama with a local chat model for Lab 05-3

**Sources:** [Sector88 — What is sovereign AI?](https://www.sector88.co/blog/what-is-sovereign-ai) · [VDF AI — data sovereignty risks](https://vdf.ai/blog/data-sovereignty-risks-2026-regulated-industries/) · Morris et al., [*Text Embeddings Reveal (Almost) As Much As Text*](https://arxiv.org/abs/2310.06816), EMNLP 2023 · Thailand PDPA B.E. 2562 · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §3

## 0 · Before you start

Lab 05-3 needs a local chat model in Ollama. The labs pick `gemma4:12b` if you have it, and otherwise another **local** model. They never pick a `:cloud` model, because a cloud passthrough is not sovereign.

```bash
# on: laptop
curl -s localhost:11434/api/tags | python3 -c "import json,sys; print([m['name'] for m in json.load(sys.stdin)['models'] if 'cloud' not in m['name']])"
```

✓ Checkpoint: you see at least one local model, or you plan to `ollama pull gemma4:12b`.

## 1 · Why the embedding step decides sovereignty

A cloud RAG pipeline sends three things out of the building: every **document** (to be embedded), every **question** (to be embedded and answered), and it stores the **vectors** on someone else's disk. For a hotel those documents are guest voicemails, CCTV, room photos and maintenance notes, which are personal data under Thailand's PDPA.

The vectors are not a safe summary. "Embeddings encode information about sensitive content; if embedding generation or vector storage happens outside the organization's control, it represents a meaningful data transfer risk" (VDF AI, 2026). Researchers have rebuilt much of the original text from embeddings alone (Morris et al., EMNLP 2023).

Embedding is also the **bulky** step: it reads every photo and every second of audio. Doing it locally removes most of the data-transfer problem at once. Then the LLM can be local too (Section 4).

✓ Checkpoint: you can list the three things a cloud RAG pipeline sends out, and why "we only send vectors" is not a privacy guarantee.

## 2 · Evidence, not claims: the egress guard

`week30_multimodal_embeddings_edge/common/egress.py` wraps a block of code so that only **loopback** (this machine) is reachable, and logs every attempt to connect or look up a host name. Lab 05-1 runs the whole pipeline inside it: load all encoders, embed notes and a photo, and search. Then it tries two leaks on purpose.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_1_egress_proof.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 2 · load + embed + search inside the guard
✓ loaded google/embeddinggemma-2 · encoders audio+image+text · 744.4M params · mps · 2.7s
✓ pipeline finished · top hit: AHU condensate leak (0.808)
✓ connection attempts during the pipeline: 0 · blocked 0 · left this machine: 0

▣ STEP 3 · now try to leak — the guard must block both
✓ fetch the model card from huggingface.co: blocked — egress guard: blocked DNS lookup of huggingface.co
✓ POST a query vector to a cloud embedding API: blocked — egress guard: blocked DNS lookup of api.example-embeddings.com
│ BLOCKED  dns      huggingface.co:443  ×1
│ BLOCKED  dns      api.example-embeddings.com:443  ×1
```

Zero connection attempts: once the model is cached, loading, embedding and searching never touch the network. `embedkit` sets `HF_HUB_OFFLINE=1` for exactly this reason.

Be precise about what the guard is. It is a **tripwire inside one Python process**, good for tests and audits. A subprocess, another program or a raw system call would get past it. Enforce sovereignty for real **outside** the app, with a host firewall, a network policy (Week 26's OpenShell policies are one way) or an air gap.

✓ Checkpoint: you can say what the guard proves and what it does not.

## 3 · One file holds the hotel's memory

`week30_multimodal_embeddings_edge/common/vectorstore.py` is a tiny vector store: SQLite for the records, numpy for the maths, no server. Each row keeps what a person needs to trust a hit (kind, title, text, source) next to a 256-d float16 vector. Lab 03 showed 256-d keeps the quality at a third of the size.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_2_local_vector_store.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · embed and store everything
✓ 28 items · {'audio': 3, 'image': 6, 'note': 18, 'video': 1} · 4.0s · 32 KiB on disk

▣ STEP 2 · reopen the file, as tomorrow's process would, and search
  ✓ water leaking under an air handler           [all kinds  ] → note:AHU condensate leak 0.836 · image:Water pooling under AHU-3 0.728
  ✓ water leaking under an air handler           [image      ] → image:Water pooling under AHU-3 0.728 · image:Plant room B2, haze 0.673
  ✕ guest called about a burning smell from the  [all kinds  ] → note:Guest room too hot 0.789 · note:แอร์มีกลิ่นไหม้ 0.747
  ✕ guest called about a burning smell from the  [audio      ] → audio:Voicemail: bathroom flooding 0.682 · audio:Voicemail: towels for 812 0.680
  ✓ แขกโทรมาบอกว่าแอร์มีกลิ่นไหม้                [audio      ] → audio:Voicemail (Thai): AC burning 0.735 · audio:Voicemail: towels for 812 0.609
  ✓ haze on the plant room camera                [image+video] → video:CCTV: plant room B2, 10 s 0.718 · image:Plant room B2, haze 0.666
⚠ 4/6 found the expected item first
◆ the English 'burning smell' question lands on an English AC note before the Thai one, and misses the Thai voicemail even inside the audio filter — a filter removes the notes, it does not close the language + modality gap. Asked in Thai, the voicemail is found first.
✓ left this machine: 0 connections (attempts 0, blocked 0)
```

A whole hotel's multimodal memory, notes, photos, voice and video, is a **32 KiB file**. You can back it up, audit it, copy it to another edge box and delete it. The two ✕ lines are Lab 04's modality gap again. The `kinds` filter stops notes crowding out media, but it does not close a **language + modality** gap: the English question still misses the Thai voicemail. Asked in Thai, it is found first. Module 07's agent handles this by searching every kind and letting the LLM read the results.

✓ Checkpoint: you know where the index file is and could copy it to another machine and search it there.

## 4 · Private RAG: a local model writes the answer

RAG in two steps. **Retrieve**: embed the question with the `QuestionAnswering` prefix and take the top 3 notes. **Generate**: give those notes to the LLM and ask it to answer **only** from them, citing each one. Both models run here, and the egress guard allows exactly one destination: Ollama on `localhost:11434`.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_3_private_rag.py
```

**Expected output** (captured on this Mac, gemma4:12b)

```
▣ STEP 1 · There's water on the floor under an air handler in the basement. What do I do first?
  [1] 0.817  AHU condensate leak
  [2] 0.670  Bathroom flooding
  [3] 0.664  Chiller plant efficiency alarm
  · ANSWER You should isolate the unit and put out wet-floor signs. Then, clear the drain before restarting the unit [1].
◆ 26 tokens in 27.9s on this machine (includes loading the model into memory)

▣ STEP 2 · แขกห้อง 1203 บอกว่าแอร์มีกลิ่นไหม้ ต้องทำอะไรบ้าง
  [1] 0.707  แอร์มีกลิ่นไหม้
  [2] 0.687  Smoke in the plant room
  [3] 0.668  Guest room too hot
  · ANSWER หากแขกแจ้งว่าแอร์มีกลิ่นไหม้ ให้ปิดเครื่องทันทีและย้ายแขกออกจากห้อง [1] จากนั้นต้องส่งช่างไฟฟ้ามาตรวจเช็คพัดลมและมอเตอร์ก่อนที่จะเปิดใช้งานอีกครั้ง [1]
◆ 54 tokens in 3.7s on this machine

▣ STEP 3 · A supplier emailed asking us to pay their invoice into a new bank account. Can I just update it?
  [1] 0.809  Supplier bank detail changes
  [2] 0.585  Bathroom flooding
  [3] 0.553  Guest room too hot
  · ANSWER No, you must not pay an invoice that asks to change the supplier's bank account. You must call the supplier using the number already on file and obtain approval from the finance director first [1].

▣ STEP 4 · what crossed the network
│ allowed  connect  ::1:11434  ×3
│ allowed  connect  127.0.0.1:11434  ×3
✓ left this machine: 0 connections
```

The Thai question got a Thai answer from the Thai note. The bank-change answer refused the fraud pattern by quoting the procedure. The only connections went to this machine's own Ollama. The first answer took 27.9 s because Ollama loaded Gemma 4 into memory; after that, answers took 3–4 s. That is the difference between "hosted in our region" and sovereign: you own every step.

✓ Checkpoint: your run shows answers with `[n]` citations and `left this machine: 0 connections`.

## Labs — run them here

**labs/lab05_1_egress_proof.py** — The full embed pipeline inside a loopback-only guard, every connection attempt logged, two deliberate leaks blocked.

**labs/lab05_2_local_vector_store.py** — Notes, photos, voicemails and CCTV in one 32 KiB SQLite file at 256-d float16, reopened and searched by kind.

**labs/lab05_3_private_rag.py** — Local retrieval plus a local Gemma 4 answering in English and Thai with citations, with network evidence.

## Try it yourself

`exercises/ex05_egress_policy.py` writes the rule the guard enforces:

1. `allowed(host, port)`: loopback only, and only the Ollama port.
2. `leaked(log)`: allowed connections that went to a non-loopback host.
3. Which artefacts must stay local in sovereign RAG?
4. In your own words: why is an in-process guard a tripwire and not a firewall?

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/exercises/ex05_egress_policy.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ allowed(): loopback (127.0.0.0/8, ::1, localhost) on port 11434 only
✓ leaked(): only ALLOWED connections to non-loopback hosts count
✓ in sovereign RAG every artefact stays: audio, questions, vectors, prompts
✓ your note says what the in-process guard cannot see AND what enforces sovereignty for real
```

<details><summary>Hint — loopback</summary>

`ipaddress.ip_address("127.8.8.8").is_loopback` is `True`: the whole 127.0.0.0/8 block is loopback, not just 127.0.0.1. The name `localhost` raises `ValueError` there, so handle it first.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Lab 05-3 exits early: "Ollama is not running" | no Ollama on localhost:11434 | `ollama serve`, then `ollama pull gemma4:12b` |
| an answer is empty | Gemma 4 is a thinking model and spent its tokens thinking | the lab sends `think: false` and 1,536 tokens; raise `max_tokens` for long answers |
| `EgressBlocked: … huggingface.co` while loading | the model is not cached yet | run Lab 02-1 once with network, then rerun |
| the first answer is slow | Ollama loads the chat model into memory on the first call | normal; later calls take seconds |

## Next

[Lab 06 — Benchmark like an edge engineer](../06_edge_benchmarking/TUTORIAL.md): p50 and p95 for every modality, why float16 is banned, and what really limits an embedding model.
