# ▶ Embed Lab 04 — One space for text, photos, voice and video

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Put photos, voicemails and a CCTV clip into the **same** 768-number space as text, with no captions and no transcripts.
- Search them with plain-English questions, and with Thai ones.
- Build one vector for a whole incident: text, a photo and a video clip in a single input.
- Budget the 8,192-token context that every modality shares.
- Meet the two limits of embedding search: the **modality gap**, and the fact that it cannot see what is **not** there.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** any laptop · ~5 GB free RAM for all encoders

**Sources:** [model card — Multimodal Input, Context Limits](https://huggingface.co/google/embeddinggemma-2) · [launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) · media: `week30_multimodal_embeddings_edge/common/media/` (synthetic, from Week 24 Module 13) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md) §1

## 0 · Before you start

Video needs a decoder. `torchvision` no longer reads video, so sentence-transformers uses **torchcodec**, which needs ffmpeg on the machine.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python -c "import torchcodec; print('torchcodec', torchcodec.__version__)"
ffmpeg -version | head -1
```

All media in this module is **synthetic**: the photos were made with an image model, the voices with text-to-speech, and the clip is generated. The voicemails are mono 16 kHz, the format the model card asks for.

✓ Checkpoint: both commands print a version.

## 1 · Cross-modal search: a sentence finds a photo

With every encoder loaded (740M), an image or an audio clip goes into the same space as text. The question is the only text; the photos and voicemails get **no prefix** (model card: "Prefixes apply to text only").

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_1_cross_modal.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 2 · embed the media — no captions, no transcripts
✓ 6 images in 1.48s (247 ms each) · 3 voicemails (8.8–12.1 s each) in 0.33s (111 ms each)

▣ STEP 3 · seven questions, ranked against all nine items
  ✓ water leaking on the floor under equipment                 → leak_under_ahu.jpg 0.725  plantroom_smoke.jpg 0.630
  ✓ smoke or haze in the plant room                            → plantroom_smoke.jpg 0.658  leak_under_ahu.jpg 0.632
  ✓ a hotel room thermostat that is too hot                    → thermostat_room1203.png 0.724  voicemail_flood_en.wav 0.648
  ✓ a suspicious invoice asking to change bank details         → invoice_bank_change.png 0.763  request_towels_en.wav 0.561
  ✓ a guest reports flooding in the bathroom                   → voicemail_flood_en.wav 0.785  request_towels_en.wav 0.672
  ✓ a guest needs more towels                                  → request_towels_en.wav 0.788  voicemail_flood_en.wav 0.671
  ✓ a Thai speaker reports a burning smell from the air condit → voicemail_burning_th.wav 0.692  invoice_bank_change.png 0.637
✓ 7/7 questions found the right item first

▣ STEP 4 · image ↔ image: which photo is closest to the normal plant room?
  0.969  plantroom_smoke.jpg
  0.848  leak_under_ahu.jpg
  0.664  thermostat_room1203.png
```

The model read the thermostat's display, recognised a bank-change invoice and understood a spoken request for towels, with no speech-to-text step. Step 4 shows a quieter fact: the smoky plant room is 0.969 similar to the normal one. To an embedding they are the same room, and the haze is a small difference. Keep that in mind for Section 4.

✓ Checkpoint: all seven questions find the right item first on your machine.

## 2 · Video and interleaved records

A CCTV clip is read as frames: by default **1 frame per second**, **140 tokens per frame**. One input can also mix modalities. Placeholders in the text (`<|image|>`, `<|video|>`, `<|audio|>`) are filled from matching keys, in order, and the model returns **one** vector for the whole record:

```python
emb = model.encode({
    "text": "Incident 0412, plant room B2, 02:10. Night shift saw this: <|image|> CCTV from the same camera: <|video|>",
    "image": ["plantroom_smoke.jpg"],
    "video": "cctv_plantroom.mp4",
})
```

All modalities share one **8,192-token** context. Costs from the model card: text 1 token per subword · image 280 · video 140 per frame · audio 25 per second.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_2_video_and_interleaved.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · budget the 8,192-token context (model card token costs)
│ part                       tokens
│ ─────────────────────────  ────────────
│ text (≈ 20 words)          ≈ 30
│ 1 photo                    280
│ CCTV clip 10s → 10 frames  1400
│ total                      1710 of 8192
◆ room left: ≈ 46 more frames, or 259 s of audio, in the same record

▣ STEP 3 · which record answers which question?
  ✓ haze appearing on the plant room camera      → incident: B2 haze (text + photo + CCTV) 0.719   (2nd: cctv_plantroom.mp4 (video  0.686)
  ✓ water leaking from an air handler            → incident: AHU-3 leak (text + photo) 0.681   (2nd: incident: B2 haze (text +  0.620)
  ✓ incident report about smoke at night         → incident: B2 haze (text + photo + CCTV) 0.711   (2nd: incident: AHU-3 leak (text 0.659)
  ✕ what does the plant room normally look like  → incident: B2 haze (text + photo + CCTV) 0.658   (2nd: cctv_plantroom.mp4 (video  0.645)
⚠ 3/4 as expected
◆ an embedding scores what IS in the input. 'Normally' asks about smoke being ABSENT — the haze record is still a plant room, plus text, so it wins. Absence and negation need a rule or a classifier.
```

One vector per incident keeps the index small and lets one question find a whole story. The ✕ is the important line. An embedding measures what **is** in an input, so "normally" and "without smoke" do not push the haze record away. Questions about absence or negation need a filter, a rule or a classifier on top.

✓ Checkpoint: you can work out how many seconds of video fit next to one photo and 30 tokens of text (8,192 − 310 = 7,882 tokens, ÷ 140 ≈ 56 s).

## 3 · Thai and English in one space

The model covers 100+ languages. A Thai question should find an English note and the other way round, without translation.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_3_cross_lingual.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · Thai questions → English notes
  ✓ ปั๊มสระว่ายน้ำตัดบ่อย ทำยังไงดี    → Pool pump trip (0.773)
  ✓ มีคนติดอยู่ในลิฟต์                 → Lift entrapment (0.766)
  ✓ ใบแจ้งหนี้ขอเปลี่ยนเลขบัญชีธนาคาร  → Supplier bank detail changes (0.737)
  ✓ ห้องพักร้อนมาก แขกบ่น              → Guest room too hot (0.677)

▣ STEP 2 · English questions → Thai notes
  ✓ the air conditioner smells like something is burning → แอร์มีกลิ่นไหม้ (0.699)
  ✓ water is dripping from the ceiling                   → น้ำรั่วจากเพดาน (0.721)
✓ 6/6 cross-lingual questions found the right note first

▣ STEP 3 · the Thai voicemail, asked in Thai and in English
  ✓ แขกโทรมาบอกว่าแอร์มีกลิ่นไหม้                    → voicemail_burning_th.wav  flood 0.575 · burning 0.707 · towels 0.588
  ✕ a guest called about a burning smell from the AC → voicemail_flood_en.wav  flood 0.654 · burning 0.628 · towels 0.651

▣ STEP 4 · does the Thai voicemail sit near its Thai note or its English twin?
│ pair                                          cosine
│ ────────────────────────────────────────────  ──────
│ Thai voicemail ↔ Thai note                    0.691
│ Thai voicemail ↔ English note (same meaning)  0.644
│ Thai voicemail ↔ towel request voicemail      0.702
```

Text to text works in both languages, 6 out of 6. Speech is harder. Asked in Thai, the Thai voicemail is found first. Asked in English, it is not, although Lab 04-1's question found it by naming "a Thai speaker".

✓ Checkpoint: you can say which of your users' questions would cross a language boundary and a modality boundary at the same time.

## 4 · The modality gap

Step 4 above shows the pattern. The Thai voicemail is closer to an unrelated **towel** voicemail (0.702) than to the English note that means exactly the same thing (0.644). Items of the same modality cluster together, whatever they mean. This **modality gap** is common to multimodal embedding models, not a bug in this one. Its practical effect: if you rank notes, photos and voicemails in one list, notes tend to crowd out the media.

Two fixes, both used later this week:

| fix | how | where |
|---|---|---|
| filter by kind | search only `audio` when someone asks "which voicemail…" | Lab 05-2's `kinds=` |
| rank per kind, then merge | take the top hits **of each kind** and show them all | Lab 07-2's `search_memory` tool |

✓ Checkpoint: you can explain why a guest voicemail might rank below a loosely related note, and name one fix.

## Labs — run them here

**labs/lab04_1_cross_modal.py** — Seven plain-English questions over six photos and three voicemails in one space, plus image-to-image similarity.

**labs/lab04_2_video_and_interleaved.py** — The token budget, a CCTV clip and two interleaved incident records, with one case it gets wrong and why.

**labs/lab04_3_cross_lingual.py** — Thai ⇄ English across notes and a Thai voicemail, and the modality gap measured.

## Try it yourself

`exercises/ex04_token_budget.py` is the ingest-side safety net for interleaved records:

1. `record_tokens(...)`: the token cost of a record.
2. `fits(tokens)` and `seconds_of_video_left(tokens)`.
3. `check_record(record)`: one placeholder per media item, per modality. A missing `<|image|>` fails silently otherwise.
4. Which task prefix goes in front of a photo?

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/exercises/ex04_token_budget.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ record_tokens: 30+280+10×140 = 1710 · 60 s audio = 1500 · 100+560+300 = 960
✓ fits(): ≤ 8,192 tokens · 1,710 used leaves room for 46 more seconds of video
✓ check_record: one placeholder per media item, per modality — problems listed, none for a good record
✓ media items get NO task prefix (prefixes are for text only)
```

<details><summary>Hint — check_record()</summary>

For each of `image`, `video`, `audio` present in the record, count `record["text"].count("<|image|>")` (and so on). A string counts as one item; a list counts as `len(list)`. Report every modality where the two numbers differ.

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torchcodec is not installed` | video decoding needs torchcodec | `uv pip install --python week30_multimodal_embeddings_edge/.venv-embed/bin/python torchcodec` |
| torchcodec cannot find ffmpeg libraries | ffmpeg is missing | `brew install ffmpeg` (macOS) or your distro's ffmpeg |
| audio vectors look random | stereo or 44.1 kHz audio | convert to mono 16 kHz: `ffmpeg -i in.wav -ac 1 -ar 16000 out.wav` |
| out of memory with all encoders | 740M in float32 peaks near 4.7 GB (Lab 02-3) | load only the encoders you need, or use bfloat16 on a GPU |

## Next

[Lab 05 — Sovereign RAG: nothing leaves the building](../05_sovereign_rag/TUTORIAL.md): prove it with a network log, store everything in one local file, and let a local Gemma 4 answer from it.
