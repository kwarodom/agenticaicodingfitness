# ▶ Embed Lab 07 — Capstone: Alto Edge Memory

> Part of Week 30 · Multimodal embeddings with EmbeddingGemma 2 on local, sovereign and edge hardware. Every lab runs on your own machine. Every number in this course was measured on the machine that ran it, or it is quoted with its source.

**What you'll actually do**
- Build **Alto Edge Memory**: one offline index of a hotel's notes, photos, voicemails, CCTV and incident records.
- Give it to an agent: a local Gemma 4 with one tool, `search_memory`, that ranks per kind to work around the modality gap.
- Ask it two night-shift questions, in English and in Thai, inside the egress guard.
- Grade the result on evidence only, then find what the grader missed.

**Time** ~60 min · **Difficulty** advanced · **Hardware** any laptop · ~5 GB free RAM · Ollama with a local chat model

**Sources:** everything in Modules 01–06 · [model card](https://huggingface.co/google/embeddinggemma-2) · [Ollama tool calling](https://ollama.com) · [`week30_multimodal_embeddings_edge/RESEARCH.md`](../RESEARCH.md)

## 0 · Before you start

The capstone runs three labs in order: **07-1 builds**, **07-2 runs the agent**, **07-3 grades**. Each one writes its evidence to `week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/`. You need the model cached (Lab 02) and Ollama with a local chat model (Lab 05).

```bash
# on: laptop
ls week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/ 2>/dev/null || echo "empty — Lab 07-1 creates it"
```

✓ Checkpoint: you have run Labs 02-1 and 05-3 successfully at least once.

## 1 · Build the memory

The capstone's design choices come from earlier measurements:

| choice | value | because |
|---|---|---|
| encoders | all (740M) | photos, voice and CCTV must be searchable (Module 04) |
| dimension | **512** | multimodal quality drops faster than text below 512 (model card table; Lab 03) |
| storage | float16 in one SQLite file | small, portable, auditable (Lab 05-2) |
| incidents | interleaved records: text + photo (+ CCTV) | one vector per story (Lab 04-2) |
| network | inside the egress guard | evidence, not a claim (Lab 05-1) |

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_1_build_memory.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 4 · summary
│ kind    items
│ ──────  ─────
│ audio   3
│ image   6
│ note    18
│ record  2
│ video   1
✓ 30 items · 52 KiB · built in 19.8s · left this machine: 0 connections
═ week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/edge_memory.sqlite is the whole memory — Lab 07-2 gives it to an agent.
```

✓ Checkpoint: `.runs/edge_memory.sqlite` and `.runs/manifest.json` exist, and the manifest shows five kinds.

## 2 · The agent and its one tool

An agent here is a loop: the LLM reads the question, decides to call `search_memory`, reads the results and answers. The tool uses one trick from Modules 04 and 05: it searches **each kind separately** (notes, photos, voice, video, incident records) and returns the best hits of every kind, each with a `source` to cite. A voicemail can then be cited even when notes score higher overall.

```python
def search_memory(model, store, query, kinds=None, per_kind=2):
    q = model.encode(query, prompt_name="SearchQuery")
    hits = []
    for k in kinds or ["note", "image", "audio", "video", "record"]:   # per-kind: the modality-gap fix
        hits += store.search(q, k=per_kind, kinds={k})
    return hits
```

The system prompt asks the model to always search first, answer only from the results, cite sources in square brackets and answer in the question's language. The lab allows at most 6 LLM calls and only one network destination, Ollama on localhost.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_2_memory_agent.py
```

**Expected output** (captured on this Mac, gemma4:12b)

```
▣ STEP 1 · Night shift handover: what happened in plant room B2, and what is our procedure for it?
  → search_memory({"query": "plant room B2 incident and procedure"})
      0.798  note   Smoke in the plant room  [hotel_kb.json#2]
      0.707  note   Chiller plant efficiency alarm  [hotel_kb.json#0]
      0.632  image  Photo: water pooling under AHU-3  [media/leak_under_ahu.jpg]
      0.615  image  Photo: plant room B2, haze  [media/plantroom_smoke.jpg]
      0.628  audio  Voicemail: towels for room 812, no rush  [media/request_towels_en.wav]
      0.614  audio  Voicemail: room 1507 bathroom flooding  [media/voicemail_flood_en.wav]
      0.692  video  CCTV: plant room B2, haze appears after ~4 s  [media/cctv_plantroom.mp4]
      0.776  record Incident 0412 · plant room B2 haze  [incident log]
      0.718  record Incident 0413 · AHU-3 condensate leak  [incident log]
  · ANSWER A haze was observed in plant room B2 at 02:10 [incident log]. If smoke or haze is detected in any plant room, it is treated as a fire event [hotel_kb.json#2]. The procedure is to press the nearest call point, evacuate the area, and contact security on extension 2222 [hotel_kb.json#2]. Do not attempt to locate the source of the smoke yourself [hotel_kb.json#2].

▣ STEP 2 · แขกห้อง 1203 แจ้งเรื่องอะไรเข้ามาบ้าง และควรทำอะไรต่อ
  → search_memory({"query": "room 1203 report issue"})
      0.698  note   Smoke in the plant room  [hotel_kb.json#2]
      0.691  note   Noise complaint at night  [hotel_kb.json#13]
      0.649  image  Photo: thermostat in room 1203 reads 29.1 °C  [media/thermostat_room1203.png]
      0.632  image  Screenshot: BMS trend for AHU-3  [media/bms_trend_ahu3.png]
      0.704  audio  Voicemail: towels for room 812, no rush  [media/request_towels_en.wav]
      0.655  audio  Voicemail (Thai): room 1203, AC not cold, burning smell  [media/voicemail_burning_th.wav]
      0.597  video  CCTV: plant room B2, haze appears after ~4 s  [media/cctv_plantroom.mp4]
      0.678  record Incident 0413 · AHU-3 condensate leak  [incident log]
      0.653  record Incident 0412 · plant room B2 haze  [incident log]
  · ANSWER แขกห้อง 1203 แจ้งว่าเครื่องปรับอากาศไม่เย็นและมีกลิ่นไหม้ [media/voicemail_burning_th.wav] คุณควรตรวจสอบระบบปรับอากาศและตรวจสอบกลิ่นไหม้ทันที หากพบกลุ่มควันหรือควันในห้องเครื่อง ให้กดปุ่มแจ้งเหตุเพลิงไหมฯ อพยพคน และแจ้งฝ่ายรักษาความปลอดภัยที่เบอร์ 2222 ทันที [hotel_kb.json#2]

▣ STEP 3 · evidence
│ allowed  connect  ::1:11434  ×4
│ allowed  connect  127.0.0.1:11434  ×4
✓ 4 LLM calls · left this machine: 0 connections · log → .runs/agent_run.json
```

Look at what per-kind search bought. In question 2, the Thai voicemail ranks below a towel voicemail and two notes, yet it reached the model because the audio list always has room. Notice what the LLM actually knew: it never heard the audio. EmbeddingGemma 2 matched the recording; the LLM only read the **title** stored with it ("AC not cold, burning smell"). The metadata you write at ingest time is what the agent can say. Read the Thai answer closely, though, before you move on to the grade.

✓ Checkpoint: your run shows both answers, with citations, and `left this machine: 0 connections`.

## 3 · Grade on evidence

Lab 07-3 reads only files in `.runs/` and re-runs a fixed retrieval check. Every point is traceable to a file or a measurement:

| area | points | evidence |
|---|---|---|
| index | 20 | manifest exists; all five kinds present |
| retrieval | 25 | 8 fixed questions, per-kind search, expected item first |
| sovereignty | 20 | 0 connections left the machine during build **and** agent run |
| agent | 35 | both answered; every citation is a real source; B2 answer cites procedure + camera/photo/incident; Thai answer cites the Thai voicemail |

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_3_grade.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 2 · retrieval (25) — per-kind search, expected item first
  ✓ [note  ] procedure for smoke in a plant room        → Smoke in the plant room
  ✓ [video ] haze seen on the plant room camera         → CCTV: plant room B2, haze appears after ~4 s
  ✓ [image ] photo of water under an air handler        → Photo: water pooling under AHU-3
  ✓ [image ] thermostat display showing a hot room      → Photo: thermostat in room 1203 reads 29.1 °C
  ✓ [audio ] แขกแจ้งว่าแอร์มีกลิ่นไหม้                  → Voicemail (Thai): room 1203, AC not cold, burn
  ✓ [audio ] guest bathroom is flooding                 → Voicemail: room 1507 bathroom flooding
  ✓ [record] incident report about haze in B2           → Incident 0412 · plant room B2 haze
  ✓ [note  ] supplier wants payment to a new bank accou → Supplier bank detail changes
✓ 8/8 · +25

▣ STEP 3 · sovereignty (20)
✓ connections that left the machine — build: 0 · agent: 0 · +20

▣ STEP 4 · agent (35)
  citations: 3 · real sources: 3 · not in the index: none
  ✓ plant-room answer cites the smoke procedure AND a camera/photo/incident item
  ✓ Thai answer cites the Thai voicemail
✓ +35

▣ STEP 5 · result
  ████████████████████████████████████████ 100/100
═ 100/100 — distinction. Every point above came from a file in .runs/ or a measurement just now.
```

✓ Checkpoint: your grade is printed, and you can point at the file behind each of the four areas.

## 4 · What the grader missed

100/100, but read the Thai answer again. It cites the right voicemail, then tells the engineer to press the fire call point and evacuate, which is the **plant-room smoke** procedure. The right note, "แอร์มีกลิ่นไหม้" (AC burning smell: switch it off, move the guest, send an electrician), was never retrieved. The agent searched for a vague "room 1203 report issue" instead of the problem itself, and the closest procedure it saw was the plant-room one.

The grader checked that citations **exist and are real**. It did not check that the procedure **fits the problem**. That is the most useful lesson in this capstone. Three ways to close the gap:

| fix | where |
|---|---|
| a two-step search: find the report, then search again for its **content** ("AC burning smell procedure") | the system prompt, or a second tool call |
| a grader check: the Thai answer must cite `hotel_kb.json#16` | `lab07_3_grade.py` |
| keep a human in the loop for any safety procedure | operations, not code |

✓ Checkpoint: you can name the flaw in one sentence and the grader check that would catch it (Exercise 07 asks for exactly this).

## Labs — run them here

**labs/lab07_1_build_memory.py** — Notes, photos, voicemails, CCTV and interleaved incident records in one 512-d SQLite memory, built inside the egress guard.

**labs/lab07_2_memory_agent.py** — A local Gemma 4 with a per-kind `search_memory` tool, answering two night-shift questions in English and Thai.

**labs/lab07_3_grade.py** — A 100-point grade from evidence files and a fixed retrieval check, with nothing scored on opinion.

## Try it yourself

`exercises/ex07_memory_tool.py` has three TODOs:

1. `per_kind_top(hits, per_kind)`: the modality-gap fix, by hand.
2. `cited(answer)` and `unknown_citations(answer, sources)`: catch made-up citations.
3. In 1–3 sentences: what did Lab 07-3 score 100 that it should not have?

Then go further: add your blind-spot check to `lab07_3_grade.py`, change the system prompt in `lab07_2_memory_agent.py` so the agent searches for the problem itself, and run 07-2 and 07-3 again.

```bash
# on: laptop
week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/exercises/ex07_memory_tool.py
```

**Expected output** (captured on this Mac, all TODOs filled)

```
✓ per_kind_top: the best hit OF EACH KIND, then sorted by score (the modality-gap fix)
✓ cited / unknown_citations: every [source] in order · flag the ones the index never returned
✓ your note names what Lab 07-3 scored 100 but should not have
```

<details><summary>Hint — the blind spot</summary>

Compare the Thai answer's advice with note #16 in `week30_multimodal_embeddings_edge/common/data/hotel_kb.json`, then look at the query the agent sent to `search_memory`.

</details>

✓ Checkpoint: all three checker lines are ✓, and your improved grader catches the Thai answer.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Lab 07-2: "run Lab 07-1 first" | no `.runs/edge_memory.sqlite` | run Lab 07-1 |
| the agent answers without calling the tool | small models sometimes skip tools | use `gemma4:12b`; the system prompt says "always call search_memory first" |
| `tool_calls` arguments arrive as a string | some models return JSON text | the lab parses both forms |
| the grade changes between runs | the LLM's wording and choices vary | the retrieval and sovereignty parts are deterministic; rerun 07-2 for the agent part |

## Next

[Lab 08 — Your Mac as the edge box](../08_mac_edge_box/TUTORIAL.md): run the same model with MLX and ONNX on Apple silicon, and turn a Mac mini into the hotel's always-on embedding server.

Then, what to build next:
- Run Lab 06-1 on a real edge box (a Raspberry Pi 5, a Jetson, a DGX Spark) and replace the ESTIMATE lines with measurements.
- Upgrade Ollama and rerun Lab 03-3: the text-only GGUF turns ✓, and you can compare it with the PyTorch model.
- Point Alto Edge Memory at your own **approved, synthetic or consented** data.

Or start again from [Lab 01 — Local, sovereign, edge](../01_local_sovereign_edge/TUTORIAL.md).
