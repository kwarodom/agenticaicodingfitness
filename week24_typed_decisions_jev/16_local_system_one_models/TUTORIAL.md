# ▶ Jev Lab 16 — Local System One models: Laya, GLiNER and GLiNER2.5-Decide vs Jev

> Part of Week 24 · Typed AI decisions with Jev. A month after Jev launched, the **System One** category has more than 25 models. Nine of them publish weights you can run yourself (systemonemodels.org, checked 2026-10-17). This module runs three of them **on your laptop**: **Laya** (Convai Innovations), **GLiNER2.5-Decide** (Fastino Labs), and **classic GLiNER**, the span extractor the GLiNER family started from. You'll see where each one fits next to Jev, then measure all of them fairly on the same tickets.

**What you'll actually do**
- Read the October 2026 landscape, and learn why three scoreboards seem to disagree.
- Run Laya offline: the same JSON as Jev, answered in tens of milliseconds. Then catch its Thai and truncation failures.
- Use classic GLiNER to **extract** room numbers with offsets, and let Jev **decide** which one has the problem.
- Feed GLiNER2.5-Decide the same Jev questions through an adapter. Compare its English and multilingual models.
- Run a fair 18-ticket bake-off: accuracy, speed, cost and calibration.
- Write the adapter and the `pick_engine()` policy for a local-first service.

**Time** ~45 min (+ a one-time ~1 min setup and ~3 GB download) · **Difficulty** intermediate · **Cost** ≈ $0.001 live (Jev side only; local models cost $0) · $0 dry

## 0 · Set up the local models once

The course `.venv` stays light (the Jev client is plain Python). The local models need `torch` and `transformers`, and classic `gliner` pins `transformers < 5.17`. So they get their own environment, `week24_typed_decisions_jev/.venv-local`. Every Module 16 lab detects it and re-runs itself there automatically. Run these two commands **once** from the ⌨ terminal. The download takes longer than the runner's 150 s lab limit.

```bash
uv venv --python 3.13 week24_typed_decisions_jev/.venv-local
uv pip install --python week24_typed_decisions_jev/.venv-local/bin/python laya "gliner2[local]" gliner tiktoken sentencepiece pillow
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab00_download_models.py
```

**Expected output**

```
▣ LOCAL models · python 3.13.13 in week24_typed_decisions_jev/.venv-local · Laya on mps (picks the best device itself) · GLiNER on cpu (its default)
✓ convaiinnovations/laya                      3.8 s
✓ convaiinnovations/laya/multilingual        25.9 s
✓ fastino/GLiNER2.5-Decide                    9.4 s
✓ fastino/GLiNER2.5-multi-Decide              8.5 s
✓ urchade/gliner_multi-v2.1                   8.5 s
◆ done in 56 s — the Module 16 labs now run offline.
```

After this, labs load weights with `HF_HUB_OFFLINE=1`, so a lab never starts a surprise download, and **no ticket text leaves your machine** on the local side. Skip the setup and every lab still runs: the local answers replay from recordings made on 2026-10-17 (labelled `RECORDED`).

✓ Checkpoint: Lab 16-0 printed five ✓ lines, or you decided to learn from the recordings, and you can explain why the local models live in a second venv.

## 1 · The landscape, and three scoreboards that "disagree"

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab01_landscape.py
```

**Expected output**

```
│ board                                     Jev    GLiNER2.5-Decide  Laya
│ Jev Decision Index (indep.) — score       60.11  9.24              4.43
│ Jev Decision Index — calibration error ↓  0.074  0.0884            0.1402
│ JevBench capability score                 77.1   —                 32.5
│ Fastino Fast Decisions (vendor-run) %     57.6   60.2              46.6
│ Banking77, Laya fine-tuned (Ollman) %     80.0   —                 79.4
```

All three rows are true:

- **Jev Decision Index** (independent, broad, zero-shot): Jev is far ahead, and base Laya is close to chance.
- **Fastino's Fast Decisions** (Fastino's own 17 English domains): GLiNER2.5-Decide edges out Jev.
- **Banking77 after fine-tuning Laya on 1,001 examples**: Laya comes within 0.6 points of Jev.

They measure different jobs. Treat every vendor-run table the way Module 10 taught you, and trust only a comparison you run yourself, on your own data, scored the same way for every model. The lab ends with a "which one for what" table for AltoTech.

The models in this module, from their own cards:

| | Laya | GLiNER2.5-Decide | GLiNER (classic) | Jev |
|---|---|---|---|---|
| job | typed decisions (choice/score/noul) | typed classification heads, plus spans | **spans** with offsets (NER) | typed decisions |
| size | 421M EN · 322M multilingual | 340M EN · 287M multi | small/medium/large | undisclosed |
| context | 512 EN · 1,024 multi (8,192 with `max_len`) | not published | short passages | 64k (262k long context) |
| runs on | your GPU/CPU (MLX, Core ML, ONNX, laya.cpp ports exist) | your CPU/GPU | your CPU/GPU | TypeSafe's API |
| licence | Apache-2.0 | Apache-2.0 | Apache-2.0 | closed, $0.042/M input tokens |

✓ Checkpoint: you can explain how GLiNER2.5-Decide can "beat Jev" and also score 9 against Jev's 60, without anyone lying.

## 2 · Laya on your laptop

Laya takes **the same request JSON as Jev**. Its `laya-serve` even exposes `POST /v1/systemone`, so TypeSafe clients can point at it. The `Router` picks a checkpoint from the text's script before the forward pass.

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab02_laya_local.py
```

**Expected output**

```
» queue                  choice  → hvac   (confidence 0.48)
◆ laya/english · LOCAL (this machine) · $0 per call · 43 ms
…
│ question    Laya (local)                Jev (hosted)
│ queue       hvac (peak 0.77)            front_desk (peak 0.60)
│ wants_move  0.93                        0.99
│ latency     43 ms (no network, mps)     448 ms (internet)
…
│ Router (auto)   multilingual  plumbing    0.95       15 ms
│ forced english  english       front_desk  0.28       27 ms
» Router (auto)   ✕ wrong AND confident — the dangerous kind
…
│ laya english (512 tok)           hvac      0.86
│ laya multilingual, max_len=8192  plumbing  0.96
│ Jev (64k context)                safety    1.00
```

What the run teaches:

- **Speed is real.** 43 ms on the Apple GPU (15 ms on the multilingual checkpoint) against ~450 ms for Jev over the internet. Most of that gap is the network round trip.
- **`confidence` means different things.** Laya's `confidence` is its own statistic, and it also returns `answer_confidence` (the peak). Compare *probabilities* across models, never the `confidence` fields.
- **Thai failed confidently.** A burning smell near the electrical cabinet went to `plumbing` at 0.95. Forcing the English checkpoint gave a wrong answer too, but an unsure one, which a gate would catch. Confidence alone can't protect you. Test the language you actually serve.
- **The truncation trap.** The English checkpoint keeps about 320 tokens of state and quietly drops the rest, so the smoke sentence never reached it. Even the 8k multilingual run, which saw everything, missed it. Base Laya is "a fast base to specialise" (Convai's words), not a zero-shot engine. Fine-tune it before you rely on it.

✓ Checkpoint: you can name the two ways Laya failed in this lab, and say which one a confidence gate would catch.

## 3 · Extract with GLiNER, then decide

Classic GLiNER answers a different question: *where* in the text is every "room number", "equipment" or "time"? It returns spans with start/end offsets, which Jev never does. The pattern is **extract, then decide**. GLiNER proposes candidates, a decision model chooses among them, and code keeps the offsets.

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab03_gliner_extract_then_decide.py
```

**Expected output**

```
▣ STEP TH-1 · extract — GLiNER over: ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง 1203 รีบมาดูด่วนค่ะ…
│ room number  3     25–26    0.59
│ room number  1203  36–40    0.64
▣ STEP TH-2 · decide — Jev chooses among the candidates
» problem_room           choice  → room_1203   (confidence 0.88)
      room_1203                  0.92  ██████████████████████░░ ◀
      none_of_these              0.06  █░░░░░░░░░░░░░░░░░░░░░░░
      room_3                     0.02  ░░░░░░░░░░░░░░░░░░░░░░░░
→ code: room 1203, mentioned at offsets 36–40
   ได้กลิ่นไหม้แถวตู้ไฟชั้น 3 ใกล้ห้อง ⟦1203⟧ รีบมาดูด่วนค่ะ
```

GLiNER marked `3` as a room number, but in Thai it is *floor* 3 (ชั้น 3). A span model finds candidates; it does not judge them. In the English ticket, room 1203 appears twice and 1408 once. Jev chose 1203 at 1.00, and code highlighted **both** mentions. Two guard rails stay in code: no candidates means no guess (send it to a person), and `none_of_these` is always an option.

✓ Checkpoint: you can explain why the room options come from GLiNER and not from Jev, and what code does when GLiNER finds nothing.

## 4 · GLiNER2.5-Decide with the same questions

GLiNER2.5-Decide answers typed classification questions in one pass on a CPU. It has no `choice` / `noul` / `score` names, so `localkit.to_gliner_schema()` translates them. A choice becomes labels with descriptions, a noul becomes `["yes", "no"]` with a prompt, and a score becomes ordered labels.

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab04_gliner_decide.py
```

**Expected output**

```
│ question  GLiNER2.5-Decide      Jev
│ queue     safety (0.56)         safety (p 1.00)
│ safety    yes (0.95)            noul 0.98
│ urgency   routine (0.34)        score 2.00 / 2
│ time      160 ms · LOCAL · CPU  518 ms · LIVE
…
│ GLiNER2.5-Decide (English)  plumbing (0.29)  yes (0.61)  149 ms · LOCAL
│ GLiNER2.5-multi-Decide      safety (0.30)    yes (0.86)  56 ms · LOCAL
│ Jev 1.13                    safety (1.00)    noul 0.94   hosted
…
» every topic's score: {'hvac': 0.97, 'billing': 0.42, 'housekeeping': 0.23, 'room_change': 0.39, 'safety': 0.01} → sum 2.02
```

Four lessons:

1. **Thai needs the multi model.** The English model sent the burning smell to `plumbing`. `GLiNER2.5-multi-Decide` got `safety`, and was faster.
2. **Smoke rated `routine`.** The ordinal urgency head is weak. Fastino's own card warns that an ordinal head "does not know that 6 is closer to 7 than 0".
3. **It does things Jev doesn't.** It returns multi-label topics in one head (`hvac` + `billing`), and **spans from the same model** (`extract_entities`).
4. **Its scores are not a distribution.** One topic call summed to 2.02. They are independent per-label scores. Don't threshold them like Jev probabilities, and never average them with Jev's.

✓ Checkpoint: you can say which GLiNER2.5-Decide model you'd use for a Thai ticket, and why its 0.95 isn't comparable to Jev's 0.98.

## 5 · A fair bake-off on 18 tickets

"Fair" means the same text, the same three questions (through one adapter), the same labels and the same scoring, with every engine zero-shot and every vendor's own multilingual advice followed for the Thai tickets.

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab05_bakeoff.py
```

**Expected output**

```
│ question         Jev 1.13  Laya (Router)  GLiNER2.5-Decide (+multi for TH)
│ queue            18/18     12/18          16/18
│ safety           17/18     17/18          16/18
│ urgency          15/18     10/18          10/18
│ queue · EN only  12/12     9/12           11/12
│ queue · TH only  6/6       3/6            5/6
» urgency mean absolute error (0 = perfect, levels 0–2): jev 0.17 · laya 0.44 · decide 0.44
…
│ Jev 1.13          476 ms       internet round trip    $0.020              LIVE
│ Laya              62 ms        this machine, mps      $0 + your hardware  LOCAL
│ GLiNER2.5-Decide  148 ms       this machine, CPU      $0 + your hardware  LOCAL
…
» jev   Brier 0.026 · confidently wrong 0/18
» laya  Brier 0.063 · confidently wrong 1/18
```

How to read it like a reviewer:

- **Direction, not a verdict.** 18 tickets is a classroom sample. The ranking agrees in direction with the independent Jev Decision Index. For a real decision, use Module 09's method on a few hundred of *your* tickets.
- **Cost isn't only per call.** Jev costs about $0.02 per 1,000 tickets. The local engines cost $0 per call, but you pay for the hardware, the updates and the engineering, which is the "you own" row in Module 10.
- **Calibration only where it exists.** Jev and Laya both return P(yes) for `safety`, so a Brier score is fair. GLiNER2.5-Decide returns one score for its winning label, so it is left out on purpose rather than forced into the comparison.
- **The gap can close.** Laya fine-tuned on 1,001 Banking77 examples reached 79.4 % against Jev's 80.0 %. If the local model is close on your data *and* the data must stay local, a fine-tune is worth a pilot.

✓ Checkpoint: you can explain why GLiNER2.5-Decide has no Brier score, and name two costs of the local engines that the "$0" column hides.

## Labs — run them here

**labs/lab00_download_models.py** — One-time setup: download Laya (EN + multilingual), GLiNER2.5-Decide (EN + multi) and GLiNER multi. Run it from the ⌨ terminal.

**labs/lab01_landscape.py** — Offline tour of the October 2026 System One landscape, the scoreboards, and a which-one-for-what table.

**labs/lab02_laya_local.py** — Laya vs Jev on the same request; a Thai ticket routed vs forced English; the truncation trap.

**labs/lab03_gliner_extract_then_decide.py** — Classic GLiNER extracts room numbers with offsets; Jev picks the one with the problem; code highlights it.

**labs/lab04_gliner_decide.py** — The Jev→GLiNER adapter, English vs multilingual Decide on Thai, multi-label topics, and why its scores don't sum to 1.

**labs/lab05_bakeoff.py** — 18 EN/TH tickets through Jev, Laya and GLiNER2.5-Decide: accuracy, urgency error, speed, cost and calibration.

Local engines run on this machine when `week24_typed_decisions_jev/.venv-local` is set up, otherwise from recordings. The Jev side runs LIVE with your key or DRY from recordings.

## Try it yourself

**Exercise 16 — the adapter and the engine picker.** Open `week24_typed_decisions_jev/16_local_system_one_models/exercises/ex16_local_or_hosted.py`. Two TODOs:

1. Write `to_gliner(questions)`. It should translate choice, noul and score into GLiNER2.5-Decide tasks, keep score levels in order, strip backticks, and raise `ValueError` for anything else (a `box` must never silently disappear).
2. Write `pick_engine(task)`. Offsets go to GLiNER. Confidential text goes to a local engine only: too long goes to a person, Thai to Laya, English to Decide. A budget under 100 ms goes to Laya. Everything else goes to Jev.

Eleven offline tests run first, for free. When they pass, one ticket goes through **your** adapter to GLiNER2.5-Decide on this machine.

```bash
.venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/exercises/ex16_local_or_hosted.py
```

**Expected output**

```
✓ to_gliner maps choice, noul and score exactly
✓ to_gliner raises ValueError for a box question
✓ score levels keep their order
✓ pick default → jev
✓ pick {'needs_offsets': True, 'data': 'confidential'} → gliner_ner
…
✓ all offline tests pass — one ticket through YOUR adapter on this machine
» {'queue': {'label': 'safety', 'confidence': 0.64…}, 'safety': {'label': 'yes', …}, 'urgency': {'label': 'routine', …}}
```

<details><summary>Hint — why "confidential + unsure" must not fall back to Jev</summary>

A rule like "if the local model is unsure, ask the cloud" sends exactly the hardest confidential texts off the machine. Module 10 called this a data-governance bug. In `pick_engine` the confidential branch only ever returns a local engine or `"human"`.

</details>

<details><summary>Stretch — Kev and laya-serve behind the same client</summary>

`laya-serve` (`pip install "laya[serve]"`) and Kev's server both expose `POST /v1/systemone`. Point a copy of `jevkit.ENDPOINT` at `http://127.0.0.1:8000/v1/systemone` and run a Module 04 lab against Laya. Then write down which answers changed, and what that does to thresholds you tuned on Jev.

</details>

✓ Checkpoint: all eleven tests pass, and your picker never returns `"jev"` for confidential data.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `◇ Laya not available here → replaying recordings` | `week24_typed_decisions_jev/.venv-local` is missing or incomplete. Run the two setup commands in §0 |
| `… LocalEntryNotFoundError` / offline error | The weights aren't downloaded yet. Run Lab 16-0 from the ⌨ terminal |
| Lab 16-0 times out from ▶ Run | Expected: the first download can take minutes. Use the terminal |
| `tiktoken is required to read a tiktoken file` | `uv pip install --python week24_typed_decisions_jev/.venv-local/bin/python tiktoken sentencepiece` |
| `laya.load()` hangs | Laya's card: transformers + TensorFlow can deadlock. `localkit` sets `USE_TF=0`; keep TF out of `.venv-local` |
| Thai answers look random | You used an English-only checkpoint: Laya `model="english"` or `GLiNER2.5-Decide`. Use the Router or `GLiNER2.5-multi-Decide` |
| A local lab is slow the first time | Loading a checkpoint takes 5–25 s; later calls are milliseconds |

## Next

**[Module 17 — Fine-tune a local System One model](../17_finetune_system_one/TUTORIAL.md)**: base Laya scored 12/18 here and was confidently wrong in Thai. Module 17 fine-tunes Laya and GLiNER2.5-multi-Decide on 456 tickets (on a Mac, or a DGX Spark), then grades them honestly on 66 held-out tickets, Module 16's 18 included.

Back to the [course overview](../README.md). Or continue the Module 15 story: Jev 1.14 now reads images, and `pplx-decider` and Clef-omni do too. Put a local text model (this module) behind an image gate (Module 15), and ask which decisions really need to leave the building.
