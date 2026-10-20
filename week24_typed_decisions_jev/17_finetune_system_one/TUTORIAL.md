# ▶ Jev Lab 17 — Fine-tune a local System One model: Mac or DGX Spark, and how to know it worked

> Part of Week 24 · Typed AI decisions with Jev. In Module 16, base Laya routed 12 of 18 tickets and sent a Thai burning smell to plumbing at 0.95. Its own authors call it "a fast base to specialise, not a zero-shot decision engine". This module specialises it. You fine-tune **Laya** (full fine-tune, Apple GPU) and **GLiNER2.5-multi-Decide** (LoRA adapter and full fine-tune) on 456 hotel tickets, on a Mac or a DGX Spark. Then you do the part most tutorials skip: decide **before** training what success means, and grade every model on tickets it has never seen.

**What you'll actually do**
- Build a dataset from 72 hand-written situations, and catch the random split that leaks 85 % of validation into training.
- Measure memory and speed on your machine, and decide Mac or DGX Spark from numbers, not vibes.
- Lock seven success criteria in a file **before** the first training run, and find out that ten lines of keywords beat every zero-shot local model.
- Fine-tune Laya in about 4 minutes on an M4 Max, and watch it become more accurate *and* overconfident.
- Train a 5 MB GLiNER LoRA adapter and a 1.1 GB full fine-tune, and see early stopping keep the right epoch.
- Grade everything on 66 held-out tickets: error bars, fixed vs broke, calibration, forgetting, speed.
- Let a gate, not your mood, decide SHIP or HOLD, and write a model card.

**Time** ~60 min (+ ~20 min of training you can run in the background) · **Difficulty** intermediate–advanced · **Cost** ≈ $0.002 live (Jev as baseline and teacher; training is $0) · $0 dry

## 0 · Before you fine-tune: is it the right tool?

Fine-tuning changes the model's weights. It is the most expensive fix, so try the cheaper ones first:

| Symptom | Cheapest fix | Module |
|---|---|---|
| Answers flip when you re-word the question | Rewrite instructions and criteria | 03 |
| Right answer, wrong confidence | Re-fit thresholds or temperatures on validation | 09, and §6 here |
| One language is weak | Use the vendor's multilingual checkpoint | 16 |
| Consistently wrong on **your** vocabulary, and the data must stay local | **Fine-tune** | 17 |

This module uses the `week24_typed_decisions_jev/.venv-local` environment from Module 16 (Laya 0.4.8, gliner2 2.1.0, PEFT 0.21). No new install is needed. If you skipped Module 16, run its §0 first. Every lab re-runs itself inside `.venv-local`. Without it, the labs replay the run recorded on 2026-10-17 (labelled `RECORDED`).

The two training labs take longer than the runner's 150 s ▶ limit. Start them from the ⌨ terminal (600 s limit) or your own terminal, and keep reading while they run:

```bash
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py --train
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train lora
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train full
```

Checkpoints go to `17_finetune_system_one/.runs/` (gitignored): about 1.3 GB for Laya, 1.1 GB for full GLiNER, and 5 MB for the LoRA adapter.

✓ Checkpoint: you can name one problem where fine-tuning is the wrong first step, and you know where the checkpoints will be written.

## 1 · Build a dataset you can trust

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab01_build_dataset.py
```

**Expected output**

```
» 72 situations × 8 variants = 576 tickets (336 English, 240 Thai)
│ split                       train  val  val with a near-copy (≥0.5)  closest pair
│ random (shuffle tickets)    461    115  85%                          1.00
│ by situation (group split)  456    120  0%                           0.44
…
│ test (48, hand-written)  0/48                0.22
│ Module 16 (18)           1/18                0.52
…
» Jev (DRY) agrees with the generator: queue 23/24 · safety 22/24 · urgency 22/24   cost ≈ $0.0005
```

How the data is built (`datagen.py`):

- **Situations, not sentences.** Each of 72 situations ("toilet overflowing", "sparks from a socket", "late checkout") is written 2–3 ways in English or Thai and carries its three labels once. Openers, closers and room numbers multiply them into 576 tickets, so 576 labels cost about 70 labelling decisions. Closers are chosen by urgency, so "No rush." never lands on a flood.
- **The random split is a trap.** Shuffling tickets puts re-worded copies of one situation on both sides: 85 % of validation tickets have a near-copy in training. Validation would then measure memory. **Splitting by situation** gives 0 % near-copies, so validation asks the real question: can the model handle a situation it has never seen?
- **The exam is written separately.** `data/test_handwritten.json` holds 48 tickets written in other styles: staff radio shorthand ("RM 1418 - guest says aircon dead"), long emails, Thai–English code-switching, negations ("the burning smell was just my toast"), and a gas leak that the written safety definition doesn't cover. Five tickets carry a `note` saying why they are hard. With Module 16's 18 tickets, that makes **66 held-out tickets** that are never trained on. The lab proves it: the closest new test ticket is 0.22 similar to anything in training. One Module 16 towel request is 0.52 similar, and the lab says so instead of hiding it.
- **Train the way you infer.** Both training files are built from one question set, `ftkit.Q`, the same one Lab 16-5 used. GLiNER's records go through Module 16's own `to_gliner_schema` adapter, so training sees the same prompts and label descriptions that `classify_text()` sends.
- **No labels? Use a teacher.** Jev labelled 24 training tickets for $0.0005 and agreed with the generator on 23 of 24 queues. The disagreements are where a person looks. One was a lost wallet: front desk or housekeeping? With real tickets and no generator, Jev-as-teacher plus a human check on its unsure answers is a cheap way to build a training set.

✓ Checkpoint: you can explain why a random split would have inflated validation accuracy, and how the lab proves the hand-written test set isn't hiding in the training data.

## 2 · Mac or DGX Spark?

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab02_mac_or_spark.py
```

**Expected output** (Apple M4 Max, 36 GiB)

```
│ Laya trainer uses     mps  (order: cuda → mps → cpu)
│ GLiNER2 trainer uses  cpu  (cuda, else cpu — never mps)
│ mixed precision       CUDA only — on a Mac both train in fp32
…
│ model                   parameters  weights fp32  full fine-tune state  LoRA r=8 state
│ Laya multilingual       322 M       1.3 GB        ≈ 5.2 GB              —
│ GLiNER2.5-multi-Decide  287 M       1.1 GB        ≈ 4.6 GB              ≈ 1.2 GB
…
│ dataset         Laya full · 3 epochs  GLiNER LoRA · 6  GLiNER full · 4  DGX Spark (Laya)
│ 456 tickets     4 min                 7 min            7 min            not measured yet
```

What decides it:

- **These are encoders, not LLMs.** A 322M-parameter model needs about 16 bytes per weight for a full fine-tune (fp32 weights, gradients, and two Adam moments): about 5 GB. That fits a 16 GB MacBook. An 8B LLM needs about 130 GB for the same thing, which is the job the 128 GB DGX Spark exists for (week 25).
- **Read what the trainer actually does.** Laya's trainer picks `cuda → mps → cpu`, so it gets the Apple GPU. GLiNER2's `ExtractorTrainer` checks for CUDA and otherwise uses the **CPU**: it has no MPS path. Mixed precision (bf16) is CUDA-only in both. On a Mac, Laya trains on the GPU in fp32 and GLiNER trains on the CPU in fp32. On a Spark, both get the GPU and bf16.
- **Measure, then project.** The lab times one real pass on 48 tickets. When the full runs from §4–§5 exist, it projects from those instead, because short passes include warm-up. For 456 tickets the Mac takes minutes. At 50,000 tickets it takes most of a day per run, and that is when the Spark earns its place.
- **The Spark column says `not measured yet`.** NVIDIA's spec gives memory (128 GB unified LPDDR5x), bandwidth (273 GB/s) and the CUDA GPU. No spec sheet says how fast *these* models train, and the course doesn't print guesses as measurements. Lab 17-8 fills the column when you run it on your Spark.

| Situation | Train on |
|---|---|
| First fine-tune, under ~5k tickets, one model | **Mac**: minutes, and the same machine you serve from |
| GLiNER2 on more than ~5k tickets, or many runs | **Spark**: its trainer gets CUDA there |
| Sweeps, several seeds, nightly re-training | **Spark**: unattended, bf16, and your laptop stays free |
| The data may not leave the building | **Either**: both are on-prem, so this is never a reason to pick the cloud |

✓ Checkpoint: you can say which device each trainer uses on your machine, and why the GLiNER rows are slower on a Mac than the Laya row.

## 3 · Lock the criteria, then measure the baselines

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab03_baselines_and_criteria.py
```

**Expected output**

```
» locked: sha256 e0b13dfbde8db0f8 · 2026-10-17 23:17:41  (.runs/criteria.lock)
…
│ engine       queue  95% CI    macro-F1  EN     TH     safety recall  false alarms  source
│ majority     15/66  14%–33%    0.07      9/42   6/24   0/14           0             RULE
│ keywords     52/66  68%–88%    0.77      31/42  21/24  8/14           2             RULE
│ laya_router  40/66  48%–73%    0.60      30/42  10/24  9/14           2             LOCAL
│ laya_base    36/66  42%–67%    0.52      26/42  10/24  8/14           1             LOCAL
│ gliner_base  40/66  48%–73%    0.57      23/42  17/24  14/14          31            LOCAL
│ jev          66/66  100%–100%  1.00      42/42  24/24  13/14          7             RECORDED
```

`criteria.json` was written before any model was trained. Lab 17-3 stores its hash in `.runs/criteria.lock`. Lab 17-7 refuses to grade if the file changed afterwards, or if the lock is younger than a checkpoint. The seven criteria, on the 66 held-out tickets:

| | Must hold | Why |
|---|---|---|
| C1 | Queue accuracy beats the zero-shot baseline by ≥ 10 points | worth the effort |
| C2 | McNemar p < 0.05 | the gain is not luck |
| C3 | Safety recall 100 % | never miss a fire |
| C4 | Thai within 10 points of English | no language left behind |
| C5 | Breaks ≤ 2 tickets the baseline got right | no silent regressions |
| C6 | Forgetting probe drops ≤ 2 of 12 | general skill kept |
| C7 | Median latency ≤ 1.5 × baseline | the reason to go local survives |

What the baselines teach:

- **Ten lines of keywords beat every zero-shot local model**: 52/66 against 40/66 for Laya's Router and GLiNER2.5-multi-Decide. Keyword rules are the floor, and a model that can't beat them isn't worth deploying.
- **Jev is the ceiling**: 66/66 on queue, though it still missed one of 14 dangerous tickets (a gas leak, which the written safety definition doesn't mention). Fine-tuning a local model makes sense when the data must stay local or every millisecond counts, not to beat Jev.
- **Two Laya baselines, on purpose.** `laya_router` (English checkpoint for English, multilingual for Thai) is what you would ship today, so the criteria compare against it. `laya_base` (multilingual only) is the exact weights you fine-tune, so the forgetting probe compares against it. Laya's own before/after table (§4) uses `laya_base`, which flatters the gain if you quote it as "the" improvement.
- **GLiNER's 14/14 safety recall is not a win.** It raised 31 false alarms on 52 safe tickets: it says "yes" to almost everything. Always read recall next to false alarms.
- **The 95 % interval is wide.** 40/66 means "somewhere between 48 % and 73 %". With 66 tickets, a few points of difference proves nothing. That's why C2 exists.

✓ Checkpoint: `.runs/criteria.lock` exists, and you can say which baseline each criterion compares against and why.

## 4 · Fine-tune Laya

```bash
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py --train   # ≈ 4 min, ⌨ terminal
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab04_finetune_laya.py                         # ▶ reads the result
```

**Expected output**

```
train items 1204, calibration items 164, eval items 360 (…laya_val.jsonl), skipped {}, device mps
optimizer budget: 1204 train item(s), effective batch up to 16 (micro_batch 8 x grad_accum 2), 228 optimizer updates over 3 epoch(s)
» 1204 train items + 164 calibration items · mps · 3.9 min · source: LOCAL (just trained)
   epoch 1  loss 0.4041  ██████████████████████████████
   epoch 2  loss 0.1941  ██████████████
   epoch 3  loss 0.0478  ████
   ⚠ laya.train: choice calibration: the fit came back unchanged at 1.0 (accuracy 1.00, mean confidence 1.000 …
…
│ accuracy (all 3 questions)  0.614              0.889       +0.275
│ loss                        1.364              2.080       +0.717
│ mean confidence             0.818              0.997       +0.179
│ choice (queue)   0.48 → 0.80  0.84 → 0.99      0.36 → 0.19
…
│ train (seen)             114      114/114        100%
│ val (unseen situations)  120      96/120         80%
│ held-out (hand-written)  66       55/66          83%
```

The whole call is one function: `laya.train.finetune(train.jsonl, "multilingual", out_dir, TrainConfig(…, eval_data=val.jsonl))`. It scores the base checkpoint on validation, trains, fits temperatures on a calibration slice, scores again and saves a checkpoint that `laya.load(out_dir)` reads. The recipe and its reasons:

- **One checkpoint for both languages.** The Router would need two fine-tunes. `multilingual` (mmBERT, 322M) learns English and Thai together.
- **Batch 8 × 2 = 16, not Laya's default 64.** With 1,204 items, 64 gives only about 19 updates per epoch. Laya warns about "collapse to the class prior" on small data, and more updates avoid it.
- **3 epochs, `shuffle_options=("choice",)`.** Few passes, because there are only 72 situations to memorise. Option shuffling re-orders the queue options each epoch, so the model can't learn "the first option is usually right".
- **`eval_data=` makes the table honest.** Without it, Laya scores its in-sample calibration slice and says so in the output ("not held-out"). A smoke run that did this on copies of its own data printed 0.98 accuracy, and the number meant nothing.

What the run teaches:

- **It worked.** Held-out queue accuracy went from 40/66 (Router) to 55/66 in under four minutes on a laptop. Urgency improved most (0.47 → 0.89 on validation), because "today" versus "immediate" is a house rule, and house rules are what examples teach.
- **The generalisation gap is the summary.** 100 % on training situations, 80 % on unseen generated ones, 83 % on hand-written held-out tickets. Training accuracy only shows the model learned its homework.
- **Accuracy up, validation loss up.** Loss rose from 1.36 to 2.08 while accuracy rose. The model makes fewer mistakes, but makes each one with near-certainty: mean confidence is 0.997.
- **Laya told you so.** The temperature fit "came back unchanged at 1.0" because the calibration slice comes from training situations the model now gets 100 % right, so there was nothing to correct. Its abstention cuts landed at about 1.000 for the same reason. The fix is in §6: fit the cut on validation.
- **The loss can go negative.** The default `rlcd` objective is not cross-entropy (it includes a reward term), so a negative step loss is normal. Don't read it as accuracy.

✓ Checkpoint: you can explain why validation loss rose while validation accuracy rose, and why the fitted temperatures stayed at 1.0.

## 5 · Fine-tune GLiNER2.5-multi-Decide: a 5 MB LoRA adapter vs a full fine-tune

```bash
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train lora   # ≈ 7 min, CPU
week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py --train full   # ≈ 7 min, CPU
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab05_finetune_gliner.py                              # ▶ reads both
```

**Expected output**

```
│ run   trainable          device  time     tickets  validation loss per epoch                      source
│ lora  1.36 M (0.5%)      cpu     7.2 min  6.3/s    0.306 → 0.160 → 0.240 → 0.154 → 0.246 → 0.480  LOCAL
│ full  287.36 M (100.0%)  cpu     6.7 min  4.5/s    0.246 → 0.167 → 0.189 → 0.169                  LOCAL
» lora: lowest validation loss at epoch 4 of 6 → that checkpoint is kept as 'best/'  (later epochs were already overfitting)
   early stopping ended it after epoch 6 of 8: two epochs in a row without a new best
» full: lowest validation loss at epoch 2 of 4 → that checkpoint is kept as 'best/'  (later epochs were already overfitting)
…
│ lora  5.5 MB           needs the 1.1 GB base next to it
│ full  1,165.5 MB       stands alone
…
│ gliner_base  65/120   45%–62%  0.53      38/80  27/40  24/24          52            …
│ gliner_lora  108/120  84%–95%  0.90      71/80  37/40  24/24          0             …
│ gliner_full  101/120  78%–90%  0.84      64/80  37/40  24/24          0             …
…
   adapter ON  → safety       1.00
   adapter OFF → safety       0.35   (the untouched base model)
```

Both runs use GLiNER2's own `ExtractorTrainer` on `data/gliner_train.jsonl`, with `data/gliner_val.jsonl` scored after every epoch:

- **LoRA freezes the 287M weights** and trains small rank-8 matrices beside each encoder layer, plus the task heads: 1.36M weights, 0.47 %. GLiNER2's guide suggests rank 4 under ~1k examples and rank 8 for 1k–10k. 456 tickets × 3 tasks is about 1.4k.
- **The full fine-tune moves every weight**, so it gets fewer epochs (4) and a small encoder learning rate (1e-5).
- **`save_best` + `early_stopping`.** Validation loss is checked every epoch. The lowest one is kept as `best/`, and training stops after two epochs without a new best. Read the LoRA curve: it improved until epoch 4 and then climbed to 0.48 as it started memorising. Without `save_best` you would ship epoch 6 (0.48). Without early stopping you pay for 8 epochs. The first recorded run had no early stopping and took 10.1 min, longer than the ⌨ terminal's 600 s limit. That's why the recipe now stops early.
- **On a Mac this is CPU work.** GLiNER2's trainer has no MPS path (§2). The LoRA run processes about 6 tickets per second here, and a Spark gives the same code a CUDA GPU (§8).

What the run teaches:

- **The 5 MB adapter won.** LoRA reached 108/120 on validation, the full fine-tune 101/120, and the base model 65/120. With 456 examples, changing fewer weights generalised better.
- **Size changes the deployment.** One 1.1 GB base plus a 5.5 MB adapter per hotel is about 1.1 GB for five hotels. Five full fine-tunes is about 5.5 GB.
- **The base model is still there.** With `pm.disable_adapter()` the same object answers like the untouched base (0.35 on the Thai smoke ticket, against 1.00 with the adapter on). Boundary checkpoints such as `GLiNER2.5-multi-Decide` load adapters through PEFT (`PeftModel.from_pretrained`). The older `.load_adapter()` works only for span checkpoints.

✓ Checkpoint: you can read the LoRA validation-loss curve, say which epoch was kept and why, and explain what `disable_adapter()` gives you.

## 6 · The exam: did it work?

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab06_evaluate.py
```

**Expected output**

```
│ engine       queue  95% CI     macro-F1  EN     TH     safety recall  false alarms  source
│ keywords     52/66  68%–88%    0.77      31/42  21/24  8/14           2             RULE
│ laya_router  40/66  48%–73%    0.60      30/42  10/24  9/14           2             …
│ laya_ft      55/66  74%–92%    0.82      34/42  21/24  10/14          2             …
│ gliner_base  40/66  48%–73%    0.57      23/42  17/24  14/14          31            …
│ gliner_lora  60/66  83%–97%    0.90      37/42  23/24  13/14          3             …
│ gliner_full  57/66  77%–94%    0.86      35/42  22/24  12/14          2             …
│ jev          66/66  100%–100%  1.00      42/42  24/24  13/14          7             RECORDED
…
│ laya_router → laya_ft      40/66 → 55/66  +23 pts  19     4      0.0026
│ gliner_base → gliner_lora  40/66 → 60/66  +30 pts  20     0      0.0000
│ gliner_base → gliner_full  40/66 → 57/66  +26 pts  18     1      0.0001
…
   gliner_lora  missed: t_en03 “Engineering: the basement car park…”
…
│ laya_ft      0.17                  1.000                     94%                15%
│ gliner_lora  0.09                  0.947                     95%                8%
…
│ gliner_base  10/12  · gliner_lora  8/12  · gliner_full  10/12      (forgetting probe)
│ laya_router → laya_ft      46 ms → 31 ms  × 0.67
```

Six questions, in the order a reviewer should ask them:

1. **Is it better?** Every fine-tune beats its zero-shot baseline by 23–30 points on 66 tickets it never saw. GLiNER + LoRA (60/66) now beats the keyword rules (52/66), and its interval (83–97 %) no longer overlaps the base model's (48–73 %).
2. **Is it luck?** McNemar looks only at the tickets where before and after disagree. LoRA fixed 20 and broke 0, so p ≈ 0.000002. Laya fixed 19 and broke 4, so p = 0.003.
3. **What did it break?** Laya broke three housekeeping/front-desk tickets written as long, polite requests ("we arrived late last night…", "can the hotel store our luggage…"), plus the candle distractor, which it now calls `safety`. That's a style the generator never used. The paired list tells you which data to write next. An accuracy number alone can't.
4. **Is it safe?** No fine-tune flags every dangerous ticket. All of them missed **t_en03**, "the basement car park drain is backing up and water is now reaching the main switchboard room door". That is water near electricity in words no training situation used. Laya also missed the gas leak, an ice machine leaking into a floor socket, and an overheated lift motor.
5. **Can you trust its confidence?** Fit the cut on **validation** (target: ≤ 5 % errors among auto-answered tickets), then check it on held-out. For LoRA the cut answered 95 % of held-out tickets at 8 % error: close, but not the promised 5 %. For Laya the cut is 1.000, because after fine-tuning nearly everything is 1.000. It answered 94 % at 15 % error. Overconfidence from §4 becomes a real cost here, and the remedy is a person in the loop or a calibration fit on held-out-like data.
6. **What else got worse?** On the forgetting probe (12 review-sentiment questions the models never trained on), Laya and the full GLiNER fine-tune kept their scores, but GLiNER **with the adapter on** dropped from 10/12 to 8/12. An adapter changes every task the model runs while it is loaded. Serve unrelated tasks with the adapter disabled. With twelve items, ±1 is noise and a fall of 3 would be a signal.

Speed didn't suffer: Laya got faster (46 → 31 ms), because for English the Router uses the larger 421M English checkpoint, and the unmerged LoRA adds about 9 % (74 → 80 ms).

✓ Checkpoint: you can name the ticket every fine-tune missed, and explain why the validation-fitted cut gave 15 % errors on held-out for Laya.

## 7 · Ship or hold: let the locked criteria decide

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab07_ship_or_hold.py
```

**Expected output**

```
✓ criteria.json matches the lock (sha256 e0b13dfbde8db0f8, locked 2026-10-17 23:17:41)
✓ laya_ft: weights written 2026-10-17 23:24 — after the criteria were locked
…
   gliner_lora  →  🔴 HOLD
│ C1  PASS    queue_gain_points = 30.3 (needs ≥ 10)
│ C2  PASS    queue_mcnemar_p = 1.90735e-06 (needs ≤ 0.05)
│ C3  FAIL    safety_recall = 0.928571 (needs ≥ 1.0)
│ C4  PASS    th_minus_en_points = 7.7 (needs ≥ -10)
│ C5  PASS    queue_broke = 0 (needs ≤ 2)
│ C6  PASS    probe_drop = 2 (needs ≤ 2)
│ C7  PASS    latency_ratio = 1.09 (needs ≤ 1.5)
…
   ✓ .runs/model_card_gliner_lora.json  (HOLD)
```

**All three candidates are on HOLD**, and that is the most useful result in the module:

| | Laya fine-tuned | GLiNER + LoRA | GLiNER full |
|---|---|---|---|
| Fails | C3 (10/14 dangerous tickets), C5 (broke 4) | C3 (13/14) | C3 (12/14) |
| Passes | C1, C2, C4, C6, C7 | C1, C2, C4, C5, C6, C7 | C1, C2, C4, C5, C6, C7 |

- **The gate decides, not the mood in the room.** "+30 points!" feels like a ship. C3 says one missed flood next to a switchboard is one too many, and C3 was written before anyone saw a number. Lab 17-7 also checks that `criteria.json` still matches its lock and that every checkpoint's weights were written *after* the lock. Moving the goalposts after seeing the results is the most common way a fine-tune "succeeds".
- **A HOLD names the next step.** Write situations for *water reaching electrical rooms and floor sockets* and *polite long-form housekeeping requests*, retrain, and grade again under the same criteria. Write them for the **category**, not the sentence. Copying t_en03 into training would just leak the exam.
- **The exam is now spent.** You studied its failures to decide what to write next, so the next SHIP needs a **fresh** held-out batch. Keep a few dozen hand-written tickets in reserve for this.
- **The model card travels with the checkpoint.** `.runs/model_card_*.json` records the base, the data hash, the criteria hash, every result, the known failures and the intended use ("NOT validated on real guests").
- **Shipping, when it comes, is gradual.** Run the model in shadow mode next to the current one, gate on the validation-fitted cut, send low-confidence answers to a person (never quietly to the cloud), and keep the previous checkpoint for rollback.

✓ Checkpoint: you can say why GLiNER + LoRA is on HOLD despite +30 points, and why the next SHIP needs new held-out tickets.

## 8 · The same fine-tune on a DGX Spark

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab08_run_on_spark.py                         # ▶ plan + status
SPARK_HOST=you@spark-abcd .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab08_run_on_spark.py --run   # ⌨ terminal
```

**Expected output** (no Spark configured, as when this module was recorded)

```
» SPARK_HOST is not set (env or week25_dgx_spark/.env.local). Set it to your ssh target, e.g.
…
│ 1  copy code + data          rsync -az --exclude .runs … $SPARK_HOST:~/agentic-week24/ (keeps paths)
│ 2  train inside NGC PyTorch  ssh $SPARK_HOST docker run --rm --gpus all --ipc=host -v ~/agentic-week24:/w …
│ 3  bring the run files back  scp $SPARK_HOST:…/lab0[45]_run.json → recorded/lab08_spark.json
…
│ Laya full fine-tune, 456 tickets × 3 epochs       3.9 min (mps)       NOT RUN
│ GLiNER LoRA, 456 tickets × 6 epochs (early sto…   7.2 min (cpu)       NOT RUN
```

The ▶ button never starts a remote job. It finds your Spark (`SPARK_HOST`, or `week25_dgx_spark/.env.local` from the week 25 course), tests it with `ssh -o BatchMode=yes`, and prints the plan. `--run` executes it:

1. `rsync` copies `week24_typed_decisions_jev/common`, Module 16's `localkit.py`, and this module (without `.runs/`) to the Spark.
2. One `docker run --gpus all` in **`nvcr.io/nvidia/pytorch:25.11-py3`**, the image week 25's NVIDIA playbooks use, installs `laya` and `gliner2[train]` *against the container's own PyTorch*, then runs `lab04 --train` and `lab05 --train lora`. The Spark is aarch64 + CUDA, and the container's PyTorch matches it, so the lab code is unchanged.
3. The run files come back to `recorded/lab08_spark.json`. Labs 17-2 and 17-8 then show **measured** Spark times next to the Mac's.

On the Spark, Laya's trainer picks CUDA and turns on bf16 autocast and gradient checkpointing. GLiNER2's trainer gets the GPU it never gets on a Mac, and Lab 17-5 switches it to bf16 there. Until you run it, the Spark column says `NOT RUN`. The course doesn't print a speed-up it hasn't measured.

✓ Checkpoint: you can list the three steps `--run` performs, and explain why the training runs inside NVIDIA's container instead of a pip-installed PyTorch.

## Labs — run them here

**labs/lab01_build_dataset.py** — Generate 576 tickets from 72 situations, compare a random split with a split by situation, leak-check the hand-written test set, and write the Laya and GLiNER training files.

**labs/lab02_mac_or_spark.py** — Which device each trainer uses, memory counted from the weight files, a timed training pass, and projections for 456, 5k and 50k tickets.

**labs/lab03_baselines_and_criteria.py** — Lock the success criteria, then measure the majority label, keyword rules, zero-shot Laya and GLiNER, and Jev on validation and held-out.

**labs/lab04_finetune_laya.py** — Fine-tune Laya's multilingual checkpoint (`--train`, ⌨ terminal), or read the run: loss, Laya's before/after table, the generalisation gap, and temperatures.

**labs/lab05_finetune_gliner.py** — GLiNER2.5-multi-Decide as a LoRA adapter and as a full fine-tune (`--train lora|full`): validation-loss curves, early stopping, size, and switching the adapter off.

**labs/lab06_evaluate.py** — The exam on 66 held-out tickets: intervals, fixed vs broke, safety misses, a validation-fitted confidence cut, forgetting and speed.

**labs/lab07_ship_or_hold.py** — Check the lock, grade C1–C7, write a model card per candidate, and see what shipping and rollback look like.

**labs/lab08_run_on_spark.py** — Find your DGX Spark, print the plan, and with `--run` train there inside NVIDIA's PyTorch container and bring the measured times back.

Training (`--train`) runs on this machine inside `week24_typed_decisions_jev/.venv-local`. Everything else runs from your checkpoints or, without them, from the 2026-10-17 recordings. The Jev side runs LIVE with your key or DRY from recordings.

## Try it yourself

**Exercise 17 — a leak-proof split and a ship gate.** Open `week24_typed_decisions_jev/17_finetune_system_one/exercises/ex17_split_and_ship.py`. Two TODOs:

1. Write `group_split(tickets, val_groups)`. Every ticket of a situation goes to the same side, order is kept, the input isn't modified, and an unknown situation id raises `ValueError` (a typo would silently shrink validation).
2. Write `ship(metrics, criteria)`. Each criterion is PASS, FAIL or MISSING. A bound value is a PASS. MISSING holds the release like FAIL does, and an empty criteria list is a HOLD.

Twelve offline tests run first. When they pass, your gate grades the 2026-10-17 fine-tunes and is compared with Lab 17-7.

```bash
.venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/exercises/ex17_split_and_ship.py
```

**Expected output**

```
✓ val holds every ticket of situation b, in order
✓ train holds everything else, in order
✓ no situation on both sides
…
✓ no criteria → HOLD

✓ all offline tests pass — grading the 2026-10-17 fine-tunes with YOUR gate
   laya_ft      your gate: HOLD  Lab 17-7: HOLD  ✓   not passing: C3, C5
   gliner_lora  your gate: HOLD  Lab 17-7: HOLD  ✓   not passing: C3
   gliner_full  your gate: HOLD  Lab 17-7: HOLD  ✓   not passing: C3
```

<details><summary>Hint — why MISSING must hold a release</summary>

If a metric wasn't measured, say the forgetting probe didn't run because a model failed to load, a gate that skips it passes by default. "We didn't check" then quietly becomes "it passed". Treat MISSING exactly like FAIL, and the only way to SHIP is to measure everything you promised to measure.

</details>

<details><summary>Stretch — round two, done honestly</summary>

Add three or four situations to `datagen.py` for the failure *categories* from §6: water reaching electrical rooms or floor sockets, and long, polite housekeeping requests. Write them in your own words, without copying any held-out ticket. Re-run Lab 17-1 (check that the leak check still says 0/48), retrain LoRA, and re-run Labs 17-6 and 17-7. Then write 20 fresh held-out tickets before you believe a SHIP, because you have already looked at the old ones.

</details>

✓ Checkpoint: all twelve tests pass, and your gate gives the same three verdicts as Lab 17-7.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `✗ data/train.json is missing` | Run Lab 17-1 first. It writes `data/` |
| `--train needs week24_typed_decisions_jev/.venv-local` | Set up Module 16 §0 (two `uv` commands and Lab 16-0) |
| A `--train` run is killed after 150 s | The ▶ button's limit. Run training from the ⌨ terminal (600 s) or your own terminal |
| `⚠ criteria.json CHANGED since it was locked` | If nothing is trained yet, delete `.runs/criteria.lock` and re-run Lab 17-3. If something is, restore the file. That's the point of the lock |
| Laya warns `fit came back unchanged at 1.0` | Expected here: the calibration slice comes from training situations the model now gets right. Use the validation-fitted cut from Lab 17-6 |
| GLiNER trains slowly on a Mac | Its trainer has no MPS path, so it runs on the CPU. Use LoRA, fewer epochs, or a Spark (§8) |
| `AttributeError: 'BoundaryExtractor' object has no attribute 'has_adapter'` | Boundary checkpoints use PEFT: `PeftModel.from_pretrained(base, path)` and `with pm.disable_adapter():` |
| Your numbers differ slightly from the recording | MPS kernels and data order aren't bit-exact across machines. Expect small drift, and read the intervals |

## Next

Back to the [course overview](../README.md). Or take the HOLD seriously: do the Stretch exercise, then point Module 16's `pick_engine()` at your checkpoint once a fresh held-out batch says SHIP. The week 25 course ([DGX Spark](../../week25_dgx_spark/README.md)) covers fine-tuning the *generative* side (LLaMA Factory, Unsloth, NeMo) when a System One model is not enough.
