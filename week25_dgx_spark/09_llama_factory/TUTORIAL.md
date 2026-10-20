# ▶ Spark Lab 09 — Fine-tune with LLaMA Factory: LoRA, QLoRA, full

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Install LLaMA Factory on your Spark exactly the way NVIDIA's playbook does: a venv, PyTorch for CUDA 13, a clone, `pip install -e`.
- Work out with arithmetic what LoRA, QLoRA and full fine-tuning each train, and which of them fit in 128 GB.
- Build a real dataset for a real task: a **hotel guest-request router** that answers with `{department, priority, reply}` in English or Thai.
- Generate the training YAML, check every key, and launch `llamafactory-cli train` under `nohup` so it survives your laptop going to sleep.
- Watch the loss from your laptop, then chat with the adapter, generate predictions and merge it into one model for Module 13.

**Time** ~55 min · **Difficulty** intermediate · **Hardware** 1 DGX Spark (or none: DRY mode + the dataset, arithmetic and config labs run on the laptop)

**Official playbooks covered:** [Fine-Tune LLMs with LLaMA Factory](https://build.nvidia.com/spark/llama-factory)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01 done | `ssh -o BatchMode=yes spark-a true` | every Spark step here runs over SSH |
| > 50 GB free on the Spark | `df -h ~` on the Spark | the playbook asks for this much for models and checkpoints |
| This repo's Python with PyYAML | `.venv/bin/python -c "import yaml"` | labs 02–03 and the exercise parse YAML and JSON |
| Nothing else big running on the Spark | `docker ps`, `nvidia-smi` | training shares the 128 GB with every other process |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python -c "import yaml; print('PyYAML', yaml.__version__)"
```

**Expected output**

```
PyYAML 6.0.3
```

> 🔐 The base model in this module, `Qwen/Qwen3-4B-Instruct-2507`, is not gated. You do not need a Hugging Face login. If you switch to a gated model (Llama, Gemma), run `hf auth login` once **on the Spark**, never in a lab.

✓ Checkpoint: PyYAML imports, and `ssh spark-a true` works (or you have decided to follow along in DRY mode).

## 1 · LoRA, QLoRA or full: decide with arithmetic

Fine-tuning means you keep training a model on your own examples. There are three ways to do it. The main difference is how many bytes of memory each **parameter** costs.

| Method | What is trained | Memory per parameter (AdamW, mixed precision) |
|---|---|---|
| **Full** | every weight | 2 (bf16 weight) + 2 (gradient) + 4 (fp32 master copy) + 8 (two fp32 Adam moments) = **16 bytes** |
| **LoRA** | two thin matrices, A (r × in) and B (out × r), next to each frozen linear layer | frozen base: **2 bytes**; only the adapter pays 16 |
| **QLoRA** | the same adapter | frozen linear layers stored in **4 bits** (~0.56 bytes with scales); embeddings stay bf16 |

LoRA trains `r × (in + out)` numbers per adapted layer. With `lora_target: all`, LLaMA Factory adapts the 7 linear layers in every block (q, k, v, o, gate, up, down). Lab 03 applies this to four real models, using the shapes from each model's `config.json`:

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py
```

**Expected output** (step 1 — arithmetic, the same on every machine; captured on this Mac)

```
▣ STEP 1 · how much does each method train, and does it fit in 128 GB?
│ LoRA rank  trainable params  of Qwen3-4B  adapter file (bf16)
│ ─────────  ────────────────  ───────────  ───────────────────  ────────────────────
│ r = 8        16.5 M           0.41 %        33.0 MB            ██░░░░░░░░░░░░░░░░░░
│ r = 16       33.0 M           0.82 %        66.1 MB            █████░░░░░░░░░░░░░░░
│ r = 32       66.1 M           1.64 %       132.1 MB            ██████████░░░░░░░░░░
│ r = 64      132.1 M           3.28 %       264.2 MB            ████████████████████
◆ Qwen3-4B has 4.02 B parameters. LoRA adds two thin matrices A (r×in) and B (out×r) to each of the 7 linear layers in all 36 blocks and trains only those.

│ model                   params   full (16 B/param)   LoRA r=16 (bf16 base)  QLoRA r=16 (4-bit base)
│ ──────────────────────  ───────  ──────────────────  ─────────────────────  ───────────────────────
│ Qwen3-4B-Instruct-2507    4.0 B      74 GB  ✓ fits       19 GB  ✓ fits          13 GB  ✓ fits
│ Qwen3-8B                  8.2 B     141 GB  ✕ > 128      27 GB  ✓ fits          17 GB  ✓ fits
│ Qwen3-32B                32.8 B     534 GB  ✕ > 128      78 GB  ✓ fits          33 GB  ✓ fits
│ Llama-3.3-70B            70.6 B    1139 GB  ✕ > 128     154 GB  ✕ > 128         56 GB  ✓ fits
◆ Rule of thumb, not a measurement: AdamW mixed precision, + 10 GB overhead, activations not counted (they grow with batch × cutoff_len; gradient checkpointing keeps them small).
```

What the table tells you:

1. **Full fine-tuning is the memory hog.** Only the 4B model can be fully trained on one Spark. An 8B model already needs 141 GB before activations.
2. **LoRA is the default.** It trains under 1 % of the weights, and the saved adapter is tens of MB, not GB. You can keep one adapter per hotel, per language or per customer.
3. **QLoRA is how a 70B model trains on one Spark.** It costs a little accuracy and speed (the 4-bit base is de-quantized on the fly), and it needs `bitsandbytes`.

NVIDIA's companion overview ([Fine-Tune Specialized LLMs with Unsloth](https://build.nvidia.com/spark/fine-tuning)) gives a data-size rule of thumb: about **100–1,000** prompt–response pairs for LoRA/QLoRA, and **1,000+** for full fine-tuning. This module's dataset has 504 pairs, so it uses LoRA.

✓ Checkpoint: you can explain why Qwen3-8B does not fit for full fine-tuning on one Spark, but Llama 3.3 70B does fit as QLoRA.

## 2 · Install LLaMA Factory, the playbook's way

The [playbook](https://build.nvidia.com/spark/llama-factory) uses a plain Python venv, with no Docker. These are its commands, in order. The course runs them in your home directory, so the paths are `~/factoryEnv` and `~/LLaMA-Factory`:

```bash
# on: spark
nvcc --version            # Step 1: CUDA 12.9 or newer
nvidia-smi
python3 --version
git --version
python3 -m venv factoryEnv                                                             # Step 2
source ./factoryEnv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130   # Step 3
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"   # Step 4
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git                       # Step 5
cd LLaMA-Factory
pip install -e ".[metrics]"                                                            # Step 6
```

The PyTorch download is several GB and can take longer than the Lab Runner lets a lab run (900 s). So lab 01 puts steps 2–6 into one script, `spark/setup_factory.sh`, and runs it under `nohup`. The script is the playbook's commands plus `set -e` and skip-if-done checks. Run the lab once to see the plan, then again with `SPARK_APPLY=1` to install:

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab01_factory_setup.py
SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab01_factory_setup.py
```

**Expected output** (captured on this Mac, DRY mode; with a Spark the ◈ rows become ✓ or ✕)

```
▣ STEP 1 · the playbook's prerequisite checks (Step 1) — read-only
…
│ prerequisite                           result     last line
│ ─────────────────────────────────────  ─────────  ─────────────────────────────────────────
│ CUDA toolkit ≥ 12.9                    ◈ example  Build cuda_13.0.r13.0/compiler.xxxxxxxx_0
│ GPU visible                            ◈ example  NVIDIA GB10, 580.95.05
│ Python 3 + venv                        ◈ example  venv ok
│ Git                                    ◈ example  git version 2.43.0
│ > 50 GB free for models + checkpoints  ◈ example  /dev/nvme0n1p2  3.7T  412G  3.1T  12% /
…
▣ STEP 3 · upload the playbook's steps 2–6 as one script, run it under nohup (opt-in)
…
$ mkdir -p ~/w25/logs && { nohup bash ~/w25/m09/setup_factory.sh > ~/w25/logs/m09_setup.log 2>&1 < /dev/null & }; sleep 3; tail -3 ~/w25/logs/m09_setup.log   [not run]
◆ Installing downloads PyTorch (several GB) and writes ~/factoryEnv and ~/LLaMA-Factory on the Spark. Re-run with SPARK_APPLY=1 to do it.
```

Re-run the lab at any time. It only reports progress, and the last step turns ✓ when `PyTorch … CUDA: True` appears. The `< /dev/null` matters: without it, ssh keeps waiting for the background job's input, and the lab hangs.

> 💡 **LLaMA Board** is LLaMA Factory's web UI. It is the same trainer with a form that writes the YAML for you. It listens on all interfaces by default, so bind it to localhost and tunnel it:
>
> ```bash
> # on: spark
> cd ~/LLaMA-Factory && source ~/factoryEnv/bin/activate
> GRADIO_SERVER_NAME=127.0.0.1 llamafactory-cli webui
> ```
>
> ```bash
> # on: laptop
> ssh -N -L 7860:localhost:7860 spark-a      # then open http://localhost:7860
> ```

Optional: the playbook's own smoke test trains Qwen3-4B on two demo datasets. Its reported result is:

```bash
# on: spark
cd ~/LLaMA-Factory && source ~/factoryEnv/bin/activate
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

**Expected output** (REFERENCE — quoted from the playbook)

```
***** train metrics *****
  epoch                    =        3.0
  total_flos               = 11076559GF
  train_loss               =     0.9993
  train_runtime            = 0:14:32.12
  train_samples_per_second =      3.749
  train_steps_per_second   =      0.471
Figure saved at: saves/qwen3-4b/lora/sft/training_loss.png
```

✓ Checkpoint: lab 01 prints `✓ PyTorch in ~/factoryEnv sees the GPU (CUDA: True)`. In DRY mode, you can say which of the playbook's steps runs under `nohup` and why.

## 3 · A real task: the hotel guest-request router

The playbook trains on demo data. You will train on a task you can **score**. A guest writes a message, and the model answers with one line of JSON:

```json
{"department": "engineering", "priority": "urgent", "reply": "Thank you for telling us. An engineer is on the way to room 1412 now. …"}
```

There are six departments (`housekeeping`, `engineering`, `front_desk`, `food_beverage`, `concierge`, `security`), two priorities, and replies in the guest's language, English or Thai. Code can check JSON output, so Module 13 can measure accuracy instead of guessing.

LLaMA Factory reads **Alpaca** records (`instruction`, `input`, `output`, optional `system`) and finds a dataset by **name** in `dataset_info.json`. Lab 02 writes both, from 45 request templates. Eight of the templates are **held out**: they appear only in the eval set, so the evaluation tests new wording, not memorised sentences.

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab02_hotel_dataset.py
```

**Expected output** (captured on this Mac — deterministic, seed 25)

```
▣ STEP 2 · write the dataset files and the dataset_info.json registry
→ wrote 09_llama_factory/data/hotel_ops.json  (341.4 KB)
→ wrote 09_llama_factory/data/hotel_ops_eval.json  (39.2 KB)
→ wrote 09_llama_factory/data/dataset_info.json  (0.4 KB)

▣ STEP 3 · validate — the checks a failed training run would otherwise teach you
✓ hotel_ops: file_name 'hotel_ops.json' exists next to dataset_info.json
✓ hotel_ops: 504 records, every mapped column present (instruction, input, output, system)
✓ hotel_ops_eval: file_name 'hotel_ops_eval.json' exists next to dataset_info.json
✓ hotel_ops_eval: 60 records, every mapped column present (instruction, input, output, system)
✓ no empty instruction or output
✓ every output is one line of valid JSON
✓ department ∈ 6 allowed values · priority ∈ normal|urgent · reply non-empty
✓ every reply repeats the guest's room number (no invented rooms)
✓ no duplicate requests, and no request appears in both train and eval
✓ longest record ≈ 177 tokens, well under cutoff_len 1024 (nothing gets truncated)

▣ STEP 4 · balance — a router trained on 90% housekeeping learns to say 'housekeeping'
│ department     train  eval  train · Thai  train · urgent
│ ─────────────  ─────  ────  ────────────  ──────────────
│ housekeeping   84     10    24            0
│ engineering    84     10    24            36
│ front_desk     84     10    14            0
│ food_beverage  84     10    28            0
│ concierge      84     10    28            0
│ security       84     10    16            67
◆ priority in train: normal 401 · urgent 103 — urgent is rarer, as in a real hotel. Watch its recall in Module 13, not just overall accuracy.
```

The registry maps LLaMA Factory's roles onto your column names:

```json
"hotel_ops": {
  "file_name": "hotel_ops.json",
  "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
}
```

> 💡 The token count is an estimate (about 4 characters per token for English, 2 for Thai script), not a real tokenizer. It only needs to show that nothing comes close to `cutoff_len`.

✓ Checkpoint: `week25/09_llama_factory/data/` holds three files, and every validation line is ✓. Modules 10 and 13 and the capstone reuse this dataset.

## 4 · The training YAML, key by key

A LLaMA Factory run is one YAML file. Lab 03 (step 3) starts from the playbook's `examples/train_lora/qwen3_lora_sft.yaml` and writes six files into `week25/09_llama_factory/configs/`:

| File | Command | Output on the Spark (under `~/w25/m09/`) |
|---|---|---|
| `hotel_lora_sft.yaml` | `llamafactory-cli train` | `saves/qwen3-4b-hotel/lora/sft/` (the adapter) |
| `hotel_qlora_sft.yaml` | `llamafactory-cli train` | `saves/qwen3-4b-hotel/qlora/sft/` |
| `hotel_full_sft.yaml` | `llamafactory-cli train` | `saves/qwen3-4b-hotel/full/sft/` |
| `hotel_chat.yaml` | `llamafactory-cli chat` | — (interactive) |
| `hotel_predict.yaml` | `llamafactory-cli train` (`do_predict`) | `saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl` |
| `hotel_merge.yaml` | `llamafactory-cli export` | `saves/qwen3-4b-hotel/merged/` |

What changed from the upstream example, and why:

| Key | Upstream | Here | Why |
|---|---|---|---|
| `dataset_dir` / `dataset` | `data` / `identity,alpaca_en_demo` | `data` / `hotel_ops` | your dataset, found by name in `data/dataset_info.json` |
| `eval_dataset` | (commented out) | `hotel_ops_eval` | an `eval_loss` curve from the held-out requests |
| `lora_rank` / `lora_alpha` | 8 / (default 2 × rank) | 16 / 32 | a bit more capacity for a 6-way decision in two languages |
| `cutoff_len` | 2048 | 1024 | the records are ~180 tokens at most |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | 1 × 8 | 2 × 4 | the same effective batch of 8, with fewer, larger forward passes |
| `logging_steps` | 10 | 5 | more loss points for a 189-step run |
| `output_dir` | `saves/qwen3-4b/lora/sft` | `saves/qwen3-4b-hotel/lora/sft` | keeps the playbook's smoke test separate |

Everything else (`template: qwen3_nothink`, `learning_rate: 1.0e-4`, cosine schedule, `bf16: true`, 3 epochs) is the playbook's example. Two course deviations: the full-FT file leaves out upstream's `deepspeed: examples/deepspeed/ds_z3_config.json` (ZeRO-3 shards across several GPUs, and one Spark has one GPU), and the QLoRA file copies `quantization_bit: 4` / `quantization_method: bnb` from upstream's `examples/train_qlora/qwen3_lora_sft_otfq.yaml`.

The lab reads its files back and checks them:

**Expected output** (step 3 — captured on this Mac)

```
✓ all 6 files parse as YAML; every key is a known LLaMA Factory argument
✓ datasets ['hotel_ops', 'hotel_ops_eval'] are registered in data/dataset_info.json
✓ one base model and one template (qwen3_nothink) across train, chat, predict and merge
✓ learning_rate is a number (1.0e-4), not the string '1e-4'
✓ chat, predict and merge all load the adapter from saves/qwen3-4b-hotel/lora/sft
✓ merge config has no quantization_bit (upstream warns against it)
```

> ⚠ `learning_rate: 1e-4` looks fine but plain YAML 1.1 parsers such as PyYAML read it as the **string** `'1e-4'`. The upstream examples write `1.0e-4`. Do the same, so every tool agrees on the value.

Step 2 of the lab checks the step arithmetic against the playbook's own numbers. The playbook's run ends at `checkpoint-411`: its two demo datasets hold 91 + 999 = 1,090 records, which gives ⌈1090 ÷ 8⌉ = 137 steps per epoch × 3 epochs = 411:

**Expected output** (step 2 — captured on this Mac)

```
│ playbook example: 1090 records ÷ (1 × 8) per step = 137 steps/epoch × 3 epochs = 411 steps  → the playbook shows 'checkpoint-411'  ✓
│ and 1090 × 3 records ÷ 872.12 s (train_runtime 0:14:32.12) = 3.749 samples/s → the playbook shows 3.749  ✓
│ your hotel run:   504 records ÷ (2 × 4) per step = 63 steps/epoch × 3 epochs = 189 steps
◆ Rough time estimate: 1512 samples ÷ 3.749 samples/s (the playbook's REFERENCE throughput) ≈ 7 min. Your records are shorter than Alpaca's, so expect it to be quicker — lab 04 shows the real time.
```

✓ Checkpoint: you can say what `dataset:` must match, why the template must be the same in all four files, and how many optimizer steps your run will take.

## 5 · Launch under nohup

Step 4 of lab 03 copies the data and the configs to `~/w25/m09/` on the Spark. It checks that `~/factoryEnv/bin/llamafactory-cli` exists and that no other LLaMA Factory job is running. It starts training **only** when you opt in:

```bash
# on: laptop
SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py
SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py --method qlora   # or full
```

The command it runs on the Spark:

```bash
# on: spark
mkdir -p ~/w25/logs && cd ~/w25/m09 && source ~/factoryEnv/bin/activate && { RECORD_VRAM=1 nohup llamafactory-cli train configs/hotel_lora_sft.yaml > ~/w25/logs/m09_hotel_lora.log 2>&1 < /dev/null & echo $! > ~/w25/logs/m09_hotel_lora.pid; }
```

- `cd ~/w25/m09` matters: `dataset_dir: data` and `output_dir: saves/…` are relative to the directory you run in.
- `nohup … &` keeps the job running after ssh disconnects. `< /dev/null` lets ssh return straight away.
- `RECORD_VRAM=1` makes LLaMA Factory add `vram_allocated` / `vram_reserved` to `trainer_log.jsonl`. Lab 04 prints the peak. Compare it with the arithmetic in Section 1.

✓ Checkpoint: in LIVE mode, the lab prints `started pid …` and `~/w25/logs/m09_hotel_lora.log` starts growing. In DRY mode, you can read the launch command and explain each part.

## 6 · Watch the loss from your laptop

LLaMA Factory writes two logs. The **console log** (your nohup file) holds progress bars and `{'loss': …}` lines. **`trainer_log.jsonl`**, in the output directory, holds one JSON line per logging step: `current_steps`, `total_steps`, `loss`, `lr`, `epoch`, `percentage`, `elapsed_time`, `remaining_time`, and `eval_loss` on eval steps. Lab 04 parses both.

Before your own run exists, the lab tests its parser on an EXAMPLE log. The log is made by a formula, and the lab labels it as one:

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py
```

**Expected output** (captured on this Mac; the loss numbers are an EXAMPLE — synthetic, generated by a formula to test the parser, not a training run)

```
▣ STEP 1 · test the parser on an EXAMPLE log first
◈ EXAMPLE — synthetic trainer_log.jsonl generated by a formula to test the parser. Not a training run.
✓ parsed 40 JSON lines (37 loss + 3 eval), skipped the half-written last line
│ step     epoch  loss    lr        loss (0 … 1.74)
│ ───────  ─────  ──────  ────────  ────────────────────────
│ 5/189    0.08   1.7356  9.98e-05  ████████████████████████
│ 30/189   0.48   0.7523  9.39e-05  ██████████░░░░░░░░░░░░░░
│ 55/189   0.87   0.3905  8.05e-05  █████░░░░░░░░░░░░░░░░░░░
│ 80/189   1.27   0.2574  6.19e-05  ████░░░░░░░░░░░░░░░░░░░░
│ 110/189  1.75   0.2033  3.73e-05  ███░░░░░░░░░░░░░░░░░░░░░
│ 135/189  2.14   0.1886  1.88e-05  ███░░░░░░░░░░░░░░░░░░░░░
│ 160/189  2.54   0.1832  5.70e-06  ███░░░░░░░░░░░░░░░░░░░░░
│ 185/189  2.94   0.1812  1.10e-07  ███░░░░░░░░░░░░░░░░░░░░░
◆ EXAMPLE: 97.88% done · elapsed 0:06:10 · left 0:00:00 · loss 1.736 → 0.181 (90% lower)
◆ eval_loss: 0.487@50 → 0.265@100 → 0.235@150
✓ console parser: 2 loss dicts · 1 eval dict · train metrics block · tqdm 180/189
✓ failure detector recognises OOM and an unregistered dataset name
```

With a Spark connected, steps 2–4 read **your** run: is the process alive, what the latest losses are, whether the log shows a known failure (out of memory, gated model, unregistered dataset, unknown template, missing bitsandbytes), and whether the output folder has what the playbook's Step 9 lists: `adapter_config.json`, a checkpoint directory, and `training_loss.png`. Lab 04 is read-only, so re-run it as often as you like. To parse a log you copied yourself, run it with `--log path/to/trainer_log.jsonl`.

How to read a loss curve:

| You see | It usually means | Try |
|---|---|---|
| loss falls fast, then flattens | the model learned the format | done. Check accuracy in Module 13 |
| `eval_loss` rises while `loss` keeps falling | over-fitting (memorising) | fewer epochs, or more varied data |
| loss stays flat from the start | learning rate too low, or labels are masked | check `learning_rate` and the column mapping |
| loss jumps to `nan` | learning rate too high | lower it (for example `5.0e-5`) |

✓ Checkpoint: the three step-1 lines are ✓. With a Spark, you have seen your own run's loss table and its peak `vram_allocated`.

## 7 · Chat, predict, merge

When training finishes, lab 04 runs three follow-ups. Each one runs only when you ask for it with its flag.

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --chat      # 3 guest requests, ~2 min
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --predict   # 60 held-out answers (nohup)
.venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --export    # merge (nohup)
```

- **`llamafactory-cli chat configs/hotel_chat.yaml`** is the playbook's Step 10 with your adapter. It is interactive: type a request, and type `exit` to leave. The lab pipes three requests in, one English, one Thai and one urgent, with `clear` between them so they do not share history.
- **`llamafactory-cli train configs/hotel_predict.yaml`** (`do_predict: true`, `predict_with_generate: true`) writes `generated_predictions.jsonl`, one `{"prompt", "predict", "label"}` line per held-out request. The config sets `do_sample: false` and `max_new_tokens: 160`: LLaMA Factory samples by default (temperature 0.95, top-p 0.7), and that would make the scores change from run to run. Module 13 scores the file.
- **`llamafactory-cli export configs/hotel_merge.yaml`** is the playbook's Step 11. It folds the adapter into the base weights and writes a plain Hugging Face model to `saves/qwen3-4b-hotel/merged/`, which vLLM can serve (Module 13). Upstream's config warns: *do not use a quantized model or `quantization_bit` when merging*. So merge from the bf16 base, even after a QLoRA run.

Where everything ends up on the Spark:

| What | Path |
|---|---|
| dataset + registry | `~/w25/m09/data/{hotel_ops.json, hotel_ops_eval.json, dataset_info.json}` |
| configs | `~/w25/m09/configs/*.yaml` |
| LoRA adapter | `~/w25/m09/saves/qwen3-4b-hotel/lora/sft/` |
| predictions | `~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl` |
| merged model | `~/w25/m09/saves/qwen3-4b-hotel/merged/` |
| logs | `~/w25/logs/m09_*.log` |

> ⚠ Cleanup (the playbook's Step 12) deletes `~/LLaMA-Factory` and `~/factoryEnv` with `rm -rf`. No lab does this for you. Keep them until Module 13 has served your merged model.

✓ Checkpoint: in LIVE mode, `--chat` answers with the JSON format you trained, and `saves/qwen3-4b-hotel/merged/` holds `config.json` plus `.safetensors` shards. In DRY mode, you can name the three follow-up commands and what each one writes.

## Labs — run them here

**labs/lab01_factory_setup.py** — Install LLaMA Factory the playbook's way (venv, PyTorch cu130, clone, pip install) under nohup, and verify that CUDA is visible.

**labs/lab02_hotel_dataset.py** — Build and validate the hotel guest-request dataset (504 train + 60 held-out eval, English and Thai) and its dataset_info.json.

**labs/lab03_config_and_launch.py** — LoRA vs QLoRA vs full arithmetic, six validated YAML files, upload, and an opt-in nohup launch of llamafactory-cli train.

**labs/lab04_monitor_and_export.py** — Parse the training logs (tested first on an EXAMPLE), diagnose failures, then chat, predict and merge on request.

Labs 02 and 03 (steps 1–3) run fully on the laptop. Labs 01, 03 (step 4) and 04 drive the Spark over SSH, or show the plan in DRY mode.

## Try it yourself

**Exercise 09 — fix the config.** Open `week25/09_llama_factory/exercises/ex09_fix_the_config.py`. A teammate's `dataset_info.json` and training YAML have four mistakes. Each would stop a run, or train an adapter that is later used with the wrong chat format:

1. `file_name` points at a file that does not exist.
2. The column mapping names a column the records do not have.
3. `dataset:` is not a name registered in `dataset_info.json`.
4. The `template` does not match the base model or the chat config.

The checker is offline. It parses the text with `json` and PyYAML and compares it with the real files in `data/`.

```bash
# on: laptop
.venv/bin/python week25/09_llama_factory/exercises/ex09_fix_the_config.py
```

**Expected output** (once all four TODOs are fixed; captured on this Mac)

```
✓ TODO 1: file_name 'hotel_ops.json' exists in data/
✓ TODO 2: every mapped column exists in the records (response → output)
✓ TODO 3: dataset 'hotel_ops' is registered in dataset_info.json
✓ TODO 4: template qwen3_nothink in both train and chat configs

▣ your config, applied: 504 records · effective batch 8 · 189 optimizer steps
│ chat loads the adapter from saves/qwen3-4b-hotel/lora/sft ✓ = train output_dir
```

<details><summary>Hint — how do I find the right column name?</summary>

Open `week25/09_llama_factory/data/hotel_ops.json` and look at the first record's keys. `columns` maps LLaMA Factory's roles (`prompt`, `query`, `response`, `system`) onto **your** key names.

</details>

<details><summary>Hint — which template does Qwen3-4B-Instruct-2507 use?</summary>

The playbook's `examples/train_lora/qwen3_lora_sft.yaml` uses `template: qwen3_nothink` for exactly this model. The chat, predict and merge configs must use the same one.

</details>

<details><summary>Stretch — add a seventh department</summary>

Add `spa` templates to `T` in lab 02 (English and Thai, and one held out), add `spa` to `DEPTS`, and re-run labs 02 and 03. How many steps does the run take now? Which checks in lab 02 would catch a template that forgot `{room}` in its reply?

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA: False` after setup | the cu130 wheel did not install. Re-run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` inside `~/factoryEnv` and read the lab 01 log |
| `Cannot open data/dataset_info.json` | you ran `llamafactory-cli` outside `~/w25/m09`. `dataset_dir: data` is relative to the directory you run in |
| `Undefined dataset hotel-ops in dataset_info.json.` | `dataset:` must be a key of `dataset_info.json` (Exercise 09) |
| `CUDA out of memory` during training | the playbook's fix: lower `per_device_train_batch_size` or raise `gradient_accumulation_steps` |
| Memory pressure while `free -g` shows room | UMA page cache. The playbook's fix: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `Cannot access gated repo` | gated model: request access on its page, then `hf auth login` on the Spark |
| QLoRA: `No module named 'bitsandbytes'` | the playbook's install does not include it: `pip install bitsandbytes` in `~/factoryEnv` (not verified on GB10 by this course) |
| The lab hangs after "started" | the background job still holds ssh's stdin. Keep `< /dev/null` in any nohup command you write |
| Training loss is not decreasing | the playbook's advice: adjust `learning_rate` or check dataset quality (lab 02) |

## Next

Continue to [Lab 10 — Unsloth: fast LoRA fine-tuning](../10_unsloth/TUTORIAL.md): train the same hotel dataset with Unsloth, and learn how to compare two frameworks fairly.
