# ▶ Spark Lab 10 — Unsloth: fast LoRA fine-tuning

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Run NVIDIA's Unsloth playbook on your Spark: its PyTorch container, its two pip installs, and its 60-step validation script.
- Train Module 09's hotel guest-request router again, this time with Unsloth, using the **same data and the same hyperparameters**. Lab 02 copies them from Module 09's YAML.
- Learn what makes a comparison between two fine-tuning frameworks fair, then check it: the settings, the timing protocol, and one scorer for both prediction files.

**Time** ~45 min · **Difficulty** intermediate · **Hardware** 1 DGX Spark (or none: DRY mode + the config, conversion and comparison checks run on the laptop)

**Official playbooks covered:** [Fine-Tune Faster with Unsloth](https://build.nvidia.com/spark/unsloth) · [Fine-Tune Specialized LLMs with Unsloth](https://build.nvidia.com/spark/fine-tuning)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 09, labs 02–03 run on the laptop | `ls week25/09_llama_factory/data week25/09_llama_factory/configs` | this module reuses that dataset and YAML |
| Docker with GPU support on the Spark | `docker info --format '{{json .Runtimes}}'` shows `nvidia` | the playbook runs in a container |
| ~20 GB free on the Spark | `df -h ~` | the container image plus the models |
| No other training job running | `pgrep -af llamafactory-cli`, `docker ps` | timings are only comparable one job at a time |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
ls week25/09_llama_factory/data week25/09_llama_factory/configs
```

**Expected output** (captured on this Mac)

```
week25/09_llama_factory/configs:
hotel_chat.yaml
hotel_full_sft.yaml
hotel_lora_sft.yaml
hotel_merge.yaml
hotel_predict.yaml
hotel_qlora_sft.yaml

week25/09_llama_factory/data:
dataset_info.json
hotel_ops.json
hotel_ops_eval.json
```

✓ Checkpoint: both folders exist. If not, run Module 09's labs 02 and 03 first. They need no Spark.

## 1 · What Unsloth is, and what it is not

Unsloth is a Python library for LoRA and QLoRA fine-tuning. It replaces parts of the Hugging Face training path with its own GPU kernels. The playbook describes it as reducing "training time and memory use compared with standard parameter-efficient fine-tuning workflows". The older version of the playbook put a number on it ("2× faster on single GPU"), as a claim. This module lets you **measure** that claim on your own Spark, with your own task.

| | LLaMA Factory (Module 09) | Unsloth (this module) |
|---|---|---|
| You write | a YAML file | a short Python script |
| Runs as | `llamafactory-cli train x.yaml` in a venv | `python script.py` in NVIDIA's PyTorch container |
| Methods | SFT, DPO, KTO, reward modelling, pre-training; LoRA, QLoRA, full | LoRA, QLoRA, full (`full_finetuning=True`), RL via TRL |
| Speed-ups | standard Hugging Face + PEFT | custom kernels and `use_gradient_checkpointing="unsloth"` |
| Web UI | LLaMA Board | none (notebooks) |
| Exports | `llamafactory-cli export` → merged Hugging Face model | `save_pretrained`, merged 16-bit, GGUF (see the Unsloth wiki) |

The companion overview playbook, [Fine-Tune Specialized LLMs with Unsloth](https://build.nvidia.com/spark/fine-tuning), is a map of methods rather than a set of commands. It says itself that "Concrete Unsloth install, launch, and training commands are **not included in this playbook**". Its guidance, quoted: LoRA/QLoRA suit "Small- to medium-sized dataset (about 100–1,000 prompt–sample pairs)", full fine-tuning suits a "Large dataset (1,000+ prompt–sample pairs)", and reinforcement learning needs "An action model, a reward model, and an environment". The hotel dataset has 504 pairs, so LoRA again.

✓ Checkpoint: you can name one thing Unsloth changes (the kernels), and one thing it must **not** change if you want to compare it with Module 09 (the data, the model, or the hyperparameters).

## 2 · The playbook's container and validation run

The [playbook](https://build.nvidia.com/spark/unsloth) pulls NVIDIA's PyTorch container, starts it interactively, installs two sets of packages inside it, downloads `test_unsloth.py`, and runs it:

```bash
# on: spark
nvcc --version                                  # Step 1: expect CUDA 13.0
nvidia-smi
docker pull nvcr.io/nvidia/pytorch:25.11-py3    # Step 2
docker run --gpus all --ulimit memlock=-1 -it --ulimit stack=67108864 --entrypoint /usr/bin/bash --rm nvcr.io/nvidia/pytorch:25.11-py3   # Step 3
```

```bash
# on: spark
# Steps 4–6, inside the container
pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
pip install --no-deps unsloth unsloth_zoo bitsandbytes
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-unsloth/assets/test_unsloth.py
python test_unsloth.py
```

`test_unsloth.py` loads `unsloth/Phi-3.5-mini-instruct` in 4 bits, adds a LoRA adapter (`r = 16`, the 7 linear layers), and runs 60 steps (`max_steps = 60`, batch 2 × accumulation 4, `optim = "adamw_8bit"`) on LAION's OIG `unified_chip2.jsonl`.

An interactive `-it` container does not work under `nohup`, because there is no terminal. So `spark/run_unsloth.sh` runs the same image, ulimits and pip commands **headless**. It changes four things, all listed in the script's header: no `-it`, a `--name` so `docker ps` shows the job, `~/w25/m10` mounted for scripts and results, and `~/.cache/huggingface` mounted (as the VLM playbook's `launch.sh` does) so models download only once. Lab 01 checks the prerequisites and the image, then runs the validation when you opt in:

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/labs/lab01_unsloth_validate.py
SPARK_APPLY=1 .venv/bin/python week25/10_unsloth/labs/lab01_unsloth_validate.py   # pull, then run (re-run to follow)
```

**Expected output** (captured on this Mac, DRY mode)

```
│ prerequisite              result     last line
│ ────────────────────────  ─────────  ────────────────────────────────────────────
│ CUDA 13.0 toolkit         ◈ example  Cuda compilation tools, release 13.0, V13.0.
│ GPU visible               ◈ example  NVIDIA GB10, 580.95.05
│ Docker without sudo       ◈ example  28.3.3
│ NVIDIA container runtime  ◈ example  "nvidia"

▣ STEP 2 · the container image (playbook Step 2: docker pull nvcr.io/nvidia/pytorch:25.11-py3)
…
$ mkdir -p ~/w25/logs && { nohup docker pull nvcr.io/nvidia/pytorch:25.11-py3 > ~/w25/logs/m10_pull.log 2>&1 < /dev/null & }; sleep 2; tail -2 ~/w25/logs/m10_pull.log   [not run]
…
▣ STEP 4 · read the validation log (playbook Step 6: expected output)
$ tail -c 30000 ~/w25/logs/m10_validate.log 2>/dev/null || echo 'no log yet'   [DRY]
◈ REFERENCE — what the playbook says to expect (not your machine):
Expected output in the terminal window:
- "Unsloth: Will patch your computer to enable 2x faster free finetuning"
- Training progress bars showing loss decreasing over 60 steps
- Final training metrics showing completion
```

(The quoted patch message comes from the older version of the playbook, `nvidia/unsloth/README.md`. The current version describes it as "A message that Unsloth will patch the environment for faster fine-tuning".) With a Spark connected, step 4 shows your own run's loss table and its `train_runtime`, and prints `validation finished` when the job ends.

✓ Checkpoint: in LIVE mode, lab 01 prints `validation finished — Unsloth works in this container`. In DRY mode, you can list the four ways `run_unsloth.sh` differs from the playbook's interactive `docker run`, and say why each is needed.

## 3 · The same hotel task, on Unsloth

A comparison is only worth something if the two runs differ in **one** thing: the framework. So lab 02 does not ask you to type hyperparameters. It reads Module 09's `hotel_lora_sft.yaml` and maps each key to its Unsloth/TRL name:

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/labs/lab02_hotel_unsloth.py
```

**Expected output** (captured on this Mac)

```
▣ STEP 1 · derive the Unsloth config from Module 09's YAML
│ LLaMA Factory key           value                        Unsloth / TRL key                           value
│ ──────────────────────────  ───────────────────────────  ──────────────────────────────────────────  ───────────────────────────
│ model_name_or_path          Qwen/Qwen3-4B-Instruct-2507  model_name                                  Qwen/Qwen3-4B-Instruct-2507
│ cutoff_len                  1024                         max_seq_length                              1024
│ quantization_bit            — (bf16)                     load_in_4bit                                False
│ lora_rank / lora_alpha      16 / 32                      r / lora_alpha                              16 / 32
│ lora_dropout                0.05                         lora_dropout                                0.05
│ lora_target                 all                          target_modules                              q k v o gate up down
│ batch × accumulation        2 × 4                        per_device … × gradient_accumulation_steps  2 × 4
│ learning_rate · schedule    0.0001 · cosine              learning_rate · lr_scheduler_type           0.0001 · cosine
│ num_train_epochs · warmup   3.0 · 0.1                    num_train_epochs · warmup_ratio             3.0 · 0.1
│ (prompt masked by default)  answer-only loss             completion_only_loss=True                   answer-only loss
→ wrote 10_unsloth/configs/hotel_unsloth_lora.json

▣ STEP 2 · convert Alpaca records → TRL conversational prompt–completion (answer-only loss)
✓ hotel_ops_chat.jsonl: 504 records, identical text to Module 09's hotel_ops.json (content sha256 d358d3c70ad8)
✓ hotel_ops_eval_chat.jsonl: 60 records, identical text to Module 09's hotel_ops_eval.json (content sha256 0da9728df351)
…
▣ STEP 3 · the training script (adapted from the playbook's test_unsloth.py)
✓ 10_unsloth/spark/train_hotel_unsloth.py compiles (syntax only — it imports unsloth, so it runs on the Spark, not here)
```

Three details matter more than they look:

1. **Answer-only loss.** LLaMA Factory computes the loss only on the response by default, and the prompt is masked. Lab 02 writes each record as `{"prompt": [system, user], "completion": [assistant]}`, the format in which TRL computes the loss on the completion only. The script also sets `completion_only_loss=True`. If you trained on the prompt too, the loss would drop fast on the long, repeated system prompt, and the run would look "better" without being better (Exercise 10).
2. **The same base model.** The script loads `Qwen/Qwen3-4B-Instruct-2507`, the exact repo Module 09 used. The playbook's script uses a pre-quantized `unsloth/…-bnb-4bit` model instead; that would be a QLoRA run, which is a different experiment. Add `--qlora` to lab 02 if you want that experiment.
3. **The same optimizer.** The playbook script uses `adamw_8bit`. The hotel script uses `adamw_torch`, Hugging Face's AdamW, to match the Hugging Face Trainer that LLaMA Factory runs on. With LoRA it hardly matters for memory (see the chart in 📊): the optimizer state for a 33 M-parameter adapter is about 0.5 GB either way.

`spark/train_hotel_unsloth.py` keeps the playbook script's calls: `FastModel.from_pretrained`, `FastLanguageModel.get_peft_model(… use_gradient_checkpointing="unsloth" …)`, `SFTTrainer(… SFTConfig(…))`. At the end it prints one `W25_RESULT {…}` line (runtime, samples/s, train and eval loss, peak memory), saves the adapter to `~/w25/m10/saves/qwen3-4b-hotel-unsloth/lora/`, and writes `generated_predictions.jsonl` for the 60 held-out requests, greedy, in LLaMA Factory's `{prompt, predict, label}` format.

Launch it when you are ready (one job at a time: the lab refuses if another training job is running):

```bash
# on: laptop
SPARK_APPLY=1 .venv/bin/python week25/10_unsloth/labs/lab02_hotel_unsloth.py
```

The command it runs on the Spark:

```bash
# on: spark
mkdir -p ~/w25/logs && { nohup bash ~/w25/m10/run_unsloth.sh hotel-lora train_hotel_unsloth.py > ~/w25/logs/m10_hotel_lora.log 2>&1 < /dev/null & }
tail -f ~/w25/logs/m10_hotel_lora.log
```

✓ Checkpoint: both conversion lines are ✓ with the same record counts as Module 09. In LIVE mode, `docker ps` shows `w25-unsloth-hotel-lora` and the log fills with `{'loss': …}` lines.

## 4 · Compare the two frameworks fairly

"Unsloth is 2× faster" and "LLaMA Factory reaches a lower loss" are only claims until you control for everything else. Lab 03 does it in four steps:

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/labs/lab03_compare_frameworks.py
```

**Expected output** (captured on this Mac, DRY mode: step 1 compares the real config files, step 4's scorer runs on an EXAMPLE — synthetic predictions edited by hand, not a model's output)

```
▣ STEP 1 · is it a fair fight? (from the real config files)
│ what                                      LLaMA Factory (Module 09)             Unsloth (lab 02)                           verdict
│ ────────────────────────────────────────  ────────────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────
│ base model                                Qwen/Qwen3-4B-Instruct-2507           Qwen/Qwen3-4B-Instruct-2507                ✓ same
│ training data                             hotel_ops (504)                       hotel_ops_chat.jsonl (504)                 ✓ same text (lab 02 checked)
│ base precision                            bf16                                  bf16                                       ✓ same
│ LoRA r / alpha / dropout                  16 / 32 / 0.05                        16 / 32 / 0.05                             ✓ same
│ LoRA targets                              all                                   7 linear layers                            ✓ same
│ effective batch                           2 × 4                                 2 × 4                                      ✓ same
│ learning rate · schedule · warmup         0.0001 · cosine · 0.1                 0.0001 · cosine · 0.1                      ✓ same
│ epochs · max length                       3.0 · 1024                            3.0 · 1024                                 ✓ same
│ loss on                                   answer only (default)                 answer only (completion_only_loss)         ✓ same
│ optimizer                                 HF Trainer default (see log: optim=)  adamw_torch                                ◆ check the LLaMA Factory log line
│ chat format                               template: qwen3_nothink               the tokenizer's own chat template          ◆ compare one rendered prompt
│ kernels, packing, gradient checkpointing  LLaMA Factory defaults                Unsloth kernels · 'unsloth' checkpointing  ◆ the difference you are measuring
✓ every setting that must match does match

▣ STEP 2 · the measurement protocol
│ 1. Same Spark, one job at a time: lab 02 refuses to launch while another training job runs.
│ 2. Time the training loop, not the download: compare train_runtime, which starts after the model is loaded.
│ 3. Run each framework twice. Report both numbers; if they differ by more than a few %, find out why.
│ 4. Peak memory from the framework itself (torch max_memory_allocated): LLaMA Factory with RECORD_VRAM=1,
│    Unsloth in the W25_RESULT line. On unified memory `free -g` also counts the page cache.
│ 5. Quality on the SAME 60 held-out requests, scored by the SAME code (step 4). Loss values are not
│    comparable across frameworks: chat templates and padding can shift them without changing answers.
…
▣ STEP 4 · one scorer for both: department, priority, urgent recall, reply language
◈ EXAMPLE — synthetic predictions written to test the scorer (5 held-out requests, answers edited by hand).
│ EXAMPLE metric  value
│ ──────────────  ─────
│ n               5
│ valid JSON      80 %
│ department      60 %
│ priority        60 %
│ urgent recall   0 %
│ reply language  80 %
✓ scorer: tolerates ```json fences, fails non-JSON answers, catches a wrong department and a missed urgent
```

The three ◆ rows cannot be forced equal, so you report them:

- **Optimizer.** LLaMA Factory prints its `TrainingArguments`, including `optim=`, near the top of its log. If it is not `adamw_torch`, note that next to your result, or set `optim:` in the YAML and train again.
- **Chat format.** LLaMA Factory builds prompts from its own `qwen3_nothink` template. Unsloth/TRL uses the chat template that ships with the tokenizer. For an instruct model the two should give the same text. Check one prompt from each before you trust a quality difference.
- **Kernels.** This is what you are measuring.

With a Spark, step 3 fills a table from **your** two runs (`train_runtime`, samples/s, peak memory, final loss). Step 4 scores both `generated_predictions.jsonl` files: Module 09's comes from `lab04_monitor_and_export.py --predict`, and the Unsloth run writes its own. Both use greedy decoding (Module 09's `hotel_predict.yaml` sets `do_sample: false`, because LLaMA Factory samples by default), so the scores are not noise.

✓ Checkpoint: every must-match row is ✓, and you can explain why the final training loss is listed as "not comparable".

## 5 · What to do with the Unsloth adapter

The adapter in `~/w25/m10/saves/qwen3-4b-hotel-unsloth/lora/` is a standard PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`), the same kind that LLaMA Factory writes. So Module 13 can serve it with vLLM's LoRA support or merge it in the same way. The playbook points to the [Unsloth wiki](https://github.com/unslothai/unsloth/wiki) for saving to GGUF (for llama.cpp and Ollama, Module 04), continuing training from a checkpoint, custom chat templates, and evaluation loops.

To remove what the playbook added, exit the container (the `--rm` flag deletes it). The image stays until you run `docker rmi nvcr.io/nvidia/pytorch:25.11-py3`. No lab runs that for you.

✓ Checkpoint: you know where both frameworks' adapters and prediction files live on the Spark, and that Module 13 can use either one.

## Labs — run them here

**labs/lab01_unsloth_validate.py** — Playbook prerequisites, the container image, and the playbook's own 60-step test_unsloth.py run headless under nohup.

**labs/lab02_hotel_unsloth.py** — Derive the Unsloth config from Module 09's YAML, convert the same 504 records to answer-only format, and launch the run (opt-in).

**labs/lab03_compare_frameworks.py** — Check that the comparison is fair, print the measurement protocol, read both runs' numbers, and score both prediction files with one scorer.

Lab 02 steps 1–3 and lab 03 steps 1, 2 and 4 run on the laptop for real. The rest drives the Spark, or shows the plan in DRY mode.

## Try it yourself

**Exercise 10 — make an unfair comparison fair.** Open `week25/10_unsloth/exercises/ex10_fair_comparison.py`. A teammate reported "Unsloth is 2× faster and has lower loss" from a run whose settings came from a notebook. Four settings differ from Module 09. The checker reads Module 09's **real** YAML and tells you which ones:

1. LoRA `r` / `lora_alpha`.
2. The effective batch (`per_device_train_batch_size × gradient_accumulation_steps`).
3. The learning rate.
4. What the loss is computed on (`completion_only_loss`).

```bash
# on: laptop
.venv/bin/python week25/10_unsloth/exercises/ex10_fair_comparison.py
```

**Expected output** (once all four TODOs are fixed; captured on this Mac)

```
✓ TODO 1: LoRA r / alpha = 16 / 32, same as Module 09
✓ TODO 2: effective batch 8, same as Module 09 → the same 189 optimizer steps
✓ TODO 3: learning rate 0.0001 · cosine · warmup 0.1
✓ TODO 4: loss on the answer only, like LLaMA Factory

▣ now a speed or accuracy difference says something about the framework, not the settings.
│ still compare on the same Spark, one job at a time, two runs each, with one scorer (lab 03).
```

<details><summary>Hint — why does the batch size change the "speed"?</summary>

The number of optimizer steps is ⌈records ÷ effective batch⌉ × epochs. A batch of 16 takes 96 steps instead of 189, and a bigger batch keeps the GPU busier. So the run finishes sooner for reasons that have nothing to do with the framework.

</details>

<details><summary>Stretch — the QLoRA experiment</summary>

Run `lab02_hotel_unsloth.py --qlora` and Module 09's `lab03_config_and_launch.py --method qlora`. Both now load a 4-bit base. Compare the peak memory with Module 09's arithmetic (13 GB for QLoRA against 19 GB for LoRA, plus activations), and the held-out accuracy with the bf16 runs. Does 4-bit cost this task anything?

</details>

✓ Checkpoint: all four checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: permission denied` | the playbook's fix: `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with a GPU error | the playbook's fix: `nvidia-ctk runtime configure --runtime=docker`, then restart Docker |
| `the input device is not a TTY` | you ran a `docker run -it …` under nohup. Use `run_unsloth.sh`, which has no `-it` |
| pip or import errors for Unsloth / Triton | the playbook's advice: use its container image and install commands exactly, and retry in a fresh container |
| `TypeError: … unexpected keyword argument` from `SFTConfig` or `SFTTrainer` | the package versions differ from the playbook's pins. Keep `"datasets==4.3.0" "trl==0.26.1"`: the playbook's own script passes `max_seq_length` and `tokenizer` with exactly those |
| CUDA out of memory | the playbook's fix: lower `per_device_train_batch_size` or `max_seq_length`, or use a 4-bit model. Lower the batch in **both** frameworks, or the comparison breaks |
| Memory pressure while `free -g` shows room | UMA page cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| Lab 03 shows `✕ differs` | you changed one framework's settings and not the other's. Re-run lab 02, which re-derives from Module 09 |

## Next

Continue to [Lab 11 — PyTorch and NeMo AutoModel fine-tuning](../11_pytorch_nemo_two_sparks/TUTORIAL.md): fine-tune with plain PyTorch and NeMo AutoModel, on one Spark and then across two.
