# ▶ Spark Lab 11 — PyTorch and NeMo AutoModel fine-tuning, on one and two Sparks

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Compare the two NVIDIA fine-tuning paths: plain **PyTorch** recipes (Transformers + PEFT + TRL) and **NeMo AutoModel** (YAML recipes).
- Work out, with arithmetic, why full fine-tuning costs ~8 bytes per parameter, LoRA ~2 and QLoRA ~0.5, and where activations blow the budget.
- Start a LoRA job on one Spark in the background, with a log file, and watch it with a parser that reads the loss curve.
- Run NeMo AutoModel's LoRA, QLoRA and full-SFT examples the same way.
- Prepare a **two-Spark** run (Docker Swarm + Accelerate + FSDP) for Llama 3.1 70B LoRA, and see what FSDP sends over the cable.

**Time** ~60 min (plus training time) · **Difficulty** advanced · **Hardware** 1 Spark (2 Sparks for Section 6; or none: DRY mode + arithmetic labs)

**Official playbooks covered:** [Fine-tune with PyTorch](https://build.nvidia.com/spark/pytorch-fine-tune) · [Fine-tune with NeMo](https://build.nvidia.com/spark/nemo-fine-tune)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 01's doctor all ✓ on Spark A | lab 01-1 | Docker without sudo, CUDA 13, ~119 GiB free |
| Module 02 done (Section 6 only) | lab 02-1 all ✓, lab 02-3 busbw ≥ 21.875 GB/s | FSDP talks over the QSFP link |
| A Hugging Face login **on the Spark** | `test -s ~/.cache/huggingface/token && echo ok` on the Spark | the Llama models are gated |
| Access to the gated models | accept the licence on each model page | otherwise: "Cannot access gated repo" |
| ~100 GB free disk | `df -h /` | two containers + model weights |

Log in to Hugging Face once, **on the Spark**. The token lands in `~/.cache/huggingface/`, which every container in this module mounts. Never put the token in a lab, a script you commit, or a chat:

```bash
# on: spark
hf auth login
```

(If `hf` is not installed on the Spark itself, run the same command once inside the PyTorch container of Section 3: the cache folder is mounted, so the login persists.) Models this module downloads: `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `unsloth/Meta-Llama-3.1-70B-bnb-4bit` (PyTorch scripts), and `meta-llama/Llama-3.1-8B`, `meta-llama/Meta-Llama-3-70B`, `Qwen/Qwen3-8B` (NeMo examples). Request access on each page you plan to use.

Then check the memory math on your laptop:

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab01_finetune_memory.py
```

**Expected output** (arithmetic, captured on this Mac; the first table)

```
▣ STEP 1 · parameters, counted from each model's shape
│ model          total   linear layers  embeddings  LoRA r=8 params (all 7 projections)
│ ─────────────  ──────  ─────────────  ──────────  ───────────────────────────────────
│ Llama 3.2 3B   3.21B   2.82B          0.39B       12,156,928
│ Llama 3.1 8B   8.03B   6.98B          1.05B       20,971,520
│ Llama 3.1 70B  70.55B  68.45B         2.10B       103,546,880
│ Qwen3 8B       8.19B   6.95B          1.24B       21,823,488
```

✓ Checkpoint: the Spark has a Hugging Face login, you have access to at least `meta-llama/Llama-3.1-8B-Instruct`, and lab 01 runs on your laptop.

## 1 · Two ways to fine-tune on a Spark

Both playbooks fine-tune Hugging Face models inside an NGC container. They differ in how much you see and write:

| | PyTorch playbook | NeMo AutoModel playbook |
|---|---|---|
| Container | `nvcr.io/nvidia/pytorch:25.11-py3` | `nvcr.io/nvidia/nemo-automodel:26.02` |
| You install | `pip install transformers peft datasets trl bitsandbytes` | nothing: AutoModel is in `/opt/Automodel` |
| A recipe is | one Python script per model (argparse flags) | a YAML file + `--section.key value` overrides |
| Recipes | Llama 3.2 3B full SFT · Llama 3.1 8B LoRA · Llama 3.1 70B LoRA (FSDP) · Llama 3.1 70B QLoRA | LoRA (Llama 3.1 8B), QLoRA (Meta-Llama-3 70B), full SFT (Qwen3 8B) |
| Data | 512 Alpaca samples by default (`--dataset_size`) | SQuAD (the recipe files are named `*_squad_*`), packed into 1024-token sequences |
| Saves a model | only the 70B LoRA script (`--output_dir`) | always: `checkpoints/LATEST/` (model, optim, rng, …) |
| Two Sparks | **yes**: Docker Swarm + Accelerate + FSDP (Section 6) | the Spark column of the playbook's matrix says "—": single-Spark examples only |

Use the PyTorch scripts to see every line of a training loop (they are ~200 lines each, in `assets/`). Use NeMo AutoModel when you want tested recipes, packed sequences and checkpoints without writing code.

✓ Checkpoint: you can say which of the two can run across two Sparks in the playbooks, and which one saves checkpoints by default.

## 2 · Memory math: full vs LoRA vs QLoRA

Training memory is **state + activations**:

```text
state        = weights + gradients + optimizer states       (bytes per parameter depend on the method)
activations  ≈ tokens in flight × hidden × layers × ~34 B    (a rough rule; checkpointing cuts it to ~2 B + one layer)
```

The PyTorch scripts load the model in `bfloat16` and train with `optim="adamw_torch"`, whose two moment buffers (m, v) take the parameter's dtype. So a full fine-tune costs **8 bytes per parameter**: weight 2 + gradient 2 + m 2 + v 2. Recipes that keep fp32 "master" weights and fp32 moments need 16. LoRA freezes the base (2 B/param, no gradients, no optimizer) and trains ~0.3% extra parameters. QLoRA also stores the frozen base in 4-bit NF4 (~0.52 B/param).

Lab 01 applies this to every recipe in this module:

**Expected output** (arithmetic, captured on this Mac; "typ" assumes Alpaca samples of at most 300 tokens, an assumption, not a measurement)

```
▣ STEP 3 · the recipes: state + activations (+ 10 GB headroom)
│ recipe                               weights  grads+opt  acts (typ)  acts (cap)  total (typ)  fits on
│ ───────────────────────────────────  ───────  ─────────  ──────────  ──────────  ───────────  ───────────────
│ PyTorch full SFT · Llama 3.2 3B         6.4     19.3        8.3        56.3        44.0 GB    1 Spark
│ PyTorch LoRA · Llama 3.1 8B            16.1      0.1       11.9        81.4        38.2 GB    1 Spark
│ PyTorch QLoRA · Llama 3.1 70B          39.7      0.6       54.7       373.5       105.1 GB    1 Spark
│ PyTorch QLoRA +ckpt · Llama 3.1 70B    39.7      0.6        5.0        34.4        55.4 GB    1 Spark
│ PyTorch LoRA (FSDP) · Llama 3.1 70B   141.3      0.6        2.5        17.2       154.5 GB    2 Sparks (FSDP)
│ NeMo full SFT · Qwen3 8B               16.4     49.1        5.8         5.8        81.3 GB    1 Spark
│ NeMo LoRA · Llama 3.1 8B               16.1      0.1        5.1         5.1        31.3 GB    1 Spark
…
▣ STEP 4 · what checkpointing and micro-batch size do to activations (Llama 3.1 70B QLoRA)
│ batch 8 × 2048 tokens · checkpointing off  ██████████████████████████░░  373.5 GB
│ batch 8 ×  300 tokens · checkpointing off  ████░░░░░░░░░░░░░░░░░░░░░░░░   54.7 GB
│ batch 8 × 2048 tokens · checkpointing on   ██░░░░░░░░░░░░░░░░░░░░░░░░░░   34.4 GB
│ batch 1 × 2048 tokens · checkpointing on   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░    4.3 GB
```

What to take from it:

1. **For most recipes the weights fit; the activations are what blow up.** `--seq_length 2048` is a cap: the scripts pad each batch only to its longest sample, and Alpaca samples are short. That is why NVIDIA's benchmark guide can run the 70B QLoRA defaults on one Spark. Point the same script at your own long documents and the "cap" column is what you get.
2. **`--gradient_checkpointing` is the big lever.** It recomputes activations in the backward pass instead of storing them: ~10× less activation memory for ~30% more compute (a common rule of thumb). The 3B full and 70B QLoRA scripts accept the flag (off by default); the 70B LoRA script has it on by default; the 8B LoRA script has no such flag.
3. **Full fine-tuning an 8B model fits one Spark** at 8 B/param (64 GB of state), but not with fp32 master weights (128 GB of state alone).
4. **Llama 3.1 70B LoRA in bf16 needs two Sparks**: 141 GB of frozen weights. With FSDP each Spark holds half. This is the playbook's multi-node example (Section 6).

The NeMo rows assume the same 8 B/param. The YAML recipe decides the real optimizer settings, so treat those rows as a sketch and check lab 03's `free -g` on your run.

✓ Checkpoint: you can explain why 70B QLoRA fits on one Spark while 70B LoRA in bf16 does not, and name the flag you turn on first when your data gets longer.

## 3 · PyTorch on one Spark: container, recipe, flags

These are the playbook's single-Spark steps. Pull and start the container, with the Hugging Face cache and the current folder mounted:

```bash
# on: spark
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker run --gpus all -it --rm --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${PWD}:/workspace" -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3
```

Inside the container, install the training stack and get the recipes:

```bash
# on: spark
pip install transformers peft datasets trl bitsandbytes
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
python Llama3_8B_LoRA_finetuning.py --dataset_size 100 --num_epochs 1 --batch_size 2
```

> ⚠ The playbook's Step 6 says `cd client-hardware-playbooks/…` after cloning `dgx-spark-playbooks`. That folder does not exist; use `dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets` as above.

The four scripts and the flags each one **actually defines** (read from the scripts in `assets/`; the README says "all scripts support" some flags that only some scripts have):

| Script | Model (default `--model_name`) | Method | Defaults | Extra flags |
|---|---|---|---|---|
| `Llama3_3B_full_finetuning.py` | `meta-llama/Llama-3.2-3B-Instruct` | full SFT | batch 8, lr 5e-5 | `--gradient_checkpointing` |
| `Llama3_8B_LoRA_finetuning.py` | `meta-llama/Llama-3.1-8B-Instruct` | LoRA r=8, all 7 projections | batch 8, lr 1e-4 | `--lora_rank` |
| `Llama3_70B_LoRA_finetuning.py` | `meta-llama/Llama-3.1-70B-Instruct` | LoRA, FSDP-ready | batch 4, checkpointing **on** | `--lora_rank`, `--use_torch_compile`, `--output_dir` |
| `Llama3_70B_qLoRA_finetuning.py` | `unsloth/Meta-Llama-3.1-70B-bnb-4bit` | QLoRA (NF4, double quant) | batch 8 | `--lora_rank`, `--gradient_checkpointing` |

All four accept `--dtype`, `--batch_size`, `--seq_length` (2048), `--num_epochs` (1), `--gradient_accumulation_steps`, `--learning_rate`, `--dataset_size` (512; 500 for 70B LoRA), `--logging_steps` (1) and `--log_dir`. Two behaviours worth knowing:

- The 3B and 8B scripts always `torch.compile` the model and run a short **warmup training pass** first, then train for real. Your log shows two sets of loss lines (lab 03 skips the first).
- Only the 70B LoRA script defines `--output_dir`, so the other runs save nothing: they prove the pipeline and give you a loss curve. Module 13 saves, merges and serves a fine-tune.

✓ Checkpoint: you can name the one PyTorch script that saves a model, and the flag you would add to the QLoRA script for long data.

## 4 · NeMo AutoModel on one Spark

The NeMo playbook checks the host, pulls its container and opens a shell in it:

```bash
# on: spark
nvcc --version && python3 --version && nvidia-smi && free -h && docker ps
docker pull nvcr.io/nvidia/nemo-automodel:26.02
docker run \
  --gpus all \
  --ulimit memlock=-1 \
  -it --ulimit stack=67108864 \
  --entrypoint /usr/bin/bash \
  --rm nvcr.io/nvidia/nemo-automodel:26.02
```

Inside, every run is `examples/llm_finetune/finetune.py` + a recipe YAML + overrides. The three playbook examples (each stops after 20 steps):

```bash
# on: spark
cd /opt/Automodel
ls examples/llm_finetune/
export HF_TOKEN=<your_huggingface_token>      # the playbook's way; lab 02 mounts your cached login instead

# LoRA on Llama 3.1 8B
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml \
--model.pretrained_model_name_or_path meta-llama/Llama-3.1-8B \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20

# QLoRA on Meta-Llama-3 70B
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml \
--model.pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B \
--loss_fn._target_ nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy \
--step_scheduler.local_batch_size 1 \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20

# full SFT on Qwen3 8B
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml \
--model.pretrained_model_name_or_path Qwen/Qwen3-8B \
--step_scheduler.local_batch_size 1 \
--step_scheduler.max_steps 20 \
--packed_sequence.packed_sequence_size 1024
```

Read the overrides as a sentence: *take this recipe, but load that model, pack sequences to 1024 tokens, micro-batch 1, stop at step 20*. The recipe file sets the rest (LoRA rank, learning rate, …); the LoRA example reuses a 1B recipe on purpose, because `--model.pretrained_model_name_or_path` decides which weights load. **Packing** concatenates short samples into full 1024-token rows, so no compute is wasted on padding.

When a run ends, the playbook checks the checkpoint:

```bash
# on: spark
ls -lah checkpoints/LATEST/
```

**Expected output** (REFERENCE — quoted from the playbook; user and group are its placeholders)

```
total 32K
drwxr-xr-x 6 username domain-users 4.0K Oct 16 22:33 .
drwxr-xr-x 4 username domain-users 4.0K Oct 16 22:33 ..
-rw-r--r-- 1 username domain-users 1.6K Oct 16 22:33 config.yaml
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 dataloader
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 model
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 optim
drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 rng
-rw-r--r-- 1 username domain-users 1.3K Oct 16 22:33 step_scheduler.pt
```

> ⚠ The playbook's container runs with `--rm` and writes `checkpoints/` inside it: exit the shell and the checkpoint is gone. Lab 02 mounts `~/w25/nemo-checkpoints` at `/opt/Automodel/checkpoints` so it survives. `checkpoints/LATEST/model` is what the playbook's optional last step uploads: `hf upload my-cool-model checkpoints/LATEST/model`.

✓ Checkpoint: you can say what `--packed_sequence.packed_sequence_size 1024` changes, and where a NeMo checkpoint ends up when you use lab 02.

## 5 · Launch in the background, then watch the log

The Lab Runner stops a foreground lab after 900 seconds, and a fine-tune takes longer. So lab 02 starts the job with `nohup … > ~/w25/logs/<job>.log 2>&1 &` and returns at once. It runs the playbook's `docker run`, made non-interactive (no `-it`), with a `--name` so you can stop it. First it checks Docker, that no other course job is running (they share the 128 GB), and that a Hugging Face login exists (it never prints the token).

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab02_launch_job.py
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab02_launch_job.py --recipe nemo-lora-8b
```

Recipes: `pytorch-lora-8b` (default: the playbook's usage example), `pytorch-full-3b`, `pytorch-qlora-70b`, `nemo-lora-8b`, `nemo-qlora-70b`, `nemo-sft-qwen3-8b`.

**Expected output** (DRY mode, captured on this Mac: the command it would run)

```
▣ STEP 2 · start it in the background
◆ Course deviations from the playbooks, on purpose: no `-it` (nothing is attached), a --name so you can stop it, nohup + a log file. The playbook clones the recipes inside the container; this clones them once to ~/w25 on the Spark and mounts them.
$ mkdir -p ~/w25/logs   [DRY]
  [ -d ~/w25/dgx-spark-playbooks/.git ] || { rm -rf ~/w25/dgx-spark-playbooks; git clone --depth 1 https://github.com/NVIDIA/dgx-spark-playbooks ~/w25/dgx-spark-playbooks; }
  cd ~/w25/dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
  nohup docker run --gpus all --rm --ipc=host --name w25-m11-pytorch-lora-8b \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "${PWD}:/workspace" -w /workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c 'pip install "transformers>=4.57.1,<5" "trl>=0.25.1,<0.26" "peft<0.18" datasets "bitsandbytes>=0.48.2" && pip uninstall -y torchao && python Llama3_8B_LoRA_finetuning.py --dataset_size 100 --num_epochs 1 --batch_size 2' \
    > ~/w25/logs/m11_pytorch-lora-8b.log 2>&1 &
  echo "started w25-m11-pytorch-lora-8b → ~/w25/logs/m11_pytorch-lora-8b.log"
```

Watch it by hand in the ⌨ terminal (Spark A), or with lab 03:

```bash
# on: spark
docker ps --filter name=w25-m11
tail -f ~/w25/logs/m11_pytorch-lora-8b.log
```

Lab 03 reads the newest log's last 400 lines and parses both trainers: the Hugging Face / TRL dict per step (`{'loss': …, 'grad_norm': …, 'learning_rate': …, 'epoch': …}`) plus the script's `TRAINING COMPLETED` block, and NeMo-style `step N … loss X` lines. It draws the loss curve, flags NaN or a loss that is not falling, spots `out of memory` or gated-model errors, and shows `free -g` and GPU use.

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab03_watch_log.py
```

**Expected output** (DRY mode, captured on this Mac. The log it parses is an EXAMPLE: the line layout follows the script's `print()` calls, every number is illustrative, not a Spark run)

```
▣ STEP 3 · parse it
◆ trainable parameters: 20,971,520  (= lab 01's count for Llama 3.1 8B at r=8)
◆ 1 warmup pass(es) for torch.compile() skipped (3 steps) — the scripts train once to compile, then again for real
│ epoch 0.02  loss  1.6801  ████████████████████████████
│ epoch 0.04  loss  1.6135  █████████████████████████░░░
…
│ epoch 1.00  loss  1.1876  █████░░░░░░░░░░░░░░░░░░░░░░░
✓ loss is finite and falling: first 2 avg 1.647 → last 2 avg 1.194 (27% lower)
│ runtime  samples/s  steps/s  train loss
│ ───────  ─────────  ───────  ──────────
│ 180.0 s  0.556      0.278    1.2764
◆ finished: yes — TRAINING COMPLETED
…
▣ STEP 5 · parser self-test — runs on every machine
✓ HF/TRL format: 3 warmup + 13 training steps, trainable count, summary and TRAINING COMPLETED all found
✓ NeMo-style `step N | loss X` lines parsed (3 steps)
```

The trainable-parameter line is the one number you can check before training starts: for Llama 3.1 8B at rank 8 on all seven projections it must be exactly 20,971,520 (32 layers × 8 × the sum of each projection's in + out size). If yours differs, LoRA targeted other modules than you think.

> 💡 NeMo AutoModel's log layout changes between releases, and the playbook shows none. The NeMo parser only looks for `step` and `loss` tokens. If lab 03 finds no steps in a NeMo log, open the log and adapt the regex in `parse_log()`.

✓ Checkpoint: LIVE, lab 03 shows your run's trainable-parameter count, a falling loss, and `TRAINING COMPLETED` (PyTorch) or 20 steps (NeMo).

## 6 · Two Sparks: Docker Swarm + Accelerate + FSDP

The PyTorch playbook's **Multi-node fine-tuning** tab scales the same scripts over two Sparks. The pieces:

| Piece | Job |
|---|---|
| **Docker Swarm** | runs one copy of the PyTorch container on each Spark (`docker-compose.yml`, `replicas: 2`, one `NVIDIA_GPU` each, host network) |
| **pytorch-ft-entrypoint.sh** | starts sshd in each container (port 2233) so the containers can reach each other |
| **Accelerate** | starts one training process per Spark from a config file (`machine_rank`, `main_process_ip`, `main_process_port`) |
| **FSDP** | shards weights, gradients and optimizer states across the two processes |
| **NCCL** (Module 02) | moves the shards over the QSFP link: `NCCL_SOCKET_IFNAME=enp1s0f1np1` in the compose file |

Lab 04 checks both Sparks (read-only), writes the two Accelerate files from the playbook's own configs, and prints the rest:

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab04_two_spark_fsdp.py              # 70B / 8B LoRA
.venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab04_two_spark_fsdp.py --config full # 3B full SFT
```

**Expected output** (DRY mode, captured on this Mac: preflight rows are EXAMPLE shapes, the files are real)

```
│ node     enp1s0f1np1     swarm     GPU UUID  daemon.json NVIDIA_GPU  swarm-resource  recipes
│ ───────  ──────────────  ────────  ────────  ──────────────────────  ──────────────  ───────
│ Spark A  192.168.100.10  inactive  ✓         ✕ step 3                ✕ step 3        ✓
│ Spark B  192.168.100.11  inactive  ✓         ✕ step 3                ✕ step 3        ✓

▣ STEP 2 · write config_fsdp_lora.yaml for each Spark (rank 0 = Spark A, the primary)
── config_fsdp_lora.spark-a.yaml: machine_rank: 0 · main_process_ip: 192.168.100.10 · main_process_port: 29500
── config_fsdp_lora.spark-b.yaml: machine_rank: 1 · main_process_ip: 192.168.100.10 · main_process_port: 29500
✓ both files parse, num_machines 2, ranks 0 and 1, the same main_process_ip on both
```

Port 29500 is a course choice; any free port on Spark A works. Then follow the playbook, step by step, in the ⌨ terminal.

**Step 1 — the interconnect IP** (on each Spark):

```bash
# on: spark
ip -br -4 address
export MN_IF_NAME="enp1s0f1np1"
export MN_IP_ADDRESS="$(ip -4 addr show "$MN_IF_NAME" | awk '/inet / {print $2}' | cut -d/ -f1)"
echo "$MN_IF_NAME $MN_IP_ADDRESS"
```

**Steps 2–3 — let Swarm hand out the GPU** (on **both** Sparks; needs sudo). Find the UUID, then edit `/etc/docker/daemon.json` so it contains the playbook's block with **your** UUID:

```bash
# on: spark
nvidia-smi -a | grep UUID
sudoedit /etc/docker/daemon.json
```

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "node-generic-resources": [
    "NVIDIA_GPU=GPU-45cbf7b3-f919-7228-7a26-b06628ebefa1"
  ]
}
```

```bash
# on: spark
sudo sed -i 's/^#\s*\(swarm-resource\s*=\s*".*"\)/\1/' /etc/nvidia-container-runtime/config.toml
sudo systemctl restart docker
```

**Steps 4–5 — form the swarm.** On Spark A: `docker swarm init --advertise-addr "$MN_IP_ADDRESS"`. It prints a `docker swarm join --token …` line; run that line on Spark B.

**Step 6 — deploy the stack** (Spark A). Both Sparks need the recipes at the same path; lab 02 cloned them to `~/w25/dgx-spark-playbooks` (run it once on Spark B too, or `git clone` there):

```bash
# on: spark
cd ~/w25/dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
chmod +x pytorch-ft-entrypoint.sh
docker stack deploy -c "$PWD/docker-compose.yml" finetuning-multinode
docker stack ps finetuning-multinode
```

**Expected output** (REFERENCE — the playbook's "healthy output"; the service really is spelled `finetunine` in `docker-compose.yml`)

```
ID             NAME                                IMAGE                              NODE         DESIRED STATE   CURRENT STATE
vlun7z9cacf9   finetuning-multinode_finetunine.1   nvcr.io/nvidia/pytorch:25.11-py3   <node-a>     Running         Running
tjl49zicvxoi   finetuning-multinode_finetunine.2   nvcr.io/nvidia/pytorch:25.11-py3   <node-b>     Running         Running
```

**Steps 7–9 — launch.** Lab 04 already put each Spark's Accelerate file in `assets/configs/` (LIVE mode). On **each** Spark, export your token in the shell and start Accelerate in that Spark's container:

```bash
# on: spark
export FINETUNING_CONTAINER=$(docker ps -q -f name=finetuning-multinode)
export HF_TOKEN=<your-huggingface-token>
docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -it "$FINETUNING_CONTAINER" bash -c '
  bash /workspace/install-requirements;
  accelerate launch --config_file=/workspace/configs/config_fsdp_lora.yaml /workspace/Llama3_70B_LoRA_finetuning.py'
```

The `run-multi-llama_3b`, `_8b` and `_70b` helpers in `assets/` are exactly this command for each script. Accelerate starts one process per machine, and each machine reads its own `machine_rank`, which is why the command runs on both Sparks (the playbook's Step 9 does not say so explicitly). Progress prints on Spark A only; watch Spark B with `nvidia-smi`.

**Step 10 — clean up** (Spark A): `docker stack rm finetuning-multinode`.

> ⚠ `pytorch-ft-entrypoint.sh` sets the container's root password to `root`, enables root SSH login on port 2233, and mounts your `~/.ssh` read-only, on the host network. Remove the stack when training ends, and do not leave it running on a shared network.

✓ Checkpoint: `docker stack ps finetuning-multinode` shows two Running tasks on two different nodes, and lab 04 prints both Accelerate files with ranks 0 and 1.

## 7 · When do two Sparks help for training?

FSDP pays for the halved memory with traffic. In `FULL_SHARD` mode each Spark gathers the other half of every layer's weights in the forward pass **and again** in the backward pass. Lab 01's last step counts it:

**Expected output** (arithmetic, captured on this Mac; link = Module 02's 21.875 GB/s pass mark)

```
▣ STEP 5 · two Sparks with FSDP: memory per Spark, and what crosses the cable per forward + backward pass
│ job                                     Accelerate config       state, 1 Spark  state per Spark  link per pass  at 21.875 GB/s
│ ──────────────────────────────────────  ──────────────────────  ──────────────  ───────────────  ─────────────  ──────────────
│ Llama 3.1 70B · LoRA (FSDP FULL_SHARD)  config_fsdp_lora.yaml    141.9 GB         71.0 GB         141.3 GB       6.46 s
│ Llama 3.1 8B · LoRA (FSDP FULL_SHARD)   config_fsdp_lora.yaml     16.2 GB          8.1 GB          16.1 GB       0.74 s
│ Llama 3.2 3B · full SFT (FSDP2)         config_finetuning.yaml    25.7 GB         12.9 GB           6.4 GB       0.29 s
```

| Situation | Use | Why |
|---|---|---|
| The job does not fit one Spark (70B LoRA in bf16, 8B full SFT with fp32 states) | **two Sparks, FSDP** | the only way to run it at that precision |
| It fits after a precision change (70B → QLoRA) | **one Spark** | ~40 GB of 4-bit weights, no link traffic; the price is QLoRA's small quality gap |
| It fits comfortably (8B LoRA, 3B full) | **one Spark**, or two for data parallelism | two only win if each pass computes for well over its link time (0.29–0.74 s here) |
| You want throughput for many small runs | **one job per Spark** | no link at all: twice the experiments |

Three rules come out of the arithmetic:

1. **The link cost is per pass, not per optimizer step.** Gradient accumulation does not reduce it. More tokens per pass (a bigger micro-batch) does.
2. **LoRA does not shrink FSDP's weight traffic.** The frozen weights still travel; only the tiny adapter gradients are cheap.
3. **Measure, then decide.** Lab 03's `steps/s` on one Spark versus the same job on two tells you whether the second Spark sped you up. The 6.46 s for 70B is a floor from bandwidth alone.

✓ Checkpoint: you can explain why gradient accumulation does not reduce FSDP's link traffic, and pick one Spark or two for each row of the table.

## Labs — run them here

**labs/lab01_finetune_memory.py** — Arithmetic: parameters, LoRA size, training state and activations for every recipe, placed on one or two Sparks, plus FSDP link traffic per pass.

**labs/lab02_launch_job.py** — Start a PyTorch or NeMo AutoModel recipe on Spark A in the background with nohup and a log file, after checking Docker, a free GPU and the Hugging Face login.

**labs/lab03_watch_log.py** — Tail and parse the training log on Spark A: loss curve, NaN and error checks, summary, memory and GPU use; its parser is self-tested on every machine.

**labs/lab04_two_spark_fsdp.py** — Preflight both Sparks for Docker Swarm + FSDP, write the two Accelerate configs from the playbook's files, and print the launch commands.

Lab 01 runs anywhere. Labs 02 and 03 run LIVE on Spark A or DRY. Lab 04 checks both Sparks and generates real config files in either mode. None of them runs `sudo` or changes Docker's configuration.

## Try it yourself

**Exercise 11 — the fine-tune budget.** Open `week25/11_pytorch_nemo_two_sparks/exercises/ex11_finetune_budget.py`. It has three `TODO`s:

1. `lora_params(layers, shapes, r)`: how many parameters LoRA adds.
2. `state_gb(frozen_b, trainable_b, frozen_bits)`: training state in GB.
3. `per_spark_gb(state, activations, n)`: memory per Spark when FSDP shards over *n* Sparks.

The checker is offline and free.

```bash
# on: laptop
.venv/bin/python week25/11_pytorch_nemo_two_sparks/exercises/ex11_finetune_budget.py
```

**Expected output** (once all three TODOs are done, captured on this Mac)

```
✓ lora_params: Llama 3.1 8B r=8 = 20,971,520 · Llama 3.1 70B r=8 = 103,546,880
✓ state_gb: 3B full = 25.7 GB · 8B LoRA = 16.2 GB · 70B QLoRA (linear layers in NF4) = 36.1 GB
✓ per_spark_gb: 70B LoRA bf16 → 154.4 GB on one Spark · 83.5 GB each on two

▣ your budget, applied (activations from lab 01's 'typ' column)
│ Llama 3.1 8B LoRA          state   16.2 GB · 1 Spark   38.1 GB · 2 Sparks   30.0 GB each → 1 Spark
│ Llama 3.1 70B LoRA, bf16   state  141.9 GB · 1 Spark  154.4 GB · 2 Sparks   83.5 GB each → 2 Sparks (FSDP)
│ Llama 3.1 8B full SFT      state   64.2 GB · 1 Spark   86.1 GB · 2 Sparks   54.0 GB each → 1 Spark
```

<details><summary>Hint — the LoRA count for one projection</summary>

A projection maps `d_in` to `d_out`. LoRA adds `A` with shape `r × d_in` and `B` with shape `d_out × r`, so `r × (d_in + d_out)` parameters. For the 8B model's `q_proj` (4096 → 4096) at r=8 that is 65,536. Sum the seven projections, multiply by the layers.

</details>

<details><summary>Stretch — rank 64 on two Sparks</summary>

Change `r` to 64 for the 70B job. How many trainable parameters now, how much state do they add, and does the plan change? Then change the activations to lab 01's "cap" value (17.2 GB) and check again.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot access gated repo for URL` | Accept the model licence on huggingface.co, then `hf auth login` on the Spark again |
| Gated-repo error in the NeMo container only | The image may use a different `HF_HOME`: check with `docker run --rm --entrypoint env nvcr.io/nvidia/nemo-automodel:26.02 \| grep HF_`, and mount the cache there |
| `docker: permission denied` | `sudo usermod -aG docker "$USER" && newgrp docker` |
| Container sees no GPU | Configure the NVIDIA Container Toolkit, restart Docker, test `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi` |
| `CUDA out of memory` | Lower `--batch_size`, add `--gradient_checkpointing` (3B full, 70B QLoRA), shorten `--seq_length`; lab 01 shows which term is too big |
| Out of memory although the job should fit | Unified memory: stop other jobs, then `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `unrecognized arguments: --use_torch_compile` | Only the 70B LoRA script has that flag (the benchmark guide passes it to others too); the 3B and 8B scripts always compile |
| QLoRA run ends with `AttributeError: … 'output_dir'` after `TRAINING COMPLETED` | In the copy we read, the QLoRA script checks `args.output_dir` without defining the flag. Training finished; nothing was saved |
| bitsandbytes CUDA error in QLoRA | NVIDIA's benchmark guide sets `export BNB_CUDA_VERSION=130` inside the container |
| Multi-node: errors or timeouts | `ACCELERATE_DEBUG_MODE=1 ACCELERATE_LOG_LEVEL=DEBUG TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL` |
| `task: non-zero exit (255)` | `docker ps -a --filter "name=finetuning-multinode"`, then `docker logs <container_id>` |
| `Cannot connect to the Docker daemon` after a swarm change | `sudo systemctl stop docker && sudo rm -rf /var/lib/docker/swarm && sudo systemctl start docker`, then init the swarm again with the interconnect IP |
| Stack tasks stay Pending | Check the GPU UUID in `daemon.json`, the `swarm-resource` line, and the interface names in `docker-compose.yml` (lab 04) |
| Can't delete `~/w25/nemo-checkpoints` | The container wrote it as root: `sudo rm -rf ~/w25/nemo-checkpoints` |

## Next

Continue to [Lab 12 — fine-tune a vision-language model and FLUX.1](../12_vlm_flux_finetune/TUTORIAL.md): the same fine-tuning ideas, applied to images.
