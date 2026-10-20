#!/usr/bin/env python3
"""Lab 09-3 · LoRA vs QLoRA vs full by arithmetic, then write the training YAML and launch it on the Spark.

Step 1 counts, from each model's real config.json numbers, how many parameters LoRA trains and how
much memory each method needs on a 128 GB Spark. Step 2 does the step-count arithmetic (and checks
it against the playbook's own `checkpoint-411`). Step 3 writes six LLaMA Factory YAML files for the
hotel dataset from lab 02 — train (LoRA / QLoRA / full), chat, predict, merge — explains every key,
and validates them. Step 4 uploads data + configs to ~/w25/m09 on the Spark and, only if you opt in,
starts `llamafactory-cli train` under nohup with a log in ~/w25/logs/.

Opt in to the launch with  --launch  or  SPARK_APPLY=1.  Choose the method with --method lora|qlora|full
(default lora). Without the opt-in, the lab prints the exact command and stops.

Run: .venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py
     SPARK_APPLY=1 .venv/bin/python week25/09_llama_factory/labs/lab03_config_and_launch.py
"""
import json
import math
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import SPEC, banner, bar, check, note, put, result, sh, step, table, warn, where  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
DATA, CONFIGS = MOD / "data", MOD / "configs"
REMOTE = "~/w25/m09"                                  # everything for this module lives here on the Spark
APPLY = "--launch" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
METHOD = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else "lora"
if METHOD not in ("lora", "qlora", "full"):
    sys.exit(f"--method must be lora, qlora or full (got {METHOD!r})")

BASE = "Qwen/Qwen3-4B-Instruct-2507"                  # the playbook's example model (not gated)
TEMPLATE = "qwen3_nothink"                            # the playbook's example template for that model
RANK, CUTOFF, EPOCHS, BATCH, ACCUM = 16, 1024, 3, 2, 4
OUT = "saves/qwen3-4b-hotel"

# ── model shapes, copied from each model's config.json on Hugging Face ────────
# name, hidden, layers, q heads × head_dim, kv heads × head_dim, intermediate, vocab, tied embeddings
MODELS = [
    ("Qwen3-4B-Instruct-2507", 2560, 36, 32 * 128, 8 * 128, 9728, 151936, True),
    ("Qwen3-8B", 4096, 36, 32 * 128, 8 * 128, 12288, 151936, False),
    ("Qwen3-32B", 5120, 64, 64 * 128, 8 * 128, 25600, 151936, False),
    ("Llama-3.3-70B", 8192, 80, 64 * 128, 8 * 128, 28672, 128256, False),
]


def shapes(h, q, kv, inter):
    """The 7 linear layers per transformer block that `lora_target: all` adapts: (in, out)."""
    return {"q_proj": (h, q), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (q, h),
            "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}


def count(m):
    """(total params, params inside the 7 linear layers, LoRA params per unit of rank)."""
    _, h, layers, q, kv, inter, vocab, tied = m
    lin = sum(i * o for i, o in shapes(h, q, kv, inter).values()) * layers
    other = vocab * h * (1 if tied else 2) + layers * (2 * h + 2 * 128) + h   # embeddings, norms (+ Qwen3 q/k norm)
    lora_per_r = sum(i + o for i, o in shapes(h, q, kv, inter).values()) * layers  # A: r×in, B: out×r
    return lin + other, lin, lora_per_r


# Bytes per parameter (rule of thumb for AdamW mixed-precision training; activations NOT included):
#   full  : bf16 weight 2 + bf16 grad 2 + fp32 master 4 + fp32 Adam m 4 + fp32 Adam v 4 = 16
#   LoRA  : frozen base in bf16 = 2; only the adapter pays the 16
#   QLoRA : frozen linear layers in 4-bit NF4 ≈ 0.5625 (4.5 bits incl. scales); embeddings stay bf16
OVERHEAD_GB = 10                                        # CUDA context, framework, a little activation memory


def mem_gb(m, method, r=RANK):
    total, lin, per_r = count(m)
    adapter = per_r * r
    if method == "full":
        return (total * 16) / 1e9 + OVERHEAD_GB
    if method == "lora":
        return (total * 2 + adapter * 16) / 1e9 + OVERHEAD_GB
    return (lin * 0.5625 + (total - lin) * 2 + adapter * 16) / 1e9 + OVERHEAD_GB


banner("Lab 09-3 · pick a method by arithmetic, write the YAML, launch on the Spark",
       f"method={METHOD} · launch={'yes (opted in)' if APPLY else 'no (add --launch or SPARK_APPLY=1)'}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "how much does each method train, and does it fit in 128 GB?")
total4, lin4, per_r4 = count(MODELS[0])
rows = []
for r in (8, 16, 32, 64):
    p = per_r4 * r
    rows.append([f"r = {r}", f"{p / 1e6:6.1f} M", f"{100 * p / total4:5.2f} %", f"{p * 2 / 1e6:6.1f} MB",
                 bar(p, per_r4 * 64, 20)])
table(rows, ["LoRA rank", "trainable params", "of Qwen3-4B", "adapter file (bf16)", ""])
note(f"Qwen3-4B has {total4 / 1e9:.2f} B parameters. LoRA adds two thin matrices A (r×in) and B (out×r) "
     "to each of the 7 linear layers in all 36 blocks and trains only those.")

print()
rows = []
for m in MODELS:
    cells = [m[0], f"{count(m)[0] / 1e9:5.1f} B"]
    for meth in ("full", "lora", "qlora"):
        need = mem_gb(m, meth)
        cells.append(f"{need:6.0f} GB  {'✓ fits' if need <= SPEC['memory_gb'] else '✕ > 128'}")
    rows.append(cells)
table(rows, ["model", "params", "full (16 B/param)", f"LoRA r={RANK} (bf16 base)", f"QLoRA r={RANK} (4-bit base)"])
note("Rule of thumb, not a measurement: AdamW mixed precision, + 10 GB overhead, activations not counted "
     "(they grow with batch × cutoff_len; gradient checkpointing keeps them small).")
note("Full fine-tuning pays 16 bytes for EVERY parameter; LoRA pays 2 for the frozen base and 16 only for "
     "the ~1 % adapter; QLoRA stores the frozen base in 4 bits. That is why a 70B model trains on one Spark only as QLoRA.")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "how many optimizer steps? (and a sanity check against the playbook)")
ref_n, ref_bs = 91 + 999, 1 * 8                        # identity.json + alpaca_en_demo.json, batch 1 × accum 8
ref_steps = math.ceil(ref_n / ref_bs) * 3
print(f"│ playbook example: {ref_n} records ÷ ({1} × {8}) per step = {math.ceil(ref_n / ref_bs)} steps/epoch × 3 epochs "
      f"= {ref_steps} steps  → the playbook shows 'checkpoint-411'  {'✓' if ref_steps == 411 else '✕'}")
print(f"│ and {ref_n} × 3 records ÷ 872.12 s (train_runtime 0:14:32.12) = {ref_n * 3 / 872.12:.3f} samples/s "
      "→ the playbook shows 3.749  ✓")
_train = json.loads((DATA / "hotel_ops.json").read_text(encoding="utf-8"))
n_train = len(_train)
SYSTEM_Q = "'" + _train[0]["system"].replace("'", "''") + "'"   # YAML single-quoted, as in hotel_chat.yaml
per_epoch = math.ceil(n_train / (BATCH * ACCUM))
steps = per_epoch * EPOCHS
print(f"│ your hotel run:   {n_train} records ÷ ({BATCH} × {ACCUM}) per step = {per_epoch} steps/epoch × {EPOCHS} epochs "
      f"= {steps} steps")
est = n_train * EPOCHS / 3.749
note(f"Rough time estimate: {n_train * EPOCHS} samples ÷ 3.749 samples/s (the playbook's REFERENCE throughput) ≈ "
     f"{est / 60:.0f} min. Your records are shorter than Alpaca's, so expect it to be quicker — lab 04 shows the real time.")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "write six YAML files, one comment per key")
MODEL_BLOCK = f"""### model
model_name_or_path: {BASE}   # base model from Hugging Face (not gated: no hf login needed)
trust_remote_code: true
"""
QUANT = """quantization_bit: 4             # QLoRA: load the frozen base in 4 bits (bitsandbytes)
quantization_method: bnb
"""


def train_yaml(method: str) -> str:
    lr = "1.0e-5" if method == "full" else "1.0e-4"
    ft = "full" if method == "full" else "lora"
    lora = "" if method == "full" else f"""lora_rank: {RANK}                    # r: width of the adapter. 8 in the playbook example; 16 for a 6-way router
lora_alpha: {2 * RANK}                   # scale = alpha / r. LLaMA Factory's default is 2 × rank
lora_dropout: 0.05
lora_target: all                 # adapt all 7 linear layers in every block (q k v o gate up down)
"""
    if method == "full":
        lora = ("# upstream examples/train_full/qwen3_full_sft.yaml also sets deepspeed: examples/deepspeed/ds_z3_config.json\n"
                "# (ZeRO-3 shards across GPUs). One Spark has one GPU and step 1 says 4B full fits, so this course drops it.\n")
    return f"""# LLaMA Factory · Week 25 Module 09 · hotel guest-request router · method: {method}
# Written by labs/lab03_config_and_launch.py — adapted from the playbook's examples/train_lora/qwen3_lora_sft.yaml.
# Run from ~/w25/m09 on the Spark:  llamafactory-cli train configs/hotel_{method}_sft.yaml
{MODEL_BLOCK}{QUANT if method == "qlora" else ""}
### method
stage: sft                       # supervised fine-tuning on (prompt, response) pairs
do_train: true
finetuning_type: {ft:<14} # {"train every weight (no adapter)" if ft == "full" else "train a small adapter, freeze the base"}
{lora}
### dataset
dataset_dir: data                # folder holding dataset_info.json (relative to where you run the command)
dataset: hotel_ops               # a NAME registered in data/dataset_info.json, not a file name
eval_dataset: hotel_ops_eval     # held-out requests: gives an eval_loss curve next to the train loss
template: {TEMPLATE}           # chat format of the base model. Must match in chat, predict and merge
cutoff_len: {CUTOFF}                 # max tokens per record; lab 02 estimates the longest at ~180
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {OUT}/{"full" if method == "full" else method}/sft   # adapter (or full model) + trainer_log.jsonl + training_loss.png
logging_steps: 5                 # one loss line every 5 optimizer steps → ~{steps // 5} points for lab 04
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none                  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: {BATCH}
gradient_accumulation_steps: {ACCUM}   # effective batch = {BATCH} × {ACCUM} = {BATCH * ACCUM}, the same as the playbook's 1 × 8
learning_rate: {lr}           # write 1.0e-4, not 1e-4: plain YAML 1.1 parsers read 1e-4 as a string
num_train_epochs: {EPOCHS}.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true                       # bfloat16 mixed precision, as in the playbook example
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
"""


FILES = {
    "hotel_lora_sft.yaml": train_yaml("lora"),
    "hotel_qlora_sft.yaml": train_yaml("qlora"),
    "hotel_full_sft.yaml": train_yaml("full"),
    "hotel_chat.yaml": f"""# llamafactory-cli chat configs/hotel_chat.yaml   (adapted from examples/inference/qwen3_lora_sft.yaml)
model_name_or_path: {BASE}
adapter_name_or_path: {OUT}/lora/sft   # the LoRA adapter the train config writes
template: {TEMPLATE}
infer_backend: huggingface       # choices: [huggingface, vllm, sglang, ktransformers]
trust_remote_code: true
# the exact system prompt every training record carries — without it the adapter answers in prose, not JSON
default_system: {SYSTEM_Q}
""",
    "hotel_predict.yaml": f"""# llamafactory-cli train configs/hotel_predict.yaml   (adapted from examples/extras/nlg_eval/llama3_lora_predict.yaml)
# Generates an answer for every held-out request → {OUT}/lora/predict/generated_predictions.jsonl (Module 13 scores it)
### model
model_name_or_path: {BASE}
adapter_name_or_path: {OUT}/lora/sft
trust_remote_code: true

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
dataset_dir: data
eval_dataset: hotel_ops_eval
template: {TEMPLATE}
cutoff_len: {CUTOFF}
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {OUT}/lora/predict
overwrite_output_dir: true
report_to: none

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
do_sample: false                 # greedy: LLaMA Factory samples by default (temperature 0.95), which makes scores noisy
max_new_tokens: 160              # an answer is ~60 tokens; the default is 1024
ddp_timeout: 180000000
""",
    "hotel_merge.yaml": f"""# llamafactory-cli export configs/hotel_merge.yaml   (adapted from examples/merge_lora/qwen3_lora_sft.yaml)
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: {BASE}
adapter_name_or_path: {OUT}/lora/sft
template: {TEMPLATE}
trust_remote_code: true

### export
export_dir: {OUT}/merged         # base + adapter folded into one plain Hugging Face model (vLLM serves it in Module 13)
export_size: 5                   # shard size in GB
export_device: cpu               # choices: [cpu, auto]
export_legacy_format: false      # write .safetensors
""",
}
CONFIGS.mkdir(parents=True, exist_ok=True)
for name, text in FILES.items():
    (CONFIGS / name).write_text(text, encoding="utf-8")
    print(f"→ wrote configs/{name:22s} ({len(text.splitlines())} lines)")
print()
print("\n".join("  " + ln for ln in FILES[f"hotel_{METHOD}_sft.yaml"].splitlines()[3:20]))
print("  …")

# ── validate what we wrote, by reading it back ────────────────────────────────
# Keys: those used in the upstream LLaMA-Factory examples this lab adapts, plus lora_alpha / lora_dropout /
# dataset_dir / do_sample / max_new_tokens (checked against src/llamafactory/hparams/*.py, commit ce9dc9e, 2026-09-28).
KNOWN = set("""default_system model_name_or_path trust_remote_code quantization_bit quantization_method stage do_train do_predict
finetuning_type lora_rank lora_alpha lora_dropout lora_target deepspeed dataset_dir dataset eval_dataset template
cutoff_len max_samples overwrite_cache preprocessing_num_workers dataloader_num_workers output_dir logging_steps
save_steps plot_loss overwrite_output_dir save_only_model report_to per_device_train_batch_size
gradient_accumulation_steps learning_rate num_train_epochs lr_scheduler_type warmup_ratio bf16 ddp_timeout
resume_from_checkpoint per_device_eval_batch_size eval_strategy eval_steps val_size predict_with_generate
adapter_name_or_path infer_backend export_dir export_size export_device export_legacy_format do_sample max_new_tokens""".split())
registry = json.loads((DATA / "dataset_info.json").read_text(encoding="utf-8"))
cfg = {n: yaml.safe_load((CONFIGS / n).read_text(encoding="utf-8")) for n in FILES}
good = True
unknown = sorted({k for c in cfg.values() for k in c if k not in KNOWN})
good &= check(not unknown, f"all {len(cfg)} files parse as YAML; every key is a known LLaMA Factory argument",
              f"unknown keys (typo?): {unknown}")
names = {c.get(k) for c in cfg.values() for k in ("dataset", "eval_dataset") if c.get(k)}
good &= check(names <= set(registry), f"datasets {sorted(names)} are registered in data/dataset_info.json",
              f"not registered: {sorted(names - set(registry))}")
good &= check(len({c["template"] for c in cfg.values()}) == 1 and len({c["model_name_or_path"] for c in cfg.values()}) == 1,
              f"one base model and one template ({TEMPLATE}) across train, chat, predict and merge",
              "template or base model differs between files — the adapter would be used with the wrong chat format")
good &= check(all(isinstance(c.get("learning_rate", 0.0), float) for c in cfg.values()),
              "learning_rate is a number (1.0e-4), not the string '1e-4'", "learning_rate parsed as a string")
lora_out = cfg["hotel_lora_sft.yaml"]["output_dir"]
good &= check(all(cfg[n]["adapter_name_or_path"] == lora_out for n in ("hotel_chat.yaml", "hotel_predict.yaml", "hotel_merge.yaml")),
              f"chat, predict and merge all load the adapter from {lora_out}",
              "adapter_name_or_path does not match the LoRA output_dir")
good &= check("quantization_bit" not in cfg["hotel_merge.yaml"], "merge config has no quantization_bit (upstream warns against it)",
              "remove quantization_bit from the merge config")
if not good:
    result("✕ the configs are inconsistent — fix lab03 before uploading.")
    sys.exit(1)

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, f"upload to {REMOTE} on the Spark and launch '{METHOD}' (opt-in)")
uploads = [(DATA / f, f"{REMOTE}/data/{f}") for f in ("hotel_ops.json", "hotel_ops_eval.json", "dataset_info.json")]
uploads += [(CONFIGS / f, f"{REMOTE}/configs/{f}") for f in FILES]
sent = [put(local, remote) for local, remote in uploads]
if where() != "dry" and not all(sent):
    warn("some uploads failed — fix ssh/scp before launching")
    sys.exit(1)

r = sh("test -x ~/factoryEnv/bin/llamafactory-cli && ~/factoryEnv/bin/llamafactory-cli version | head -4 "
       "|| echo 'MISSING: run lab 01 first'",
       example="----------------------------------------------------------\n"
               "| Welcome to LLaMA Factory, version 0.9.x                |\n"
               "----------------------------------------------------------")
if r.live and "MISSING" in r.out:
    warn("LLaMA Factory is not installed in ~/factoryEnv. Run lab 01 with SPARK_APPLY=1 first.")
    sys.exit(0)
busy = sh("pgrep -af 'llamafactory-cli (train|export)' || echo 'no training running'",
          example="no training running")
if busy.live and "no training running" not in busy.out:
    warn("a LLaMA Factory job is already running on this Spark — let it finish (lab 04 watches it) before starting another.")
    sys.exit(0)

log = f"~/w25/logs/m09_hotel_{METHOD}.log"
launch = (f"mkdir -p ~/w25/logs && cd {REMOTE} && source ~/factoryEnv/bin/activate && "
          f"{{ RECORD_VRAM=1 nohup llamafactory-cli train configs/hotel_{METHOD}_sft.yaml > {log} 2>&1 < /dev/null & "
          f"echo $! > ~/w25/logs/m09_hotel_{METHOD}.pid; }}; sleep 2; echo \"started pid $(cat ~/w25/logs/m09_hotel_{METHOD}.pid)\"")
if METHOD == "qlora":
    note("QLoRA needs bitsandbytes, which the playbook's `pip install -e \".[metrics]\"` does not install. If the log "
         "says it is missing: source ~/factoryEnv/bin/activate && pip install bitsandbytes (not verified on GB10 here).")
if not APPLY:
    print(f"$ {launch}   [not run]")
    result(f"configs written and checked. To start training for real: SPARK_APPLY=1 (or --launch). "
           f"Then run lab 04 to watch {log}.")
    sys.exit(0)
sh(launch, example="started pid 48213")
result(f"training '{METHOD}' is running under nohup. It survives this lab and your laptop sleeping. "
       f"Watch it: .venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py")
