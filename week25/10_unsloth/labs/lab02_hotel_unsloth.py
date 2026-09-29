#!/usr/bin/env python3
"""Lab 10-2 · Train Module 09's hotel dataset with Unsloth, using exactly Module 09's hyperparameters.

Step 1 reads Module 09's LLaMA Factory YAML and turns it into hotel_unsloth_config.json, key by key,
so the two runs cannot drift apart by accident. Step 2 converts hotel_ops.json into TRL's
conversational prompt–completion format (loss on the answer only, as in LLaMA Factory) and checks
the conversion loses nothing. Step 3 syntax-checks spark/train_hotel_unsloth.py (adapted from the
playbook's test_unsloth.py). Step 4 uploads everything to ~/w25/m10 and, only if you opt in, starts
the run inside the playbook container under nohup. Lab 03 watches it and compares the two frameworks.

Opt in with  --launch  or  SPARK_APPLY=1.  Add  --qlora  to load the base in 4 bits instead.

Run: .venv/bin/python week25/10_unsloth/labs/lab02_hotel_unsloth.py
     SPARK_APPLY=1 .venv/bin/python week25/10_unsloth/labs/lab02_hotel_unsloth.py
"""
import hashlib
import json
import os
import py_compile
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, put, result, sh, step, table, warn, where  # noqa: E402

WEEK = Path(__file__).resolve().parents[2]
MOD, M09 = WEEK / "10_unsloth", WEEK / "09_llama_factory"
APPLY = "--launch" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
QLORA = "--qlora" in sys.argv
KIND = "qlora" if QLORA else "lora"
REMOTE = "~/w25/m10"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]   # = lora_target: all

banner("Lab 10-2 · the hotel dataset on Unsloth — same data, same numbers as Module 09",
       f"{KIND} · launch={'yes (opted in)' if APPLY else 'no (add --launch or SPARK_APPLY=1)'}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "derive the Unsloth config from Module 09's YAML")
src = M09 / "configs" / "hotel_lora_sft.yaml"
if not src.is_file():
    sys.exit("✕ Module 09's configs/hotel_lora_sft.yaml is missing — run week25/09_llama_factory/labs/lab03 first.")
lf = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg = {
    "model": lf["model_name_or_path"],
    "max_seq_length": lf["cutoff_len"],
    "load_in_4bit": QLORA,
    "r": lf["lora_rank"],
    "lora_alpha": lf.get("lora_alpha", 2 * lf["lora_rank"]),
    "lora_dropout": lf.get("lora_dropout", 0.0),
    "target_modules": TARGETS,
    "per_device_train_batch_size": lf["per_device_train_batch_size"],
    "gradient_accumulation_steps": lf["gradient_accumulation_steps"],
    "num_train_epochs": lf["num_train_epochs"],
    "learning_rate": lf["learning_rate"],
    "lr_scheduler_type": lf["lr_scheduler_type"],
    "warmup_ratio": lf["warmup_ratio"],
    "logging_steps": lf["logging_steps"],
    "eval_steps": lf["eval_steps"],
    "optim": "adamw_torch",               # Hugging Face's AdamW; the playbook's test script uses adamw_8bit
    "seed": 42,                           # Hugging Face's default seed, which LLaMA Factory also uses
    "output_dir": f"outputs/hotel_{KIND}",
    "adapter_dir": f"saves/qwen3-4b-hotel-unsloth/{KIND}",
    "predictions": f"saves/qwen3-4b-hotel-unsloth/{KIND}/generated_predictions.jsonl",
}
rows = [
    ["model_name_or_path", lf["model_name_or_path"], "model_name", cfg["model"]],
    ["cutoff_len", lf["cutoff_len"], "max_seq_length", cfg["max_seq_length"]],
    ["quantization_bit", lf.get("quantization_bit", "— (bf16)"), "load_in_4bit", cfg["load_in_4bit"]],
    ["lora_rank / lora_alpha", f"{lf['lora_rank']} / {lf.get('lora_alpha')}", "r / lora_alpha", f"{cfg['r']} / {cfg['lora_alpha']}"],
    ["lora_dropout", lf.get("lora_dropout"), "lora_dropout", cfg["lora_dropout"]],
    ["lora_target", lf["lora_target"], "target_modules", "q k v o gate up down"],
    ["batch × accumulation", f"{lf['per_device_train_batch_size']} × {lf['gradient_accumulation_steps']}",
     "per_device … × gradient_accumulation_steps", f"{cfg['per_device_train_batch_size']} × {cfg['gradient_accumulation_steps']}"],
    ["learning_rate · schedule", f"{lf['learning_rate']} · {lf['lr_scheduler_type']}", "learning_rate · lr_scheduler_type",
     f"{cfg['learning_rate']} · {cfg['lr_scheduler_type']}"],
    ["num_train_epochs · warmup", f"{lf['num_train_epochs']} · {lf['warmup_ratio']}", "num_train_epochs · warmup_ratio",
     f"{cfg['num_train_epochs']} · {cfg['warmup_ratio']}"],
    ["(prompt masked by default)", "answer-only loss", "completion_only_loss=True", "answer-only loss"],
]
table(rows, ["LLaMA Factory key", "value", "Unsloth / TRL key", "value"])
(MOD / "configs").mkdir(exist_ok=True)
cfg_path = MOD / "configs" / f"hotel_unsloth_{KIND}.json"
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print(f"→ wrote {cfg_path.relative_to(WEEK)}")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "convert Alpaca records → TRL conversational prompt–completion (answer-only loss)")
(MOD / "data").mkdir(exist_ok=True)
good = True
for name in ("hotel_ops", "hotel_ops_eval"):
    records = json.loads((M09 / "data" / f"{name}.json").read_text(encoding="utf-8"))
    out = MOD / "data" / f"{name}_chat.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            user = r["instruction"] + (("\n" + r["input"]) if r["input"] else "")   # LLaMA Factory joins them the same way
            f.write(json.dumps({"prompt": [{"role": "system", "content": r["system"]},
                                           {"role": "user", "content": user}],
                                "completion": [{"role": "assistant", "content": r["output"]}]},
                               ensure_ascii=False) + "\n")
    back = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    same = all(b["prompt"][1]["content"] == r["instruction"] and b["completion"][0]["content"] == r["output"]
               and b["prompt"][0]["content"] == r["system"] for b, r in zip(back, records))
    digest = hashlib.sha256(json.dumps([(r["instruction"], r["output"]) for r in records]).encode()).hexdigest()[:12]
    good &= check(len(back) == len(records) and same,
                  f"{out.name}: {len(back)} records, identical text to Module 09's {name}.json (content sha256 {digest})",
                  f"{out.name}: conversion changed or lost records")
print(json.dumps(back[0], ensure_ascii=False)[:300] + " …")
if not good:
    sys.exit(1)

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "the training script (adapted from the playbook's test_unsloth.py)")
train = MOD / "spark" / "train_hotel_unsloth.py"
py_compile.compile(str(train), doraise=True)
check(True, f"{train.relative_to(WEEK)} compiles (syntax only — it imports unsloth, so it runs on the Spark, not here)", "")
for ln in train.read_text(encoding="utf-8").split('"""', 2)[2].splitlines():      # skip the docstring
    if any(k in ln for k in ("FastModel.from_pretrained(", "get_peft_model(", "SFTTrainer(", "completion_only_loss",
                             "W25_RESULT", "save_pretrained(cfg")):
        print("  " + ln.strip())

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, f"upload to {REMOTE} and launch inside the playbook container (opt-in)")
uploads = [(MOD / "spark" / "run_unsloth.sh", f"{REMOTE}/run_unsloth.sh"),
           (train, f"{REMOTE}/train_hotel_unsloth.py"),
           (cfg_path, f"{REMOTE}/hotel_unsloth_config.json"),
           (MOD / "data" / "hotel_ops_chat.jsonl", f"{REMOTE}/data/hotel_ops_chat.jsonl"),
           (MOD / "data" / "hotel_ops_eval_chat.jsonl", f"{REMOTE}/data/hotel_ops_eval_chat.jsonl")]
log = f"~/w25/logs/m10_hotel_{KIND}.log"
launch = (f"mkdir -p ~/w25/logs && {{ nohup bash {REMOTE}/run_unsloth.sh hotel-{KIND} train_hotel_unsloth.py > {log} "
          f"2>&1 < /dev/null & }}; sleep 3; tail -2 {log}")
if not APPLY:
    for local, remote in uploads:
        print(f"$ scp {local.name} <spark>:{remote}   [not run]")
    print(f"$ {launch}   [not run]")
    result(f"config and data ready. To train for real: SPARK_APPLY=1. Then lab 03 watches {log}.")
    sys.exit(0)
busy = sh("docker ps --filter name=w25-unsloth --format '{{.Names}}'; pgrep -af '[l]lamafactory-cli train' || true",
          example="")
if busy.live and busy.out.strip():
    warn(f"another training job is running ({busy.out.strip()}). One at a time, or the timings are not comparable.")
    sys.exit(0)
if not all(put(local, remote) for local, remote in uploads) and where() != "dry":
    warn("upload failed — check ssh/scp")
    sys.exit(1)
sh(launch, example="Collecting transformers\n…")
result(f"Unsloth run '{KIND}' started under nohup. Watch and compare: .venv/bin/python week25/10_unsloth/labs/lab03_compare_frameworks.py")
