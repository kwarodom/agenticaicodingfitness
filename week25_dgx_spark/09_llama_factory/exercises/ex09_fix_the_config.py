#!/usr/bin/env python3
"""Exercise 09 · Fix a broken dataset_info.json and training YAML before they waste a GPU run.

A teammate wrote the registry and the configs below for the hotel dataset. Each one would crash
`llamafactory-cli train` (or, worse, train fine and then be used with the wrong chat format).
Fix the four TODOs by editing the text, save, then run:
    .venv/bin/python week25/09_llama_factory/exercises/ex09_fix_the_config.py

The checker is free and offline: it parses the text with json / PyYAML and checks it against the
real files in week25/09_llama_factory/data/ (run lab 02 first if they are missing).
Stuck? Compare with exercises/solutions/.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

# ── TODO 1 ── the file name. Look at what lab 02 actually wrote into data/.
# ── TODO 2 ── the column mapping. Open one record of data/hotel_ops.json: which key holds the answer?
DATASET_INFO = """
{
  "hotel_ops": {
    "file_name": "hotel_ops.jsonl",
    "columns": {"prompt": "instruction", "query": "input", "response": "answer", "system": "system"}
  }
}
"""

# ── TODO 3 ── `dataset:` must be a NAME registered in DATASET_INFO (not a file name, not a guess).
# ── TODO 4 ── the chat format must match the base model — and match hotel_chat.yaml below.
TRAIN_YAML = """
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
dataset_dir: data
dataset: hotel-ops
template: llama3
cutoff_len: 1024
output_dir: saves/qwen3-4b-hotel/lora/sft
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
"""

CHAT_YAML = """
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-4b-hotel/lora/sft
template: qwen3_nothink
infer_backend: huggingface
trust_remote_code: true
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 09 · fix the config", "offline checker · json + PyYAML · no Spark needed", status=False)
    if not (DATA / "hotel_ops.json").is_file():
        print("✕ data/hotel_ops.json is missing — run labs/lab02_hotel_dataset.py first.")
        sys.exit(1)
    info = json.loads(DATASET_INFO)
    train, chat = yaml.safe_load(TRAIN_YAML), yaml.safe_load(CHAT_YAML)
    entry = info.get("hotel_ops", {})
    ok = True

    fname = entry.get("file_name", "")
    ok &= check((DATA / fname).is_file(), f"TODO 1: file_name '{fname}' exists in data/",
                f"TODO 1: file_name '{fname}' is not in data/ — ls week25/09_llama_factory/data")

    rows = json.loads((DATA / fname).read_text(encoding="utf-8")) if (DATA / fname).is_file() else \
        json.loads((DATA / "hotel_ops.json").read_text(encoding="utf-8"))
    cols = entry.get("columns", {})
    missing = sorted({v for v in cols.values() if v not in rows[0]})
    ok &= check(not missing and cols.get("response") == "output",
                "TODO 2: every mapped column exists in the records (response → output)",
                f"TODO 2: columns {missing} are not keys of the records; the records have {sorted(rows[0])}")

    ok &= check(train.get("dataset") in info,
                f"TODO 3: dataset '{train.get('dataset')}' is registered in dataset_info.json",
                f"TODO 3: dataset '{train.get('dataset')}' is not a key of dataset_info.json {sorted(info)} "
                "(LLaMA Factory would stop with: Undefined dataset … in dataset_info.json.)")

    ok &= check(train.get("template") == chat.get("template") == "qwen3_nothink",
                "TODO 4: template qwen3_nothink in both train and chat configs",
                f"TODO 4: train uses template '{train.get('template')}', chat uses '{chat.get('template')}' — "
                "Qwen3-4B-Instruct-2507 uses qwen3_nothink, and both must agree")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    eff = train["per_device_train_batch_size"] * train["gradient_accumulation_steps"]
    steps = -(-len(rows) // eff) * int(train["num_train_epochs"])
    print(f"\n▣ your config, applied: {len(rows)} records · effective batch {eff} · {steps} optimizer steps")
    print(f"│ chat loads the adapter from {chat['adapter_name_or_path']} "
          f"{'✓ = train output_dir' if chat['adapter_name_or_path'] == train['output_dir'] else '✕ ≠ train output_dir'}")
    print("\n═ Each of these four mistakes stops (or silently spoils) a first run. Lab 03 validates all four before it uploads.")


if __name__ == "__main__":
    main()
