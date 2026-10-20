#!/usr/bin/env python3
"""STEP 1 · 💻 Build the dataset + training script (on your laptop)  [BEGINNER]

REAL and runs locally: turns domain (instruction, response) pairs into an
instruction-tuning dataset (OpenAI chat-messages JSONL, train/val splits) AND
writes the actual Unsloth training script — all into .sandbox/. STEP 2 pushes
these to the DGX over SSH; STEP 4 runs the script there for real.

Domain: smart-hotel HVAC operations.

Run:  python demos/step01_dataset_prep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

SYSTEM = ("You are HotelHVAC-Assistant, an expert in smart-hotel HVAC operations. "
          "Answer with specific setpoints, thresholds, and alarm priorities.")

PAIRS = [
    ("A guest room reads 26 °C with the fan-coil at max. What do I do?",
     "Drop chilled-water supply to 6.5 °C and stage the second compressor. If the "
     "room doesn't fall below 24 °C in 20 min, raise a CRITICAL alarm (guest-impacting)."),
    ("Filter differential pressure is 260 Pa. Alarm priority?",
     "ΔP > 250 Pa means a fouled filter: dispatch ROUTINE maintenance, not CRITICAL. "
     "Schedule replacement within 48h; log the run-hours."),
    ("What's the night setback for unoccupied rooms?",
     "Set unoccupied rooms to 25.5 °C cooling / 20 °C heating with the fan on AUTO; "
     "resume comfort band 30 min before check-in based on the PMS occupancy feed."),
    ("Chiller COP dropped from 5.1 to 3.8. Cause?",
     "A 25% COP drop usually means condenser fouling or low refrigerant charge. "
     "Check approach temperature first; dispatch ROUTINE inspection, trend for 24h."),
]

# The REAL Unsloth training script that runs on the DGX (inside the NGC container).
TRAIN_SCRIPT = '''\
#!/usr/bin/env python3
"""Unsloth LoRA fine-tune — runs ON THE DGX. Reads train.jsonl, writes hvac-model GGUF."""
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = os.environ.get("HF_MODEL", "unsloth/Llama-3.2-1B-Instruct")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60"))
print(f"[unsloth] loading {MODEL} in 4-bit (QLoRA)…", flush=True)
model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

ds = load_dataset("json", data_files={"train": "train.jsonl"})["train"]
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    args=SFTConfig(max_steps=MAX_STEPS, learning_rate=2e-4, warmup_steps=5,
                   per_device_train_batch_size=1, gradient_accumulation_steps=4,
                   logging_steps=1, output_dir="out", report_to=[]),
).train()

print("[unsloth] exporting GGUF (q4_k_m) → ./hvac-model/ …", flush=True)
model.save_pretrained_gguf("hvac-model", tok, quantization_method="q4_k_m")
print("[unsloth] DONE — adapter + GGUF written.", flush=True)
'''


def to_record(instr: str, resp: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instr},
        {"role": "assistant", "content": resp},
    ]}


def main() -> None:
    print("━" * 64)
    print("  STEP 1 · 💻 Build dataset + training script (laptop)")
    print("━" * 64, "\n")

    sb = config.ensure_sandbox()
    records = [to_record(i, r) for i, r in PAIRS]
    train, val = records[:3], records[3:]
    (sb / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (sb / "val.jsonl").write_text("\n".join(json.dumps(r) for r in val) + "\n")
    (sb / "train_hvac_unsloth.py").write_text(TRAIN_SCRIPT)

    print(f"Domain: {config.DOMAIN}")
    print("Wrote REAL files into .sandbox/ (on your laptop):")
    print(f"  • train.jsonl              ({len(train)} examples)")
    print(f"  • val.jsonl                ({len(val)} examples)")
    print("  • train_hvac_unsloth.py    (the Unsloth LoRA script that runs on the DGX)\n")
    print("One training record:\n")
    print(json.dumps(records[0], indent=2))

    print("\nData-prep rules that matter:")
    print("  • Consistent SYSTEM prompt → the behaviour you bake in.")
    print("  • Diverse REAL examples > near-duplicates. Hold out a val split.")
    print("  • 50–500 good examples already move a base model a lot with LoRA.")
    print("  • Your data stays local until YOU push it to YOUR DGX — that's the point.")

    print("\nTakeaway: dataset + script are ready on your laptop. STEP 2 connects to")
    print("your DGX over SSH and pushes them there.")


if __name__ == "__main__":
    main()
