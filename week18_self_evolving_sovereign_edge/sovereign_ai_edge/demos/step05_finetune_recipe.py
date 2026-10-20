#!/usr/bin/env python3
"""PART 6 · Fine-tuning on the edge (NeMo AutoModel LoRA)  [ADVANCED]

Fine-tuning adapts a base model to YOUR exact domain — your HVAC fault codes,
your internal docs, your local dialect — without ever sending that data to a
cloud trainer. On a DGX Spark, a 31B LoRA run is 45–90 minutes.

You can't run the 31B training job on a Mac, so this demo does two real things
instead:
  1. GENERATES the exact NeMo AutoModel artifacts (finetune YAML + SFT dataset)
     into .sandbox/ — the real recipe, ready to copy to a Spark.
  2. DEMONSTRATES the *effect* of domain adaptation cheaply and locally: it asks
     the base model an HVAC question with and without a domain system prompt, so
     you can see what fine-tuning bakes permanently into the weights.

Run:  python demos/step05_finetune_recipe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import edgeview  # noqa: E402

FINETUNE_YAML = """\
# finetune_gemma4.yaml — NeMo AutoModel LoRA recipe for an HVAC domain expert
model:
  pretrained_model_name_or_path: google/gemma-4-31b-it
  precision: bf16            # use fp8 on Blackwell for 50% less memory

training:
  method: lora               # parameter-efficient: fine-tune 31B in 128 GB
  lora_rank: 16              # 16–64; higher = more capacity, more memory
  lora_alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj]

data:
  train_file: dataset.jsonl
  max_seq_length: 4096

optimizer:
  lr: 2.0e-4                 # too high = catastrophic forgetting
  epochs: 3                  # 2–5; stop early if eval loss climbs

output:
  checkpoint_dir: ./checkpoints/gemma4-hvac-lora
"""

# SFT format: JSONL with "input"/"output" fields (the PDF's recipe).
SFT_ROWS = [
    {"input": "What is the chiller fault CH-F03?",
     "output": "CH-F03 is condenser approach temperature high — usually fouled "
               "condenser tubes or low condenser water flow. Inspect the cooling "
               "tower and condenser water pump first."},
    {"input": "Recommend action for AHU-F02",
     "output": "AHU-F02 is filter differential pressure high. Schedule a filter "
               "replacement; until then, expect reduced airflow and rising space "
               "temperature."},
    {"input": "FCU in room 412 won't cool. What do I check?",
     "output": "Check the chilled-water valve actuator, verify the room thermostat "
               "setpoint, and confirm the FCU fan is energized. Most 'no cool' "
               "FCU calls are a stuck valve or a tripped fan."},
]


def main() -> None:
    edgeview.banner("PART 6", "Fine-tuning on the edge (NeMo AutoModel LoRA)", "ADVANCED")

    sb = config.ensure_sandbox()
    (sb / "finetune_gemma4.yaml").write_text(FINETUNE_YAML)
    import json
    (sb / "dataset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in SFT_ROWS) + "\n")

    print("Generated the real NeMo AutoModel artifacts into .sandbox/:")
    print(f"  • {sb / 'finetune_gemma4.yaml'}   (LoRA recipe)")
    print(f"  • {sb / 'dataset.jsonl'}          ({len(SFT_ROWS)} SFT examples)\n")
    print("On a DGX Spark you'd then run (no cloud, fully sovereign training):")
    print("  python -m nemo.collections.llm.train --config finetune_gemma4.yaml")
    print("  # ~8 min load + 3 epochs ≈ 68 min total for a 31B LoRA\n")
    print("Estimated wall-clock for this dataset on a DGX Spark GB10:")
    print("  load ~8m · 3 epochs ~55m · save ~5m  →  ~68 min, one-time, on-prem.\n")

    if not edgeview.require_local():
        return

    print("Now SEE what domain adaptation does — same local model, same question,")
    print("with vs without the HVAC-expert framing that LoRA would bake into weights:\n")
    edgeview.sovereignty_line()

    q = "A hotel reports fault CH-F03 on chiller 2. What does it mean and what do I do?"

    print(">>> BASE model (no domain knowledge):")
    edgeview.generate(q, max_tokens=650, show_reasoning=False,
                      title="generic assistant")
    print()

    print(">>> DOMAIN-ADAPTED (system prompt standing in for LoRA weights):")
    edgeview.generate(
        [
            {"role": "system", "content":
             "You are an expert HVAC fault-diagnosis assistant for hotels. "
             "Fault CH-F03 = condenser approach temperature high. Be specific and "
             "give a concrete first action."},
            {"role": "user", "content": q},
        ],
        max_tokens=650, show_reasoning=False, title="HVAC domain expert",
    )

    print("\nTakeaway: a system prompt steers behaviour per-call; LoRA makes that")
    print("expertise PERMANENT in the weights — smaller prompts, faster, consistent,")
    print("and trained on data that never left your DGX Spark.")


if __name__ == "__main__":
    main()
