#!/usr/bin/env python3
"""Lab 10-3 · Compare LLaMA Factory and Unsloth on the same task, fairly.

"Framework X is 2× faster" means nothing unless both runs trained the same model on the same data with
the same numbers. Step 1 checks that, from the real config files of Module 09 and lab 02. Step 2 prints
the measurement protocol. Step 3 reads both runs' results from the Spark (runtime, samples/s, peak
memory, final loss). Step 4 scores both runs' answers on the 60 held-out requests with one scorer; the
scorer is tested first on a clearly labelled EXAMPLE file (synthetic, not a model's output).

Read-only everywhere. Run it while the runs train, and again when both have finished.

Run: .venv/bin/python week25/10_unsloth/labs/lab03_compare_frameworks.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, result, sh, step, table  # noqa: E402

WEEK = Path(__file__).resolve().parents[2]
LF_YAML = WEEK / "09_llama_factory" / "configs" / "hotel_lora_sft.yaml"
US_JSON = WEEK / "10_unsloth" / "configs" / "hotel_unsloth_lora.json"
LF_LOG, US_LOG = "~/w25/logs/m09_hotel_lora.log", "~/w25/logs/m10_hotel_lora.log"
LF_TRAINER_LOG = "~/w25/m09/saves/qwen3-4b-hotel/lora/sft/trainer_log.jsonl"
LF_PRED = "~/w25/m09/saves/qwen3-4b-hotel/lora/predict/generated_predictions.jsonl"
US_PRED = "~/w25/m10/saves/qwen3-4b-hotel-unsloth/lora/generated_predictions.jsonl"
THAI = re.compile(r"[฀-๿]")


def fairness(lf: dict, us: dict) -> list[list[str]]:
    """(what, LLaMA Factory, Unsloth, verdict) — equal where it must be, reported where it cannot be."""
    same = lambda a, b: "✓ same" if a == b else "✕ differs"                   # noqa: E731
    return [
        ["base model", lf["model_name_or_path"], us["model"], same(lf["model_name_or_path"], us["model"])],
        ["training data", f"{lf['dataset']} (504)", "hotel_ops_chat.jsonl (504)", "✓ same text (lab 02 checked)"],
        ["base precision", "4-bit" if lf.get("quantization_bit") else "bf16", "4-bit" if us["load_in_4bit"] else "bf16",
         same(bool(lf.get("quantization_bit")), us["load_in_4bit"])],
        ["LoRA r / alpha / dropout", f"{lf['lora_rank']} / {lf.get('lora_alpha')} / {lf.get('lora_dropout')}",
         f"{us['r']} / {us['lora_alpha']} / {us['lora_dropout']}",
         same((lf["lora_rank"], lf.get("lora_alpha"), lf.get("lora_dropout")), (us["r"], us["lora_alpha"], us["lora_dropout"]))],
        ["LoRA targets", lf["lora_target"], f"{len(us['target_modules'])} linear layers",
         same(lf["lora_target"] == "all", len(us["target_modules"]) == 7)],
        ["effective batch", f"{lf['per_device_train_batch_size']} × {lf['gradient_accumulation_steps']}",
         f"{us['per_device_train_batch_size']} × {us['gradient_accumulation_steps']}",
         same(lf["per_device_train_batch_size"] * lf["gradient_accumulation_steps"],
              us["per_device_train_batch_size"] * us["gradient_accumulation_steps"])],
        ["learning rate · schedule · warmup", f"{lf['learning_rate']} · {lf['lr_scheduler_type']} · {lf['warmup_ratio']}",
         f"{us['learning_rate']} · {us['lr_scheduler_type']} · {us['warmup_ratio']}",
         same((lf["learning_rate"], lf["lr_scheduler_type"], lf["warmup_ratio"]),
              (us["learning_rate"], us["lr_scheduler_type"], us["warmup_ratio"]))],
        ["epochs · max length", f"{lf['num_train_epochs']} · {lf['cutoff_len']}", f"{us['num_train_epochs']} · {us['max_seq_length']}",
         same((lf["num_train_epochs"], lf["cutoff_len"]), (us["num_train_epochs"], us["max_seq_length"]))],
        ["loss on", "answer only (default)", "answer only (completion_only_loss)", "✓ same"],
        ["optimizer", "HF Trainer default (see log: optim=)", us["optim"], "◆ check the LLaMA Factory log line"],
        ["chat format", f"template: {lf['template']}", "the tokenizer's own chat template", "◆ compare one rendered prompt"],
        ["kernels, packing, gradient checkpointing", "LLaMA Factory defaults", "Unsloth kernels · 'unsloth' checkpointing",
         "◆ the difference you are measuring"],
    ]


def parse_pred(text: str) -> dict | None:
    """A model's answer → dict, tolerating ```json fences and text around one JSON object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def score(lines: list[dict]) -> dict:
    """Score generated_predictions.jsonl rows ({prompt, predict, label}) — one scorer for both frameworks."""
    n = len(lines)
    valid = dept = prio = lang = urgent_hit = urgent_all = 0
    for row in lines:
        want, got = json.loads(row["label"]), parse_pred(row["predict"])
        urgent_all += want["priority"] == "urgent"
        if not isinstance(got, dict):
            continue
        valid += 1
        dept += got.get("department") == want["department"]
        prio += got.get("priority") == want["priority"]
        urgent_hit += want["priority"] == "urgent" and got.get("priority") == "urgent"
        lang += bool(THAI.search(got.get("reply", ""))) == bool(THAI.search(want["reply"]))
    pct = lambda k, d=n: f"{100 * k / d:.0f} %" if d else "—"                 # noqa: E731
    return {"n": n, "valid JSON": pct(valid), "department": pct(dept), "priority": pct(prio),
            "urgent recall": pct(urgent_hit, urgent_all), "reply language": pct(lang)}


def seconds(hms: str) -> float:
    """'0:06:19.00' → 379.0 (LLaMA Factory prints train_runtime like this)."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


banner("Lab 10-3 · LLaMA Factory vs Unsloth — the same task, compared fairly",
       "configs checked locally · results read from the Spark · one scorer for both")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "is it a fair fight? (from the real config files)")
if not (LF_YAML.is_file() and US_JSON.is_file()):
    sys.exit("✕ run week25/09_llama_factory/labs/lab03 and week25/10_unsloth/labs/lab02 first — their configs are compared here.")
lf, us = yaml.safe_load(LF_YAML.read_text(encoding="utf-8")), json.loads(US_JSON.read_text(encoding="utf-8"))
rows = fairness(lf, us)
table(rows, ["what", "LLaMA Factory (Module 09)", "Unsloth (lab 02)", "verdict"])
broken = [r[0] for r in rows if r[3].startswith("✕")]
check(not broken, "every setting that must match does match", f"these differ, so any speed or accuracy gap is not "
      f"only the framework: {', '.join(broken)}")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "the measurement protocol")
for ln in ["1. Same Spark, one job at a time: lab 02 refuses to launch while another training job runs.",
           "2. Time the training loop, not the download: compare train_runtime, which starts after the model is loaded.",
           "3. Run each framework twice. Report both numbers; if they differ by more than a few %, find out why.",
           "4. Peak memory from the framework itself (torch max_memory_allocated): LLaMA Factory with RECORD_VRAM=1,",
           "   Unsloth in the W25_RESULT line. On unified memory `free -g` also counts the page cache.",
           "5. Quality on the SAME 60 held-out requests, scored by the SAME code (step 4). Loss values are not",
           "   comparable across frameworks: chat templates and padding can shift them without changing answers."]:
    print("│ " + ln)

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "the two runs' numbers, read from the Spark")
lf_log = sh(f"grep -E 'train_runtime|train_samples_per_second|train_loss|optim=' {LF_LOG} 2>/dev/null | tail -5 || true",
            example="  train_loss               =     …\n  train_runtime            = …")
lf_tl = sh(f"tail -n 3 {LF_TRAINER_LOG} 2>/dev/null || true", quiet=True, example="")
us_log = sh(f"grep W25_RESULT {US_LOG} 2>/dev/null | tail -1 || true",
            example='W25_RESULT {"framework": "unsloth", "train_runtime_s": …, "peak_mem_allocated_gb": …}')
if lf_log.live:
    lfm = dict(re.findall(r"(train_runtime|train_samples_per_second|train_loss)\s*=\s*(\S+)", lf_log.out))
    vram = [json.loads(x).get("vram_allocated") for x in lf_tl.out.splitlines() if x.startswith("{")]
    usm = json.loads(us_log.out.split("W25_RESULT", 1)[1]) if "W25_RESULT" in us_log.out else {}
    table([["train_runtime", f"{seconds(lfm['train_runtime']):.0f} s" if "train_runtime" in lfm else "—",
            f"{usm['train_runtime_s']:.0f} s" if usm else "—"],
           ["samples / s", lfm.get("train_samples_per_second", "—"), usm.get("train_samples_per_second", "—")],
           ["peak memory allocated", f"{max(v for v in vram if v)} GiB" if any(vram) else "— (set RECORD_VRAM=1)",
            f"{usm['peak_mem_allocated_gb']} GiB" if usm else "—"],
           ["final train loss (not comparable)", lfm.get("train_loss", "—"), usm.get("train_loss", "—")]],
          ["metric (LIVE, your Spark)", "LLaMA Factory", "Unsloth"])
else:
    print("◈ DRY — no runs to read. With a Spark this is a 4-row table of YOUR two runs; nothing is filled in for you.")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, "one scorer for both: department, priority, urgent recall, reply language")
evalset = json.loads((WEEK / "09_llama_factory" / "data" / "hotel_ops_eval.json").read_text(encoding="utf-8"))
print("◈ EXAMPLE — synthetic predictions written to test the scorer (5 held-out requests, answers edited by hand).")
picked = [r for r in evalset if json.loads(r["output"])["priority"] == "normal"][:4] + \
         [next(r for r in evalset if json.loads(r["output"])["priority"] == "urgent")]
ex = [{"prompt": r["instruction"], "label": r["output"], "predict": r["output"]} for r in picked]
ex[1]["predict"] = "```json\n" + ex[1]["predict"] + "\n```"                                     # fenced, still valid
d2 = json.loads(ex[2]["label"])["department"]
ex[2]["predict"] = ex[2]["predict"].replace(f'"{d2}"', '"concierge"' if d2 != "concierge" else '"front_desk"', 1)
ex[3]["predict"] = "Sure! I will send someone."                                                  # not JSON
ex[4]["predict"] = ex[4]["predict"].replace('"urgent"', '"normal"', 1)                          # a missed emergency
s = score(ex)
table([[k, v] for k, v in s.items()], ["EXAMPLE metric", "value"])
good = check(s == {"n": 5, "valid JSON": "80 %", "department": "60 %", "priority": "60 %", "urgent recall": "0 %",
                   "reply language": "80 %"},
             "scorer: tolerates ```json fences, fails non-JSON answers, catches a wrong department and a missed urgent",
             f"scorer returned {s}")
if not good:
    sys.exit(1)
preds = {}
for name, path in (("LLaMA Factory", LF_PRED), ("Unsloth", US_PRED)):
    r = sh(f"cat {path} 2>/dev/null || true", quiet=True, example="")
    if r.live and r.out.strip():
        preds[name] = score([json.loads(x) for x in r.out.splitlines() if x.strip()])
if preds:
    keys = list(next(iter(preds.values())))
    table([[k] + [preds.get(n, {}).get(k, "—") for n in ("LLaMA Factory", "Unsloth")] for k in keys],
          ["held-out metric (LIVE)", "LLaMA Factory", "Unsloth"])
    result("same model, same data, same numbers, same scorer: what differs now is the framework.")
else:
    note("no prediction files yet: Module 09 lab 04 --predict writes LLaMA Factory's; the Unsloth run writes its own at the end.")
    result("configs compared and scorer tested. Re-run when both runs have finished.")
