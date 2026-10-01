#!/usr/bin/env python3
"""Lab 09-4 · Watch the training run, read its loss, then chat with, predict with and merge the adapter.

LLaMA Factory writes two logs: the nohup console log (~/w25/logs/m09_hotel_<method>.log) and, next to
the adapter, trainer_log.jsonl — one JSON line per logging step (loss, lr, epoch, % done, time left,
eval_loss). This lab parses both. Step 1 tests the parser on a clearly labelled EXAMPLE log (synthetic,
made by a formula — not a run), so you can trust it before your own run exists. Step 2 reads your real
run over ssh. Step 3 diagnoses known failures. Step 4 validates the output folder the way the playbook
does. Step 5 runs the follow-ups you ask for:

  --chat      llamafactory-cli chat, fed three guest requests (foreground, ~2 min)
  --predict   llamafactory-cli train configs/hotel_predict.yaml → generated_predictions.jsonl (nohup)
  --export    llamafactory-cli export configs/hotel_merge.yaml → one merged model for vLLM (nohup)

Run: .venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py
     .venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --export
     .venv/bin/python week25/09_llama_factory/labs/lab04_monitor_and_export.py --log my_trainer_log.jsonl
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, bar, check, note, result, sh, step, table, warn  # noqa: E402

METHOD = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else "lora"
REMOTE = "~/w25/m09"
OUT = f"{REMOTE}/saves/qwen3-4b-hotel/{METHOD}/sft"
LOG = f"~/w25/logs/m09_hotel_{METHOD}.log"
ACT = f"cd {REMOTE} && source ~/factoryEnv/bin/activate"


# ── the parser (this is the part worth copying into your own tools) ───────────
def parse_trainer_log(text: str) -> list[dict]:
    """trainer_log.jsonl → list of dicts. Skips partial lines (the file grows while you read it)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def parse_console(text: str) -> dict:
    """The nohup console log → {'steps': [{'loss':…, 'epoch':…}], 'evals': […], 'metrics': {…}, 'progress': 'a/b'}.

    Handles the Hugging Face Trainer's dict lines ("{'loss': 1.23, 'grad_norm': …}"), the final
    '***** train metrics *****' block and the last tqdm progress counter.
    """
    steps, evals, metrics = [], [], {}
    for m in re.finditer(r"\{'(?:loss|eval_loss)'[^{}]*\}", text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        (evals if "eval_loss" in d else steps).append(d)
    block = re.search(r"\*\*\*\*\* train metrics \*\*\*\*\*(.*?)(?:\n\s*\n|\Z|Figure saved)", text, re.S)
    if block:
        for k, v in re.findall(r"^\s*(\w+)\s*=\s*(\S+)", block.group(1), re.M):
            metrics[k] = v
    prog = re.findall(r"(\d+)/(\d+) \[", text)
    return {"steps": steps, "evals": evals, "metrics": metrics,
            "progress": f"{prog[-1][0]}/{prog[-1][1]}" if prog else ""}


FAILURES = [  # (pattern in the console log, what it means, the fix)
    (r"CUDA out of memory|OutOfMemoryError", "out of memory",
     "lower per_device_train_batch_size (raise gradient_accumulation_steps to keep the effective batch), or flush the "
     "UMA page cache: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"),
    (r"Cannot access gated repo|401 Client Error", "gated model / Hugging Face auth",
     "run `hf auth login` on the Spark and request access on the model page"),
    (r"Cannot open data/dataset_info\.json", "dataset_info.json not found",
     "run the command from ~/w25/m09 (dataset_dir: data is relative) — re-run lab 03"),
    (r"Undefined dataset (\S+) in dataset_info\.json", "dataset name not registered",
     "the `dataset:` value must be a key in data/dataset_info.json — see Exercise 09"),
    (r"Template (\S+) does not exist", "unknown template", "use template: qwen3_nothink for Qwen3-4B-Instruct-2507"),
    (r"No module named '?bitsandbytes", "bitsandbytes missing (QLoRA)", "pip install bitsandbytes inside ~/factoryEnv"),
    (r"Traceback \(most recent call last\)", "the trainer crashed", "read the last 40 lines of the log for the cause"),
]


def diagnose(text: str) -> list[tuple[str, str]]:
    found = []
    for pat, what, fix in FAILURES:
        if re.search(pat, text):
            found.append((what, fix))
            if what != "the trainer crashed":
                break                                  # the specific cause beats the generic traceback line
    return found


def show_run(rows: list[dict], label: str) -> None:
    losses = [r for r in rows if r.get("loss") is not None]
    evals = [r for r in rows if r.get("eval_loss") is not None]
    if not losses:
        warn(f"{label}: no loss lines yet — the model may still be downloading or tokenising the dataset")
        return
    hi = max(r["loss"] for r in losses)
    pick = losses if len(losses) <= 8 else [losses[i] for i in sorted({round(i * (len(losses) - 1) / 7) for i in range(8)})]
    table([[f"{r.get('current_steps', '?')}/{r.get('total_steps', '?')}", f"{r.get('epoch', 0):.2f}",
            f"{r['loss']:.4f}", f"{r.get('lr', 0):.2e}", bar(r["loss"], hi, 24)] for r in pick],
          ["step", "epoch", "loss", "lr", f"loss (0 … {hi:.2f})"])
    last = rows[-1]
    first_l, last_l = losses[0]["loss"], losses[-1]["loss"]
    note(f"{label}: {last.get('percentage', '?')}% done · elapsed {last.get('elapsed_time', '?')} · "
         f"left {last.get('remaining_time', '?')} · loss {first_l:.3f} → {last_l:.3f} "
         f"({100 * (first_l - last_l) / first_l:.0f}% lower)")
    if evals:
        note("eval_loss: " + " → ".join(f"{r['eval_loss']:.3f}@{r.get('current_steps')}" for r in evals)
             + ("  ⚠ rising: over-fitting? fewer epochs" if len(evals) > 1 and evals[-1]["eval_loss"] > min(e["eval_loss"] for e in evals) * 1.1 else ""))
    if any("vram_allocated" in r for r in rows):
        note(f"peak GPU memory allocated: {max(r.get('vram_allocated', 0) for r in rows)} GB (RECORD_VRAM=1)")


# ── EXAMPLE logs: synthetic, generated by a formula to test the parser. NOT a training run. ──
def example_trainer_log() -> str:
    total, out = 189, []
    for s in range(5, total + 1, 5):
        loss = round(0.18 + 1.9 * math.exp(-s / 25), 4)             # a made-up smooth curve
        rec = {"current_steps": s, "total_steps": total, "loss": loss,
               "lr": round(1e-4 * 0.5 * (1 + math.cos(math.pi * s / total)), 8), "epoch": round(s / 63, 2),
               "percentage": round(100 * s / total, 2), "elapsed_time": f"0:{s // 30:02d}:{(2 * s) % 60:02d}",
               "remaining_time": f"0:{(total - s) // 30:02d}:00"}
        out.append(json.dumps(rec))
        if s % 50 == 0:
            out.append(json.dumps({"current_steps": s, "total_steps": total, "eval_loss": round(loss + 0.05, 4),
                                   "epoch": rec["epoch"], "percentage": rec["percentage"]}))
    return "\n".join(out) + '\n{"current_steps": 190, "total_st'        # a half-written last line, as in real life


EXAMPLE_CONSOLE = """[INFO|trainer.py] ***** Running training *****
{'loss': 1.6523, 'grad_norm': 2.1, 'learning_rate': 9.9e-05, 'epoch': 0.08}
{'loss': 0.4127, 'grad_norm': 0.9, 'learning_rate': 7.5e-05, 'epoch': 0.79}
{'eval_loss': 0.4402, 'eval_runtime': 3.1, 'epoch': 0.79}
 95%|█████████▌| 180/189 [06:01<00:18,  2.01s/it]
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     0.4321
  train_runtime            = 0:06:19.00

Figure saved at: saves/qwen3-4b-hotel/lora/sft/training_loss.png"""

banner("Lab 09-4 · monitor the run, then chat · predict · merge",
       f"method={METHOD} · log {LOG} · adapter {OUT}")

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "test the parser on an EXAMPLE log first")
local = sys.argv[sys.argv.index("--log") + 1] if "--log" in sys.argv else ""
if local:
    text = Path(local).read_text(encoding="utf-8", errors="replace")
    print(f"◆ parsing YOUR file {local}")
    show_run(parse_trainer_log(text), local)
else:
    print("◈ EXAMPLE — synthetic trainer_log.jsonl generated by a formula to test the parser. Not a training run.")
    rows = parse_trainer_log(example_trainer_log())
    good = check(len(rows) == 40 and sum("eval_loss" in r for r in rows) == 3,
                 f"parsed {len(rows)} JSON lines (37 loss + 3 eval), skipped the half-written last line",
                 f"parser returned {len(rows)} rows, expected 40")
    show_run(rows, "EXAMPLE")
    c = parse_console(EXAMPLE_CONSOLE)
    good &= check(len(c["steps"]) == 2 and len(c["evals"]) == 1 and c["metrics"].get("train_loss") == "0.4321"
                  and c["progress"] == "180/189",
                  "console parser: 2 loss dicts · 1 eval dict · train metrics block · tqdm 180/189",
                  f"console parser returned {c}")
    good &= check([w for w, _ in diagnose("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")] == ["out of memory"]
                  and [w for w, _ in diagnose("ValueError: Undefined dataset hotel-ops in dataset_info.json.")]
                  == ["dataset name not registered"],
                  "failure detector recognises OOM and an unregistered dataset name",
                  "failure detector missed a known failure")
    if not good:
        sys.exit(1)

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "your run on the Spark: is it alive, and what does the loss say?")
alive = sh("pgrep -af '[l]lamafactory-cli' || echo 'no llamafactory-cli process'",
           example="no llamafactory-cli process")
console = sh(f"tail -c 20000 {LOG} 2>/dev/null || echo 'no log yet: start training with lab 03 (SPARK_APPLY=1)'",
             quiet=True, example="◈ DRY: no log — the parser was tested on the EXAMPLE in step 1")
print("\n".join((console.out.strip().splitlines() or [""])[-6:]))
tl = sh(f"cat {OUT}/trainer_log.jsonl 2>/dev/null || true", quiet=True, example="")
if tl.live:
    run = parse_trainer_log(tl.out)
    show_run(run, "YOUR RUN") if run else note("no trainer_log.jsonl yet (it appears after the first logging step)")
    c = parse_console(console.out)
    if c["progress"]:
        note(f"tqdm progress in the console log: {c['progress']}")
else:
    print("◈ DRY — no run to read. Step 1 showed what the parser prints for one.")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "diagnose")
if console.live:
    problems = diagnose(console.out)
    finished = "***** train metrics *****" in console.out
    running = "no llamafactory-cli process" not in alive.out
    for what, fix in problems:
        check(False, "", f"{what} → {fix}")
    if finished:
        m = parse_console(console.out)["metrics"]
        result(f"training finished · train_loss {m.get('train_loss', '?')} · runtime {m.get('train_runtime', '?')}")
    elif running and not problems:
        result("training is running — re-run this lab to refresh (it is read-only).")
    elif not problems:
        warn("no process and no final metrics: not started yet, or it stopped. Read the whole log.")
else:
    print("◈ DRY — the detector was tested in step 1 (OOM, unregistered dataset).")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, "validate the output folder (the playbook's Step 9)")
note("The playbook's Step 9 lists what to expect rather than showing it — the REFERENCE below is that list.")
ls = sh(f"ls -la {OUT}/ 2>/dev/null | head -30",
        reference="- A final checkpoint directory (`checkpoint-411` or similar)\n"
                  "- Model configuration files such as `adapter_config.json`\n"
                  "- Training metrics showing decreasing loss values\n"
                  "- A training loss plot saved as a PNG file")
want = ["adapter_config.json", "adapter_model.safetensors", "trainer_log.jsonl", "training_loss.png"]
if METHOD == "full":
    want = ["config.json", "trainer_log.jsonl", "training_loss.png"]
if ls.live:
    rows = [[w, "✓" if w in ls.out else "✕"] for w in want]
    rows.append(["checkpoint-*", "✓ " + ", ".join(sorted(set(re.findall(r"checkpoint-\d+", ls.out))))
                 if "checkpoint-" in ls.out else "✕"])
    table(rows, ["expected file", "present"])
    ready = all(w in ls.out for w in want)
else:
    ready = False

# ── STEP 5 ────────────────────────────────────────────────────────────────────
step(5, "use the adapter: chat · predict · merge (each only when you ask)")
ASKS = ["The air conditioning in room 1412 is making a loud noise.",
        "ขอหมอนเพิ่มหนึ่งใบที่ห้อง 905 ค่ะ",
        "There is smoke coming from the kitchen near the lobby!"]
chat = (f"{ACT} && printf '%s\\nclear\\n%s\\nclear\\n%s\\nexit\\n' " + " ".join(f"'{a}'" for a in ASKS)
        + " | llamafactory-cli chat configs/hotel_chat.yaml 2>&1 | grep -E 'Assistant:' ")
predict = (f"{ACT} && {{ nohup llamafactory-cli train configs/hotel_predict.yaml > ~/w25/logs/m09_hotel_predict.log "
           "2>&1 < /dev/null & }; sleep 2; echo started")
export = (f"{ACT} && {{ nohup llamafactory-cli export configs/hotel_merge.yaml > ~/w25/logs/m09_hotel_merge.log "
          "2>&1 < /dev/null & }; sleep 2; echo started")
if METHOD != "lora":
    note("chat / predict / merge configs point at the LoRA adapter. For qlora/full, edit adapter_name_or_path "
         "(QLoRA) or model_name_or_path (full) first.")
for flag, cmd, blurb, ex in [
        ("--chat", chat, "three guest requests through llamafactory-cli chat (Step 10)",
         'Assistant: {"department": "engineering", "priority": "normal", "reply": "…"}'),
        ("--predict", predict, "answers for all 60 held-out requests → saves/qwen3-4b-hotel/lora/predict/", "started"),
        ("--export", export, "base + adapter merged → saves/qwen3-4b-hotel/merged (Step 11)", "started")]:
    if flag not in sys.argv:
        print(f"$ {cmd}   [not run — add {flag}]")
        continue
    if ls.live and not ready:
        warn(f"{flag}: the adapter is not complete yet (step 4). Wait for training to finish.")
        continue
    print(f"→ {blurb}")
    sh(cmd, example=ex, timeout=600)
sh(f"ls {REMOTE}/saves/qwen3-4b-hotel/merged 2>/dev/null | head -12 || echo 'not merged yet'", quiet="--export" not in sys.argv,
   example="config.json\ngeneration_config.json\nmodel-00001-of-00002.safetensors\nmodel-00002-of-00002.safetensors\n"
           "model.safetensors.index.json\ntokenizer.json\ntokenizer_config.json")
note("Watch predict/merge with: tail -f ~/w25/logs/m09_hotel_predict.log  ·  tail -f ~/w25/logs/m09_hotel_merge.log")
result(f"adapter: {OUT} · merged model: {REMOTE}/saves/qwen3-4b-hotel/merged · Module 13 serves and scores them.")
