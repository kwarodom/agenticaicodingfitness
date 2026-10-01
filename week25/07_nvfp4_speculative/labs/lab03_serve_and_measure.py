#!/usr/bin/env python3
"""Lab 07-3 · Serve with TensorRT-LLM (NVFP4, EAGLE-3 or Draft-Target) and measure tokens per second.

One lab for the three servers this module starts on the Spark, all with the playbooks' own
`trtllm-serve` commands. Default mode is read-only: which server is running, which models it
serves, and a timed request (the speculative-decoding playbook's prompt) at temperature 0.
Every measurement is saved with its source, so you can compare "baseline" with "EAGLE-3"
on the same Spark — and a laptop stand-in number is never put next to a Spark number.

    lab03_serve_and_measure.py --start nvfp4          # your lab 02 checkpoint (nvfp4-quantization playbook)
    lab03_serve_and_measure.py --start eagle3         # gpt-oss-120b + EAGLE-3 head (speculative-decoding)
    lab03_serve_and_measure.py --start draft          # Llama 3.3 70B FP4 + 8B FP4 draft (speculative-decoding)
    lab03_serve_and_measure.py --start baseline       # gpt-oss-120b with the SAME settings, speculation off
    lab03_serve_and_measure.py --start bf16           # the BF16 original of lab 02's model, for --quality
    lab03_serve_and_measure.py --quality --label X    # six checkable questions → a score saved under label X
    lab03_serve_and_measure.py --stop                 # docker stop w25-trtllm
    lab03_serve_and_measure.py [--label NAME]         # status + one timed measurement

Run: .venv/bin/python week25/07_nvfp4_speculative/labs/lab03_serve_and_measure.py
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (LAPTOP_OLLAMA, banner, chat, http_json, mode, models, note, pick_laptop_model, result,  # noqa: E402
                      sh, step, table, up, url, warn, where)

NAME, LOG, PORT = "w25-trtllm", "~/w25/logs/w25-trtllm.log", 8355
RUNS = Path(__file__).resolve().parents[1] / ".runs"                   # gitignored
RUNS_FILE = RUNS / "tok_s.json"
DOCKER = (f"docker run --name {NAME} -e HF_TOKEN -v \"$HOME/.cache/huggingface/:/root/.cache/huggingface/\" "
          "--rm --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all --ipc=host --network host")

# Commands from the playbooks. Course changes (stated in the tutorial): --name w25-trtllm, no -it, nohup + log,
# and --port 8355 (the trt-llm playbook's port) so these never collide with vLLM on :8000 from Module 05.
SERVERS = {
    "nvfp4": ("nvfp4-quantization Step 8 (DGX Spark)", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B (your NVFP4 export)",
              f"""export MODEL_PATH="$HOME/w25/nvfp4/output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"
mkdir -p ~/w25/logs && {{ nohup {DOCKER} \\
  -v "$MODEL_PATH:/workspace/model" \\
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \\
  trtllm-serve /workspace/model \\
    --backend pytorch \\
    --max_batch_size 4 \\
    --port {PORT} > {LOG} 2>&1 < /dev/null & }}
echo "started {NAME} (pid $!) → {LOG}\""""),
    "eagle3": ("speculative-decoding Option 1 (EAGLE-3)", "openai/gpt-oss-120b + nvidia/gpt-oss-120b-Eagle3-long-context",
               f"""export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"
mkdir -p ~/w25/logs && {{ nohup {DOCKER} \\
  "$TRTLLM_IMAGE" \\
  bash -c '
    hf download openai/gpt-oss-120b && \\
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \\
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \\
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \\
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \\
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \\
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \\
      --backend pytorch --tp_size 1 \\
      --max_batch_size 1 \\
      --extra_llm_api_options /tmp/extra-llm-api-config.yml \\
      --port {PORT}' > {LOG} 2>&1 < /dev/null & }}
echo "started {NAME} (pid $!) → {LOG}\""""),
    "draft": ("speculative-decoding Option 2 (Draft-Target)", "nvidia/Llama-3.3-70B-Instruct-FP4 + nvidia/Llama-3.1-8B-Instruct-FP4",
              f"""export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"
mkdir -p ~/w25/logs && {{ nohup {DOCKER} \\
  "$TRTLLM_IMAGE" \\
  bash -c "
    hf download nvidia/Llama-3.3-70B-Instruct-FP4 && \\
    hf download nvidia/Llama-3.1-8B-Instruct-FP4 \\
    --local-dir /opt/Llama-3.1-8B-Instruct-FP4/ && \\
    cat <<EOF > extra-llm-api-config.yml
print_iter_log: false
disable_overlap_scheduler: true
speculative_config:
  decoding_type: DraftTarget
  max_draft_len: 4
  speculative_model_dir: /opt/Llama-3.1-8B-Instruct-FP4/
kv_cache_config:
  enable_block_reuse: false
EOF
    trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \\
      --backend pytorch --tp_size 1 \\
      --max_batch_size 1 \\
      --kv_cache_free_gpu_memory_fraction 0.9 \\
      --extra_llm_api_options ./extra-llm-api-config.yml \\
      --port {PORT}
  " > {LOG} 2>&1 < /dev/null & }}
echo "started {NAME} (pid $!) → {LOG}\""""),
}
# The BF16 original of lab 02's model, for the quality comparison (a course variant of the NVFP4 serve command:
# the Hugging Face id instead of the exported folder).
SERVERS["bf16"] = ("course variant of nvfp4-quantization Step 8, BF16 original", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B (BF16)",
                   f"""mkdir -p ~/w25/logs && {{ nohup {DOCKER} \\
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \\
  trtllm-serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --backend pytorch \\
    --max_batch_size 4 \\
    --port {PORT} > {LOG} 2>&1 < /dev/null & }}
echo "started {NAME} (pid $!) → {LOG}\"""")
# The fair baseline for EAGLE-3: the SAME command with the draft head removed (a course variant, not a playbook
# block). Same model, container, batch size and KV settings — only speculation is off.
_e3 = SERVERS["eagle3"][2]
SERVERS["baseline"] = ("course variant of Option 1 with speculative_config removed", "openai/gpt-oss-120b, no draft head",
                       _e3.replace("""    hf download nvidia/gpt-oss-120b-Eagle3-long-context \\
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \\
""", "").replace("""speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/
""", ""))
assert SERVERS["baseline"][2] != _e3 and "Eagle" not in SERVERS["baseline"][2]
# The speculative-decoding playbook's EAGLE-3 test prompt (a long, reasoning-style answer).
PROMPT = ("Solve the following problem step by step. If a train travels 180 km in 3 hours, and then slows down by "
          "20% for the next 2 hours, what is the total distance traveled? Show all intermediate calculations and "
          "provide a final numeric answer.")

# --quality: six questions with one checkable answer each. Same questions for every server → comparable scores.
QUESTIONS = [
    ("What is 17 × 23? Answer with the number only.", "391"),
    ("What is the capital of Australia? Answer in one word.", "canberra"),
    ("Which planet is known as the Red Planet? Answer in one word.", "mars"),
    ("What is the chemical symbol for gold? Answer with the symbol only.", "au"),
    ("A train travels 180 km in 3 hours. What is its speed in km/h? Answer with the number only.", "60"),
    ("Spell the word 'spark' backwards. Answer with the word only.", "kraps"),
]


def arg(flag: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv) else ""


def complete(base: str, model: str, max_tokens: int) -> dict:
    """POST {base}/completions (the playbook's endpoint), non-streaming, timed end to end."""
    body = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter()
    d = http_json("POST", base.rstrip("/") + "/completions", body, timeout=300)
    secs = time.perf_counter() - t0
    toks = int((d.get("usage") or {}).get("completion_tokens") or 0)
    text = ((d.get("choices") or [{}])[0].get("text") or "").strip()
    return {"secs": round(secs, 2), "tokens": toks, "tok_s": round(toks / secs, 1) if secs and toks else 0.0,
            "text": text}


banner("Lab 07-3 · serve and measure (TensorRT-LLM)", f"NVFP4 · EAGLE-3 · Draft-Target — one container ({NAME}) on :{PORT}")
start, label = arg("--start"), arg("--label")

if "--stop" in sys.argv:
    step(1, f"stop {NAME}")
    sh(f"docker stop {NAME} 2>/dev/null || echo 'not running'", timeout=120, example=NAME)
    result("stopped. The model cache in ~/.cache/huggingface stays, so the next start skips the download.")
    sys.exit(0)

if start:
    if start not in SERVERS:
        warn(f"--start takes one of: {', '.join(SERVERS)}")
        sys.exit(2)
    src, what, cmd = SERVERS[start]
    step(1, f"start {start}: {what}  (playbook: {src})")
    if where("a") == "dry":
        sh(cmd, example=f"started {NAME} (pid <pid>) → {LOG}")
        result("DRY: nothing started. Connect a Spark and run the same command.")
        sys.exit(0)
    sh(f"docker stop {NAME} >/dev/null 2>&1; true", timeout=120, quiet=True, echo=False)
    sh(cmd, timeout=60)
    note(f"Loading takes minutes (the first run downloads the weights). Watch: tail -f {LOG}  — "
         "then run this lab again without --start to measure.")
    sys.exit(0)

step(1, f"what is running on the Spark ({NAME})")
sh(f"docker ps --filter name={NAME} --format '{{{{.Names}}}}  {{{{.Image}}}}  {{{{.Status}}}}'; "
   f"tail -n 4 {LOG} 2>/dev/null || true", timeout=30,
   example=f"{NAME}  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12  Up 6 minutes\n…\nINFO:     Application startup complete.")

step(2, f"the endpoint — GET /v1/models on :{PORT}")
base = url("trtllm")
if mode() == "live" and up(base):
    ids = models(base)
    source, model = "spark", (ids[0] if ids else "default")
    print(f"│ {base} → {', '.join(ids) or '(no ids)'}")
else:
    print(f"│ {base or '(no Spark host)'} → ○ down")
    model = pick_laptop_model(["gemma3:4b"])
    source = "laptop" if model else "dry"
    if model:
        note(f"No TensorRT-LLM on the Spark, so the measurement below uses Ollama on THIS laptop ({model}). "
             "It shows how the measurement works; its speed says nothing about a Spark.")
        base = LAPTOP_OLLAMA

if "--quality" in sys.argv and source != "dry":
    step(3, "quality — six questions with one right answer each, temperature 0")
    q_tokens = 1024 if source == "spark" else 60          # R1-distill thinks before answering; the laptop stays light
    extra = None if source == "spark" else {"reasoning_effort": "none"}
    score, rows = 0, []
    for question, want in QUESTIONS:
        try:
            r = chat(base, model, [{"role": "user", "content": question}], max_tokens=q_tokens, temperature=0,
                     stream=False, timeout=300, extra=extra)
            answer = re.sub(r"(?s)<think>.*?</think>", "", r["text"]).strip()
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            answer = f"(error: {type(e).__name__})"
        hit = want in answer.lower()
        score += hit
        rows.append(["✓" if hit else "✕", question[:52], answer.replace("\n", " ")[:40]])
    table(rows, ["", "question", "answer (thinking removed)"])
    tag = "LIVE on the Spark" if source == "spark" else "LAPTOP STAND-IN (not the Spark's model)"
    print(f"◆ {tag} · {model} · score {score}/{len(QUESTIONS)}")
    RUNS.mkdir(exist_ok=True)
    qfile = RUNS / "quality.json"
    try:
        saved = json.loads(qfile.read_text())
    except (OSError, json.JSONDecodeError):
        saved = []
    saved.append({"source": source, "label": label or model, "model": model, "score": score, "of": len(QUESTIONS),
                  "date": time.strftime("%Y-%m-%d %H:%M")})
    qfile.write_text(json.dumps(saved[-30:], indent=1))
    table([[r["source"], r["label"], r["model"][:34], f"{r['score']}/{r['of']}", r["date"]] for r in saved[-30:]],
          ["where", "label", "model", "score", "when"])
    result("Six questions catch a broken quantization, not a 1 % accuracy change. For a real verdict run an "
           "evaluation suite on your own task (the playbook: 'Run evaluations for your use case').")
    sys.exit(0)

step(3, "one timed request — the speculative-decoding playbook's prompt, temperature 0")
if source == "dry":
    print("◈ DRY — no endpoint to time. On a Spark this prints tokens, seconds and tok/s.")
    result("DRY: start a server with --start nvfp4 | eagle3 | draft, then run this lab again.")
    sys.exit(0)
max_tokens = 300 if source == "spark" else 120                       # keep the shared laptop Ollama light
tag = "LIVE on the Spark" if source == "spark" else "LAPTOP STAND-IN (not Spark numbers)"
print(f"→ POST {base}/completions · model={model} · max_tokens={max_tokens} · {tag}")
try:
    m = complete(base, model, max_tokens)
except (HTTPError, URLError, TimeoutError, OSError) as e:
    warn(f"request failed: {type(e).__name__}: {e}")
    sys.exit(1)
print("· ANSWER  " + m["text"][:400].replace("\n", "\n          ") + (" …" if len(m["text"]) > 400 else ""))
print(f"◆ {tag} · {model} · {m['tokens']} tokens in {m['secs']} s · {m['tok_s']} tok/s (end to end, incl. prefill)")

step(4, "your measurements so far (grouped by where they ran — never compare across groups)")
RUNS.mkdir(exist_ok=True)
try:
    saved = json.loads(RUNS_FILE.read_text())
except (OSError, json.JSONDecodeError):
    saved = []
saved.append({"source": source, "label": label or ("laptop-stand-in" if source == "laptop" else model),
              "model": model, "date": time.strftime("%Y-%m-%d %H:%M"), **{k: m[k] for k in ("tokens", "secs", "tok_s")}})
RUNS_FILE.write_text(json.dumps(saved[-30:], indent=1))
rows = [[r["source"], r["label"], r["model"][:34], r["tokens"], r["secs"], r["tok_s"], r["date"]]
        for r in sorted(saved[-30:], key=lambda r: (r["source"], r["date"]))]
table(rows, ["where", "label", "model", "tokens", "seconds", "tok/s", "when"])
note(f"Saved to {RUNS_FILE.relative_to(RUNS.parents[1])}. On the Spark: --start baseline, measure with --label "
     "baseline; then --start eagle3, measure with --label eagle3. Same model, prompt and max_tokens.")
result("speed-up = tok/s with speculation ÷ tok/s without, on the SAME Spark and model. Lab 04 predicts it "
       "from the acceptance rate.")
