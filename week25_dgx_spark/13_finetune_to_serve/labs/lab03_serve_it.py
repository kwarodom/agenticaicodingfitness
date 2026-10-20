#!/usr/bin/env python3
"""Lab 13-3 · Serve the fine-tune: the LoRA adapter on vLLM (no merge), or the merged model, or GGUF on Ollama.

Step 1 reads the adapter's own adapter_config.json on the Spark (read-only): the base model and rank r
decide vLLM's flags. Step 2 prints the three ways to serve it. Only with --launch (or SPARK_APPLY=1)
does it copy the adapter to ~/w25/adapters/hotel-ft and start vLLM (path A) in the background, then
wait for /v1/models to list the adapter and ask it one guest message.

Path A's LoRA flags are vLLM's own CLI, not an NVIDIA playbook (see Module 05, section 7).
Run: .venv/bin/python week25/13_finetune_to_serve/labs/lab03_serve_it.py [--launch]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _hotel_eval as H  # noqa: E402
from sparkkit import banner, chat, check, models, note, result, sh, show_chat, step, url, warn, where  # noqa: E402

APPLY = "--launch" in sys.argv or os.environ.get("SPARK_APPLY") == "1"
ADAPTER = "~/w25/m09/saves/qwen3-4b-hotel/lora/sft"
MERGED = "~/w25/m09/saves/qwen3-4b-hotel/merged"
NAME = "hotel-ft"

banner("Lab 13-3 · serve the fine-tune", "adapter on vLLM · merged model on vLLM · GGUF on Ollama")

step(1, "read the adapter's own config: which base, which rank?")
r = sh(f"cat {ADAPTER}/adapter_config.json && ls -la {ADAPTER} | grep -E 'safetensors|tokenizer' ",
       example='{"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507", "r": 16, "lora_alpha": 32, '
               '"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], …}\n'
               "-rw-rw-r-- 1 you you 66M … adapter_model.safetensors")
base, rank = "Qwen/Qwen3-4B-Instruct-2507", 16
try:
    cfg = json.loads(r.out[r.out.index("{"):r.out.rindex("}") + 1])
    base, rank = cfg.get("base_model_name_or_path", base), int(cfg.get("r", rank))
except (ValueError, json.JSONDecodeError):
    pass
check(bool(base) and rank > 0, f"base = {base} · rank r = {rank} → --max-lora-rank must be ≥ {rank}",
      "could not read adapter_config.json — did Module 09's training finish?")

VLLM_LORA = f"""docker run -d --name w25-hotel-vllm --gpus all --ipc host \\
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/w25/adapters:/adapters" \\
  --entrypoint '' vllm/vllm-openai:latest \\
  vllm serve {base} --max-model-len 8192 --gpu-memory-utilization 0.3 \\
    --enable-lora --lora-modules {NAME}=/adapters/{NAME} --max-lora-rank {max(rank, 16)}"""
VLLM_MERGED = f"""cd ~/w25/m09 && llamafactory-cli export configs/hotel_merge.yaml     # writes {MERGED}
docker run -d --name w25-hotel-merged --gpus all --ipc host -p 8000:8000 \\
  -v "$HOME/w25/m09/saves/qwen3-4b-hotel/merged:/models/hotel-merged" \\
  --entrypoint '' vllm/vllm-openai:latest \\
  vllm serve /models/hotel-merged --served-model-name hotel-merged --max-model-len 8192 --gpu-memory-utilization 0.3"""
OLLAMA_GGUF = f"""pip install -r ~/llama.cpp/requirements.txt          # once, in a venv
python ~/llama.cpp/convert_hf_to_gguf.py {MERGED} --outtype q8_0 --outfile ~/w25/hotel-router-q8_0.gguf
printf 'FROM ./hotel-router-q8_0.gguf\\nPARAMETER temperature 0\\n' > ~/w25/Modelfile
cd ~/w25 && ollama create hotel-router -f Modelfile && ollama run hotel-router 'The shower has no hot water'"""

step(2, "three ways to serve it")
for title, cmd in (("A · adapter on vLLM, no merge (one base can carry many adapters)", VLLM_LORA),
                   ("B · merged model on vLLM (one plain model, no LoRA flags)", VLLM_MERGED),
                   ("C · GGUF on Ollama (smallest, simplest, slowest to update)", OLLAMA_GGUF)):
    print(f"\n→ {title}")
    for line in cmd.splitlines():
        print(f"  {line}")

step(3, "start path A on the Spark (opt-in)")
if not APPLY:
    note("Not launched. Re-run with --launch to copy the adapter to ~/w25/adapters/hotel-ft and start vLLM.")
elif where() == "dry":
    warn("--launch needs a reachable Spark (DRY now).")
else:
    sh(f"mkdir -p ~/w25/adapters && rm -rf ~/w25/adapters/{NAME}.new && cp -r {ADAPTER} ~/w25/adapters/{NAME}.new "
       f"&& mv -T ~/w25/adapters/{NAME}.new ~/w25/adapters/{NAME} && ls ~/w25/adapters/{NAME}")
    sh("docker rm -f w25-hotel-vllm 2>/dev/null; " + VLLM_LORA.replace("\\\n", " "))
    base_url = url("vllm")
    print(f"│ waiting for {base_url}/models to list '{NAME}' (first start downloads the base model) …")
    ids: list[str] = []
    for _ in range(90):
        ids = models(base_url, timeout=3)
        if NAME in ids:
            break
        time.sleep(10)
    check(NAME in ids, f"vLLM lists: {', '.join(ids)}", f"'{NAME}' not listed after 15 min — docker logs w25-hotel-vllm")
    if NAME in ids:
        it = H.load_eval(1)[0]
        ans = chat(base_url, NAME, [{"role": "system", "content": it["system"]},
                                    {"role": "user", "content": it["instruction"]}], max_tokens=160, temperature=0)
        ans["source"] = "spark"
        show_chat(ans)

result("Serve the adapter (A) while you iterate: retraining only swaps a ~60 MB folder. Merge (B/C) when "
       "you ship one model and want no LoRA overhead. Lab 13-4 routes clients to whichever you serve.")
