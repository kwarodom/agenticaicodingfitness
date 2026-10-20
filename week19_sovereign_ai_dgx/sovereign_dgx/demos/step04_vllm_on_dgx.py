#!/usr/bin/env python3
"""PART 4 · vLLM on DGX — stand up a throughput server, step by step  [INTERMEDIATE]

vLLM is a SEPARATE server from Ollama. You start it on the DGX (port 8000); it
serves ONE model behind the same OpenAI API, with PagedAttention + continuous
batching for high throughput under concurrency.

This chapter is a hands-on guide: the steps to RUN vLLM on the DGX, lots of
examples to CALL it, and then a live measurement of whatever endpoint this app is
currently pointed at.

Run:  python demos/step04_vllm_on_dgx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxview  # noqa: E402

# ── the step-by-step DGX guide (🖥️ = run on the DGX) ──────────────────────────
STEP1 = """\
# 🖥️ STEP 1 (on the DGX) — pull NVIDIA's vLLM container
#    find the latest tag: catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
export HF_TOKEN="hf_..."                     # from huggingface.co/settings/tokens
export VLLM_TAG=26.05.post1-py3              # use the latest you find
docker pull nvcr.io/nvidia/vllm:$VLLM_TAG
"""

STEP2 = """\
# 🖥️ STEP 2 (on the DGX) — serve a model that FITS a single Spark (128 GB).
#    Start small/medium; go bigger later (see "GOING BIGGER" below).
export MODEL=openai/gpt-oss-20b              # ~20B, fits comfortably, open weights

docker run -d --name vllm --gpus all --ipc host \\
  -p 8000:8000 \\
  -e HF_TOKEN="$HF_TOKEN" \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  nvcr.io/nvidia/vllm:$VLLM_TAG \\
  vllm serve "$MODEL" \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192
#    first run downloads the weights (minutes). Watch:  docker logs -f vllm
"""

STEP3 = """\
# 🖥️ STEP 3 (on the DGX) — verify it locally BEFORE involving the tunnel
curl http://localhost:8000/v1/models                       # lists the served model
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"openai/gpt-oss-20b",
       "messages":[{"role":"user","content":"12*17"}]}'     # expect 204
"""

STEP4 = """\
# 🖥️ STEP 4 (on the DGX) — expose port 8000 (vLLM needs NO --host-header)
ngrok http 8000 --basic-auth="dgx:agenticai"
# 💻 then on your LAPTOP, point this tutorial at vLLM and reload:
export DGX_CONN=tunnel
export DGX_TUNNEL_URL="https://dgx:agenticai@<new-vllm-id>.ngrok-free.app/v1"
# (or on the LAN:  export DGX_BASE_URL=http://<dgx-host>:8000/v1)
"""

# ── ways to CALL the vLLM server (same OpenAI API) ────────────────────────────
CALL_CURL = """\
# A) curl — one completion
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"Hello"}],
       "max_tokens":128}'
"""

CALL_CURL_STREAM = """\
# B) curl — STREAMING (tokens arrive as they're generated)
curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"Write a haiku"}],
       "stream":true}'
"""

CALL_SDK = """\
# C) Python — the OpenAI SDK, only base_url changes
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain PagedAttention in 1 sentence."}])
print(r.choices[0].message.content)
"""

CALL_SDK_STREAM = """\
# D) Python — streaming
stream = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Count to 20."}], stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
"""

CALL_MODELS = """\
# E) list what the server is serving
curl http://localhost:8000/v1/models
"""

CALL_CONCURRENCY = """\
# F) the vLLM WIN — fire many requests at once (continuous batching).
#    Single-stream tok/s ≈ Ollama; AGGREGATE throughput scales with concurrency.
seq 20 | xargs -P 20 -I{} curl -s http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hi {}"}],
       "max_tokens":64}' > /dev/null
#    watch GPU stay busy across all 20:  docker logs -f vllm   (or nvidia-smi)
"""

BIGGER = """\
# GOING BIGGER (when 20B isn't enough)
#  • 70B on ONE Spark — use an NVFP4/FP8 build so it fits in 128 GB:
#      vllm serve nvidia/Llama-3.3-70B-Instruct-FP4 --max-model-len 8192
#  • 235B across TWO linked Sparks — tensor parallelism over the 200GbE link:
#      mpirun -H spark-0,spark-1 -np 2 \\
#        vllm serve nvidia/Qwen3-235B-A22B-NVFP4 --tensor-parallel-size 2
#  • OOM? lower --max-model-len and add --max-num-seqs 4.
"""


def _measure(model: str, prompt: str):
    client = dgxview._client()
    start = time.time(); ttft = None; chars = 0
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=300, temperature=0.3, stream=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        piece = (d.content or "") + (dgxview._extract_reasoning(d) or "")
        if piece:
            if ttft is None:
                ttft = time.time() - start
            chars += len(piece)
    elapsed = time.time() - start
    toks = max(1, round(chars / 4))
    decode = elapsed - (ttft or 0)
    return (ttft or elapsed), (toks / decode if decode > 0 else 0), toks, elapsed


def main() -> None:
    dgxview.banner("PART 4", "vLLM on DGX — stand up a throughput server", "INTERMEDIATE")
    dgxview.mode_line()

    print("vLLM is a SEPARATE server from Ollama — you start it on the DGX (port 8000)")
    print("and it serves ONE model behind the same OpenAI API. Follow these steps on")
    print("the DGX, then point this app at it (Step 4).\n")
    print("═" * 64)
    print(STEP1); print(STEP2); print(STEP3); print(STEP4)
    print("═" * 64)
    print("\nCALLING THE vLLM SERVER — same OpenAI API, many ways:\n")
    for block in (CALL_CURL, CALL_CURL_STREAM, CALL_SDK, CALL_SDK_STREAM,
                  CALL_MODELS, CALL_CONCURRENCY):
        print(block)
    print(BIGGER)

    print("Why vLLM (vs Ollama):")
    print("  • PagedAttention   — kv-cache in pages → less waste, more requests resident")
    print("  • Continuous batch — new requests join the running batch each step")
    print("  • Net effect       — single-user tok/s ≈ Ollama; AGGREGATE throughput scales")
    print("                       ~linearly with concurrent users (that's the reason to use it)\n")

    # ── live measurement of whatever this app is connected to right now ──
    # Detect the runtime by PROBING (not by port): Ollama answers the native
    # /api/tags; vLLM / llama.cpp / other OpenAI servers return 404 there.
    is_ollama = False
    try:
        config._open(config._native_base() + "/tags", timeout=3)
        is_ollama = True
    except Exception:
        is_ollama = False
    served = "Ollama" if is_ollama else "vLLM / OpenAI-compatible server"
    print("═" * 64)
    if dgxview.is_sim():
        print("SIM mode — representative DGX Spark serving numbers:")
        print("  ◆ time-to-first-token:  ~120 ms")
        print("  ◆ single-stream decode: ~48 tok/s   (qwen3.6 35B MoE, Q8)")
        print("  ◆ 8 concurrent users:   ~310 tok/s aggregate (PagedAttention win)")
    else:
        dgxview.sovereignty_line()
        print(f"Measuring the connected endpoint — runtime: {served}, model: {config.MODEL}\n")
        ttft, tps, toks, total = _measure(config.MODEL,
                                          "Explain PagedAttention in 2 sentences.")
        print(f"  ◆ time-to-first-token:  {ttft*1000:.0f} ms")
        print(f"  ◆ single-stream decode: ~{tps:.1f} tok/s  ({toks} tokens, {total:.1f}s total)")
        print("  ◆ network hops to cloud: 0  (served from your hardware)")
        if is_ollama:
            print("\n  ℹ This is your Ollama endpoint. To measure vLLM instead, point the")
            print("    🔌 Connection at the vLLM endpoint (e.g. http://<dgx>:8800/v1).")
        else:
            print("\n  ✓ This is a vLLM / OpenAI-compatible endpoint — try the concurrency")
            print("    test (block F) to see aggregate throughput climb under load.")

    print("\nTakeaway: vLLM = production serving for MANY users. Stand it up on the DGX,")
    print("call it with the SAME OpenAI code, and watch aggregate throughput scale.")


if __name__ == "__main__":
    main()
