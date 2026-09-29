#!/usr/bin/env bash
# Pre-download what the Week 25 labs pull, so 20 laptops don't all start a 20 GB download in class.
#   week25/spark_host/prefetch.sh pull   # Ollama models + container images (shared by every account)
#   week25/spark_host/prefetch.sh hf     # Hugging Face weights into sparklab's cache (needs sudo; after hf auth login)
# Safe to re-run: everything already present is skipped.
set -uo pipefail

OLLAMA=(gpt-oss:20b llama3.1:8b gemma3:4b qwen2.5:32b llama3.2-vision:11b
        qwen3.6:35b-a3b-q8_0 qwen3.6:35b-a3b-mtp-q4_K_M gemma4:12b nemotron-3-nano:latest qwen3.6:35b-a3b)
IMAGES=(nvcr.io/nvidia/pytorch:25.11-py3 nvcr.io/nvidia/vllm:26.05-py3 vllm/vllm-openai:latest
        lmsysorg/sglang:latest-cu130 nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12
        nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
        nvcr.io/nvidia/nemo-automodel:26.02 ghcr.io/open-webui/open-webui:ollama)
HF=(Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B nvidia/Llama-3.1-8B-Instruct-FP8 nvidia/Qwen3.6-35B-A3B-NVFP4
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B nvidia/Llama-3.1-8B-Instruct-FP4)

fail=()
# flaky networks/DNS: 8 tries, 30 s apart (ollama and docker resume partial layers)
retry() { local n; for n in 1 2 3 4 5 6 7 8; do "$@" && return 0; ((n < 8)) && { echo "  … retry $n"; sleep 30; }; done; return 1; }
case "${1:-}" in
  pull)
    have=$(ollama list | awk 'NR>1{print $1}')
    for m in "${OLLAMA[@]}"; do
      grep -qx "$m" <<<"$have" && { echo "= ollama $m"; continue; }
      echo "↓ ollama $m"; retry ollama pull "$m" >/dev/null 2>&1 || fail+=("ollama $m")
    done
    for i in "${IMAGES[@]}"; do
      docker image inspect "$i" >/dev/null 2>&1 && { echo "= image $i"; continue; }
      echo "↓ image $i"; retry docker pull -q "$i" >/dev/null || fail+=("image $i")
    done ;;
  hf)
    HFBIN=/home/sparklab/.local/bin/hf
    run=(sudo -u sparklab -H); [[ $(id -un) == sparklab ]] && run=()
    for r in "${HF[@]}"; do
      echo "↓ hf $r"; retry "${run[@]}" "$HFBIN" download "$r" >/dev/null || fail+=("hf $r (gated? → hf auth login as sparklab)")
    done ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac

if ((${#fail[@]})); then printf '✗ %s\n' "${fail[@]}"; exit 1; fi
echo "✓ prefetch $1 done"
