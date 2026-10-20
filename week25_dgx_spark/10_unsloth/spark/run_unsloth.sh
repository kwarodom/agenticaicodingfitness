#!/usr/bin/env bash
# Week 25 · Module 10 — run one Python script inside NVIDIA's Unsloth playbook container, headless.
# (https://build.nvidia.com/spark/unsloth)
#
#   usage:  bash ~/w25/m10/run_unsloth.sh <job-name> <script.py> [args…]
#   e.g.    nohup bash ~/w25/m10/run_unsloth.sh validate test_unsloth.py > ~/w25/logs/m10_validate.log 2>&1 < /dev/null &
#
# The image, the ulimits and the two pip commands are the playbook's Steps 2–4, copied exactly.
# Course deviations, so it can run under nohup and keep its results:
#   • no `-it` (there is no terminal under nohup) and a --name, so `docker ps` shows the job
#   • ~/w25/m10 is mounted at /workspace/m10 (scripts in, adapters and logs out)
#   • ~/.cache/huggingface is mounted, as the VLM playbook's launch.sh does, so models download once
#   • the container is --rm, like the playbook's, so the pip installs repeat on every run (a few minutes)
set -euo pipefail
NAME="$1"; shift
mkdir -p "$HOME/w25/m10" "$HOME/.cache/huggingface"
docker run --gpus all --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  --entrypoint /usr/bin/bash --name "w25-unsloth-$NAME" \
  -v "$HOME/w25/m10:/workspace/m10" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/m10 \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  -c 'set -e
      pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
      pip install --no-deps unsloth unsloth_zoo bitsandbytes
      if [ ! -f test_unsloth.py ]; then
        curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-unsloth/assets/test_unsloth.py
      fi
      echo "== W25 running: python $*"
      python "$@"
      echo "W25_JOB_DONE"' w25 "$@"
