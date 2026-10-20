#!/usr/bin/env bash
# Week 25 · Module 09 — LLaMA Factory setup on a DGX Spark, steps 2–6 of NVIDIA's playbook
# (https://build.nvidia.com/spark/llama-factory), run in your home directory.
# The commands are the playbook's; this script only adds `set -e`, skip-if-done checks and echo lines.
# lab01_factory_setup.py uploads it to ~/w25/m09/ and runs it under nohup:
#   nohup bash ~/w25/m09/setup_factory.sh > ~/w25/logs/m09_setup.log 2>&1 &
set -euo pipefail
cd "$HOME"

echo "== Step 2 · Python virtual environment (~/factoryEnv)"
if [ ! -d factoryEnv ]; then
  python3 -m venv factoryEnv
fi
source ./factoryEnv/bin/activate

echo "== Step 3 · PyTorch with CUDA 13 support"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "== Step 4 · verify PyTorch sees the GPU"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "== Step 5 · clone LLaMA Factory (~/LLaMA-Factory)"
if [ ! -d LLaMA-Factory ]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

echo "== Step 6 · install LLaMA Factory with metrics support"
pip install -e ".[metrics]"
# Course change: newer LLaMA Factory dropped the "metrics" extra (pip only warns) and moved jieba / nltk /
# rouge-chinese to requirements/metrics.txt. Without them `--predict` (predict_with_generate) crashes.
if [ -f requirements/metrics.txt ]; then pip install -r requirements/metrics.txt; fi

llamafactory-cli version
echo "SETUP_DONE"
