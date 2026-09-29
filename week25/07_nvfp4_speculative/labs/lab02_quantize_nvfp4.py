#!/usr/bin/env python3
"""Lab 07-2 · Quantize DeepSeek-R1-Distill-Llama-8B to NVFP4 with Model Optimizer, on your Spark.

Follows NVIDIA's NVFP4 quantization playbook. By default the lab is read-only: it checks the
Spark (Docker, disk, Hugging Face login), shows the quantization job's log and output folder,
reads hf_quant_config.json, and compares the checkpoint size with the arithmetic from lab 01.

The quantization itself takes 45–90 minutes, so it is started only when you ask, in the
background with nohup (the runner kills foreground labs after 900 s):

    .venv/bin/python week25/07_nvfp4_speculative/labs/lab02_quantize_nvfp4.py --start
    .venv/bin/python week25/07_nvfp4_speculative/labs/lab02_quantize_nvfp4.py          # status, any time

Without a Spark (DRY) you see every command and an illustrative output shape, labelled EXAMPLE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, note, result, sh, step, table, weights_gb, where, warn  # noqa: E402

IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"      # playbook, DGX Spark row
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
WORK = "~/w25/nvfp4"
OUT = f"{WORK}/output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf"
LOG = "~/w25/logs/nvfp4_quant.log"

# The playbook's Step 5 command for DGX Spark, with two course changes (stated in the tutorial):
#   · no `-it` (a nohup job has no terminal) and `nohup … > log &` around it
#   · `-e HF_TOKEN` passes the variable only if it is set; the model is public, and a gated model can use
#     the token that `hf auth login` stored in ~/.cache/huggingface (mounted below)
QUANT_CMD = f"""mkdir -p {WORK}/output_models ~/w25/logs && cd {WORK} && \\
{{ nohup docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v "./output_models:/workspace/output_models" \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -e HF_TOKEN \\
  {IMAGE} \\
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && \\
    cd /app/Model-Optimizer && pip install -e '.[dev]' && \\
    export ROOT_SAVE_PATH='/workspace/output_models' && \\
    /app/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \\
    --model '{MODEL}' \\
    --quant nvfp4 \\
    --tp 1 \\
    --export_fmt hf
  " > {LOG} 2>&1 < /dev/null & }}
echo "started quantization job (pid $!) → {LOG}\""""

banner("Lab 07-2 · quantize to NVFP4 with Model Optimizer",
       "playbook: nvfp4-quantization · DGX Spark path (TensorRT-LLM container, Model Optimizer 0.35.0)")
start = "--start" in sys.argv

step(1, "preflight — Docker, disk, Hugging Face login (read-only)")
checks = [
    ("docker", "docker version --format '{{.Server.Version}}'", "28.3.3"),
    ("disk", "df -h ~ | tail -1", "/dev/nvme0n1p2  3.7T  412G  3.1T  12% /"),
    ("hf login", "test -s ~/.cache/huggingface/token && echo 'token file present' || echo 'no token file (fine for this public model)'",
     "token file present"),
    ("image", f"docker image inspect {IMAGE} --format '{{{{.Size}}}}' 2>/dev/null | numfmt --to=si || echo 'not pulled yet (the job pulls it)'",
     "not pulled yet (the job pulls it)"),
]
rows = []
for name, cmd, ex in checks:
    r = sh(cmd, timeout=60, example=ex)
    rows.append([name, "◈ example" if r.source != "live" else ("✓" if r.ok else "✕"),
                 (r.out.strip().splitlines() or ["—"])[-1][:60]])
table(rows, ["check", "result", "output"])

step(2, "the quantization job (the playbook's Step 5, run in the background)")
if where("a") == "dry":
    sh(QUANT_CMD, example=f"started quantization job (pid <pid>) → {LOG}")
elif start:
    sh(QUANT_CMD, timeout=60)
    note(f"Follow it: ssh to the Spark and run  tail -f {LOG}   — or run this lab again for a status check.")
else:
    print(QUANT_CMD)
    note("Not started (read-only by default). Add --start to launch the 45–90 minute job in the background.")

step(3, "status — the log tail and the exported checkpoint")
sh(f"tail -n 6 {LOG} 2>/dev/null || echo 'no log yet — start the job with --start'", timeout=30,
   example="…\nQuantizing model...\nExporting to HF format ...\nQuantized model saved to /workspace/output_models/"
           "saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf")
sh(f"ls {OUT}/ 2>/dev/null || echo 'not exported yet'", timeout=30,
   example="config.json\ngeneration_config.json\nhf_quant_config.json\nmodel-00001-of-00002.safetensors\n"
           "model-00002-of-00002.safetensors\nmodel.safetensors.index.json\nspecial_tokens_map.json\n"
           "tokenizer.json\ntokenizer_config.json")
q = sh(f"cat {OUT}/hf_quant_config.json 2>/dev/null || echo 'no hf_quant_config.json yet'", timeout=30,
       example='{\n  "producer": {"name": "modelopt", "version": "0.35.0"},\n  "quantization": {\n'
               '    "quant_algo": "NVFP4",\n    "group_size": 16,\n    "exclude_modules": ["lm_head"]\n  }\n}')
note("hf_quant_config.json is how serving engines (TensorRT-LLM, vLLM, SGLang) recognise a Model Optimizer "
     "checkpoint: quant_algo NVFP4, group_size 16 — the 16-value blocks from lab 01.")

step(4, "size on disk — measured vs the 4.5-bit arithmetic")
sizes = sh(f"du -sh {OUT} 2>/dev/null; du -sh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B "
           "2>/dev/null || true", timeout=60,
           example=f"<size>\t{OUT}\n<size>\t~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B")
bf16, fp4 = weights_gb(8.03, "bf16"), weights_gb(8.03, "nvfp4")
print(f"│ arithmetic: BF16 weights {bf16:.1f} GB · NVFP4 weights {fp4:.1f} GB (every weight at 4.5 bits)")
print(f"│ the embeddings (128,256 × 4,096 ≈ 0.53 B params) and lm_head usually stay BF16, so expect the real "
      f"checkpoint above {fp4:.1f} GB")
if sizes.source == "live":
    note("Compare the two du lines: the ratio is your measured compression. It is less than 3.56× because "
         "some layers are excluded from quantization (see exclude_modules above).")
else:
    note("DRY: no sizes measured. Run on a Spark to fill in the two du lines.")

if q.source == "live" and "NVFP4" in q.out:
    result("the NVFP4 checkpoint exists. Serve it and measure it: lab 03 (--start nvfp4).")
elif where("a") == "dry":
    result("DRY: commands shown, outputs are EXAMPLE shapes. Connect a Spark and run with --start.")
else:
    warn("no NVFP4 checkpoint yet — start the job with --start, or wait for it to finish.")
    result("run this lab again to check progress; quantization takes 45–90 minutes on the first run.")
