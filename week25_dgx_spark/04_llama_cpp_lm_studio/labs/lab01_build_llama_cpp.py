#!/usr/bin/env python3
"""Lab 04-1 · Build llama.cpp for the GB10 and start llama-server on port 30080.

Follows the llama.cpp playbook on your Spark: check the build tools, clone llama.cpp,
build `llama-server` with CUDA for architecture 121a-real, then serve the playbook's GGUF
(unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL). One change from the playbook: the course
serves on port 30080, not 30000, because SGLang (Module 06) uses 30000.

Read-only by default. The build (5–10 min) starts only with --build, the server (plus the
playbook's "~35 GB order of magnitude" first download) only with --serve. Both run in the background with a log in
~/w25/logs/, so the Lab Runner's 15-minute limit never kills them. Installing apt
packages needs sudo, so the lab prints that command for you to type.

Run: .venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab01_build_llama_cpp.py [--build] [--serve]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import PORTS, banner, note, result, sh, step  # noqa: E402

PORT = PORTS["llamacpp"]                          # 30080 in this course (playbook: 30000)
MODEL = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL"
CMAKE = ("cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON "
         "-DCMAKE_CUDA_ARCHITECTURES=121a-real")
BUILD = "cmake --build build --config Release --target llama-server -j"
SERVE = f"./bin/llama-server -hf {MODEL} --host 0.0.0.0 --port {PORT}"
# Quoted from the playbook's "You should see log lines similar to" block (its port is 30000; yours will say 30080).
REF_LOG = """0.14.342.935 I srv  llama_server: model loaded
0.14.342.939 I srv  llama_server: server is listening on http://0.0.0.0:30000
0.14.342.944 I srv  update_slots: all slots are idle"""

banner("Lab 04-1 · build llama.cpp for CUDA and serve a GGUF", f"playbook steps 1–5 · llama-server on :{PORT}")

step(1, "build tools: git, cmake ≥ 3.14, the CUDA compiler, an Arm CPU")
sh("uname -m && git --version && cmake --version | head -1 && (nvcc --version || /usr/local/cuda/bin/nvcc --version) | tail -1",
   example="aarch64\ngit version 2.43.0\ncmake version 3.28.3\nBuild cuda_13.0.r13.0/compiler.xxxxxxxx_0")
note("nvcc not found? The playbook's fix: export PATH=/usr/local/cuda/bin:$PATH, then run CMake again from a clean build dir.")

step(2, "apt packages (the playbook's step 1 — needs sudo, so you type it)")
sh("dpkg-query -W -f='${Package} ${Status}\\n' git clang cmake libcurl4-openssl-dev libssl-dev 2>&1 | sed 's/ install ok//'",
   example="git installed\nclang installed\ncmake installed\nlibcurl4-openssl-dev installed\nlibssl-dev installed")
print("→ anything missing? on the Spark: sudo apt update && sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev")

step(3, "clone and build llama-server with CUDA for GB10 (sm_121)")
build_cmd = (f"mkdir -p ~/w25/logs && ([ -d ~/llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp) "
             f"&& cd ~/llama.cpp && {{ nohup bash -c '{CMAKE} && {BUILD}' > ~/w25/logs/llama-build.log 2>&1 < /dev/null & }}; "
             f"echo build started, pid $!")
if "--build" in sys.argv:
    sh(build_cmd, example="build started, pid 23456")
    note("follow it: ssh spark-a tail -f ~/w25/logs/llama-build.log   — the playbook says 5–10 minutes")
else:
    print("→ add --build to run, in the background:")
    print(f"  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp\n  {CMAKE}\n  {BUILD}")

step(4, "is llama-server built?")
sh("ls -la ~/llama.cpp/build/bin/llama-server 2>/dev/null && ~/llama.cpp/build/bin/llama-server --version 2>&1 | tail -2 "
   "|| (echo 'not built yet'; tail -3 ~/w25/logs/llama-build.log 2>/dev/null)",
   example="-rwxrwxr-x 1 you you <size> <date> /home/you/llama.cpp/build/bin/llama-server\n"
           "version: <build number> (<commit>)\nbuilt with <compiler> for Linux aarch64")

step(5, f"serve the playbook's GGUF on :{PORT}")
serve_cmd = (f"mkdir -p ~/w25/logs && cd ~/llama.cpp/build && {{ nohup {SERVE} > ~/w25/logs/llama-server.log 2>&1 < /dev/null & }}; "
             f"echo llama-server started, pid $!")
if "--serve" in sys.argv:
    sh(serve_cmd, example="llama-server started, pid 34567")
    note("the first start downloads the GGUF into ~/.cache/huggingface/hub (tens of GB). Watch the log until "
         "'server is listening'.")
else:
    print(f"→ add --serve to run, in the background, from ~/llama.cpp/build:\n  {SERVE}")
sh("tail -3 ~/w25/logs/llama-server.log 2>/dev/null || echo 'no server log yet'", reference=REF_LOG)
note(f"the playbook's log shows port 30000; this course passes --port {PORT}, so yours says {PORT}.")

step(6, "health check (the playbook's /health endpoint)")
sh(f"curl -sf http://127.0.0.1:{PORT}/health && echo || echo 'not ready — still loading, or not started'",
   example='{"status":"ok"}')

result(f"llama.cpp = one C++ binary + one GGUF file + an OpenAI-compatible API on :{PORT}. "
       "Lab 04 calls it; lab 02 opens the GGUF it serves.")
