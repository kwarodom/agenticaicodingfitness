#!/usr/bin/env bash
# Is the classroom ready? Read-only checks of both Sparks, as the `sparklab` account students use.
#   week25/spark_host/ready_check.sh          # run on Spark A (or any machine with ssh to both)
# Spark A/B ssh targets come from SPARK_HOST / SPARK_HOST2 (env or week25/.env.local), like the runner.
# ✓ ready · ⚠ fine for single-Spark labs or fix before class · ✕ a lab will fail
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cfg() { local v=${!1:-}; [[ -n $v ]] && { echo "$v"; return; }; sed -n "s/^$1=//p" "$HERE/../.env.local" 2>/dev/null | tail -1; }
A=$(cfg SPARK_HOST); B=$(cfg SPARK_HOST2)
[[ -n $A ]] || { echo "✕ SPARK_HOST is not set (week25/.env.local or env)"; exit 1; }
bad=0; ok() { echo "  ✓ $*"; }; warn() { echo "  ⚠ $*"; }; no() { echo "  ✕ $*"; bad=1; }

# One ssh per Spark; the remote side prints key=value facts, the checks below read them.
PROBE='
echo "host=$(hostname)"
echo "user=$(id -un)"
echo "keys=$(grep -cE "^(ssh-|ecdsa-|sk-)" ~/.ssh/authorized_keys 2>/dev/null || echo 0)"
echo "hf=$(~/.local/bin/hf auth whoami >/dev/null 2>&1 && echo yes || echo no)"
echo "ngc=$(grep -q nvcr.io ~/.docker/config.json 2>/dev/null && echo yes || echo no)"
echo "nccl=$(ls ~/nccl/build/lib/libnccl.so.2 ~/nccl-tests/build/all_gather_perf >/dev/null 2>&1 && echo yes || echo no)"
echo "ncclt=$(cd ~/nccl-tests 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo -)"
echo "mem=$(free -g | awk "/Mem:/{print \$7}")"
echo "tags=$(tailscale status --json 2>/dev/null | python3 -c "import sys,json;print(\",\".join(json.load(sys.stdin)[\"Self\"].get(\"Tags\") or []) or \"none\")" 2>/dev/null)"
for m in meta-llama--Llama-3.3-70B-Instruct nvidia--Qwen3.6-35B-A3B-NVFP4 Qwen--Qwen3-4B-Instruct-2507; do
  s=$(ls -d ~/.cache/huggingface/hub/models--$m/snapshots/* 2>/dev/null | head -1)
  echo "model_$m=$([ -n "$s" ] && du -sLBG "$s" 2>/dev/null | cut -f1 | tr -d G || echo 0)"
done
for i in vllm/vllm-openai:latest nvcr.io/nvidia/vllm:26.05-py3 nvcr.io/nvidia/pytorch:25.11-py3; do
  echo "img_$i=$(docker image inspect "$i" >/dev/null 2>&1 && echo yes || echo no)"
done
echo "runcluster=$(grep -q "pip install -q" ~/run_cluster.sh 2>/dev/null && echo yes || echo no)"
echo "adapter=$(test -f ~/w25/m09/saves/qwen3-4b-hotel/lora/sft/adapter_model.safetensors && echo yes || echo no)"
dev=$(ibdev2netdev 2>/dev/null | sed -n "s/.* ==> \(\S*\) (Up)/\1/p" | head -1)
echo "linkip=$(ip -4 -o addr show "$dev" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -1)"
'
declare -A FA FB
probe() {  # $1 = ssh target, $2 = name of the assoc array to fill
  local out; out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "bash -s" <<<"$PROBE" 2>/dev/null) || return 1
  while IFS='=' read -r k v; do [[ -n $k ]] && eval "$2[\$k]=\$v"; done <<<"$out"
}

check_spark() {  # $1 = A|B, $2 = target, $3 = array name
  local -n F=$3
  echo "━━ Spark $1 · $2"
  [[ -n ${F[host]:-} ]] || { no "ssh as $2 failed: key, tailnet, or the Spark is off"; return; }
  ok "ssh works → ${F[host]} as ${F[user]}"
  local k=${F[keys]}; ((k > 1)) && ok "$k keys can log in as sparklab" || warn "only $k key(s) for sparklab: add the students' with sudo week25/spark_host/add_key.sh on BOTH Sparks"
  [[ ${F[hf]} == yes ]] && ok "Hugging Face login" || no "no Hugging Face login: sudo -u sparklab -H bash -lc '~/.local/bin/hf auth login'"
  [[ ${F[ngc]} == yes ]] && ok "nvcr.io login" || warn "no nvcr.io login (only Module 06 NIM needs it): week25/spark_host/ngc_login.sh"
  [[ ${F[tags]} == *tag:dgx-spark* ]] && ok "tailnet tag tag:dgx-spark" || warn "not tagged tag:dgx-spark: students' tailnet grant won't reach it (admin console → Machines → Edit ACL tags)"
  ((${F[mem]:-0} >= 100)) && ok "${F[mem]} GB memory available" || warn "${F[mem]} GB available: run lab_mode.sh on before the GPU labs"
  for i in vllm/vllm-openai:latest nvcr.io/nvidia/vllm:26.05-py3 nvcr.io/nvidia/pytorch:25.11-py3; do
    [[ ${F[img_$i]:-no} == yes ]] && ok "image $i" || warn "image $i missing: week25/spark_host/prefetch.sh pull"
  done
}

probe "$A" FA; [[ -n $B ]] && probe "$B" FB
check_spark A "$A" FA
gb() { local v=${1:-0}; ((v >= $2)); }
gb "${FA[model_meta-llama--Llama-3.3-70B-Instruct]:-0}" 130 && ok "Llama 3.3 70B cached (Modules 05 §6, 11 §6)" || warn "Llama 3.3 70B not cached (two-Spark Modules 05 §6, 11 §6 need it on both)"
gb "${FA[model_nvidia--Qwen3.6-35B-A3B-NVFP4]:-0}" 15 && ok "Qwen3.6-35B-A3B NVFP4 cached (capstone brain)" || no "Qwen3.6 NVFP4 not cached: sudo week25/spark_host/prefetch.sh hf"
if [[ -z $B ]]; then
  echo; echo "━━ Spark B · not configured (set SPARK_HOST2): single-Spark course only"
else
  echo; check_spark B "$B" FB
  if [[ -n ${FB[host]:-} ]]; then
    gb "${FB[model_meta-llama--Llama-3.3-70B-Instruct]:-0}" 130 && ok "Llama 3.3 70B cached" || warn "Llama 3.3 70B not cached (TWO_SPARKS.md: copy it over the cable)"
    gb "${FB[model_Qwen--Qwen3-4B-Instruct-2507]:-0}" 5 && ok "Qwen3-4B-Instruct cached (capstone router base)" || no "Qwen3-4B-Instruct not cached on B: capstone router can't start"
    [[ ${FB[adapter]} == yes ]] && ok "hotel-ft adapter trained (capstone router)" || warn "no hotel-ft adapter on B yet: run Module 09 labs 01–04 with SPARK_HOST pointed at Spark B (START_HERE.md)"
    echo; echo "━━ Two Sparks"
    [[ ${FA[user]} == "${FB[user]}" ]] && ok "same user on both (${FA[user]})" || no "users differ (${FA[user]} vs ${FB[user]}): use sparklab@ for both"
    [[ ${FA[runcluster]} == yes && ${FB[runcluster]} == yes ]] && ok "patched run_cluster.sh on both (Module 05 §6)" || warn "~/run_cluster.sh missing or unpatched for sparklab (Module 05 §6 Step 1)"
    [[ ${FA[nccl]} == yes && ${FB[nccl]} == yes ]] && ok "NCCL + nccl-tests built on both" || no "NCCL not built on both (TWO_SPARKS.md Step 5)"
    [[ ${FA[ncclt]} == "${FB[ncclt]}" ]] && ok "same nccl-tests build (${FA[ncclt]})" || warn "nccl-tests differ (${FA[ncclt]} vs ${FB[ncclt]}): pin both to b4d5bee"
    if [[ -n ${FB[linkip]} ]]; then
      r=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$A" "ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new ${FB[linkip]} hostname" 2>/dev/null)
      [[ $r == "${FB[host]}" ]] && ok "Spark A → B over the cable (${FB[linkip]}) as ${FA[user]}" || no "Spark A can't ssh to B over the cable as ${FA[user]} (TWO_SPARKS.md Step 4)"
    else
      no "no ConnectX-7 link is Up on Spark B: check the QSFP cable (lab 02-1)"
    fi
  fi
fi
echo
((bad)) && echo "═ fix the ✕ lines, then run this again" || echo "═ ready. Before class: week25/spark_host/lab_mode.sh on"
exit $bad
