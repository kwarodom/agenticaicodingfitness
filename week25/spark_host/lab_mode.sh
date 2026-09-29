#!/usr/bin/env bash
# Switch this Spark between its day job and the Week 25 classroom.
#   week25/spark_host/lab_mode.sh on       # stop the containers that hold lab ports / GPU memory
#   week25/spark_host/lab_mode.sh off      # start exactly the ones `on` stopped
#   week25/spark_host/lab_mode.sh status   # ports, memory, tailnet URL — run before class
#   week25/spark_host/lab_mode.sh fix-perms  # give back files lab containers left root-owned (also run by on/off)
set -euo pipefail

# name → why it has to go during class
declare -A HOLDS=(
  [nemotron-lightning]="~100 GB unified memory (vLLM gpu-memory-utilization)"
  [supabase-kong]=":8000 — vLLM / NIM default port"
  [alto-backend]=":8001 — second vLLM in the two-model labs"
)
STATE=${XDG_STATE_HOME:-$HOME/.local/state}/spark-lab-mode.stopped
# Ollama :11434 · vLLM :8000/:8001 · SGLang :30000 · TRT-LLM :8355 · llama.cpp :30080 · LM Studio :1234
# LiteLLM :4000 · Open WebUI :12000 · labs' misc :8080
LAB_PORTS=(11434 8000 8001 30000 8355 30080 1234 4000 12000 8080)

running() { docker ps --format '{{.Names}}' | grep -qx "$1"; }

# Lab containers run as root and mount ~/.cache/huggingface and ~/w25, so they leave root-owned files there.
# Later `hf download` then fails on its lock files and students (no sudo) cannot delete checkpoints.
# docker group membership is enough to chown them back — no sudo needed.
fix_perms() {
  local u home
  for u in "$(id -un)" sparklab; do
    home=$(getent passwd "$u" | cut -d: -f6) || continue
    docker run --rm -v "$home:/h" ubuntu:24.04 bash -c \
      "n=\$(find /h/.cache/huggingface /h/w25 -uid 0 2>/dev/null | wc -l); \
       [ \$n -gt 0 ] && find /h/.cache/huggingface /h/w25 -uid 0 -exec chown $(id -u "$u"):$(id -g "$u") {} + 2>/dev/null; \
       echo \"  $u: \$n root-owned file(s) fixed\""
  done
}

case "${1:-status}" in
  on)
    mkdir -p "$(dirname "$STATE")"; : >"$STATE.new"
    for c in "${!HOLDS[@]}"; do
      if running "$c"; then echo "■ stopping $c  (${HOLDS[$c]})"; docker stop -t 30 "$c" >/dev/null; echo "$c" >>"$STATE.new"; fi
    done
    cat "$STATE.new" >>"$STATE"; rm -f "$STATE.new"; sort -u -o "$STATE" "$STATE"
    ollama ps 2>/dev/null | awk 'NR>1{print $1}' | xargs -r -n1 ollama stop 2>/dev/null || true
    echo "▪ repairing file ownership"; fix_perms
    echo "✓ lab mode ON — run '$0 off' after class"; exec "$0" status ;;
  off)
    [[ -s $STATE ]] || { echo "nothing to restore (lab mode was not switched on by this script)"; exit 0; }
    while read -r c; do echo "▶ starting $c"; docker start "$c" >/dev/null; done <"$STATE"
    rm -f "$STATE"; echo "▪ repairing file ownership"; fix_perms
    echo "✓ lab mode OFF — day-job containers restored" ;;
  fix-perms) fix_perms ;;
  status)
    dns=$(tailscale status --json 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo "?")
    echo "Tailnet host : $dns   ($(tailscale ip -4 2>/dev/null || echo 'tailscale down?'))"
    echo "SPARK_HOST   : sparklab@$dns"
    getent passwd sparklab >/dev/null && echo "sparklab     : account exists" || echo "sparklab     : ✗ missing — sudo $(dirname "$0")/setup_sparklab_user.sh"
    echo "Memory       : $(free -g | awk '/Mem:/{print $7" GB available of "$2" GB"}')"
    echo "Ports        :"
    for p in "${LAB_PORTS[@]}"; do
      who=$(docker ps --format '{{.Names}} {{.Ports}}' | grep -E "(0\.0\.0\.0|\[::\]):$p->" | awk '{print $1}' | head -1 || true)
      if ss -ltn "sport = :$p" | grep -q LISTEN; then printf '  :%-6s in use %s\n' "$p" "${who:+($who)}"; else printf '  :%-6s free\n' "$p"; fi
    done
    echo "Day-job      :"; for c in "${!HOLDS[@]}"; do printf '  %-20s %s\n' "$c" "$(running "$c" && echo running || echo stopped)"; done ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
