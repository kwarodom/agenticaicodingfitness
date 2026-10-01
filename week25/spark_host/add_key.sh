#!/usr/bin/env bash
# Manage which laptops can ssh into sparklab@<this spark>.
#   sudo week25/spark_host/add_key.sh alice.pub [bob.pub ...]   # or a quoted "ssh-ed25519 AAAA… alice" line
#   sudo week25/spark_host/add_key.sh --list
#   sudo week25/spark_host/add_key.sh --remove alice            # removes keys whose comment contains "alice"
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
AK=/home/sparklab/.ssh/authorized_keys
[[ -f $AK ]] || { echo "run setup_sparklab_user.sh first" >&2; exit 1; }

case "${1:-}" in
  ""|-h|--help) sed -n 2,5p "$0"; exit 0 ;;
  --list) ssh-keygen -l -f "$AK" 2>/dev/null || echo "(no keys)"; exit 0 ;;
  --remove)
    [[ -n "${2:-}" ]] || { echo "--remove needs a name" >&2; exit 1; }
    before=$(wc -l <"$AK"); grep -vF -- "$2" "$AK" >"$AK.new" || true
    cat "$AK.new" >"$AK"; rm -f "$AK.new"
    echo "removed $((before - $(wc -l <"$AK"))) key(s) matching '$2'"; exit 0 ;;
esac

for arg in "$@"; do
  if [[ -f $arg ]]; then line=$(grep -m1 -E '^(ssh-|ecdsa-|sk-)' "$arg"); else line=$arg; fi
  printf '%s\n' "$line" | ssh-keygen -l -f /dev/stdin >/dev/null 2>&1 || { echo "✗ not a public key: $arg" >&2; continue; }
  key=$(awk '{print $2}' <<<"$line")
  if grep -qF -- "$key" "$AK"; then echo "= already present: $(awk '{print $3}' <<<"$line")"; continue; fi
  printf '%s\n' "$line" >>"$AK"; echo "+ added: $(printf '%s\n' "$line" | ssh-keygen -l -f /dev/stdin)"
done
chown sparklab:sparklab "$AK"; chmod 600 "$AK"
