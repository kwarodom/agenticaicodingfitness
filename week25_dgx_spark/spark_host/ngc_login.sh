#!/usr/bin/env bash
# Log the sparklab account in to nvcr.io (only NIM containers, Module 06, need this).
# Reads the key silently, checks it, tests it against nvcr.io, and only then runs docker login.
#   week25/spark_host/ngc_login.sh
set -uo pipefail

read -rs -p "NGC Personal API Key (input hidden): " K; echo
# strip what copy-paste tends to add: CR/LF, spaces, surrounding quotes
K=$(printf '%s' "$K" | tr -d '\r\n\t ' | sed -e "s/^[\"']//" -e "s/[\"']\$//")

echo "· length ${#K}, starts with '${K:0:6}…'"
[[ -n $K ]] || { echo "✗ empty — nothing was pasted (try right-click → Paste, or Ctrl+Shift+V)"; exit 1; }
[[ $K == nvapi-* ]] || echo "⚠ not an nvapi- key — a legacy NGC key can still work; an HF token (hf_…) will not"

code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -u "\$oauthtoken:$K" \
  "https://nvcr.io/proxy_auth?account=%24oauthtoken&offline_token=true&service=registry")
case $code in
  200) echo "✓ nvcr.io accepts this key" ;;
  401|403)
    echo "✗ nvcr.io rejects this key ($code). Fix at https://org.ngc.nvidia.com/setup/api-keys :"
    echo "  Generate Personal Key → Services Included must list 'NGC Catalog' (a key made only for"
    echo "  build.nvidia.com / Cloud Functions fails exactly like this). Check it hasn't expired."
    unset K; exit 1 ;;
  000) echo "✗ could not reach nvcr.io (DNS/network) — run again"; unset K; exit 1 ;;
  *)   echo "? unexpected HTTP $code from nvcr.io — trying docker login anyway" ;;
esac

printf '%s' "$K" | sudo -u sparklab -H docker login nvcr.io -u '$oauthtoken' --password-stdin
rc=$?; unset K; exit $rc
