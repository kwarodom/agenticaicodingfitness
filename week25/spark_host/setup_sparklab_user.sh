#!/usr/bin/env bash
# One-time, on the Spark: create the shared `sparklab` account that every Spark Lab Runner laptop
# ssh-es into (SPARK_HOST=sparklab@<spark>.<tailnet>.ts.net). Key-only login; docker + GPU access.
#   sudo week25/spark_host/setup_sparklab_user.sh
# NOTE: docker group membership is root-equivalent on this host. Only add keys of people you trust.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
U=sparklab

# build deps the labs expect but students (no sudo) cannot install: Module 04's llama.cpp build needs the
# OpenSSL/curl headers, or llama-server builds without HTTPS and `-hf` downloads fail; Module 02 needs MPI + perftest.
apt-get install -y -q git clang cmake libcurl4-openssl-dev libssl-dev libopenmpi-dev perftest >/dev/null

id "$U" &>/dev/null || adduser --disabled-password --gecos "Week 25 Spark Lab" "$U"
usermod -aG docker,video,render "$U"

install -d -m 700 -o "$U" -g "$U" "/home/$U/.ssh"
install -m 600 -o "$U" -g "$U" /dev/null "/home/$U/.ssh/authorized_keys.tmp"
[[ -f "/home/$U/.ssh/authorized_keys" ]] || mv "/home/$U/.ssh/authorized_keys.tmp" "/home/$U/.ssh/authorized_keys"
rm -f "/home/$U/.ssh/authorized_keys.tmp"

# sparklab: keys only, but allow ssh -L (Module 01 tunnels the localhost-only dashboard)
cat > /etc/ssh/sshd_config.d/60-sparklab.conf <<'CONF'
Match User sparklab
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AllowTcpForwarding yes
    X11Forwarding no
CONF
install -d -m 755 /run/sshd   # absent until the socket-activated ssh.service first runs
sshd -t
systemctl try-reload-or-restart ssh

# uv + the `hf` CLI for the account (labs expect `hf auth login` / `hf download` on the Spark)
sudo -u "$U" -H bash -lc 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv tool install --upgrade "huggingface_hub[cli]"'

echo
echo "✓ $U ready. Next:"
echo "  sudo -u $U -H bash -lc '~/.local/bin/hf auth login'        # read-only HF token"
echo "  sudo -u $U -H docker login nvcr.io -u '\$oauthtoken'        # NGC API key as password"
echo "  sudo week25/spark_host/add_key.sh <student.pub>             # once per laptop"
