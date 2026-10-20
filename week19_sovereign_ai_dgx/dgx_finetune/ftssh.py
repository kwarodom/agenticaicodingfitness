#!/usr/bin/env python3
"""Run real commands ON the DGX over SSH, streaming output back to the tutorial.

The laptop tutorial can't run docker/training locally — those live on the DGX.
This helper shells out to the system ``ssh``/``scp`` (so it uses your existing keys
and ssh-agent) to execute remote commands and stream their stdout/stderr live, and
to push files (the dataset + training script) to the DGX.

SSH config comes from env (injected by the server from the UI panel):
    FT_SSH_HOST · FT_SSH_USER · FT_SSH_PORT (opt) · FT_SSH_KEY (opt path)
    FT_WORKDIR  — remote working dir (default ~/dgx_finetune_demo)
Key-based auth only (BatchMode=yes) — no interactive password prompts.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _bootstrap_from_settings() -> None:
    """Load saved SSH config from .settings.json when it isn't in the environment.

    So running a demo directly in a terminal works the same as via the web app
    (which sets these env vars in its own process). Secrets are never persisted,
    so only non-secret SSH fields are restored here.
    """
    import json
    f = Path(__file__).resolve().parent / ".settings.json"
    if not f.exists():
        return
    try:
        ssh = (json.loads(f.read_text()) or {}).get("ssh") or {}
    except Exception:
        return
    for k, env in {"host": "FT_SSH_HOST", "user": "FT_SSH_USER", "port": "FT_SSH_PORT",
                   "key": "FT_SSH_KEY", "workdir": "FT_WORKDIR",
                   "hf_model": "FT_HF_MODEL"}.items():
        v = ssh.get(k)
        if v and not os.environ.get(env):
            os.environ[env] = v


_bootstrap_from_settings()

# Validate the workdir against a strict allowlist (no shell metacharacters), so it
# is safe to interpolate unquoted into remote commands AND ~ still expands.
_WORKDIR_RE = re.compile(r"^[A-Za-z0-9._/~-]+$")


def _safe_workdir(w: str) -> str:
    w = (w or "").strip() or "~/dgx_finetune_demo"
    return w if _WORKDIR_RE.match(w) else "~/dgx_finetune_demo"


WORKDIR = _safe_workdir(os.environ.get("FT_WORKDIR", "~/dgx_finetune_demo"))


def cfg() -> dict:
    return {
        "host": os.environ.get("FT_SSH_HOST", "").strip(),
        "user": os.environ.get("FT_SSH_USER", "").strip(),
        "port": os.environ.get("FT_SSH_PORT", "").strip(),
        "key": os.environ.get("FT_SSH_KEY", "").strip(),
    }


def ready() -> bool:
    c = cfg()
    return bool(c["host"] and c["user"])


def target() -> str:
    c = cfg()
    return f"{c['user']}@{c['host']}"


def _ssh_base() -> list[str]:
    c = cfg()
    parts = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=12"]
    if c["port"]:
        parts += ["-p", c["port"]]
    if c["key"]:
        parts += ["-i", os.path.expanduser(c["key"])]
    parts.append(target())
    return parts


def _scp_base() -> list[str]:
    c = cfg()
    parts = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=12"]
    if c["port"]:
        parts += ["-P", c["port"]]
    if c["key"]:
        parts += ["-i", os.path.expanduser(c["key"])]
    return parts


def _p(line: str = "") -> None:
    print(line, flush=True)


def require() -> bool:
    """Print guidance (and return False) if SSH isn't configured."""
    if ready():
        return True
    _p("⚠  No DGX SSH connection configured for fine-tuning.")
    _p("")
    _p("   ❗ Use the  🖥️ DGX SSH  button (top-right) — NOT the 🔌 Connection panel.")
    _p("      • 🔌 Connection = the inference URL used for EVAL (HTTP API)")
    _p("      • 🖥️ DGX SSH    = a real shell on the DGX, used to TRAIN (this step)")
    _p("")
    _p("   In the 🖥️ DGX SSH panel, set:")
    _p("     • Host  (e.g. 192.168.68.123)     • User  (your DGX login)     • key path (optional)")
    _p("   then click Save (the pill turns green).  Key-based SSH only — test first:")
    _p("     ssh <user>@<host> nvidia-smi -L")
    return False


def run(remote_cmd: str, *, echo: bool = True) -> int:
    """Run one remote command over SSH, streaming stdout+stderr. Returns exit code."""
    if not ready():
        _p("✗ SSH not configured"); return 2
    if echo:
        _p(f"🖥️  $ ssh {target()} \"{_short(remote_cmd)}\"")
    proc = subprocess.Popen(_ssh_base() + [remote_cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        _p(f"   ↳ remote exit code {proc.returncode}")
    return proc.returncode


def write_remote(remote_path: str, content: str) -> int:
    """Write `content` to a remote file with strict (077) perms, via STDIN.

    Used for secrets (e.g. an HF token env-file): the value never appears in argv,
    the command echo, or the remote process list. `remote_path` must be caller-
    controlled and free of shell metacharacters.
    """
    if not ready():
        _p("✗ SSH not configured"); return 2
    cmd = _ssh_base() + [f"umask 077; cat > {remote_path}"]
    try:
        return subprocess.run(cmd, input=content, text=True).returncode
    except Exception as e:  # noqa: BLE001
        _p(f"   write_remote failed: {e}"); return 1


def push(local_path: str, remote_rel: str) -> int:
    """scp a local file to FT_WORKDIR/remote_rel on the DGX (mkdir -p first)."""
    if not ready():
        _p("✗ SSH not configured"); return 2
    run(f"mkdir -p {WORKDIR}", echo=False)
    dest = f"{target()}:{WORKDIR}/{remote_rel}"
    _p(f"💻→🖥️  scp {os.path.basename(local_path)}  →  {WORKDIR}/{remote_rel}")
    r = subprocess.run(_scp_base() + [local_path, dest])
    return r.returncode


def test() -> int:
    """Confirm SSH works and the GPU is visible."""
    _p(f"Testing SSH to {target()} …")
    return run("echo '✓ ssh ok'; nvidia-smi -L || echo '(no nvidia-smi — is this a GPU box?)'")


def _short(s: str, n: int = 90) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


if __name__ == "__main__":
    sys.exit(0 if test() == 0 else 1)
