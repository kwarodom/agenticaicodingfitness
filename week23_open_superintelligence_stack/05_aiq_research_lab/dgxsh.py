#!/usr/bin/env python3
"""🖥️ DGX console backend — run commands ON the DGX over SSH (Tailscale), stream back.

Every Week 23 app ships this file. The web app's "🖥️ DGX console" panel posts
commands here; we shell out to the system ``ssh`` (so your existing keys and
ssh-agent are used) and stream stdout/stderr back to the browser live. The grey
"run on the DGX" blocks in the chapters stop being screenshots.

Config from env (the UI panel can set these at runtime via /api/dgx/config):
    DGX_SSH_HOST  — default your-spark.your-tailnet.ts.net  (your Spark on Tailscale)
    DGX_SSH_USER  — remote username (default your-dgx-user)
    DGX_SSH_PORT  — optional
    DGX_SSH_KEY   — optional identity file path

Key-based auth only (BatchMode=yes) — no password prompts can reach a web app.
First-time setup, once, from a terminal (needs your DGX password):
    ssh-copy-id <user>@your-spark.your-tailnet.ts.net
"""
from __future__ import annotations

import os
import subprocess

DEFAULT_HOST = "your-spark.your-tailnet.ts.net"
DEFAULT_USER = "your-dgx-user"


def target() -> str:
    host = os.environ.get("DGX_SSH_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    user = os.environ.get("DGX_SSH_USER", DEFAULT_USER).strip()
    return f"{user}@{host}" if user else host


def _ssh_argv(command: str) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8"]
    port = os.environ.get("DGX_SSH_PORT", "").strip()
    if port:
        argv += ["-p", port]
    key = os.environ.get("DGX_SSH_KEY", "").strip()
    if key:
        argv += ["-i", os.path.expanduser(key)]
    argv += [target(), command]
    return argv


def popen(command: str) -> subprocess.Popen:
    """Start the remote command; caller streams .stdout and may .kill()."""
    return subprocess.Popen(_ssh_argv(command), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=False)


def status() -> dict:
    """Quick reachability + GPU probe (2 short remote commands in one hop)."""
    try:
        out = subprocess.run(
            _ssh_argv("echo __OK__ $(whoami)@$(hostname); nvidia-smi -L 2>/dev/null | head -4"),
            capture_output=True, text=True, timeout=12)
        text = (out.stdout or "") + (out.stderr or "")
        ok = "__OK__" in text
        return {"target": target(), "ok": ok,
                "info": text.replace("__OK__", "✓ connected as").strip()[:500],
                "hint": "" if ok else
                        f"key auth failed/unreachable — from a terminal run:  "
                        f"ssh-copy-id {target()}   (then retry)"}
    except subprocess.TimeoutExpired:
        return {"target": target(), "ok": False, "info": "",
                "hint": "timeout — is the DGX awake and on your tailnet? (tailscale status)"}
    except Exception as e:  # noqa: BLE001
        return {"target": target(), "ok": False, "info": "", "hint": str(e)}


def apply_config(p: dict) -> None:
    for field, env in (("host", "DGX_SSH_HOST"), ("user", "DGX_SSH_USER"),
                       ("port", "DGX_SSH_PORT"), ("key", "DGX_SSH_KEY")):
        v = (p.get(field) or "").strip()
        if v:
            os.environ[env] = v
        else:
            os.environ.pop(env, None)
