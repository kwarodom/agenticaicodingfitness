#!/usr/bin/env python3
"""natkit — Module 04's small helper on top of clawkit: run the laptop NAT CLI and read what it says.

NAT 1.9 prints a lot: an authlib deprecation warning on every start, a configuration summary, colour codes and
the full LangChain message object for every tool call. `nat()` runs the real CLI (through clawkit.laptop), keeps
the whole log in .runs/, and prints the lines a learner needs: the agent's thoughts, the tool calls, the tool
responses, the workflow result and any error. Nothing is invented — every printed line came from NAT.

    from natkit import nat, workflow_result, RUNS, CONFIGS
    r, secs = nat(["run", "--config_file", CONFIGS / "hello.laptop.yml", "--input", "What time is it?"])
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from clawkit import ANSI, NAT, NAT_PY, ROOT, WEEK, laptop, note, warn  # noqa: E402

MOD = Path(__file__).resolve().parent
RUNS = MOD / ".runs"
CONFIGS = MOD / "configs"
TOOLS = MOD / "tools"
ALTO_CONFIGS = WEEK / "common" / "alto_ops" / "src" / "alto_ops" / "configs"
LAPTOP_YML = ALTO_CONFIGS / "workflow.laptop.yml"
SANDBOX_YML = ALTO_CONFIGS / "workflow.sandbox.yml"
BMS_SERVER = WEEK / "common" / "bms_mcp_server.py"
CSV = WEEK / "common" / "data" / "chiller_plant.csv"
QUIET_ENV = {"PYTHONWARNINGS": "ignore"}          # silences the authlib DeprecationWarning NAT 1.9 prints

NOISE = ("AuthlibDeprecationWarning", "It will be compatible before version 2.0.0", "from authlib.jose import",
         "Using provided input_schema for multi-argument function", "Received session ID",
         "Negotiated protocol version", "Shared workflow built", "Starting NAT from config file",
         "Execution complete.")
KEEP_INFO = ("Calling tools", "Agent input", "Agent's thoughts", "Tool's input", "Tool's response",
             "Workflow Result", "Adding tool", "Calling tool", "Configured to use MCP server")


def rel(p) -> str:
    """Paths as a learner types them from the repo root."""
    s = str(p)
    return s.replace(str(ROOT) + "/", "")


def clean(text: str) -> str:
    return ANSI.sub("", text).replace("\x1b", "")


def filtered(out: str, *, summary: bool = False, width: int = 200) -> list[str]:
    """The lines worth reading from a NAT log (the rest stays in the .runs log)."""
    keep, skip_summary = [], False
    has_final_error = "\nError:" in clean(out)
    for ln in clean(out).splitlines():
        s = ln.rstrip()
        if any(n in s for n in NOISE):
            continue
        if s.strip() == "Configuration Summary:":
            skip_summary = not summary
        if skip_summary:
            if not s.strip():
                skip_summary = False
            continue
        if has_final_error and " - ERROR " in s:          # NAT logs the same exception 5×; keep the last line
            continue
        if " - WARNING " in s:
            s = "WARNING " + s.split(" - ", 3)[-1]
        if " - INFO " in s or " - DEBUG " in s:
            if not any(k in s for k in KEEP_INFO):
                continue
            s = s.split(" - ", 3)[-1]                       # drop timestamp + logger name
        if not s.strip() or set(s.strip()) <= set("-"):
            continue
        if "Tool's input: content=" in s:                  # the whole LangChain AIMessage — show the calls only
            calls = re.findall(r"'function': \{'name': '([^']+)', 'arguments': '([^']*)'", s)
            s = "Tool's input: " + ", ".join(f"{n}({a})" for n, a in calls) if calls else s
        keep.append(s if len(s) <= width else s[:width - 1] + "…")
    return keep


def nat(args: list, *, show: str = "", env: dict | None = None, timeout: float = 600, summary: bool = False,
        log: str = "", prefix: list | None = None, echo: bool = True) -> tuple:
    """Run `nat <args>` on this laptop for real. Returns (Result, seconds). Full log → .runs/<log>."""
    RUNS.mkdir(parents=True, exist_ok=True)
    argv = (prefix or [NAT]) + [str(a) for a in args]
    shown = show or "week26/.venv-nat/bin/nat " + " ".join(
        (f"'{rel(a)}'" if (" " in rel(a) or "?" in rel(a)) else rel(a)) for a in args)
    t0 = time.perf_counter()
    r = laptop(argv, quiet=True, timeout=timeout, env={**QUIET_ENV, **(env or {})}, cwd=ROOT, show=shown)
    secs = time.perf_counter() - t0
    if log:
        (RUNS / log).write_text(clean(r.out), encoding="utf-8")
    if echo:
        for ln in filtered(r.out, summary=summary):
            print("  " + ln)
    return r, secs


def nat_with_tools(args: list, modules: list[str], **kw) -> tuple:
    """`nat <args>` with extra module-local tool modules imported first (registration happens on import).

    NAT finds installed packages through the `nat.components` entry point; a module that is NOT installed can
    still register itself if it is imported before the CLI starts. That is all this does."""
    code = ("import sys; sys.path.insert(0, %r); " % str(TOOLS) + "".join(f"import {m}; " for m in modules)
            + "from nat.cli.main import run_cli; sys.argv = ['nat'] + sys.argv[1:]; sys.exit(run_cli())")
    return nat(args, prefix=[NAT_PY, "-c", code], **kw)


def workflow_result(out: str) -> str:
    """The text NAT prints after 'Workflow Result:' (console front end)."""
    t = clean(out)
    if "Workflow Result:" not in t:
        return ""
    body = t.split("Workflow Result:", 1)[1]
    return body.split("\n-----", 1)[0].strip()


def error_line(out: str) -> str:
    for ln in reversed(clean(out).splitlines()):
        if ln.startswith("Error:") or "Error:" in ln[:60]:
            return ln.strip()
    return ""


def error_kind(out: str) -> str:
    """'ReActAgentParsingFailedError' from 'Error: ReActAgentParsingFailedError: …'."""
    m = re.search(r"Error: (\w+(?:Error|Exception))", error_line(out))
    return m.group(1) if m else (error_line(out)[:40] or "no answer")


def greenlet_env() -> tuple[dict | None, str]:
    """`nat serve` in NAT 1.9.0 imports SQLAlchemy's asyncio extension, which needs `greenlet`.

    Returns (env, how): {} when week26/.venv-nat already has greenlet; a PYTHONPATH shim when a matching
    greenlet exists in week25/.venv-nat (same Python 3.12); None when neither — the caller prints the fix."""
    import subprocess
    probe = subprocess.run([str(NAT_PY), "-c", "import greenlet"], capture_output=True, text=True)
    if probe.returncode == 0:
        return {}, "week26/.venv-nat has greenlet"
    donor = WEEK.parent / "week25" / ".venv-nat" / "lib" / "python3.12" / "site-packages" / "greenlet"
    if donor.is_dir():
        shim = RUNS / "pyshim"
        shim.mkdir(parents=True, exist_ok=True)
        link = shim / "greenlet"
        if not link.exists():
            os.symlink(donor, link)
        return {"PYTHONPATH": str(shim)}, "borrowed greenlet from week25/.venv-nat via .runs/pyshim"
    return None, "missing"


GREENLET_FIX = "uv pip install --python week26/.venv-nat/bin/python greenlet"


def post(url: str, body: dict, *, timeout: float = 300) -> tuple[bytes, float]:
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urlopen(req, timeout=timeout) as r:                    # noqa: S310 — our own laptop server
        raw = r.read()
    return raw, time.perf_counter() - t0


def get_json(url: str, timeout: float = 10) -> dict:
    with urlopen(url, timeout=timeout) as r:                    # noqa: S310
        return json.loads(r.read())


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    """NAT's /v1/workflow/full stream: `data: {...}` answer chunks and `intermediate_data: {...}` steps."""
    out = []
    for ln in raw.decode("utf-8", errors="replace").splitlines():
        if ":" not in ln or not ln.strip():
            continue
        kind, payload = ln.split(":", 1)
        try:
            out.append((kind.strip(), json.loads(payload)))
        except json.JSONDecodeError:
            continue
    return out


def docker_daemon() -> str:
    """'' when the Docker daemon answers, else the reason (client missing / daemon not running)."""
    import shutil
    import subprocess
    if not shutil.which("docker"):
        return "docker client not installed"
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                           timeout=20)
    except Exception as e:  # noqa: BLE001
        return f"docker info failed: {e}"
    return "" if r.returncode == 0 else "Docker client installed, daemon not running"


if __name__ == "__main__":
    env, how = greenlet_env()
    note(f"NAT CLI: {rel(NAT)} ({'✓' if NAT.exists() else '✕ missing'}) · nat serve greenlet: {how}")
    if env is None:
        warn(f"fix: {GREENLET_FIX}")
