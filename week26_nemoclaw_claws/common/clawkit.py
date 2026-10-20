#!/usr/bin/env python3
"""clawkit — the small, standard-library helper every Week 26 lab uses to drive a claw on your DGX Spark.

Week 26 = NemoClaw on DGX Spark, beginner to expert: OpenShell sandboxes and policy, NeMo Agent Toolkit
(NAT) claws, tracing, benchmarking and hardening. clawkit is Week 25's sparkkit plus the claw stack:
OpenShell / NemoClaw / NAT / Phoenix / OTel ports, a laptop runner for the local NAT and OpenShell CLIs,
and `change()` — the opt-in gate every lab uses before it mutates anything on the Spark.

Everything here is plain Python you can read in ten minutes. Nothing is hidden:
each helper prints the exact shell command or HTTP call it makes.

    from clawkit import banner, step, sh, change, laptop, chat_any

    sh("openshell sandbox list", example=EXAMPLE_LIST)             # read-only → runs in LIVE
    change("nemoclaw my-assistant policy add pypi --yes")           # mutation → only with CLAW_APPLY=1
    laptop([NAT, "--version"])                                      # the NAT CLI on THIS laptop, for real

WHERE a command runs (decided per call, printed on every line of output):

  • ON THE SPARK   — this script is running on the Spark itself (nvidia-smi reports a GB10) → run locally.
  • OVER SSH       — SPARK_HOST is set (e.g. `spark-abcd` or `me@spark-abcd.tailnet.ts.net`) and answers
                     `ssh -o BatchMode=yes` → run there. SPARK_HOST2 is an optional second Spark.
  • DRY            — no Spark reachable, or SPARK_MODE=dry → nothing runs. You see the command, then either
                     a RECORDED transcript (captured from a real Spark with SPARK_RECORD=1) or the
                     REFERENCE output quoted from NVIDIA's playbooks, or an EXAMPLE shape written for the
                     course. All three are clearly labelled; none of them is your machine.
  • ON THIS LAPTOP — `laptop(argv)` runs the course's local tools for real: the NAT CLI
                     (week26/.venv-nat) and the OpenShell CLI parser (week26/.venv-openshell).

HTTP endpoints of a claw on the Spark: vLLM :8000 · Ollama :11434 · NAT REST :8001 · NAT MCP :9901 ·
Phoenix :6006 · OTel collector :4318 · OpenClaw Control UI :18789 · Hermes API :8642 · OpenShell gateway :8080.
When the Spark endpoint is down, `chat_any()` may use Ollama on THIS laptop as a labelled
"LAPTOP STAND-IN" so client-side ideas (tool calls, NAT agents, MCP) still run for real.

Configuration lookup order (never printed): process env → week26/.env.local (the runner's 🖥 Spark dialog
writes it, gitignored) → week25/.env.local (reuse the Spark hosts you set up last week) → repo-root .env.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COMMON = Path(__file__).resolve().parent            # …/week26/common
WEEK = COMMON.parent                                # …/week26
PREV = WEEK.parent / "week25"                       # last week: its .env.local holds your Spark hosts
ROOT = WEEK.parent                                  # …/agenticaicodingfitness
RECORDED = COMMON / "recorded"                      # real Spark transcripts, replayed in DRY mode

# DGX Spark at a glance (NVIDIA DGX Spark spec sheet). Labs use these for "does it fit?" math.
SPEC = {
    "gpu": "NVIDIA GB10 Grace Blackwell Superchip",
    "memory_gb": 128,            # LPDDR5x, unified: CPU and GPU share it
    "mem_bw_gbs": 273,           # the number that bounds single-stream decode speed
    "fp4_pflops": 1.0,           # sparse FP4 peak
    "cpu": "20-core Arm (10× Cortex-X925 + 10× Cortex-A725)",
    "nic": "ConnectX-7, 2× QSFP, 200 Gb/s",
    "storage_tb": 4,
    "os": "DGX OS (Ubuntu 24.04, aarch64)",
}

# The claw stack's ports (research tutorial §4.1 port plan, from the OpenShell / NAT / OpenClaw docs).
PORTS = {"vllm": 8000, "ollama": 11434, "nat": 8001, "nat_mcp": 9901, "phoenix": 6006, "otel": 4318,
         "openclaw": 18789, "hermes": 8642, "gateway": 8080, "litellm": 4000}
OPENAI_KINDS = ("vllm", "ollama", "nat", "hermes", "litellm")    # speak /v1/chat/completions
CHAT_KINDS = ("vllm", "ollama", "nat", "hermes")                 # targets an inline ```spark block may use
LAPTOP_OLLAMA = "http://localhost:11434/v1"
LAPTOP_NAT = "http://localhost:8001/v1"                          # a `nat serve` a Module 04 lab started here

if hasattr(sys.stdout, "reconfigure"):              # Thai text + box glyphs on any terminal
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


# ── configuration ─────────────────────────────────────────────────────────────
def _clean(raw: str) -> str:
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v.strip()


def _file_value(path: Path, name: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(rf"\s*(?:export\s+)?{re.escape(name)}\s*=(.*)$", line)
            if m and not line.lstrip().startswith("#"):
                return _clean(m.group(1))
    except OSError:
        pass
    return ""


def cfg(name: str, default: str = "") -> str:
    """env → week26/.env.local → week25/.env.local → repo-root .env → default."""
    v = os.environ.get(name, "").strip()
    return (v or _file_value(WEEK / ".env.local", name) or _file_value(PREV / ".env.local", name)
            or _file_value(ROOT / ".env", name) or default)


def cfg_source(name: str) -> str:
    if os.environ.get(name, "").strip():
        return "environment"
    if _file_value(WEEK / ".env.local", name):
        return "week26/.env.local"
    if _file_value(PREV / ".env.local", name):
        return "week25/.env.local"
    if _file_value(ROOT / ".env", name):
        return "repo-root .env"
    return ""


HOST_RE = re.compile(r"^[A-Za-z0-9_.@:\-]{1,120}$")


def host(which: str = "a") -> str:
    """ssh target for Spark A (SPARK_HOST) or Spark B (SPARK_HOST2). '' if not configured."""
    h = cfg("SPARK_HOST2" if which == "b" else "SPARK_HOST")
    return h if HOST_RE.match(h or "") else ""


_CACHE: dict = {}


def on_spark() -> bool:
    """True when this script runs ON a DGX Spark (GB10 GPU visible to nvidia-smi)."""
    if "on_spark" not in _CACHE:
        ok = False
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout
                ok = "GB10" in out
            except Exception:  # noqa: BLE001
                ok = False
        _CACHE["on_spark"] = ok or os.environ.get("SPARK_ON_SPARK") == "1"
    return _CACHE["on_spark"]


def _ssh_base(h: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "ServerAliveInterval=15",
            "-o", "StrictHostKeyChecking=accept-new", h]


def reachable(which: str = "a") -> bool:
    """Can we run commands on that Spark right now? (cached for this process)"""
    key = f"reach_{which}"
    if key not in _CACHE:
        if which == "a" and on_spark():
            _CACHE[key] = True
        elif not host(which) or not shutil.which("ssh"):
            _CACHE[key] = False
        else:
            try:
                r = subprocess.run(_ssh_base(host(which)) + ["true"], capture_output=True, timeout=12)
                _CACHE[key] = r.returncode == 0
            except Exception:  # noqa: BLE001
                _CACHE[key] = False
    return _CACHE[key]


def mode() -> str:
    """'live' or 'dry'. SPARK_MODE forces it; otherwise live iff Spark A is reachable."""
    m = os.environ.get("SPARK_MODE", "").lower()
    if m == "dry":
        return "dry"
    if m == "live":
        return "live"
    return "live" if reachable("a") else "dry"


def where(which: str = "a") -> str:
    """'local' (on the Spark) | 'ssh' | 'dry' — where sh() would run a command right now."""
    if mode() == "dry":
        return "dry"
    if which == "a" and on_spark():
        return "local"
    return "ssh" if reachable(which) else "dry"


def label(which: str = "a") -> str:
    w = where(which)
    if w == "local":
        return f"Spark {which.upper()} (this machine)"
    if w == "ssh":
        return f"Spark {which.upper()} · {host(which)}"
    return f"Spark {which.upper()} · DRY"


def allow_laptop() -> bool:
    return os.environ.get("SPARK_ALLOW_LAPTOP", "1") != "0"


# ── pretty output (the Lab Runner colours lines by their first glyph) ─────────
def banner(title: str, sub: str = "", *, status: bool = True) -> None:
    """Lab header. status=False for offline labs/exercises that never touch a Spark."""
    print("━" * 72)
    print(f"━━ {title}")
    if sub:
        print(f"   {sub}")
    print("━" * 72)
    if not status:
        return
    m = mode()
    if m == "live":
        a = label("a")
        b = f" · {label('b')}" if host("b") else ""
        print(f"▣ LIVE · {a}{b}")
    else:
        why = "SPARK_MODE=dry" if os.environ.get("SPARK_MODE", "").lower() == "dry" else \
            ("SPARK_HOST not set" if not host("a") else f"{host('a')} unreachable over ssh")
        print(f"◈ DRY · {why} · commands are shown, not run; outputs are RECORDED, REFERENCE or EXAMPLE (labelled)")


def step(n, text: str) -> None:
    print(f"\n▣ STEP {n} · {text}", flush=True)


def ok(msg: str) -> None:
    print(f"✓ {msg}")


def warn(msg: str) -> None:
    print(f"⚠ {msg}")


def note(msg: str) -> None:
    print(f"◆ {msg}")


def result(msg: str) -> None:
    print(f"═ {msg}")


def table(rows: list[list], headers: list[str]) -> None:
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [min(max(len(r[i]) for r in cols), 52) for i in range(len(headers))]
    clip = lambda s, w: s if len(s) <= w else s[:w - 1] + "…"          # noqa: E731
    fmt = lambda r: "  ".join(clip(c, widths[i]).ljust(widths[i]) for i, c in enumerate(r))  # noqa: E731
    print("│ " + fmt(headers))
    print("│ " + "  ".join("─" * w for w in widths))
    for r in cols[1:]:
        print("│ " + fmt(r))


def bar(value: float, vmax: float, width: int = 28) -> str:
    n = 0 if vmax <= 0 else max(0, min(width, round(width * value / vmax)))
    return "█" * n + "░" * (width - n)


def check(cond: bool, good: str, bad: str) -> bool:
    """Exercise checker line: ✓ or ✕."""
    print(("✓ " + good) if cond else ("✕ " + bad))
    return bool(cond)


# ── running shell commands on a Spark ─────────────────────────────────────────
class Result:
    def __init__(self, code: int, out: str, source: str):
        self.code, self.out, self.source = code, out, source     # source: live | recorded | reference

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def live(self) -> bool:
        return self.source == "live"

    def __repr__(self) -> str:
        return f"Result(code={self.code}, source={self.source!r}, {len(self.out)} chars)"


def _rec_key(which: str, cmd: str) -> str:
    return "sh_" + hashlib.sha1(f"{which}\n{cmd}".encode()).hexdigest()[:20]


def _record(key: str, payload: dict) -> None:
    if os.environ.get("SPARK_RECORD") != "1":
        return
    RECORDED.mkdir(parents=True, exist_ok=True)
    (RECORDED / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _replay(key: str) -> dict | None:
    p = RECORDED / f"{key}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def _redact(text: str) -> str:
    """Strip anything token-shaped before a transcript is written to disk."""
    text = re.sub(r"(hf_|nvapi-|sk-lf-|pk-lf-|sk-|tvly-)[A-Za-z0-9_\-]{8,}", r"\1•••", text)
    text = re.sub(r"(#token=)[A-Za-z0-9_\-.]+", r"\1•••", text)          # OpenClaw dashboard URLs
    return re.sub(r"(?i)((?:token|key|password|secret)\s*[=:]\s*)\S+", r"\1•••", text)


def sh(cmd: str, which: str = "a", *, reference: str = "", example: str = "", timeout: float = 600,
       quiet: bool = False, echo: bool = True) -> Result:
    """Run `cmd` (bash) on Spark A or B, streaming output. DRY → print the command + RECORDED/REFERENCE/EXAMPLE.

    reference: expected output as NVIDIA's playbook documents it — labelled REFERENCE in DRY mode.
    example:   illustrative output shape written for this course (not from a playbook, not a measurement)
               — labelled EXAMPLE. Use it only when the playbook shows no output.
    """
    w = where(which)
    tag = {"local": f"[{label(which)}]", "ssh": f"[ssh {host(which)}]", "dry": "[DRY]"}[w]
    if echo:
        first, *rest = cmd.strip().splitlines() or [""]
        print(f"$ {first}   {tag}")
        for ln in rest:
            print(f"  {ln}")
    key = _rec_key(which, cmd.strip())
    if w == "dry":
        rec = _replay(key)
        if rec:
            if not quiet:
                print(f"◈ RECORDED on {rec.get('host', 'a Spark')} · {rec.get('date', '?')} — replay, not your machine")
                print(rec.get("out", "").rstrip())
            return Result(int(rec.get("code", 0)), rec.get("out", ""), "recorded")
        if not quiet:
            if reference:
                print("◈ REFERENCE — quoted from NVIDIA's playbook / docs (not your machine):")
                print(reference.rstrip("\n"))
            elif example:
                print("◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):")
                print(example.rstrip("\n"))
            else:
                print("◈ (dry run — nothing executed; connect a Spark to see real output)")
        return Result(0, reference or example, "reference" if reference else "example")
    argv = ["bash", "-lc", cmd] if w == "local" else _ssh_base(host(which)) + [f"bash -lc {shlex.quote(cmd)}"]
    start, lines = time.time(), []
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1)
    except FileNotFoundError as e:
        warn(f"could not start: {e}")
        return Result(127, "", "live")
    try:
        for line in proc.stdout:                                # stream as it arrives
            lines.append(line)
            if not quiet:
                print(line, end="", flush=True)
            if time.time() - start > timeout:
                proc.kill()
                warn(f"timed out after {timeout:.0f}s — killed")
                break
        proc.wait(timeout=10)
    except KeyboardInterrupt:
        proc.kill()
        raise
    out = "".join(lines)
    code = proc.returncode if proc.returncode is not None else 124
    _record(key, {"cmd": cmd.strip(), "which": which, "host": label(which), "code": code,
                  "date": time.strftime("%Y-%m-%d"), "out": _redact(out)})
    return Result(code, out, "live")


def put(local: str | Path, remote: str, which: str = "a") -> bool:
    """Copy a file this lab generated onto the Spark (scp). DRY → show the command only."""
    w = where(which)
    local = str(local)
    if w == "local":
        dest = os.path.expanduser(remote)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        print(f"$ cp {local} {remote}   [{label(which)}]")
        return True
    print(f"$ scp {Path(local).name} {host(which) or '<spark>'}:{remote}   [{'DRY' if w == 'dry' else 'scp'}]")
    if w == "dry":
        return False
    rel = remote[2:] if remote.startswith("~/") else remote      # scp and ssh both start in $HOME
    parent = os.path.dirname(rel) or "."
    mk = f'mkdir -p "$HOME"/{shlex.quote(parent)}' if not rel.startswith("/") else f"mkdir -p {shlex.quote(parent)}"
    subprocess.run(_ssh_base(host(which)) + [mk], capture_output=True, timeout=30)
    r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", local, f"{host(which)}:{rel}"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        warn(f"scp failed: {r.stderr.strip()[:200]}")
    return r.returncode == 0


# ── OpenAI-compatible HTTP endpoints on the Spark ─────────────────────────────
def api_host(which: str = "a") -> str:
    """Hostname for HTTP calls. SPARK_API_HOST[2] wins (use 'localhost' with an ssh -L tunnel)."""
    h = cfg("SPARK_API_HOST2" if which == "b" else "SPARK_API_HOST")
    if h and HOST_RE.match(h):
        return h
    if which == "a" and on_spark():
        return "localhost"
    ssh_h = host(which)
    return ssh_h.split("@", 1)[-1] if ssh_h else ""


def url(kind: str, which: str = "a") -> str:
    """Base URL (…/v1 for OpenAI-style servers) for `kind` on Spark A/B. SPARK_URL_<KIND> overrides."""
    override = cfg(f"SPARK_URL_{kind.upper()}" + ("2" if which == "b" else ""))
    if override:
        return override.rstrip("/")
    h = api_host(which)
    if not h:
        return ""
    port = PORTS.get(kind, 8000)
    suffix = "/v1" if kind in OPENAI_KINDS else ""
    return f"http://{h}:{port}{suffix}"


def http_json(method: str, full_url: str, body: dict | None = None, *, timeout: float = 30,
              headers: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(full_url, data=data, method=method,
                  headers={"Content-Type": "application/json", **(headers or {})})
    with urlopen(req, timeout=timeout) as r:                    # noqa: S310 — course-local endpoints
        raw = r.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def models(base: str, *, timeout: float = 4, api_key: str = "") -> list[str]:
    """GET {base}/models → model ids. [] when the server is down."""
    if not base:
        return []
    try:
        hdr = {"Authorization": f"Bearer {api_key}"} if api_key else None
        d = http_json("GET", base.rstrip("/") + "/models", timeout=timeout, headers=hdr)
        return [m.get("id", "") for m in d.get("data", []) if m.get("id")]
    except Exception:  # noqa: BLE001
        return []


def up(base: str, timeout: float = 3) -> bool:
    if not base:
        return False
    try:
        http_json("GET", base.rstrip("/") + "/models", timeout=timeout)
        return True
    except HTTPError as e:                                     # 401 still means "a server is listening"
        return e.code in (401, 403)
    except Exception:  # noqa: BLE001
        return False


def chat(base: str, model: str, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.2,
         tools: list | None = None, stream: bool = True, timeout: float = 180, api_key: str = "",
         extra: dict | None = None, echo: bool = False) -> dict:
    """POST {base}/chat/completions. Streams by default so we can measure time-to-first-token.

    Returns {text, reasoning, tool_calls, ttft_ms, total_ms, out_tokens, in_tokens, tok_s, model}.
    Thinking models (Qwen3, Nemotron, gemma4 …) may put their thoughts in a `reasoning`/`reasoning_content`
    field — collected separately so `text` is only the answer.
    """
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": bool(stream and not tools)}
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
    if extra:
        body.update(extra)
    if echo:
        print(f"→ POST {base}/chat/completions  model={model}  stream={body['stream']}")
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = f"Bearer {api_key}"
    req = Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                  headers=hdr)
    t0 = time.perf_counter()
    ttft = None
    text, reasoning, usage, tool_calls, served = [], [], {}, [], model
    n_chunks = 0
    with urlopen(req, timeout=timeout) as r:                    # noqa: S310
        if not body["stream"]:
            d = json.loads(r.read().decode("utf-8", errors="replace"))
            ttft = (time.perf_counter() - t0) * 1000
            msg = (d.get("choices") or [{}])[0].get("message") or {}
            text.append(msg.get("content") or "")
            reasoning.append(msg.get("reasoning_content") or msg.get("reasoning") or "")
            tool_calls = msg.get("tool_calls") or []
            usage, served = d.get("usage") or {}, d.get("model", model)
        else:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                served = d.get("model", served)
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (piece or think) and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    if piece or think:
                        n_chunks += 1
                    text.append(piece)
                    reasoning.append(think)
    total = (time.perf_counter() - t0) * 1000
    out_tok = int(usage.get("completion_tokens") or n_chunks or 0)
    gen_ms = max(1.0, total - (ttft or 0)) if body["stream"] else max(1.0, total)   # no TTFT without streaming
    return {"text": "".join(text).strip(), "reasoning": "".join(reasoning).strip(), "tool_calls": tool_calls,
            "ttft_ms": round(ttft or total, 1), "total_ms": round(total, 1), "out_tokens": out_tok,
            "in_tokens": int(usage.get("prompt_tokens") or 0),
            "tok_s": round(out_tok / (gen_ms / 1000), 1) if out_tok else 0.0, "model": served,
            "usage_exact": bool(usage.get("completion_tokens"))}


def _http_key(kind: str, model: str, messages: list[dict], max_tokens: int, tools) -> str:
    blob = json.dumps([kind, model, messages, max_tokens, tools], sort_keys=True, ensure_ascii=False)
    return "http_" + hashlib.sha1(blob.encode()).hexdigest()[:20]


def laptop_models() -> list[str]:
    if "laptop" not in _CACHE:
        _CACHE["laptop"] = models(LAPTOP_OLLAMA, timeout=2) if allow_laptop() else []
    return _CACHE["laptop"]


def pick_laptop_model(preferred: list[str] | None = None) -> str:
    """A model that is actually pulled on this laptop's Ollama (prefers small, non-cloud tags)."""
    have = [m for m in laptop_models() if not m.endswith(":cloud") and "-cloud" not in m]
    for want in (preferred or []) + ["nemotron-3-nano", "gemma4:12b", "gemma3:4b", "llama3.2", "qwen3"]:
        for m in have:
            if m == want or m.startswith(want):
                return m
    return have[0] if have else ""


def resolve(kind: str, which: str = "a") -> tuple[str, str]:
    """(base_url, source) for `kind`: the Spark endpoint if it answers, else the laptop stand-in, else ('','dry')."""
    if mode() == "live":
        base = url(kind, which)
        if up(base):
            return base, "spark"
    if kind == "nat" and up(LAPTOP_NAT, 1.5):                   # a real NAT server on this laptop (Module 04)
        return LAPTOP_NAT, "laptop"
    # DRY means "do not touch the Spark". The laptop stand-in is separate: it follows its own switch
    # (SPARK_ALLOW_LAPTOP, the runner's 💻 toggle), so laptop labs still run for real without a Spark.
    if allow_laptop() and laptop_models():
        return LAPTOP_OLLAMA, "laptop"
    return "", "dry"


def chat_any(kind: str, model: str, messages: list[dict], *, which: str = "a", max_tokens: int = 256,
             laptop_model: str | None = None, tools: list | None = None, reference: str = "",
             quiet: bool = False, think: bool = False, **kw) -> dict:
    """chat() against the Spark's `kind` server; fall back to the laptop stand-in, then to DRY replay.

    Every return value carries `source`: spark | laptop | recorded | reference — print it, always.
    think=False sends reasoning_effort="none" to Ollama so thinking models (nemotron, qwen3, gemma4) answer
    directly instead of spending the whole token budget on hidden reasoning.
    """
    base, src = resolve(kind, which)
    key = _http_key(kind, model, messages, max_tokens, tools)
    if src in ("spark", "laptop"):
        use_model = model if (src == "spark" or base == LAPTOP_NAT) else (laptop_model or pick_laptop_model())
        if not quiet:
            where_s = (f"{kind} on {label(which)}" if src == "spark" else
                       "NAT server on THIS laptop (stand-in, not the Spark)" if base == LAPTOP_NAT else
                       "Ollama on THIS laptop (stand-in, not the Spark)")
            print(f"→ POST {base}/chat/completions · model={use_model} · {where_s}")
        if not think and base != LAPTOP_NAT and (src == "laptop" or kind == "ollama"):
            kw["extra"] = {"reasoning_effort": "none", **(kw.get("extra") or {})}
        try:
            r = chat(base, use_model, messages, max_tokens=max_tokens, tools=tools, **kw)
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            detail = e.read().decode(errors="replace")[:300] if isinstance(e, HTTPError) else str(e)
            warn(f"{kind} call failed: {type(e).__name__}: {detail}")
            r = None
        if r is not None:
            r["source"] = src
            if src == "spark":
                _record(key, {"kind": kind, "host": label(which), "date": time.strftime("%Y-%m-%d"), **r})
            return r
    rec = _replay(key)
    if rec:
        rec["source"] = "recorded"
        if not quiet:
            print(f"◈ RECORDED {kind} answer from {rec.get('host', 'a Spark')} · {rec.get('date', '?')} — replay")
        return rec
    if not quiet:
        print(f"◈ DRY — no {kind} endpoint reachable and no recording; showing the REFERENCE text")
    return {"text": reference or "(dry run — start the server on your Spark to get a real answer)",
            "reasoning": "", "tool_calls": [], "ttft_ms": 0, "total_ms": 0, "out_tokens": 0, "in_tokens": 0,
            "tok_s": 0.0, "model": model, "source": "reference", "usage_exact": False}


def show_chat(r: dict, *, max_chars: int = 900) -> None:
    """Print one chat result with its source and speed — the line every serving lab ends with."""
    src = {"spark": "LIVE on the Spark", "laptop": "LAPTOP STAND-IN (not Spark numbers)",
           "recorded": "RECORDED from a Spark", "reference": "REFERENCE (playbook text)"}.get(r.get("source"), "?")
    if r.get("reasoning"):
        print(f"~ REASONING ({len(r['reasoning'])} chars, hidden) — thinking model")
    txt = (r.get("text") or "").strip()
    print("· ANSWER  " + (txt[:max_chars] + (" …" if len(txt) > max_chars else "")).replace("\n", "\n          "))
    if r.get("tool_calls"):
        for tc in r["tool_calls"]:
            fn = tc.get("function") or {}
            print(f"→ tool_call {fn.get('name')}({fn.get('arguments')})")
    if r.get("source") in ("spark", "laptop", "recorded") and r.get("total_ms"):
        approx = "" if r.get("usage_exact") else " (≈, chunk count)"
        print(f"◆ {src} · {r.get('model')} · TTFT {r['ttft_ms']:.0f} ms · {r['out_tokens']} tok{approx} "
              f"in {r['total_ms'] / 1000:.1f}s · {r['tok_s']} tok/s")
    else:
        print(f"◆ {src}")


# ── memory math: "will it fit in 128 GB?" ─────────────────────────────────────
BITS = {"fp32": 32, "bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "nvfp4": 4.5, "mxfp4": 4.25, "int4": 4.5,
        "q8_0": 8.5, "q6_k": 6.6, "q5_k_m": 5.7, "q4_k_m": 4.85, "q3_k_m": 3.9}
"""Effective bits per weight INCLUDING scales (NVFP4 = 4-bit values + one FP8 scale per 16 → ~4.5)."""


def weights_gb(params_b: float, fmt: str = "bf16") -> float:
    bits = BITS.get(fmt.lower(), fmt if isinstance(fmt, (int, float)) else 16)
    return params_b * 1e9 * float(bits) / 8 / 1e9


def kv_cache_gb(layers: int, kv_heads: int, head_dim: int, ctx: int, batch: int = 1, bytes_per: float = 2) -> float:
    """K and V, every layer, every token in the context, every concurrent sequence."""
    return 2 * layers * kv_heads * head_dim * ctx * batch * bytes_per / 1e9


def decode_ceiling_tok_s(active_params_b: float, fmt: str = "bf16", bw_gbs: float = SPEC["mem_bw_gbs"]) -> float:
    """Upper bound for single-stream decode: every token reads every ACTIVE weight once from memory."""
    gb = weights_gb(active_params_b, fmt)
    return bw_gbs / gb if gb else 0.0


# ── the claw stack: laptop tools, sandbox names, the change() gate ────────────
NAT = WEEK / ".venv-nat" / "bin" / "nat"                  # NeMo Agent Toolkit CLI on this laptop
NAT_PY = WEEK / ".venv-nat" / "bin" / "python"
OPENSHELL = WEEK / ".venv-openshell" / "bin" / "openshell"  # OpenShell CLI 0.0.111 — the last PyPI wheel with a
                                                          # macOS binary; used here as an offline policy PARSER
OPENSHELL_PINNED = "0.0.116"                              # what NemoClaw pins on the Spark (research tutorial)
SANDBOX_RE = re.compile(r"^[a-z0-9-]{1,40}$")             # the runner's name rule (spec §4.4)
RESEARCH = WEEK / "NemoClaw on DGX Spark — Beginner to Expert Tutorial with NAT.md"
PLAYBOOKS = ROOT / "dgx-spark-playbooks" / "nvidia"       # local clone (gitignored), used by audit_references


def sandbox(default: str = "my-assistant") -> str:
    """The sandbox the labs talk about: CLAW_SANDBOX from the 🖥 dialog, else `my-assistant` (the docs' name)."""
    v = cfg("CLAW_SANDBOX") or default
    if not SANDBOX_RE.match(v):
        raise SystemExit(f"✕ CLAW_SANDBOX={v!r} is not a valid sandbox name (^[a-z0-9-]{{1,40}}$)")
    return v


def apply_enabled() -> bool:
    """True only when the learner opted in to changes on the Spark (runner 🔓 toggle → CLAW_APPLY=1)."""
    return os.environ.get("CLAW_APPLY") == "1"


def change(cmd: str, which: str = "a", *, preview: str = "", example: str = "", reference: str = "",
           timeout: float = 900) -> Result:
    """A command that CHANGES the Spark (install, onboard, policy add/set, snapshot, create/delete).

    Two-step, like the Alto Reef spec's policy flow: `preview` (a --dry-run or read-only command) always
    runs first when given; the real `cmd` only runs in LIVE mode AND with CLAW_APPLY=1. Otherwise the
    command is printed with the reason it did not run, and the EXAMPLE/REFERENCE is shown as usual.
    """
    if preview:
        sh(preview, which, example=example, reference=reference, timeout=timeout)
    if where(which) != "dry" and not apply_enabled():
        first = cmd.strip().splitlines()[0] if cmd.strip() else ""
        print(f"$ {first}   [NOT RUN]")
        print("→ this changes your Spark — turn on 🔓 Allow changes in the runner (CLAW_APPLY=1) to run it")
        return Result(0, "", "skipped")
    if preview and where(which) == "dry":
        first = cmd.strip().splitlines()[0] if cmd.strip() else ""
        print(f"$ {first}   [DRY]")
        print("◈ (dry run — the change above would be applied here)")
        return Result(0, "", "example")
    return sh(cmd, which, example=example, reference=reference, timeout=timeout)


def laptop(argv: list, *, timeout: float = 600, env: dict | None = None, cwd: str | Path | None = None,
           quiet: bool = False, show: str = "") -> Result:
    """Run a local tool on THIS laptop for real (NAT, the OpenShell parser, docker …), streaming output.

    argv is a list — no shell, no string interpolation. `show` overrides the echoed command line (e.g. to
    print `nat` instead of the full venv path).
    """
    argv = [str(a) for a in argv]
    shown = show or " ".join(shlex.quote(a) for a in argv).replace(str(WEEK) + "/", "week26/")
    print(f"$ {shown}   [this laptop]")
    if not shutil.which(argv[0]) and not Path(argv[0]).exists():
        warn(f"{Path(argv[0]).name} is not installed on this laptop — see the module's '0 · Before you start'")
        return Result(127, "", "laptop")
    run_env = {**os.environ, "NO_COLOR": "1", "PYTHONUNBUFFERED": "1", **(env or {})}
    start, lines = time.time(), []
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            bufsize=1, env=run_env, cwd=str(cwd) if cwd else None)
    try:
        for line in proc.stdout:
            lines.append(line)
            if not quiet:
                print(line, end="", flush=True)
            if time.time() - start > timeout:
                proc.kill()
                warn(f"timed out after {timeout:.0f}s — killed")
                break
        proc.wait(timeout=10)
    except KeyboardInterrupt:
        proc.kill()
        raise
    return Result(proc.returncode if proc.returncode is not None else 124, "".join(lines), "laptop")


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def openshell_offline(args: list[str], *, home: Path | None = None, quiet: bool = True) -> Result:
    """The laptop OpenShell CLI with no gateway behind it (a dead endpoint). Syntax errors come back
    immediately; anything that needs a gateway fails with a connection error — which proves it PARSED."""
    home = home or (WEEK / ".runs" / "openshell-home")
    home.mkdir(parents=True, exist_ok=True)
    r = laptop([OPENSHELL, *args], quiet=quiet, timeout=60, show="openshell " + " ".join(shlex.quote(a) for a in args),
               env={"HOME": str(home), "OPENSHELL_GATEWAY_ENDPOINT": "http://127.0.0.1:9"})
    r.out = ANSI.sub("", r.out)
    return r


def parsed_ok(r: Result) -> bool:
    """True when the offline CLI got past argument/YAML parsing and only failed to reach the gateway."""
    t = " ".join(r.out.split())
    return r.code != 127 and ("Connection refused" in t or "tcp connect error" in t or "transport error" in t)


def free_port(preferred: int, span: int = 50) -> int:
    """The documented port if it is free on this laptop, else the next free one (printed by the caller)."""
    import socket
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as so:
            so.settimeout(0.2)
            if so.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


class background:
    """Start a local server (nat serve, nat mcp serve, the mock BMS …), wait until `ready_url` answers,
    and ALWAYS stop it when the `with` block ends — even on error. Its log goes to `log` (a .runs file).

        with background([NAT, "serve", "--config_file", cfg, "--port", str(port)],
                        ready_url=f"http://localhost:{port}/docs", log=RUNS / "nat_serve.log"):
            …
    """

    def __init__(self, argv: list, *, ready_url: str, log: Path, timeout: float = 180, env: dict | None = None,
                 cwd: str | Path | None = None, show: str = ""):
        self.argv, self.ready_url, self.log, self.timeout = [str(a) for a in argv], ready_url, Path(log), timeout
        self.env, self.cwd, self.show, self.proc = env, cwd, show, None

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        shown = self.show or " ".join(shlex.quote(a) for a in self.argv).replace(str(WEEK) + "/", "week26/")
        print(f"$ {shown} &   [this laptop, background → {self.log.name}]")
        fh = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(self.argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(self.cwd or ROOT),
                                     env={**os.environ, "NO_COLOR": "1", "PYTHONUNBUFFERED": "1", **(self.env or {})},
                                     start_new_session=True)
        t0 = time.time()
        from urllib.error import HTTPError as _HE
        while time.time() - t0 < self.timeout:
            if self.proc.poll() is not None:
                tail = self.log.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"server exited with code {self.proc.returncode} before it was ready:\n{tail}")
            try:
                with urlopen(Request(self.ready_url, method="GET"), timeout=2):      # noqa: S310
                    break
            except _HE:
                break                                           # any HTTP answer = listening
            except Exception:  # noqa: BLE001
                time.sleep(0.7)
        else:
            self.__exit__(None, None, None)
            raise RuntimeError(f"{self.ready_url} did not answer within {self.timeout:.0f}s — see {self.log}")
        ok(f"ready in {time.time() - t0:.1f}s → {self.ready_url}")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            import signal as _sig
            try:
                os.killpg(os.getpgid(self.proc.pid), _sig.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.proc.pid), _sig.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            print(f"■ stopped {Path(self.argv[0]).name} (pid {self.proc.pid})")
        return False


if __name__ == "__main__":                          # python clawkit.py → where am I, what is installed?
    banner("clawkit self-check", "where would commands run, which claw endpoints answer, which laptop tools exist?")
    print(f"◆ on a Spark: {on_spark()} · SPARK_HOST={host('a') or '—'} ({cfg_source('SPARK_HOST') or 'not set'}) · "
          f"SPARK_HOST2={host('b') or '—'}")
    print(f"◆ mode: {mode()} · sandbox: {sandbox()} · changes allowed: {apply_enabled()}")
    for k in ("vllm", "ollama", "nat", "nat_mcp", "phoenix", "otel", "hermes", "openclaw"):
        u = url(k)
        print(f"│ {k:9s} {u or '(no host)':42s} {'● up' if (up(u, 1.5) if k in OPENAI_KINDS else False) else '○ down / not probed'}")
    print(f"│ laptop    {LAPTOP_OLLAMA:42s} {', '.join(laptop_models()[:4]) or '○ none'}")
    print(f"│ NAT CLI   {'week26/.venv-nat/bin/nat':42s} {'✓' if NAT.exists() else '✕ missing'}")
    print(f"│ OpenShell {'week26/.venv-openshell/bin/openshell':42s} {'✓' if OPENSHELL.exists() else '✕ missing'}")
