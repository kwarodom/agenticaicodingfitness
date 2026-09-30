#!/usr/bin/env python3
"""benchkit — small helpers shared by the Module 06 benchmarking labs (module-local, on top of clawkit).

Nothing here talks to a Spark. It reads the files `nat eval` / `nat sizing calc` write, computes percentiles
the same way everywhere, keeps one JSON summary per benchmark layer in 06_benchmarking/.runs/, and runs two
small local helpers a lab may need: the greenlet stand-in `nat serve` needs on this laptop, and a tiny
reverse proxy (lab 06-4's stand-in for an extra network hop).
"""
from __future__ import annotations

import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from clawkit import NAT_PY, ROOT, laptop, warn  # noqa: E402

MOD = Path(__file__).resolve().parent               # …/week26/06_benchmarking
RUNS = MOD / ".runs"
CONFIGS = MOD / "configs"
DATA = MOD / "data"
CSV = ROOT / "week26" / "common" / "data" / "chiller_plant.csv"


def rel(p: Path) -> str:
    """Path relative to the repo root — NAT configs here use repo-root-relative paths."""
    return str(Path(p).resolve().relative_to(ROOT))


# ── percentiles: one definition for every lab (linear interpolation, like numpy's default) ──
def percentile(xs: list[float], p: float) -> float:
    s = sorted(float(x) for x in xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def awk_percentile(xs: list[float], p: float) -> float:
    """The research tutorial's `sort -n | awk '{a[NR]=$1} END {print a[int(NR*p)]}'` — nearest rank, 1-based."""
    s = sorted(xs)
    i = int(len(s) * p / 100.0)
    return s[max(0, min(len(s) - 1, i - 1))] if s else float("nan")


# ── the chiller CSV, computed the same way the chiller_kpi tool does ──
def chiller_kpi(hours: int) -> dict:
    with CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-max(1, int(hours)) * 4:]
    kw = sum(float(r["kw"]) for r in rows) / len(rows)
    rt = sum(float(r["rt"]) for r in rows) / len(rows)
    return {"hours": hours, "kw": kw, "rt": rt, "kw_per_rt": kw / rt, "status": "ALARM" if kw / rt > 0.80 else "OK"}


# ── reading what `nat eval` wrote ──
def read_profile(out_dir: Path) -> dict:
    """p90/p95/p99 + CI from inference_optimization.json, p50 + tokens from standardized_data_all.csv."""
    out_dir = Path(out_dir)
    res: dict = {"dir": str(out_dir)}
    io = out_dir / "inference_optimization.json"
    if io.is_file():
        d = json.loads(io.read_text(encoding="utf-8"))
        ci = d.get("confidence_intervals", {})
        wf = ci.get("workflow_run_time_confidence_intervals") or {}
        ll = ci.get("llm_latency_confidence_intervals") or {}
        res["wf"] = {k: wf.get(k) for k in ("n", "mean", "p90", "p95", "p99", "ninety_fifth_interval")}
        res["llm"] = {k: ll.get(k) for k in ("n", "mean", "p90", "p95", "p99")}
        res["token_uniqueness"] = d.get("token_uniqueness")
    sd = out_dir / "standardized_data_all.csv"
    if sd.is_file():
        starts, ends, prompt, completion = {}, {}, [], []
        with sd.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                et, ex = r.get("event_type"), r.get("example_number")
                ts = float(r.get("event_timestamp") or 0)
                if et == "WORKFLOW_START":
                    starts[ex] = min(ts, starts.get(ex, ts))
                elif et == "WORKFLOW_END":
                    ends[ex] = max(ts, ends.get(ex, ts))
                elif et == "LLM_END":
                    prompt.append(int(float(r.get("prompt_tokens") or 0)))
                    completion.append(int(float(r.get("completion_tokens") or 0)))
        runtimes = [ends[k] - starts[k] for k in ends if k in starts]
        res["runtimes"] = runtimes
        res["p50_runtime"] = percentile(runtimes, 50) if runtimes else None
        res["llm_calls"] = len(prompt)
        res["prompt_tokens"] = prompt
        res["completion_tokens"] = completion
    return res


# ── one JSON summary per benchmark layer (lab 06-5 puts them side by side) ──
def save_summary(layer: str, data: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"summary_{layer}.json"
    p.write_text(json.dumps({"layer": layer, "date": time.strftime("%Y-%m-%d %H:%M"),
                             "source": "LAPTOP STAND-IN", **data}, indent=1), encoding="utf-8")
    return p


def load_summary(layer: str) -> dict | None:
    p = RUNS / f"summary_{layer}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except Exception:  # noqa: BLE001
        return None


# ── `nat serve` on this laptop needs greenlet; .venv-nat does not ship it ──
GREENLET_STUB = '''"""Stand-in for the missing `greenlet` package (Week 26 · Module 06).
NAT 1.9's FastAPI front end imports sqlalchemy.ext.asyncio at start-up, which refuses to import without greenlet.
The async job store that would use it only starts when Dask is installed — it is not in .venv-nat — so nothing
ever calls into this stub. On the Spark, `uv pip install greenlet` into the NAT environment instead."""


class greenlet:  # noqa: N801
    def __init__(self, *a, **k):
        raise RuntimeError("greenlet stand-in: async job store is not available in this environment")


def getcurrent():
    return object.__new__(greenlet)
'''


def nat_serve_env() -> dict:
    """{} if .venv-nat can import greenlet, else a PYTHONPATH that adds a clearly labelled stand-in."""
    r = laptop([NAT_PY, "-c", "import greenlet"], quiet=True, show="week26/.venv-nat/bin/python -c 'import greenlet'")
    if r.ok:
        return {}
    shim = RUNS / "pyshim"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "greenlet.py").write_text(GREENLET_STUB, encoding="utf-8")
    warn("greenlet is missing from week26/.venv-nat, and `nat serve` will not start without it. This lab adds a stub "
         f"on PYTHONPATH ({rel(shim)}/greenlet.py). No async jobs run here, so the stub is never called.")
    return {"PYTHONPATH": str(shim)}


# ── lab 06-4's stand-in for "one more hop + a policy check" between the agent and the model ──
class HopProxy:
    """A tiny reverse proxy: allow-list check on method+path, forward to `upstream`, time its own overhead.

    It is NOT OpenShell (no TLS interception, no Landlock, no network namespace). It only gives the laptop
    a second path to the same model so the tax *method* (same eval, two paths, compare p95) runs for real.
    """
    ALLOW = {("POST", "/v1/chat/completions"), ("GET", "/v1/models")}

    def __init__(self, upstream: str, port: int):
        self.upstream, self.port = upstream.rstrip("/"), port
        self.allowed = self.denied = 0
        self.overhead_ms: list[float] = []
        self.upstream_ms: list[float] = []
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _go(self, method: str):
                t0 = time.perf_counter()
                path = self.path.split("?", 1)[0]
                if (method, path) not in proxy.ALLOW:
                    proxy.denied += 1
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b'{"error":"denied by hop proxy policy"}')
                    return
                proxy.allowed += 1
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                req = Request(proxy.upstream + path.removeprefix("/v1"), data=body, method=method,
                              headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
                t1 = time.perf_counter()
                try:
                    with urlopen(req, timeout=600) as r:            # noqa: S310 — laptop Ollama only
                        data, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "application/json")
                except HTTPError as e:
                    data, status, ctype = e.read(), e.code, "application/json"
                t2 = time.perf_counter()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                t3 = time.perf_counter()
                proxy.upstream_ms.append((t2 - t1) * 1000)
                proxy.overhead_ms.append(((t1 - t0) + (t3 - t2)) * 1000)

            def do_POST(self):  # noqa: N802
                self._go("POST")

            def do_GET(self):  # noqa: N802
                self._go("GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        print(f"■ stopped hop proxy on :{self.port}")
        return False
