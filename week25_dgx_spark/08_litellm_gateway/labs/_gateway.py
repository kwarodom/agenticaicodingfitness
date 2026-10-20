"""Module 08 helper (not a lab): build the LiteLLM config and run the proxy as a child process.

Shared by lab01–lab03 so there is ONE config generator. Standard library + PyYAML only.
The proxy binary is week25/.venv-litellm/bin/litellm (LiteLLM 1.89 with the [proxy] extra).
"""
from __future__ import annotations

import atexit
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import PORTS, WEEK, mode, models, note, up, url, warn  # noqa: E402

LITELLM = WEEK / ".venv-litellm" / "bin" / "litellm"
RUNS = Path(__file__).resolve().parents[1] / ".runs"            # gitignored (**/.runs/)
CONFIG = RUNS / "litellm.config.yaml"
LAPTOP = "http://localhost:11434/v1"

# What each engine serves by default in this course (the model id the engine reports at /v1/models).
# lab01 asks the live engine first and uses these only when it is down.
DEFAULT_MODEL = {
    "ollama": "gpt-oss:20b",                         # Module 03 (Open WebUI playbook's first model)
    "vllm": "nvidia/Llama-3.1-8B-Instruct-FP8",       # Module 05 (vLLM playbook support matrix)
    "sglang": "Qwen/Qwen3-8B",                        # Module 06 (SGLang playbook MODEL_HANDLE)
    "trtllm": "nvidia/Llama-3.1-8B-Instruct-FP4",     # Modules 06–07 (TRT-LLM playbook default)
}
# LiteLLM provider prefix per engine. All four speak the OpenAI API; vLLM has its own provider.
PROVIDER = {"ollama": "openai", "vllm": "hosted_vllm", "sglang": "openai", "trtllm": "openai"}


def spark_base(kind: str, which: str) -> str:
    """The engine's URL on Spark A/B, or a placeholder host (spark-a / spark-b) when none is configured."""
    return url(kind, which) or f"http://spark-{which}:{PORTS[kind]}/v1"


def served_model(kind: str, which: str) -> tuple[str, str]:
    """(model id, where it came from): ask the live engine, else the course default."""
    base = url(kind, which)
    if base and mode() == "live" and up(base, 2):
        ids = models(base, timeout=3)
        if ids:
            return ids[0], "live /v1/models"
    return DEFAULT_MODEL[kind], "course default"


def deployment(alias: str, kind: str, which: str, model: str) -> dict:
    return {"model_name": alias,
            "litellm_params": {"model": f"{PROVIDER[kind]}/{model}", "api_base": spark_base(kind, which),
                               "api_key": "none"}}


def laptop_deployments() -> list[dict]:
    """The laptop fallbacks. nemotron-3-nano thinks by default: extra_body turns that off in Ollama."""
    return [
        {"model_name": "laptop-fast",
         "litellm_params": {"model": "openai/gemma3:4b", "api_base": LAPTOP, "api_key": "none"}},
        {"model_name": "laptop-nemotron",
         "litellm_params": {"model": "openai/nemotron-3-nano:latest", "api_base": LAPTOP, "api_key": "none",
                            "extra_body": {"reasoning_effort": "none"}}},
    ]


def build_config(*, demo: bool = False) -> tuple[dict, list[list[str]]]:
    """The whole gateway config + a table of where each alias points. demo=True adds `laptop-pair`."""
    ml, rows = [], []
    for which in ("a", "b"):
        for kind in ("ollama", "vllm", "sglang", "trtllm"):
            model, src = served_model(kind, which)
            ml.append(deployment(f"spark-{which}/{kind}", kind, which, model))
            rows.append([f"spark-{which}/{kind}", f"{PROVIDER[kind]}/{model}", spark_base(kind, which), src])
    # `chat`: ONE alias, TWO deployments (vLLM on both Sparks) → LiteLLM load-balances between them.
    for which in ("a", "b"):
        model, _ = served_model("vllm", which)
        ml.append(deployment("chat", "vllm", which, model))
        rows.append(["chat", f"hosted_vllm/{model}", spark_base("vllm", which), "load-balanced A + B"])
    ml += laptop_deployments()
    rows += [["laptop-fast", "openai/gemma3:4b", LAPTOP, "fallback"],
             ["laptop-nemotron", "openai/nemotron-3-nano:latest", LAPTOP, "fallback"]]
    if demo:
        # two names for the SAME laptop Ollama, standing in for two Sparks, to watch the balancer pick
        for base in (LAPTOP, "http://127.0.0.1:11434/v1"):
            ml.append({"model_name": "laptop-pair",
                       "litellm_params": {"model": "openai/gemma3:4b", "api_base": base, "api_key": "none"}})
            rows.append(["laptop-pair", "openai/gemma3:4b", base, "demo: same Ollama, two names"])
        # one group, two deployments: Spark A's vLLM (down unless your Spark serves it) + the laptop → cooldown demo
        model, _ = served_model("vllm", "a")
        ml.append(deployment("one-spark-down", "vllm", "a", model))
        ml.append({"model_name": "one-spark-down",
                   "litellm_params": {"model": "openai/gemma3:4b", "api_base": LAPTOP, "api_key": "none"}})
        rows += [["one-spark-down", f"hosted_vllm/{model}", spark_base("vllm", "a"), "demo: cooldown"],
                 ["one-spark-down", "openai/gemma3:4b", LAPTOP, "demo: cooldown"]]
    fallbacks = [{"chat": ["laptop-fast"]}] + [{f"spark-{w}/{k}": ["laptop-fast"]}
                                               for w in ("a", "b") for k in ("ollama", "vllm", "sglang", "trtllm")]
    cfg = {
        "model_list": ml,
        "router_settings": {"routing_strategy": "simple-shuffle", "num_retries": 1, "timeout": 120,
                            "allowed_fails": 1, "cooldown_time": 30, "fallbacks": fallbacks},
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }
    return cfg, rows


def for_spark(cfg: dict, laptop_url: str = "") -> dict:
    """The same config for a gateway running ON Spark A: Spark A's engines become localhost, Spark B stays
    remote, and the laptop fallback points at laptop_url (the laptop's tailnet name) — or is removed."""
    import copy
    out = copy.deepcopy(cfg)
    a_host = f"//{url('vllm', 'a').split('//')[-1].split(':')[0]}:" if url("vllm", "a") else "//spark-a:"
    keep = []
    for d in out["model_list"]:
        p = d["litellm_params"]
        if p["api_base"].startswith(LAPTOP) or p["api_base"].startswith("http://127.0.0.1:11434"):
            if not laptop_url:
                continue                                   # no route back to the laptop: drop the fallback
            p["api_base"] = laptop_url.rstrip("/")
        else:
            p["api_base"] = p["api_base"].replace(a_host, "//localhost:")
        keep.append(d)
    out["model_list"] = keep
    if not laptop_url:
        out["router_settings"]["fallbacks"] = [{"chat": ["spark-a/ollama"]}]     # stay on the Spark
    return out


def write_config(cfg: dict, path: Path = CONFIG) -> Path:
    import yaml
    path.parent.mkdir(exist_ok=True)
    head = ("# LiteLLM gateway for Week 25 — generated by week25/08_litellm_gateway/labs/lab01_build_config.py\n"
            "# Start:  LITELLM_MASTER_KEY=sk-… litellm --config litellm.config.yaml --port 4000\n")
    path.write_text(head + yaml.safe_dump(cfg, sort_keys=False, width=110), encoding="utf-8")
    return path


# ── the proxy process ─────────────────────────────────────────────────────────
def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def pick_port(preferred: int = PORTS["litellm"]) -> int:
    for p in range(preferred, preferred + 50):
        if port_free(p):
            if p != preferred:
                warn(f"port {preferred} is taken on this laptop — using {p} instead")
            return p
    raise RuntimeError("no free port near 4000")


def redact(key: str) -> str:
    return key[:7] + "…" if key else "(none)"


class Proxy:
    """`litellm --config … --port …` as a child process. Always stopped on exit (atexit + SIGTERM/SIGINT)."""

    def __init__(self, config: Path, port: int | None = None):
        self.config, self.port = config, port or pick_port()
        self.key = "sk-w25-" + secrets.token_urlsafe(12)          # a fresh master key per run, never printed
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = RUNS / f"litellm-{self.port}.log"
        self.proc: subprocess.Popen | None = None

    def start(self, wait_s: float = 90) -> "Proxy":
        if not LITELLM.exists():
            raise SystemExit(f"✕ {LITELLM} not found — create it: uv venv week25/.venv-litellm && "
                             "uv pip install --python week25/.venv-litellm/bin/python 'litellm[proxy]==1.89.0'")
        cmd = [str(LITELLM), "--config", str(self.config), "--host", "127.0.0.1", "--port", str(self.port),
               "--telemetry", "False"]
        shown = " ".join(cmd).replace(str(WEEK.parent) + "/", "")       # repo-relative paths, key never on argv
        print(f"$ LITELLM_MASTER_KEY={redact(self.key)} {shown}")
        env = {**os.environ, "LITELLM_MASTER_KEY": self.key, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
        self.log.parent.mkdir(exist_ok=True)
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log.open("w"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
        atexit.register(self.stop)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: sys.exit(130))
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.proc.poll() is not None:
                tail = self.log.read_text(errors="replace")[-800:]
                raise SystemExit(f"✕ litellm exited with code {self.proc.returncode}:\n{tail}")
            try:
                with urlopen(self.base + "/health/liveliness", timeout=1) as r:   # noqa: S310
                    if r.status == 200:
                        note(f"gateway up on {self.base} after {time.time() - t0:.1f} s (pid {self.proc.pid}, "
                             f"log {self.log.relative_to(WEEK)})")
                        return self
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise SystemExit(f"✕ litellm did not answer /health/liveliness within {wait_s:.0f} s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            print(f"◆ gateway stopped (pid {self.proc.pid}); port {self.port} is free again: {port_free(self.port)}")
        self.proc = None

    def __enter__(self) -> "Proxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def raw(method: str, full_url: str, key: str | None, body: dict | None = None, timeout: float = 180):
    """HTTP call that returns (status, headers, json) instead of raising on 4xx — to SHOW rejections."""
    import json
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    req = Request(full_url, data=json.dumps(body).encode() if body else None, method=method, headers=hdr)
    try:
        with urlopen(req, timeout=timeout) as r:                  # noqa: S310
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except ValueError:
            payload = {}
        return e.code, dict(e.headers), payload
