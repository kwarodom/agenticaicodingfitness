"""OpenShell CLI adapter. Every method is a fixed argv template.
Flags marked // VERIFY should be checked against https://docs.nvidia.com/openshell/ before M1 sign-off."""
from __future__ import annotations
import re
from ..security import run_argv, validate_name, CmdResult

def status() -> CmdResult:
    r = run_argv(["openshell", "status"])
    r.parsed = {"connected": "Connected" in r.stdout}
    return r

def inference_get() -> CmdResult:
    r = run_argv(["openshell", "inference", "get"])
    prov = re.search(r"provider:\s*(\S+)", r.stdout); model = re.search(r"model:\s*(\S+)", r.stdout)
    r.parsed = {"provider": prov.group(1) if prov else None, "model": model.group(1) if model else None}
    return r

def sandbox_list() -> CmdResult:
    r = run_argv(["openshell", "sandbox", "list"])
    rows = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append({"name": parts[0], "status": parts[1].lower()})  # // VERIFY column order
    r.parsed = rows
    return r

def policy_list(sandbox: str) -> CmdResult:
    return run_argv(["openshell", "policy", "list", validate_name(sandbox)])

def logs_tail(sandbox: str, lines: int = 200) -> CmdResult:
    return run_argv(["openshell", "logs", validate_name(sandbox), "--source", "sandbox"], timeout=10)  # // VERIFY --tail is follow-mode

def forward_start(port: int, sandbox: str) -> CmdResult:
    return run_argv(["openshell", "forward", "start", "--background", str(int(port)), validate_name(sandbox)])
