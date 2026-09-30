from __future__ import annotations
from ..security import run_argv, validate_name, CmdResult

def version() -> CmdResult:
    return run_argv(["nemoclaw", "--version"])

def status(sandbox: str) -> CmdResult:
    return run_argv(["nemoclaw", validate_name(sandbox), "status"])

def policy_list(sandbox: str) -> CmdResult:
    r = run_argv(["nemoclaw", validate_name(sandbox), "policy", "list"])
    r.parsed = [l.strip().split()[0] for l in r.stdout.splitlines() if l.strip() and not l.lower().startswith("name")]  # // VERIFY output format
    return r

def policy_add_dry_run(sandbox: str, preset: str) -> CmdResult:
    return run_argv(["nemoclaw", validate_name(sandbox), "policy", "add", validate_name(preset), "--dry-run"])

def policy_add(sandbox: str, preset: str) -> CmdResult:
    return run_argv(["nemoclaw", validate_name(sandbox), "policy", "add", validate_name(preset), "--yes"])
