"""Name validation and argv-template execution. No shell strings, ever."""
from __future__ import annotations
import re, subprocess, shutil, time
from dataclasses import dataclass, field, asdict

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

def validate_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValueError(f"invalid name {name!r}: must match {NAME_RE.pattern}")
    return name

@dataclass
class CmdResult:
    command: list[str]
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    at: float = field(default_factory=time.time)
    parsed: object | None = None
    def to_dict(self):
        d = asdict(self); return d

def run_argv(argv: list[str], timeout: float = 30) -> CmdResult:
    """Run a fixed argv (list) with no shell. Missing binary -> exit 127."""
    t0 = time.time()
    if shutil.which(argv[0]) is None:
        return CmdResult(argv, False, 127, "", f"{argv[0]}: not found", 0)
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return CmdResult(argv, p.returncode == 0, p.returncode, p.stdout, p.stderr, int((time.time()-t0)*1000))
    except subprocess.TimeoutExpired as e:
        return CmdResult(argv, False, 124, e.stdout or "", "timeout", int((time.time()-t0)*1000))
