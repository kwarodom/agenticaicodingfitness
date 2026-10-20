#!/usr/bin/env python3
"""Exercise 15 · reference solution — write the RUNTIME policy for the NAT hotel agent.

After setup the agent needs no package index: only its files and its model. Fill in the three TODOs,
save, then run:
    .venv/bin/python week25/15_openshell_sandbox/exercises/ex15_lock_down_the_agent.py

The checker is offline and free. It validates your policy with policykit (the playbook rules), then
replays twelve actions through policykit.decide — a course-made teaching model of OpenShell's
semantics, not OpenShell — and compares each decision with what a locked-down agent should get.
Stuck? Compare with exercises/solutions/ and policies/hotel_agent_policy.yaml.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import policykit as pk  # noqa: E402
from sparkkit import banner, check, table  # noqa: E402


# ── TODO 1 ── filesystem_policy: the OS is readable but never writable (/usr /lib /proc /dev/urandom /etc);
#   the agent may write only /sandbox, /tmp and /dev/null. Keep include_workdir: true.
def filesystem() -> dict:
    return {"include_workdir": True,
            "read_only": ["/usr", "/lib", "/proc", "/dev/urandom", "/etc"],
            "read_write": ["/sandbox", "/tmp", "/dev/null"]}


# ── TODO 2 ── ONE network group named "inference": endpoint inference.local port 443, and a binaries
#   allow-list with the NAT venv's Python ("/sandbox/.venv-nat/bin/python*"). Use pk.group(name, endpoints, binaries).
#   No other groups: no pypi, no curl, nothing else.
def network_policies() -> dict:
    return {"inference": pk.group("inference", [{"host": "inference.local", "port": 443}],
                                  ["/sandbox/.venv-nat/bin/python*"])}


# ── TODO 3 ── process: run as the unprivileged user and group "sandbox".
def process() -> dict:
    return {"run_as_user": "sandbox", "run_as_group": "sandbox"}


# ─────────────────────────── checker — no need to edit below ────────────────
PY = "/sandbox/.venv-nat/bin/python3.12"
EXPECT = [
    ({"op": "write", "path": "/sandbox/tickets.jsonl"}, "allow"),
    ({"op": "read", "path": "/sandbox/14_nat_agents/configs/hotel_agent.yml"}, "allow"),
    ({"op": "read", "path": "/etc/ssl/certs/ca-certificates.crt"}, "allow"),
    ({"op": "write", "path": "/tmp/trace.jsonl"}, "allow"),
    ({"op": "connect", "host": "inference.local", "port": 443, "binary": PY}, "inspect_for_inference"),
    ({"op": "connect", "host": "pypi.org", "port": 443, "binary": "/sandbox/.venv-nat/bin/pip"}, "deny"),
    ({"op": "connect", "host": "api.openai.com", "port": 443, "binary": PY}, "deny"),
    ({"op": "connect", "host": "github.com", "port": 443, "binary": "/usr/bin/curl"}, "deny"),
    ({"op": "read", "path": "/home/sandbox/.ssh/id_ed25519"}, "deny"),
    ({"op": "write", "path": "/etc/cron.d/backdoor"}, "deny"),
    ({"op": "write", "path": "/usr/lib/python3/dist-packages/sitecustomize.py"}, "deny"),
    ({"op": "run_as", "user": "root"}, "deny"),
]


def main() -> None:
    banner("Exercise 15 · lock down the hotel agent", "offline checker · policykit (course-made model, not OpenShell)",
           status=False)
    fs, net, pr = filesystem(), network_policies(), process()
    ok = True
    ok &= check(isinstance(fs, dict) and "/sandbox" in (fs.get("read_write") or []) and "/etc" in (fs.get("read_only") or [])
                and "/etc" not in (fs.get("read_write") or []),
                "filesystem: /sandbox writable · /etc read-only",
                f"TODO 1: filesystem() gave {fs!r}")
    ok &= check(isinstance(net, dict) and list(net) == ["inference"],
                "network_policies: exactly one group, 'inference'",
                f"TODO 2: network_policies() should have one key 'inference' (got {list(net) if isinstance(net, dict) else net!r})")
    ok &= check(isinstance(pr, dict) and pr.get("run_as_user") == "sandbox" and pr.get("run_as_group") == "sandbox",
                "process: runs as sandbox:sandbox", f"TODO 3: process() gave {pr!r}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    policy = {"version": 1, "filesystem_policy": fs, "landlock": {"compatibility": "best_effort"},
              "process": pr, "network_policies": net}
    errs, warns = pk.validate(policy)
    ok &= check(not errs and not warns, "policykit.validate: 0 errors, 0 warnings",
                "validate: " + "; ".join(errs + warns))

    print("\n▣ twelve actions, replayed through policykit.decide")
    rows, wrong = [], 0
    for action, want in EXPECT:
        got, why = pk.decide(policy, action)
        wrong += got != want
        rows.append([pk.fmt_action(action), want, ("✓ " if got == want else "✕ ") + got, why[:48]])
    table(rows, ["action", "should be", "your policy", "why"])
    ok &= check(wrong == 0, "every decision matches a locked-down agent", f"{wrong} decision(s) differ — see ✕ rows")
    if not ok:
        sys.exit(1)
    print("\n═ Compare with policies/hotel_agent_policy.yaml: yours is its runtime form (setup-only pypi removed). "
          "Push it with `openshell policy set hotel-agent --policy <file> --wait`.")


if __name__ == "__main__":
    main()
