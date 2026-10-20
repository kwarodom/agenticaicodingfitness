#!/usr/bin/env python3
"""policykit — a small, course-made toolkit for OpenShell sandbox policies (Week 25 · Module 15).

Three things, all plain Python (PyYAML only):

  build_policy(...)      write a policy YAML in the shape the NVIDIA playbooks use
  validate(policy)       check it against the rules the playbooks document (schema + semantics)
  decide(policy, action) a TEACHING MODEL of what the policy would allow — read/write a path,
                         connect host:port from a binary, an HTTP method on an L7 endpoint

`decide()` is NOT OpenShell. The real enforcement is Landlock (filesystem), seccomp/process
controls, and the sandbox's egress proxy — kernel and proxy code on the Spark. This model follows
the semantics the playbooks describe so you can reason about a policy BEFORE you push it. Where it
makes an assumption the playbooks do not state, the reason string says "(course assumption)".

Sources (dgx-spark-playbooks, nvidia/…):
  playbook-openshell/README.md            troubleshooting table: path rules, run_as_user, host/port, `Version`
  playbook-healthcare-agent/assets/sandbox-policy.yaml   a complete shipped policy (shape, landlock, binaries)
  playbook-nemoclaw-applications/README.md network_policies is a map; access mode + binaries required
  the OpenShell CLI 0.0.111 parser         the field names (see lab 15-2, which asks the real parser)
"""
from __future__ import annotations

import fnmatch
import ipaddress
import re
from pathlib import PurePosixPath

import yaml

TOP_KEYS = ("version", "filesystem_policy", "landlock", "process", "network_policies", "network_middlewares")
FS_KEYS = ("include_workdir", "read_only", "read_write")
PROCESS_KEYS = ("run_as_user", "run_as_group")
GROUP_KEYS = ("name", "endpoints", "binaries")
ENDPOINT_KEYS = ("host", "path", "port", "ports", "protocol", "tls", "enforcement", "access", "rules", "allowed_ips",
                 "deny_rules", "allow_encoded_slash", "websocket_credential_rewrite", "request_body_credential_rewrite",
                 "allow_uninspected_credentials", "persisted_queries", "graphql_persisted_queries",
                 "graphql_max_body_bytes", "credential_signing", "signing_service", "signing_region",
                 "credential_binding", "json_rpc", "mcp")
LANDLOCK_MODES = ("best_effort", "hard_requirement")
ACCESS_MODES = ("read-only", "read-write", "full")          # from `openshell policy update --help` examples
READ_METHODS = ("GET", "HEAD", "OPTIONS")                   # what "read-only" permits (course assumption)
WORKDIR = "/sandbox"                                        # include_workdir → the sandbox working dir
INFERENCE_HOST = "inference.local"


# ── build ─────────────────────────────────────────────────────────────────────
def group(name: str, endpoints: list[dict], binaries: list[str]) -> dict:
    return {"name": name, "endpoints": endpoints, "binaries": [{"path": b} for b in binaries]}


def build_policy(*, read_only: list[str], read_write: list[str], groups: dict[str, dict],
                 user: str = "sandbox", landlock: str = "best_effort", include_workdir: bool = True) -> dict:
    """A policy dict in the shipped layout: version · filesystem_policy · landlock · process · network_policies."""
    return {"version": 1,
            "filesystem_policy": {"include_workdir": include_workdir, "read_only": list(read_only),
                                  "read_write": list(read_write)},
            "landlock": {"compatibility": landlock},
            "process": {"run_as_user": user, "run_as_group": user},
            "network_policies": groups}


def dump(policy: dict) -> str:
    return yaml.safe_dump(policy, sort_keys=False, default_flow_style=None, width=110)


def load(text_or_path) -> dict:
    s = text_or_path if "\n" in str(text_or_path) else open(text_or_path, encoding="utf-8").read()
    return yaml.safe_load(s) or {}


# ── validate ──────────────────────────────────────────────────────────────────
def _path_problems(where: str, p) -> list[str]:
    if not isinstance(p, str) or not p.startswith("/"):
        return [f"{where}: {p!r} must be an absolute path starting with /"]
    if ".." in PurePosixPath(p).parts:
        return [f"{where}: {p!r} uses .. traversal"]
    return []


def _is_private_ip(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def validate(policy) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors = what the playbooks say OpenShell rejects; warnings = what applies cleanly but
    then blocks your agent (the playbooks' "most common reason" rows)."""
    errs, warns = [], []
    if not isinstance(policy, dict):
        return ["policy must be a YAML mapping"], []
    for k in policy:
        if k not in TOP_KEYS:
            hint = " (lowercase it: `version:`)" if k == "Version" else ""
            errs.append(f"unknown top-level field {k!r}{hint}; expected one of {', '.join(TOP_KEYS)}")
    if policy.get("version") != 1:
        errs.append(f"version must be 1 (got {policy.get('version')!r})")

    fs = policy.get("filesystem_policy") or {}
    if not isinstance(fs, dict):
        errs.append("filesystem_policy must be a mapping")
        fs = {}
    for k in fs:
        if k not in FS_KEYS:
            errs.append(f"filesystem_policy: unknown field {k!r}")
    for kind in ("read_only", "read_write"):
        for p in fs.get(kind) or []:
            errs += _path_problems(f"filesystem_policy.{kind}", p)
    if not fs.get("read_write") and not fs.get("include_workdir"):
        warns.append("nothing is writable: the agent cannot write any file (add /sandbox or /tmp to read_write)")

    ll = policy.get("landlock") or {}
    if ll and ll.get("compatibility") not in LANDLOCK_MODES:
        errs.append(f"landlock.compatibility must be one of {LANDLOCK_MODES}")

    pr = policy.get("process") or {}
    for k in pr:
        if k not in PROCESS_KEYS:
            errs.append(f"process: unknown field {k!r}")
    if str(pr.get("run_as_user", "")).lower() == "root" or str(pr.get("run_as_group", "")).lower() == "root":
        errs.append("process.run_as_user / run_as_group must not be root")

    net = policy.get("network_policies", {})
    if isinstance(net, list):
        errs.append("network_policies must be a map keyed by group name, not a list "
                    "(OpenShell: invalid type: sequence, expected a map)")
        net = {}
    for gname, g in (net or {}).items():
        where = f"network_policies.{gname}"
        if not re.fullmatch(r"[a-z0-9]([a-z0-9_-]*[a-z0-9])?", str(gname)):
            warns.append(f"{where}: use lowercase letters, digits, - or _ in group names")
        if not isinstance(g, dict):
            errs.append(f"{where} must be a mapping with name / endpoints / binaries")
            continue
        for k in g:
            if k not in GROUP_KEYS:
                errs.append(f"{where}: unknown field {k!r}; expected name, endpoints, binaries")
        if not g.get("binaries"):
            warns.append(f"{where}: no binaries — no program is authorised to use these endpoints (every call → 403)")
        for b in g.get("binaries") or []:
            if not isinstance(b, dict) or "path" not in b:
                errs.append(f"{where}.binaries: each entry needs a path, e.g. {{path: /usr/bin/curl}}")
            else:
                errs += _path_problems(f"{where}.binaries", b["path"])
        for i, e in enumerate(g.get("endpoints") or []):
            ew = f"{where}.endpoints[{i}]"
            if not isinstance(e, dict):
                errs.append(f"{ew} must be a mapping with host and port")
                continue
            for k in e:
                if k not in ENDPOINT_KEYS:
                    errs.append(f"{ew}: unknown field {k!r}")
            if not e.get("host"):
                errs.append(f"{ew}: missing host")
            port = e.get("port")
            if port is None and not e.get("ports"):
                errs.append(f"{ew}: missing port")
            elif port is not None and (not isinstance(port, int) or not 0 < port < 65536):
                errs.append(f"{ew}: port must be an integer 1–65535 (got {port!r})")
            if e.get("access") and e["access"] not in ACCESS_MODES:
                errs.append(f"{ew}: access must be one of {ACCESS_MODES}")
            if e.get("host") != INFERENCE_HOST and not (e.get("access") or e.get("protocol") or e.get("rules")):
                warns.append(f"{ew} ({e.get('host')}): no access mode (e.g. access: full + tls: skip, or protocol: rest"
                             " + enforcement: enforce) — the proxy denies it with CONNECT 403")
            if _is_private_ip(str(e.get("host", ""))) and not e.get("allowed_ips"):
                warns.append(f"{ew}: {e['host']} is a private/loopback IP — the sandbox blocks those ranges unless "
                             "allowed_ips opens them; route models through inference.local instead")
    return errs, warns


# ── decide: a teaching model of the semantics ─────────────────────────────────
def _under(path: str, prefix: str) -> bool:
    p, q = PurePosixPath(path), PurePosixPath(prefix)
    return p == q or q in p.parents


def _bin_ok(binary: str, g: dict) -> str | None:
    for b in g.get("binaries") or []:
        pat = str(b.get("path", ""))
        if fnmatch.fnmatch(binary, pat) or fnmatch.fnmatch(binary, pat.replace("/**", "/*")) or \
                (pat.endswith("/**") and binary.startswith(pat[:-2])):
            return pat
    return None


def _ip_opened(policy: dict, host: str) -> bool:
    ip = ipaddress.ip_address(host)
    for g in (policy.get("network_policies") or {}).values():
        for e in g.get("endpoints") or []:
            for net in e.get("allowed_ips") or []:
                try:
                    if ip in ipaddress.ip_network(str(net), strict=False):
                        return True
                except ValueError:
                    pass
    return False


def decide(policy: dict, action: dict) -> tuple[str, str]:
    """(decision, reason). decision ∈ allow | deny | inspect_for_inference.

    action examples:
      {"op": "read",  "path": "/etc/hosts"}
      {"op": "write", "path": "/sandbox/tickets.jsonl"}
      {"op": "connect", "host": "api.openai.com", "port": 443, "binary": "/usr/bin/curl"}
      {"op": "http", "host": "pypi.org", "port": 443, "binary": "/usr/local/bin/uv", "method": "GET", "path": "/simple/"}
      {"op": "run_as", "user": "root"}
    """
    op = action.get("op")
    fs = policy.get("filesystem_policy") or {}
    ro = list(fs.get("read_only") or [])
    rw = list(fs.get("read_write") or []) + ([WORKDIR] if fs.get("include_workdir") else [])
    if op in ("read", "write"):
        path = action["path"]
        hit_rw = next((p for p in rw if _under(path, p)), None)
        hit_ro = next((p for p in ro if _under(path, p)), None)
        if op == "write":
            if hit_rw:
                return "allow", f"under read_write {hit_rw}"
            if hit_ro:
                return "deny", f"{hit_ro} is read_only (Landlock: Permission denied)"
            return "deny", "not in read_only or read_write (Landlock: Permission denied)"
        if hit_rw or hit_ro:
            return "allow", f"under {'read_write ' + hit_rw if hit_rw else 'read_only ' + hit_ro}"
        return "deny", "not in read_only or read_write (Landlock: Permission denied)"

    if op == "run_as":
        want = (policy.get("process") or {}).get("run_as_user", "?")
        return ("allow", f"process.run_as_user is {want}") if action.get("user") == want else \
               ("deny", f"the sandbox runs as {want!r}; it cannot become {action.get('user')!r}")

    if op in ("connect", "http"):
        host, port, binary = action["host"], int(action.get("port", 443)), action.get("binary", "")
        if host == INFERENCE_HOST:
            return "inspect_for_inference", ("inference.local is handled by the proxy's inference routing, "
                                             "forwarded to the provider set with `openshell inference set`")
        if _is_private_ip(host) and not _ip_opened(policy, host):
            return "deny", ("private/loopback IPs are blocked unless an endpoint opens them with allowed_ips "
                            "(course assumption — the playbooks disagree, see TUTORIAL §4)")
        matched_host = False
        for gname, g in (policy.get("network_policies") or {}).items():
            for e in g.get("endpoints") or []:
                ports = [e["port"]] if e.get("port") is not None else list(e.get("ports") or [])
                if not fnmatch.fnmatch(host, str(e.get("host", ""))) or port not in ports:
                    continue
                matched_host = True
                pat = _bin_ok(binary, g)
                if not pat:
                    continue                                    # maybe another group lists this binary
                if op == "http":
                    method, rpath = action.get("method", "GET").upper(), action.get("path", "/")
                    for r in e.get("rules") or []:
                        a = r.get("allow") or {}
                        if a.get("method", method).upper() == method and fnmatch.fnmatch(rpath, a.get("path", "*")):
                            return "allow", f"{gname}: rule allow {method} {a.get('path')}"
                    if e.get("rules"):
                        return "deny", f"{gname}: no rule allows {method} {rpath} (L7 enforce)"
                    if e.get("access") == "read-only" and method not in READ_METHODS:
                        return "deny", f"{gname}: access read-only allows only {'/'.join(READ_METHODS)} (course assumption)"
                return "allow", f"network_policies.{gname}: {host}:{port} for binary {pat}"
        if matched_host:
            return "deny", f"{host}:{port} is listed, but not for binary {binary} (binaries allow-list)"
        return "deny", f"{host}:{port} is not in network_policies (default deny egress)"
    return "deny", f"unknown action {op!r}"


def fmt_action(a: dict) -> str:
    if a["op"] in ("read", "write"):
        return f"{a['op']:5s} {a['path']}"
    if a["op"] == "run_as":
        return f"run as {a['user']}"
    b = a.get("binary", "?").rsplit("/", 1)[-1]
    extra = f" {a.get('method', 'GET')} {a.get('path', '/')}" if a["op"] == "http" else ""
    return f"{b} → {a['host']}:{a.get('port', 443)}{extra}"
