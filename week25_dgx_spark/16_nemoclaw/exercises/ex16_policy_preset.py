#!/usr/bin/env python3
"""Exercise 16 · Write a least-privilege NemoClaw network preset, and the checker that guards it.

Fill in the four TODOs, save, then run:
    .venv/bin/python week25/16_nemoclaw/exercises/ex16_policy_preset.py

The checker is free and offline: it compares your functions with the rules the NemoClaw
applications playbook documents, then prints the preset YAML you built. Stuck? Compare
with exercises/solutions/.
"""
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402


# ── TODO 1 ── is `name` a lowercase, hyphenated RFC 1123 label?
#   letters a-z, digits, hyphens; 1–63 characters; must not start or end with a hyphen. No underscores.
def is_rfc1123(name: str) -> bool:
    return None


# ── TODO 2 ── one HTTPS endpoint the egress proxy will actually let through:
#   {"host": host, "port": 443, "access": "full", "tls": "skip"}
def endpoint(host: str) -> dict:
    return None


# ── TODO 3 ── the whole preset as a Python dict (yaml.safe_dump turns it into the file):
#   {"preset": {"name": name, "description": description},
#    "network_policies": {name: {"name": name, "endpoints": [endpoint(h) ...], "binaries": [{"path": b} ...]}}}
def build_preset(name: str, description: str, hosts: list[str], binaries: list[str]) -> dict:
    return None


# ── TODO 4 ── return a list of error strings (empty list = valid). Catch the playbook's four failures:
#   • preset.name is not RFC 1123                      → mention "name"
#   • network_policies is not a dict (e.g. a list)      → mention "map"
#   • an endpoint has no access mode (access: full)     → mention "access"
#   • a group has no non-empty binaries list            → mention "binaries"
def errors(doc: dict) -> list[str]:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
HOSTS = ["developer.nvidia.com", "blogs.nvidia.com", "news.ycombinator.com"]
BINS = ["/usr/local/bin/openclaw", "/usr/local/bin/node", "/usr/bin/node", "/usr/bin/curl"]


def has(errs, word: str) -> bool:
    return isinstance(errs, list) and any(word in str(e).lower() for e in errs)


def main() -> None:
    banner("Exercise 16 · least-privilege policy preset", "offline checker · free · no Spark needed", status=False)
    ok = True
    names = {"news-sources": True, "brave": True, "news_sources": False, "News": False, "-news": False,
             "news-": False, "a" * 64: False}
    got = {n: is_rfc1123(n) for n in names}
    ok &= check(got == names, "is_rfc1123: news-sources ✓ · news_sources ✕ · News ✕ · -news ✕ · 64 chars ✕",
                "TODO 1: wrong for " + ", ".join(f"{n[:12]!r} (want {w}, got {got[n]!r})"
                                                  for n, w in names.items() if got[n] != w)[:160])

    want_ep = {"host": "blogs.nvidia.com", "port": 443, "access": "full", "tls": "skip"}
    ok &= check(endpoint("blogs.nvidia.com") == want_ep, "endpoint: host + port 443 + access full + tls skip",
                f"TODO 2: endpoint('blogs.nvidia.com') should be {want_ep} (got {endpoint('blogs.nvidia.com')!r})")

    doc = build_preset("news-sources", "Daily news digest source allowlist", HOSTS, BINS)
    group = ((doc or {}).get("network_policies") or {}).get("news-sources") if isinstance(doc, dict) else None
    shape_ok = (isinstance(doc, dict) and doc.get("preset") == {"name": "news-sources",
                                                                "description": "Daily news digest source allowlist"}
                and isinstance(group, dict) and group.get("name") == "news-sources"
                and [e.get("host") for e in group.get("endpoints", [])] == HOSTS
                and group.get("binaries") == [{"path": b} for b in BINS])
    ok &= check(shape_ok, "build_preset: a map keyed by group, 3 endpoints, 4 binaries",
                "TODO 3: build_preset must return {'preset': …, 'network_policies': {'news-sources': {…}}}")

    good = errors(doc) if shape_ok else None
    bad_name = {**(doc or {}), "preset": {"name": "news_sources", "description": "x"}} if shape_ok else {}
    as_list = {"preset": {"name": "news-sources"}, "network_policies": [{"host": "blogs.nvidia.com", "port": 443}]}
    no_access = {"preset": {"name": "x"}, "network_policies": {"x": {"name": "x", "binaries": [{"path": BINS[0]}],
                                                                    "endpoints": [{"host": "a.com", "port": 443}]}}}
    no_bins = {"preset": {"name": "x"}, "network_policies": {"x": {"name": "x", "endpoints": [endpoint("a.com")]}}}
    try:
        verdicts = [good == [], has(errors(bad_name), "name"), has(errors(as_list), "map"),
                    has(errors(no_access), "access"), has(errors(no_bins), "binaries")]
    except Exception as e:  # noqa: BLE001 — a half-written errors() should fail the check, not crash it
        verdicts = [False] * 5
        print(f"  (errors() raised {type(e).__name__}: {e})")
    ok &= check(all(verdicts), "errors: valid → [] · catches name · map · access · binaries",
                f"TODO 4: [valid→[], name, map, access, binaries] = {verdicts}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your preset, as the file you would pass to `nemoclaw <sandbox> policy-add --from-file`")
    for line in yaml.safe_dump(doc, sort_keys=False).splitlines():
        print("│ " + line)
    print("\n═ Now add a fourth host with an underscore in the group key, or remove /usr/bin/curl: "
          "what does each change allow or break?")


if __name__ == "__main__":
    main()
