#!/usr/bin/env python3
"""Lab 16-3 · Policy preset builder: grant an agent exactly the egress it needs, and prove it.

Runs on your laptop (real, offline). It writes the News Digest's `news-sources.yaml`
network preset in the exact shape the NemoClaw applications playbook documents, parses
it back, and validates it against every rule the playbook spells out (a map not a
list, an RFC 1123 preset name, host + port, an access mode, a `binaries` allow-list),
plus a few least-privilege checks this course adds (no wildcards, no shells, no
secrets in the file). Then it feeds the validator six broken presets (four the
playbook documents, plus two this course adds) and shows each one caught.

With a Spark, the file is copied to ~/w25/nemoclaw/ (a plain copy). Applying it to the
sandbox changes live policy, so that only happens with --apply.

Run: .venv/bin/python week25/16_nemoclaw/labs/lab16_3_policy_preset.py [--host blogs.example.org] [--apply]
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, cfg, check, note, put, result, sh, step, table, where  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / ".runs"                                   # gitignored (week25/.gitignore: **/.runs/)

# From the playbook's News Digest recipe: the three example sources and the four binaries.
PLAYBOOK_HOSTS = ["developer.nvidia.com", "blogs.nvidia.com", "news.ycombinator.com"]
PLAYBOOK_BINARIES = ["/usr/local/bin/openclaw", "/usr/local/bin/node", "/usr/bin/node", "/usr/bin/curl"]

RFC1123 = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
IP_LITERAL = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SHELLS = {"/bin/bash", "/bin/sh", "/usr/bin/bash", "/usr/bin/sh", "/bin/zsh", "/usr/bin/python3", "/usr/bin/python"}
SECRET = re.compile(r"(sk-[A-Za-z0-9]{8,}|nvapi-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|xox[bap]-[A-Za-z0-9\-]{8,}"
                    r"|\b\d{8,10}:[A-Za-z0-9_\-]{30,}\b)")
MAX_HOSTS = 10


def render(name: str, description: str, hosts: list[str], binaries: list[str]) -> str:
    """The preset YAML, laid out like the playbook's news-sources.yaml."""
    lines = ["preset:", f"  name: {name}", f'  description: "{description}"', "", "network_policies:",
             f"  {name}:", f"    name: {name}", "    endpoints:"]
    for h in hosts:
        h = h if HOSTNAME.match(h) or IP_LITERAL.match(h) else f'"{h}"'      # quote '*.x' so YAML parses it
        lines += [f"      - host: {h}", "        port: 443", "        access: full", "        tls: skip"]
    lines.append("    binaries:")
    lines += [f"      - {{ path: {b} }}" for b in binaries]
    return "\n".join(lines) + "\n"


def validate(text: str) -> list[tuple[str, str]]:
    """[(level, message)] — level 'error' (the playbook says it fails) or 'warn' (least-privilege advice)."""
    found: list[tuple[str, str]] = []
    if SECRET.search(text):
        found.append(("error", "a credential-shaped string is in the file — never put tokens in a policy preset"))
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return found + [("error", f"not valid YAML: {str(e).splitlines()[0]}")]
    if not isinstance(doc, dict):
        return found + [("error", "top level must be a mapping with preset: and network_policies:")]
    name = ((doc.get("preset") or {}).get("name") if isinstance(doc.get("preset"), dict) else None) or ""
    if not RFC1123.match(str(name)):
        found.append(("error", f"preset.name {name!r}: must be a lowercase, hyphenated RFC 1123 label"))
    groups = doc.get("network_policies")
    if isinstance(groups, list):
        return found + [("error", "network_policies is a list: 'invalid type: sequence, expected a map'")]
    if not isinstance(groups, dict) or not groups:
        return found + [("error", "network_policies must be a map of groups")]
    n_hosts = 0
    for key, g in groups.items():
        if "_" in str(key):
            found.append(("warn", f"group key {key!r} has an underscore — the playbook disagrees with itself here; "
                                  "use hyphens to be safe"))
        if not isinstance(g, dict) or not isinstance(g.get("endpoints"), list) or not g["endpoints"]:
            found.append(("error", f"group {key!r} needs an endpoints list"))
            continue
        for ep in g["endpoints"]:
            host, port = (ep or {}).get("host"), (ep or {}).get("port")
            if not host or port is None:
                found.append(("error", f"endpoint {ep!r}: needs both host and port"))
                continue
            n_hosts += 1
            if not (ep.get("access") == "full" or (ep.get("protocol") == "rest" and ep.get("rules"))):
                found.append(("error", f"{host}: no access mode → the proxy answers 'CONNECT tunnel failed, response 403'"))
            if "*" in str(host):
                found.append(("warn", f"{host}: wildcard host — name each site instead"))
            elif IP_LITERAL.match(str(host)):
                found.append(("warn", f"{host}: raw IP — prefer a hostname you can audit"))
            elif not HOSTNAME.match(str(host)):
                found.append(("error", f"{host!r}: not a valid lowercase hostname"))
            if port != 443:
                found.append(("warn", f"{host}:{port} — not HTTPS; is plain port {port} really needed?"))
        bins = g.get("binaries")
        if not isinstance(bins, list) or not bins:
            found.append(("error", f"group {key!r} has no binaries allow-list → no program may use it (every fetch 403s)"))
            continue
        for b in bins:
            path = (b or {}).get("path", "") if isinstance(b, dict) else ""
            if not path.startswith("/") or ".." in path:
                found.append(("error", f"binary {b!r}: paths must be absolute with no '..'"))
            elif path in SHELLS:
                found.append(("warn", f"{path}: a shell/interpreter lets ANY script in the sandbox use this egress"))
    if n_hosts > MAX_HOSTS:
        found.append(("warn", f"{n_hosts} hosts — every host widens the egress surface; keep the list short"))
    return found


def show(findings: list[tuple[str, str]]) -> None:
    if not findings:
        print("✓ no findings: valid shape, least-privilege checks pass")
    for level, msg in findings:
        print(("✕ " if level == "error" else "⚠ ") + msg)


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--host", action="append", default=[], help="extra news host to allow (repeatable)")
ap.add_argument("--apply", action="store_true", help="also run policy-add on the Spark sandbox (changes live policy)")
args = ap.parse_args()
sandbox = cfg("NEMOCLAW_SANDBOX", "my-assistant")
if not RFC1123.match(sandbox):
    sandbox = "my-assistant"

banner("Lab 16-3 · policy preset builder", "the News Digest egress allow-list, generated and validated on this laptop",
       status=False)

step(1, "generate news-sources.yaml (the playbook's shape)")
hosts = PLAYBOOK_HOSTS + [h.lower() for h in args.host if h.lower() not in PLAYBOOK_HOSTS]
text = render("news-sources", "Daily news digest source allowlist", hosts, PLAYBOOK_BINARIES)
OUT.mkdir(exist_ok=True)
path = OUT / "news-sources.yaml"
path.write_text(text, encoding="utf-8")
print(text.rstrip())
note(f"written to {path.relative_to(HERE.parents[1])}")

step(2, "parse it back and validate")
findings = validate(text)
show(findings)
good = not any(level == "error" for level, _ in findings)

step(3, "the validator must catch every documented mistake (rows 1-4: playbook · 5-6: this course)")
bad_cases = [
    ("underscore in preset.name", text.replace("  name: news-sources\n  description", "  name: news_sources\n  description", 1)),
    ("list instead of a map", "preset:\n  name: news-sources\nnetwork_policies:\n  - host: blogs.nvidia.com\n    port: 443\n"),
    ("bare {host, port} (no access mode)", render("news-sources", "x", ["blogs.nvidia.com"], PLAYBOOK_BINARIES)
     .replace("        access: full\n        tls: skip\n", "")),
    ("no binaries allow-list", render("news-sources", "x", ["blogs.nvidia.com"], PLAYBOOK_BINARIES).split("    binaries:")[0]),
    ("a token pasted into the file", render("news-sources", "key sk-FAKEFAKEnotarealkey0000", ["blogs.nvidia.com"],
                                            PLAYBOOK_BINARIES)),
    ("wildcard host + a shell binary", render("news-sources", "x", ["*.example.com"], ["/bin/bash"])),
]
rows = []
for label, bad in bad_cases:
    f = validate(bad)
    errors = [m for lv, m in f if lv == "error"]
    warns = [m for lv, m in f if lv == "warn"]
    caught = bool(errors or warns)
    rows.append([label, "✓ caught" if caught else "✕ missed", (errors or warns or ["—"])[0][:58]])
table(rows, ["broken preset", "validator", "first finding"])
all_caught = all(r[1] == "✓ caught" for r in rows)

step(4, "put it on the Spark and (only with --apply) add it to the sandbox")
remote = "~/w25/nemoclaw/news-sources.yaml"
copied = put(path, remote)
apply_cmd = f"nemoclaw {sandbox} policy-add --from-file {remote} --yes"
verify_cmd = f"openshell policy get {sandbox} --full | grep -E \"host:|port:\""
if args.apply and copied and good:
    sh(apply_cmd, timeout=120)
    sh(verify_cmd, timeout=60)
else:
    why = ("DRY — no Spark" if where() == "dry" else
           "the preset has errors" if not good else "--apply not given (it changes live sandbox policy)")
    print(f"→ not applied ({why}). On the Spark, run:")
    print(f"$ {apply_cmd}")
    print(f"$ {verify_cmd}")
note("Undo later with: nemoclaw <sandbox> policy-remove news-sources --yes  (network changes hot-reload, no rebuild)")

print()
check(good, "news-sources.yaml is valid and least-privilege", "news-sources.yaml has errors — see step 2")
check(all_caught, "every documented mistake was caught", "the validator missed a documented mistake")
result("Three hosts, four named binaries, HTTPS only. Every extra host is another place a prompt-injected agent "
       "could send your data.")
sys.exit(0 if good and all_caught else 1)
