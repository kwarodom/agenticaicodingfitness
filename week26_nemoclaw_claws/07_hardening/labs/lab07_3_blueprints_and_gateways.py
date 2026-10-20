#!/usr/bin/env python3
"""Lab 07-3 · L6.2 + L6.3 — Custom images, external-gateway blueprints, remote gateways.

Mostly laptop-real:
  1. generate the Alto Ops Claw image Dockerfile for `nemoclaw onboard --from` (the research tutorial's L3.8
     composition), and lint it with a small course-made checker (non-root USER, tools baked in, no secrets);
  2. validate the tutorial's external-gateway blueprint.yaml against the rules the tutorial lists (bare HTTPS
     origin, exact OpenShell release with min = max, PEM-only CA bundle ≤ 1 MiB that is a regular file and not
     a symlink, absolute authentication path). The CA and credential files are throwaway ones this lab
     generates in .runs/ with openssl. Then eleven broken copies, one rule each;
  3. ask the real laptop OpenShell CLI (0.0.111) about the remote-gateway commands;
  4. on the Spark: `nemoclaw-blueprint-runner plan` / `status --external-target` (read-only, sh()) and
     `openshell gateway start --remote` (a change, only with 🔓 CLAW_APPLY=1 and a second Spark configured).

The Dockerfile lint and the blueprint validator are COURSE code that follows the tutorial's wording. The real
checks are done by nemoclaw-blueprint-runner and the NemoClaw SDK on the Spark.

Run: .venv/bin/python week26/07_hardening/labs/lab07_3_blueprints_and_gateways.py
"""
import copy
import hashlib
import os
import re
import shutil
import ssl
import stat
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import policykit as pk  # noqa: E402
import yaml  # noqa: E402
from clawkit import (banner, change, host, laptop, note, ok, openshell_offline, put, result, sh, step,  # noqa: E402
                     table, warn)

MOD = Path(__file__).resolve().parents[1]
RUNS = MOD / ".runs"
FAKEROOT = RUNS / "fakeroot"                   # the blueprint's absolute paths are resolved under here
THROW = RUNS / "throwaway"                     # the private key never goes into the fake root
BLUEPRINT = MOD / "blueprint" / "blueprint.yaml"
POLICY = pk.load(MOD / "policies" / "prod.yaml")
MIB = 1024 * 1024
RUNS.mkdir(parents=True, exist_ok=True)

# Quoted verbatim from dgx-spark-playbooks/nvidia/playbook-openshell/README.md (the audit checks these lines).
REF_REMOTE = """To manage a gateway on remote hardware from a separate workstation, ensure passwordless SSH works first, then use `openshell gateway start --remote <username>@<hostname>`
| TLS / certificate errors when adding a remote gateway by LAN IP | Gateway certificate is valid for `openshell`, `localhost`, and `127.0.0.1` — not the LAN IP | Map `openshell` to the hardware IP in `/etc/hosts`, then register with `openshell gateway add https://openshell:8080 --remote <user>@<hardware-ip>` |"""

banner("Lab 07-3 · blueprints and gateways (L6.2 · L6.3)",
       "laptop: Dockerfile + blueprint validators, real CLI parser · Spark: blueprint runner (read-only)")

# ── 1 · the image ─────────────────────────────────────────────────────────────
step(1, "a custom image for `nemoclaw onboard --from` — tools baked in, non-root")
SPEC = {"base": "ubuntu:24.04", "apt": ["python3.12", "python3-pip", "curl"],
        "pip": "'nvidia-nat[langchain,mcp,profiler,opentelemetry]'", "workflow": "workflows/alto_ops",
        "config": "workflow.sandbox.yml", "user": POLICY["process"]["run_as_user"], "workdir": "/sandbox"}


def dockerfile(s: dict) -> str:
    """The research tutorial's L3.8 Dockerfile, generated from a spec so the USER always matches the policy."""
    return "\n".join([
        f"FROM {s['base']}",
        f"RUN apt-get update && apt-get install -y {' '.join(s['apt'])} && rm -rf /var/lib/apt/lists/*",
        f"RUN pip3 install --break-system-packages uv && uv pip install --system {s['pip']}",
        f"COPY {s['workflow']} /app/alto_ops",
        "RUN uv pip install --system -e /app/alto_ops",
        f"COPY {s['config']} /app/workflow.yml",
        f"USER {s['user']}",
        f"WORKDIR {s['workdir']}",
    ]) + "\n"


def lint_dockerfile(text: str, policy: dict) -> list[tuple[str, str]]:
    """Course-made checks, each tied to a line of the tutorial. (level, message); level ∈ ✕ ⚠ ✓."""
    out, lines = [], [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    users = [ln.split(None, 1)[1] for ln in lines if ln.upper().startswith("USER ")]
    last = users[-1].split(":")[0] if users else ""
    if not last or last in ("root", "0"):
        out.append(("✕", f"final USER is {last or 'unset (root)'}: OpenShell requires a non-root identity"))
    else:
        out.append(("✓", f"final USER {last} (non-root)"))
    want = str((policy.get("process") or {}).get("run_as_user", ""))
    if want and last and last != want:
        out.append(("⚠", f"USER {last} ≠ policy process.run_as_user {want}"))
    for ln in lines:
        head = ln.split(None, 1)[0].upper()
        if head in ("CMD", "ENTRYPOINT") and re.search(r"\b(pip3?|uv pip|npm|apt-get|apt)\s+install\b", ln):
            out.append(("✕", f"installs at runtime ({head}) — bake tools in at build time, so no egress is needed"))
        if re.search(r"curl[^|]*\|\s*(ba)?sh", ln):
            out.append(("⚠", "pipes a download into a shell — pin and verify what you fetch"))
        if head in ("ENV", "ARG") and re.search(r"(?i)(key|token|secret|password)\s*[= ]", ln):
            out.append(("✕", "a secret in ENV/ARG ends up in the image — use an OpenShell provider"))
        if head == "ADD" and re.search(r"https?://", ln):
            out.append(("⚠", "ADD from a URL — fetch at build with a checksum instead"))
    apt = " ".join(ln for ln in lines if "apt-get install" in ln)
    for g in (policy.get("network_policies") or {}).values():
        for b in g.get("binaries") or []:
            name = Path(b["path"]).name
            if name not in apt and name not in text:
                out.append(("⚠", f"the policy trusts {b['path']} but this image does not obviously install it"))
    if not any(lv != "✓" for lv, _ in out):
        out.append(("✓", f"no runtime installs, no secrets, every policy binary ({', '.join(sorted({Path(b['path']).name for g in POLICY['network_policies'].values() for b in g['binaries']}))}) comes from apt"))
    return out


good = dockerfile(SPEC)
(RUNS / "image").mkdir(exist_ok=True)
(RUNS / "image" / "Dockerfile").write_text(good, encoding="utf-8")
print(good.rstrip())
for lv, msg in lint_dockerfile(good, POLICY):
    print(f"{lv} {msg}")
note("The tutorial calls this Dockerfile its own composition: verify the NAT install line on your base image. "
     "Course assumption: apt's python3.12 on ubuntu:24.04 is /usr/bin/python3.12, the binary prod.yaml trusts.")

bad = good.replace(f"USER {SPEC['user']}", "USER root").replace(
    "WORKDIR /sandbox", "WORKDIR /sandbox\nENV OPENAI_API_KEY=<your-key>\n"
                        "CMD pip install pandas && nat serve --config_file /app/workflow.yml")
bad = bad.replace("RUN pip3 install", "RUN curl -fsSL https://example.invalid/setup.sh | bash\nRUN pip3 install")
print("\n→ the same image, 'just make it work' edition:")
for lv, msg in lint_dockerfile(bad, POLICY):
    print(f"{lv} {msg}")
print("$ nemoclaw onboard --from ./Dockerfile   [type it in the ⌨ terminal on the Spark — the wizard is interactive]")
put(RUNS / "image" / "Dockerfile", "~/week26/07_hardening/image/Dockerfile")
note("The build context also needs workflows/alto_ops and workflow.sandbox.yml next to the Dockerfile. "
     "Module 08 (capstone) assembles the full context and builds it.")

# ── 2 · the blueprint validator ───────────────────────────────────────────────
step(2, "throwaway CA + credential files for the blueprint (openssl, on this laptop)")
tgt = FAKEROOT / "var" / "run" / "openshell-target"
tgt.mkdir(parents=True, exist_ok=True)
THROW.mkdir(parents=True, exist_ok=True)
ca, key = tgt / "ca.pem", THROW / "ca.key"
openssl = shutil.which("openssl")
if not openssl:
    warn("openssl not found on this laptop — the CA checks below will report the file as missing")
else:
    laptop([openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
            "-keyout", key, "-out", ca, "-days", "1", "-subj", "/CN=openshell.alto.local"], quiet=True,
           show="openssl req -x509 -newkey ec -nodes -days 1 -subj /CN=openshell.alto.local -out …/ca.pem")
    laptop([openssl, "x509", "-in", ca, "-outform", "der", "-out", THROW / "ca.der"], quiet=True,
           show="openssl x509 -in ca.pem -outform der -out ca.der")
cred = tgt / "authentication"
cred.write_text("course-throwaway-credential — not a real secret\n", encoding="utf-8")
cred.chmod(0o600)
ok(f"wrote {ca.relative_to(RUNS)} ({ca.stat().st_size if ca.exists() else 0} bytes) and "
   f"{cred.relative_to(RUNS)} (mode 600) under .runs/fakeroot — never used anywhere")

step(3, "validate the tutorial's blueprint.yaml (course validator, the tutorial's rules)")
EXACT = re.compile(r"^\d+\.\d+\.\d+$")
PEM_BLOCK = re.compile(r"-----BEGIN ([A-Z0-9 ]+)-----\s+[A-Za-z0-9+/=\s]+?-----END \1-----", re.S)


def under_root(p: str, root: Path) -> Path:
    return root / p.lstrip("/")


def validate_blueprint(bp: dict, root: Path = FAKEROOT) -> list[tuple[str, str, str]]:
    """[(level, rule, message)] — ✕ error · ⚠ warning · ✓ pass."""
    out = []
    t = bp.get("openshell_target") or {}
    # R1 · bare HTTPS origin
    u = urlsplit(str(t.get("endpoint", "")))
    bare = u.scheme == "https" and u.hostname and not u.username and u.path == "" and not u.query and not u.fragment
    out.append(("✓" if bare else "✕", "bare HTTPS origin",
                f"{t.get('endpoint')}" + ("" if bare else " — want https://host[:port] with no path, user, query")))
    # R2 · exact release, min = max = expected
    mn, mx, ex = (str(bp.get("min_openshell_version", "")), str(bp.get("max_openshell_version", "")),
                  str(t.get("expected_release", "")))
    exact = all(EXACT.match(v) for v in (mn, mx, ex)) and mn == mx == ex
    out.append(("✓" if exact else "✕", "exact OpenShell release",
                f"min {mn} · max {mx} · expected {ex}" + ("" if exact else " — must be one X.Y.Z, min = max = expected")))
    # R3 · the CA bundle
    ca_p = str((t.get("trust") or {}).get("ca_file", ""))
    f = under_root(ca_p, root)
    if not ca_p.startswith("/"):
        out.append(("✕", "CA bundle", f"{ca_p!r} is not an absolute path"))
    elif not os.path.lexists(f):
        out.append(("✕", "CA bundle", f"{ca_p} does not exist"))
    else:
        st = os.lstat(f)
        if stat.S_ISLNK(st.st_mode):
            out.append(("✕", "CA bundle", f"{ca_p} is a symlink — must be a regular file"))
        elif not stat.S_ISREG(st.st_mode):
            out.append(("✕", "CA bundle", f"{ca_p} is not a regular file"))
        elif st.st_size > MIB:
            out.append(("✕", "CA bundle", f"{ca_p} is {st.st_size / MIB:.2f} MiB — limit 1 MiB"))
        else:
            raw = f.read_bytes()
            try:
                txt = raw.decode("ascii")
            except UnicodeDecodeError:
                txt = ""
            kinds = [m.group(1) for m in PEM_BLOCK.finditer(txt)]
            rest = PEM_BLOCK.sub("", txt).strip()
            if not kinds or rest:
                out.append(("✕", "CA bundle", f"{ca_p} is not PEM-only (binary/DER or stray text)"))
            elif set(kinds) != {"CERTIFICATE"}:
                out.append(("✕", "CA bundle", f"{ca_p} holds {', '.join(sorted(set(kinds) - {'CERTIFICATE'}))} — "
                                              "certificates only (course check)"))
            else:
                try:
                    ssl.create_default_context().load_verify_locations(cafile=str(f))
                    fp = hashlib.sha256(ssl.PEM_cert_to_DER_cert(txt[:txt.index("-----END CERTIFICATE-----") + 25])
                                        ).hexdigest()
                    out.append(("✓", "CA bundle", f"{len(kinds)} PEM certificate(s), {st.st_size} B, regular file · "
                                                  f"sha256 {fp[:16]}…"))
                except ssl.SSLError as e:
                    out.append(("✕", "CA bundle", f"{ca_p} does not load as a CA: {e.reason}"))
    # R4 · absolute authentication path
    a_p = str((t.get("authentication") or {}).get("credential_file", ""))
    if not a_p.startswith("/"):
        out.append(("✕", "authentication path", f"{a_p!r} is not absolute"))
    elif not under_root(a_p, root).is_file():
        out.append(("⚠", "authentication path", f"{a_p} is absolute but missing here"))
    else:
        mode = under_root(a_p, root).stat().st_mode & 0o777
        out.append(("✓" if not mode & 0o077 else "⚠", "authentication path",
                    f"{a_p} (mode {mode:o})" + ("" if not mode & 0o077 else " — readable by others (course check)")))
    if t.get("lifecycle") != "external":
        out.append(("⚠", "lifecycle", f"{t.get('lifecycle')!r} — the tutorial's external-target form uses external"))
    return out


bp = yaml.safe_load(BLUEPRINT.read_text(encoding="utf-8"))
print(BLUEPRINT.read_text(encoding="utf-8").rstrip())
for lv, rule, msg in validate_blueprint(bp):
    print(f"{lv} {rule}: {msg}")
note("The sha256 is this lab's analog of what `plan` does ('fingerprints CA'). The runner's own output is on "
     "the Spark, below.")

step(4, "eleven broken copies — one rule each")
(tgt / "ca-link.pem").unlink(missing_ok=True)
(tgt / "ca-link.pem").symlink_to("ca.pem")
if ca.exists():
    pem = ca.read_text(encoding="ascii")
    (tgt / "ca-big.pem").write_text(pem * (int(1.1 * MIB) // len(pem) + 1), encoding="ascii")
    (tgt / "ca-with-key.pem").write_text(pem + key.read_text(encoding="ascii"), encoding="ascii")
    if (THROW / "ca.der").exists():
        shutil.copyfile(THROW / "ca.der", tgt / "ca.der")
(tgt / "ca-dir.pem").mkdir(exist_ok=True)


def with_(**changes):
    b = copy.deepcopy(bp)
    for dotted, v in changes.items():
        node, *path = dotted.split("__")
        cur = b
        keys = [node, *path]
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = v
    return b


D = "/var/run/openshell-target/"
BROKEN = [
    ("endpoint has a path", with_(openshell_target__endpoint="https://openshell.alto.local:8443/api/v1")),
    ("endpoint is http://", with_(openshell_target__endpoint="http://openshell.alto.local:8080")),
    ("endpoint has a user", with_(openshell_target__endpoint="https://admin@openshell.alto.local:8443")),
    ("a version range", with_(max_openshell_version="0.1.2")),
    ("expected_release not exact", with_(openshell_target__expected_release=">=0.0.116")),
    ("CA is a symlink", with_(openshell_target__trust__ca_file=D + "ca-link.pem")),
    ("CA over 1 MiB", with_(openshell_target__trust__ca_file=D + "ca-big.pem")),
    ("CA bundle has a private key", with_(openshell_target__trust__ca_file=D + "ca-with-key.pem")),
    ("CA is DER, not PEM", with_(openshell_target__trust__ca_file=D + "ca.der")),
    ("CA path is a directory", with_(openshell_target__trust__ca_file=D + "ca-dir.pem")),
    ("relative credential path", with_(openshell_target__authentication__credential_file="authentication")),
]
rows = []
for name, b in BROKEN:
    fails = [(rule, msg) for lv, rule, msg in validate_blueprint(b) if lv == "✕"]
    rows.append([name, "✕ " + fails[0][0] if fails else "✓ passed (!)", fails[0][1][:66] if fails else ""])
table(rows, ["broken copy", "caught by", "message"])
caught = sum(r[1].startswith("✕") for r in rows)
(ok if caught == len(BROKEN) else warn)(f"{caught}/{len(BROKEN)} broken copies caught by the rule they break")

# ── 3 · remote gateways ───────────────────────────────────────────────────────
step(5, "remote gateways — ask the real laptop CLI (0.0.111) about the tutorial's commands")
HOME = RUNS / "openshell-home"
for args, what in (
        (["gateway", "add", "https://openshell:8080", "--remote"], "the tutorial's line, as printed"),
        (["gateway", "add", "https://openshell:8080", "--remote", "me@spark-01.alto.local"], "with the SSH target"),
        (["gateway", "start", "--remote", "user@spark-01.alto.local"], "provision a remote gateway")):
    r = openshell_offline(args, home=HOME)
    first = next((ln.strip() for ln in r.out.splitlines() if ln.strip()), "")
    print(f"  → {what}: exit {r.code} · {first[:110]}")
note("Two real findings. (1) The research tutorial prints `openshell gateway add https://openshell:8080 --remote` "
     "with no value; the CLI requires one, and the playbook writes `--remote <user>@<hardware-ip>`. (2) The "
     "laptop CLI 0.0.111 has no `gateway start` at all. The playbook documents `gateway start --remote` for the "
     "OpenShell it installs, so check `openshell gateway --help` on your Spark before you script it.")
print("◈ REFERENCE — quoted from NVIDIA's playbook (playbook-openshell/README.md):")
print(REF_REMOTE)
note("So on your workstation: map `openshell` to the Spark's IP in /etc/hosts (the lab never edits it), then "
     "run `gateway add` with the SSH target. The certificate is valid for `openshell`, not the LAN IP.")

step(6, "the Spark — is its gateway up, and (optionally) provision one on Spark B")
sh("openshell status", reference="`openshell status` should report the gateway as **Connected**.")
if host("b"):
    change(f"openshell gateway start --remote {host('b')}", preview="openshell status",
           example="(the gateway on Spark B is provisioned over SSH — see `openshell status` there)")
else:
    print("$ openshell gateway start --remote user@spark-01.alto.local   [NOT RUN — no SPARK_HOST2 configured]")
    note("With a second Spark in 🖥 Spark setup (SPARK_HOST2) this step provisions its gateway through change().")

step(7, "the external target — nemoclaw-blueprint-runner on the Spark (read-only)")
put(BLUEPRINT, "~/week26/07_hardening/blueprint/blueprint.yaml")
BP_DIR = "$HOME/week26/07_hardening/blueprint"
sh("command -v nemoclaw-blueprint-runner || echo 'nemoclaw-blueprint-runner: not on PATH'",
   example="/home/<you>/.local/bin/nemoclaw-blueprint-runner")
sh(f"NEMOCLAW_BLUEPRINT_PATH={BP_DIR} nemoclaw-blueprint-runner plan",
   example="""plan: blueprint 1.0.0 · target https://openshell.alto.local:8443 · release 0.0.116
  CA bundle: /var/run/openshell-target/ca.pem · sha256 <fingerprint>
  no network requests made""")
sh(f"NEMOCLAW_BLUEPRINT_PATH={BP_DIR} nemoclaw-blueprint-runner status --external-target",
   example="""status: https://openshell.alto.local:8443 · <reachable | unreachable> · release <reported release>""")
note("Both EXAMPLE blocks are shapes, not the runner's real wording. `plan` 'validates, fingerprints CA, no "
     "network'; `status --external-target` makes 'one credential-free health request' (research tutorial). "
     "The tutorial writes NEMOCLAW_BLUEPRINT_PATH=/abs/path/blueprint without saying whether it wants the "
     "folder or the file; check `nemoclaw-blueprint-runner --help`. On a real Spark, `plan` will also look for "
     "/var/run/openshell-target/ca.pem, which only exists once your platform team hands you the bundle.")

step(8, "two ways to reach a remote gateway — who owns what (Part 6, exercise 3)")
table([
    ["who runs the gateway", "you, provisioned from your CLI over SSH", "the platform team (Kubernetes / Helm)"],
    ["lifecycle", "yours: start, stop, destroy, upgrade", "theirs: you get plan + status only"],
    ["how you authenticate", "SSH + mTLS certs the gateway provisions", "a credential file + a pinned CA bundle"],
    ["OpenShell version", "whatever you installed", "pinned: min = max = expected_release"],
    ["name resolution", "`openshell` in your /etc/hosts", "platform DNS: you must control the hostname"],
    ["trust boundary", "your workstation ↔ your Spark", "your claw ↔ someone else's control plane"],
], ["", "openshell gateway start --remote", "nemoclaw-blueprint-runner + external target"])
result(f"Dockerfile lint ✓ · blueprint ✓ · {caught}/{len(BROKEN)} broken blueprints caught · the tutorial's "
       "`gateway add … --remote` needs its SSH target.")
