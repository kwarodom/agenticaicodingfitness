#!/usr/bin/env python3
"""Lab 21-1 · Playbook atlas: every official Spark playbook in one table, and where this week covers it.

Reads the local clone of NVIDIA's playbooks (dgx-spark-playbooks/nvidia/playbook-*/README.md) and, for each
one, pulls out the title, the "Estimated time" line, the "Supported hardware platforms" table and the disk
figure from Prerequisites. Then it reads week25/AUTHORING.md to find which Week 25 module covers it.
Everything not covered by Modules 01–20 belongs to this module, grouped into themes.

Runs on the laptop, offline. It only reads files. The other labs and the exercise import its parser.

Run: .venv/bin/python week25/21_playbook_atlas/labs/lab21_1_playbook_atlas.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, WEEK, bar, banner, check, note, result, step, table, warn  # noqa: E402

PLAYBOOKS = ROOT / "dgx-spark-playbooks" / "nvidia"
SKIP = {"test"}                                      # playbook-test is the catalogue's template, not a playbook

# The themes this module groups the "rest" into (course editorial choice, not NVIDIA's).
THEMES = {
    "images & video":        ["comfyui", "multi-modal-inference", "visual-gen-ai", "video-gen-guide"],
    "vision & video agents": ["live-vlm-webui", "vss"],
    "robotics":              ["isaac", "gr00t", "spark-reachy-photo-booth"],
    "data science & HPC":    ["cuda-x-data-science", "portfolio-optimization", "single-cell", "topic-modeling",
                              "jax", "cupynumeric"],
    "kernels & pretraining": ["cutile-kernels", "kernel-dev-ft", "nanochat", "nvfp4-pretraining"],
    "platform & multi-GPU":  ["brev", "mig", "multi-gpu", "connect-two-stations"],
    "Station serving/agents": ["sglang-inference", "dgx-station-ai-skills", "healthcare-agent"],
}
# Older un-prefixed copies of a playbook that were renamed (nvidia/<old-name>/README.md).
TWIN_ALIASES = {"comfyui": "comfy-ui", "brev": "register-to-brev"}
# Slugs listed under https://build.nvidia.com/spark/ — build.nvidia.com/spark index fetched 2026-09-29 by the
# course lead. Not re-fetched by this lab (it stays offline); update the list when the catalogue changes.
SPARK_INDEX = {
    "connect-to-your-spark", "open-webui", "comfyui", "dgx-dashboard", "vllm", "connect-multiple-sparks", "pair",
    "hermes-agent", "cutile-kernels", "cli-coding-agent", "speculative-decoding", "llama-cpp", "nemotron", "sglang",
    "trt-llm", "nvfp4-quantization", "multi-modal-inference", "nim-llm", "lm-studio", "single-cell",
    "portfolio-optimization", "cuda-x-data-science", "txt2kg", "jax", "flux-finetuning", "llama-factory",
    "nemo-fine-tune", "pytorch-fine-tune", "unsloth", "vlm-finetuning", "nemoclaw", "nemoclaw-applications",
    "live-vlm-webui", "isaac", "vibe-coding", "multi-agent-chatbot", "connect-two-sparks", "nccl", "vss",
    "spark-reachy-photo-booth", "openshell", "openclaw", "brev", "rag-ai-workbench", "tailscale", "vscode",
    "connect-three-sparks", "multi-sparks-through-switch",
}
SECTION = {t: i for i, t in enumerate(THEMES, 2)}   # tutorial section number per theme
THEME_OF = {slug: t for t, slugs in THEMES.items() for slug in slugs}


# ── README parsing ─────────────────────────────────────────────────────────────
def section(md: str, heading: str) -> str:
    """Body of the first '## <heading>' up to the next '## ' heading (fence-aware)."""
    out, inside, fence = [], False, False
    for ln in md.splitlines():
        if ln.lstrip().startswith("```"):
            fence = not fence
        if not fence and ln.startswith("## "):
            if inside:
                break
            inside = ln[3:].strip().lower().startswith(heading.lower())
            continue
        if inside:
            out.append(ln)
    return "\n".join(out)


def overview(md: str) -> str:
    """Everything before the first Instructions / Step heading — the 'what is it' part of a README."""
    m = re.search(r"^## (Instructions|Step 1|Image Gen Quick Start|Kernel Benchmarks|Multi-node|Pretrain)",
                  md, re.M)
    return md[:m.start()] if m else md[:6000]


def to_minutes(line: str) -> int | None:
    """First duration in the 'Estimated time' line, upper end of a range: '5–10 MIN' → 10, '2 HOURS' → 120."""
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?\s*(MIN|minutes?|HOURS?|hours?)", line)
    if not m:
        return None
    n = float(m.group(2) or m.group(1))
    return int(n * 60 if m.group(3).lower().startswith("hour") else n)


def platforms(md: str) -> tuple[list[str], str]:
    """Rows of the 'Supported hardware platforms' table → (['DGX Spark', …], 'table'); else inferred."""
    rows = re.findall(r"^\| \*\*([^*]+)\*\*", section(md, "Supported hardware platforms"), re.M)
    rows = list(dict.fromkeys(r.strip() for r in rows))
    if rows:
        return rows, "table"
    seen = [p for p in ("DGX Spark", "DGX Station", "RTX") if p in overview(md)]
    return seen, "inferred"


def disk_gb(md: str) -> int | None:
    """Largest GB/TB figure on the first Prerequisites line that talks about storage or disk."""
    for ln in section(md, "Prerequisites").splitlines():
        if re.search(r"storage|disk", ln, re.I) and re.search(r"\d+\s*(GB|TB)", ln):
            nums = [float(n) * (1000 if u == "TB" else 1) for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*(GB|TB)", ln)]
            return int(max(nums))
    return None


def course_map() -> dict[str, str]:
    """playbook slug → 'NN' from the module table in week25/AUTHORING.md."""
    out = {}
    for ln in (WEEK / "AUTHORING.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d\d)\s*\|\s*`[^`]+`\s*\|[^|]*\|\s*(.+?)\s*\|\s*$", ln)
        if not m:
            continue
        for tok in re.sub(r"\([^)]*\)", "", m.group(2)).split(","):
            tok = tok.strip()
            if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", tok):
                out[tok] = m.group(1)
    return out


def repo_links() -> dict[str, set[str]]:
    """Every build.nvidia.com/<namespace>/<slug> link written anywhere in the clone's READMEs."""
    links: dict[str, set[str]] = {}
    for p in PLAYBOOKS.glob("*/README.md"):
        for ns, slug in re.findall(r"https://build\.nvidia\.com/(spark|station|playbooks)/([a-z0-9-]+)",
                                   p.read_text(encoding="utf-8", errors="replace")):
            links.setdefault(slug, set()).add(ns)
    return links


def load_playbooks() -> list[dict]:
    """One dict per nvidia/playbook-*/README.md (minus the template), with everything the labs need."""
    if not PLAYBOOKS.is_dir():
        sys.exit(f"✕ {PLAYBOOKS} not found — clone https://github.com/NVIDIA/dgx-spark-playbooks at the repo root")
    cmap, out = course_map(), []
    for p in sorted(PLAYBOOKS.glob("playbook-*/README.md")):
        slug = p.parent.name.removeprefix("playbook-")
        if slug in SKIP:
            continue
        md = p.read_text(encoding="utf-8", errors="replace")
        time_line = next((ln for ln in md.splitlines()
                          if re.search(r"\*\*(Estimated time|Duration):\*\*", ln)), "")
        time_line = re.sub(r".*\*\*(Estimated time|Duration):\*\*\s*", "", time_line).strip()
        plats, src = platforms(md)
        tagline = re.search(r"^> (.+)$", md, re.M)
        out.append({
            "slug": slug,
            "title": re.search(r"^# (.+)$", md, re.M).group(1).strip(),
            "tagline": tagline.group(1).strip() if tagline else "",
            "time_line": time_line,
            "minutes": to_minutes(time_line),
            "platforms": plats,
            "platform_source": src,
            "spark": "DGX Spark" in plats,
            "disk_gb": disk_gb(md),
            "module": cmap.get(slug, "21"),
            "theme": THEME_OF.get(slug, ""),
            "overview": overview(md),
            "twin": next((t for t in (slug, TWIN_ALIASES.get(slug, "")) if t and (PLAYBOOKS / t).is_dir()), ""),
            "path": p,
        })
    return out


def plat_label(pb: dict) -> str:
    short = {"DGX Spark": "Spark", "DGX Station": "Station", "RTX or RTX PRO": "RTX", "RTX PRO": "RTX PRO"}
    s = " + ".join(short.get(x, x) for x in pb["platforms"]) or "?"
    return s + (" (inferred)" if pb["platform_source"] == "inferred" else "")


def build_link(pb: dict) -> str:
    """build.nvidia.com URL: /spark/ for Spark playbooks, /station/ for Station-only, /playbooks/ otherwise.

    The namespace follows the links the clone itself uses (/spark/…, /station/…, /playbooks/…); the slug is the
    directory name without 'playbook-'. Lab step 4 checks it against SPARK_INDEX and the README links.
    """
    ns = "spark" if pb["spark"] else ("station" if any("Station" in x for x in pb["platforms"]) else "playbooks")
    return f"https://build.nvidia.com/{ns}/{pb['slug']}"


def fmt_time(m: int | None) -> str:
    if m is None:
        return "—"
    return f"{m} min" if m < 120 else f"{m / 60:g} h"


# ── the lab ────────────────────────────────────────────────────────────────────
def main() -> None:
    banner("Lab 21-1 · playbook atlas", "every nvidia/playbook-*/README.md, parsed on this laptop", status=False)

    step(1, "scan the playbook clone")
    pbs = load_playbooks()
    print(f"$ ls dgx-spark-playbooks/nvidia/playbook-*/README.md   → {len(pbs) + len(SKIP)} files "
          f"({', '.join('playbook-' + s for s in SKIP)} skipped: it is the catalogue template)")
    note(f"{len(pbs)} playbooks · {sum(p['spark'] for p in pbs)} list DGX Spark in 'Supported hardware platforms'")

    step(2, "every playbook: time, platforms, and the Week 25 module that covers it")
    rows = []
    for pb in sorted(pbs, key=lambda p: (p["module"], p["slug"])):
        rows.append([pb["slug"], pb["title"][:40], fmt_time(pb["minutes"]), "✓" if pb["spark"] else "✕",
                     plat_label(pb), "Module " + pb["module"]])
    table(rows, ["playbook", "title", "time", "Spark?", "platforms", "covered by"])

    atlas = [p for p in pbs if p["module"] == "21"]
    step(3, f"the {len(atlas)} playbooks Modules 01–20 do not cover, by theme (this module)")
    trows, counts = [], []
    for theme, slugs in THEMES.items():
        mine = [p for p in atlas if p["theme"] == theme]
        n_spark = sum(p["spark"] for p in mine)
        n_station = sum(1 for p in mine if not p["spark"] and any("Station" in x for x in p["platforms"]))
        n_rtx = len(mine) - n_spark - n_station
        counts.append((theme, len(mine), n_spark, n_station, n_rtx))
        trows.append([f"§{SECTION[theme]} {theme}", len(mine), n_spark, n_station, n_rtx,
                      ", ".join(p["slug"] for p in mine)])
    table(trows, ["theme (tutorial section)", "all", "Spark", "Station only", "RTX only", "playbooks"])
    for theme, n, n_spark, *_ in counts:
        print(f"│ {theme:24s} {bar(n_spark, n, 14)} {n_spark} of {n} run on a Spark")

    step(4, "build.nvidia.com slugs — the /spark index (fetched 2026-09-29) and links inside the clone")
    links, srows = repo_links(), []
    for pb in atlas:
        ns = links.get(pb["slug"])
        in_readme = f"/{'/'.join(sorted(ns))}/{pb['slug']}" if ns else \
            ("/playbooks/playbook-cutile-kernels" if pb["slug"] == "cutile-kernels" else "—")
        if pb["slug"] in SPARK_INDEX:
            verdict = "✓ confirmed /spark/" + pb["slug"]
        elif pb["spark"]:
            verdict = "✕ Spark playbook, not in /spark index"
        else:
            verdict = "unverified (" + ("Station" if "Station" in plat_label(pb) else "RTX") + " only)"
        srows.append([pb["slug"], verdict, in_readme])
    table(srows, ["dir slug", "build.nvidia.com/spark index", "linked in a README as"])
    confirmed = sum(p["slug"] in SPARK_INDEX for p in atlas)
    no_dir = sorted(SPARK_INDEX - {p["slug"] for p in pbs})
    spark_missing = sorted(p["slug"] for p in pbs if p["spark"] and p["slug"] not in SPARK_INDEX)
    note(f"{confirmed} of {len(atlas)} atlas slugs are confirmed on the /spark index. "
         f"Station/RTX slugs stay unverified: the index lists Spark playbooks only.")
    note(f"on the /spark index but no playbook-* dir in this clone: {', '.join(no_dir) or '—'}")
    note(f"list DGX Spark in their README but not on the /spark index: {', '.join(spark_missing) or '—'}")

    step(5, "consistency checks")
    unmapped = [p["slug"] for p in atlas if not p["theme"]]
    missing = sorted(set(course_map()) - {p["slug"] for p in pbs})
    ok = check(not unmapped, f"every uncovered playbook has a theme ({len(atlas)} of {len(atlas)})",
               f"no theme yet: {', '.join(unmapped)}")
    ok &= check(not missing, "every playbook named in AUTHORING.md exists in the clone",
                f"named in AUTHORING.md but not in the clone: {', '.join(missing)}")
    if not ok:
        warn("the clone moved on since this module was written — update THEMES or AUTHORING.md")
    result(f"{len(pbs)} playbooks: {len(pbs) - len(atlas)} covered in Modules 01–20, {len(atlas)} in this atlas "
           f"({sum(p['spark'] for p in atlas)} of them run on a DGX Spark).")


if __name__ == "__main__":
    main()
