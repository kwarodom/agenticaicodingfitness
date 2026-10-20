"""Module 17 web-frontend tests for the Jev Lab Runner (DRY mode). Usage: e2e17.py OUT [--thai-only]"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8124/"
OUT = Path(sys.argv[1])
THAI_ONLY = "--thai-only" in sys.argv
SHOTS = OUT / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)
RESULTS = []
INIT = "localStorage.setItem('jl.mode','dry');localStorage.setItem('jl.lang','en');localStorage.removeItem('jl.last');"
M = "17_finetune_system_one"


def record(name, ok, detail="", shots=()):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail, "shots": list(shots)})
    print(("✓ " if ok else "✕ ") + name + (f" — {detail}" if detail else ""), flush=True)


def ready(pg):
    pg.goto(BASE)
    pg.wait_for_function("typeof COURSE!=='undefined' && COURSE.length>0 && document.querySelector('#content h2')")
    pg.wait_for_timeout(600)


def goto(pg, fi, si):
    pg.evaluate(f"gotoSec({fi},{si})")
    pg.wait_for_timeout(500)


def run_lab(pg, fi, rel, timeout=150_000):
    i = pg.evaluate(f"fid({fi},{json.dumps(rel)})")
    t0 = time.time()
    pg.click(f"#runbtn-{i}")
    pg.wait_for_selector(f"#foot-{i}.show", timeout=timeout)
    return i, pg.inner_text(f"#foot-{i}"), pg.inner_text(f"#term-{i}"), round(time.time() - t0, 1)


def card(pg, i):
    return pg.locator(f"#runbtn-{i}").locator("xpath=ancestor::article")


def shot(pg, name, locator=None):
    path = SHOTS / f"{name}.png"
    (locator.screenshot(path=str(path)) if locator is not None else pg.screenshot(path=str(path)))
    return path.name


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        ctx.add_init_script(INIT)
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        ready(pg)
        mods = pg.evaluate("COURSE.map(c=>c.folder)")
        FI = mods.index(M)
        secs = pg.evaluate(f"COURSE[{FI}].sections.map(s=>[s.kind,s.title])")
        kind = [k for k, _ in secs]
        LABS, EX = kind.index("labs"), kind.index("exercises")
        if not THAI_ONLY:
            record("Course loads with 17 modules, Module 17 last", len(mods) == 17 and mods[-1] == M, f"{len(mods)} modules")
            goto(pg, FI, 0)
            txt = pg.inner_text("#content")
            record("M17 intro renders", "Fine-tune" in txt and "456" in txt, "", [shot(pg, "m17_01_intro")])
            for si, name in ((2, "m17_02_dataset"), (4, "m17_03_criteria"), (7, "m17_04_exam"), (8, "m17_04_ship")):
                goto(pg, FI, si)
                shot(pg, name)
            diag = kind.index("diagrams") if "diagrams" in kind else None
            if diag is not None:
                goto(pg, FI, diag)
                pg.wait_for_timeout(2500)
                n = pg.evaluate("document.querySelectorAll('#content svg').length")
                record("M17 📊 diagrams + 3 charts render", n >= 5, f"{n} SVGs", [shot(pg, "m17_05_diagrams")])
                pg.evaluate("document.getElementById('content').scrollTop=document.getElementById('content').scrollHeight")
                pg.wait_for_timeout(3000)
                shot(pg, "m17_06_charts")
            else:
                record("M17 📊 diagrams + 3 charts render", False, f"no diagrams section in {kind}")
            goto(pg, FI, LABS)
            for rel, key in [("labs/lab01_build_dataset.py", "0/48"), ("labs/lab02_mac_or_spark.py", "never mps"),
                             ("labs/lab03_baselines_and_criteria.py", "locked: sha256"),
                             ("labs/lab04_finetune_laya.py", "55/66"), ("labs/lab05_finetune_gliner.py", "108/120"),
                             ("labs/lab06_evaluate.py", "McNemar"), ("labs/lab07_ship_or_hold.py", "HOLD"),
                             ("labs/lab08_run_on_spark.py", "NOT RUN")]:
                i, foot, term, s = run_lab(pg, FI, rel)
                card(pg, i).scroll_into_view_if_needed()
                ok = "exit 0" in foot and key in term and "Traceback" not in term
                record(f"M17 {rel.split('/')[1][:5]} runs in the browser", ok,
                       f"{foot.strip()} · {s}s · {'LOCAL models' if 'LOCAL models' in term else 'recorded'} · found '{key}'",
                       [shot(pg, f"m17_lab_{rel.split('/')[1][:5]}", card(pg, i))])
            goto(pg, FI, EX)
            i, foot, term, s = run_lab(pg, FI, "exercises/solutions/ex17_split_and_ship.py")
            record("M17 exercise solution passes all 12 checks", "exit 0" in foot and "all offline tests pass" in term,
                   f"{foot.strip()} · {s}s", [shot(pg, "m17_ex_solution", card(pg, i))])
            i, foot, term, s = run_lab(pg, FI, "exercises/ex17_split_and_ship.py")
            record("M17 exercise starter shows the TODOs", "✕ TODO 1" in term and "✕ TODO 2" in term,
                   "starter fails with ✕ TODO lines", [shot(pg, "m17_ex_starter", card(pg, i))])
        # Thai
        goto(pg, FI, 5)
        pg.evaluate("setLang('th')")
        pg.wait_for_timeout(1200)
        th = pg.inner_text("#content")
        thai_chars = sum(1 for ch in th if "฀" <= ch <= "๿")
        record("M17 Thai toggle shows the Thai lesson", thai_chars > 200, f"{thai_chars} Thai characters on §4 (Laya)",
               [shot(pg, "m17_th_laya")])
        goto(pg, FI, 8)
        shot(pg, "m17_th_ship")
        pg.evaluate("setLang('en')")
        record("No JavaScript errors in the page", not errs, "; ".join(errs)[:200])
        ctx.close()
        browser.close()
    prev = json.loads((OUT / "results.json").read_text()) if THAI_ONLY and (OUT / "results.json").is_file() else []
    keep = [r for r in prev if not r["test"].startswith(("M17 Thai", "No JavaScript"))] if THAI_ONLY else []
    allr = keep + RESULTS
    (OUT / "results.json").write_text(json.dumps(allr, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n◆ {sum(r['pass'] for r in allr)}/{len(allr)} checks passed")


if __name__ == "__main__":
    main()
