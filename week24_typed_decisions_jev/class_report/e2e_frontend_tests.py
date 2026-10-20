"""Week 24 web-frontend tests for the Jev Lab Runner — screenshots + videos for the class report.

Runs against http://127.0.0.1:8124 in DRY mode (Jev answers replay recordings, $0; Module 16's
local models still run for real on this Mac). Writes:
  OUT/shots/*.png   OUT/clips/*.webm   OUT/results.json
"""
import json
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8124/"
OUT = Path(sys.argv[1])
SHOTS, CLIPS = OUT / "shots", OUT / "clips"
for d in (SHOTS, CLIPS):
    d.mkdir(parents=True, exist_ok=True)
RESULTS = []
INIT = "localStorage.setItem('jl.mode','dry');localStorage.setItem('jl.lang','en');localStorage.removeItem('jl.last');"


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


def fid(pg, fi, rel):
    return pg.evaluate(f"fid({fi},{json.dumps(rel)})")


def run_lab(pg, fi, rel, timeout=150_000):
    i = fid(pg, fi, rel)
    t0 = time.time()
    pg.click(f"#runbtn-{i}")
    pg.wait_for_selector(f"#foot-{i}.show", timeout=timeout)
    foot = pg.inner_text(f"#foot-{i}")
    term = pg.inner_text(f"#term-{i}")
    return i, foot, term, round(time.time() - t0, 1)


def card(pg, i):
    return pg.locator(f"#runbtn-{i}").locator("xpath=ancestor::article")


def shot(pg, name, locator=None, full=False):
    path = SHOTS / f"{name}.png"
    (locator.screenshot(path=str(path)) if locator is not None else pg.screenshot(path=str(path), full_page=full))
    return path.name


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # ───────────────────────── screenshots + assertions ─────────────────────────
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        ctx.add_init_script(INIT)
        pg = ctx.new_page()
        console_errors = []
        pg.on("pageerror", lambda e: console_errors.append(str(e)))
        ready(pg)

        mods = pg.evaluate("COURSE.map(c=>c.folder)")
        record("Course loads with 16 modules", len(mods) == 16 and mods[-2:] == ["15_jev_sees_images", "16_local_system_one_models"],
               f"{len(mods)} modules; last two: {mods[-2:]}", [shot(pg, "01_home")])
        models = pg.evaluate("[...document.querySelectorAll('#model-sel option')].map(o=>o.textContent)")
        record("Model picker offers jev-1.14.0", any("jev-1.14.0" in m for m in models), " · ".join(models))
        mode = pg.inner_text("#pill-mode") if pg.query_selector("#pill-mode") else ""
        record("Header shows DRY mode ($0 replays)", "DRY" in mode.upper() or "dry" in mode.lower(), mode.strip()[:60])

        FI15 = mods.index("15_jev_sees_images")
        FI16 = mods.index("16_local_system_one_models")

        # Module 15 — intro, box section, ask-jev, diagrams
        goto(pg, FI15, 0)
        chips = pg.inner_text(".metachips") if pg.query_selector(".metachips") else ""
        record("M15 intro renders with time/difficulty/cost chips", "40 min" in chips, chips.replace("\n", " · "),
               [shot(pg, "02_m15_intro")])
        goto(pg, FI15, 1)
        record("M15 §0 'what 1.14 adds' table renders", pg.query_selector("#content table") is not None, "",
               [shot(pg, "03_m15_what_is_new")])
        goto(pg, FI15, 4)
        img = pg.query_selector("#content img[src*='puddle_box_example']")
        loaded = img and pg.evaluate("(i)=>i.complete && i.naturalWidth>0", img)
        if img:
            img.scroll_into_view_if_needed()
            pg.wait_for_timeout(300)
        record("M15 §3 Box section shows the drawn-box photo", bool(loaded), "media/puddle_box_example.jpg loaded" if loaded else "image missing",
               [shot(pg, "04_m15_box_section")])
        goto(pg, FI15, 6)
        pg.locator(".jevblk").first.scroll_into_view_if_needed()
        pg.click("#jb-ask-0")
        pg.wait_for_selector("#jb-out-0.show", timeout=30_000)
        out = pg.inner_text("#jb-out-0")
        record("M15 inline ⚡ Ask Jev returns typed answers", "alarm" in out and "PLACEHOLDER" not in out,
               out.splitlines()[0][:90], [shot(pg, "05_m15_ask_jev", pg.locator(".jevblk").first)])
        goto(pg, FI15, 7)
        pg.wait_for_timeout(2500)
        svgs = pg.evaluate("document.querySelectorAll('#content svg').length")
        record("M15 📊 diagrams render (architecture · sequence · charts)", svgs >= 3, f"{svgs} SVG diagrams",
               [shot(pg, "06_m15_diagrams", full=False)])
        pg.evaluate("document.getElementById('content').scrollTop=document.getElementById('content').scrollHeight")
        pg.wait_for_timeout(3000)
        shot(pg, "06b_m15_charts")

        # Module 15 labs
        goto(pg, FI15, 8)
        for n, rel in enumerate(["labs/lab01_first_image_call.py", "labs/lab02_where_images_go.py",
                                 "labs/lab03_box_and_crop.py", "labs/lab04_direct_vs_pipeline.py",
                                 "labs/lab05_versions_and_long_context.py"], 1):
            i, foot, term, secs = run_lab(pg, FI15, rel)
            ok = "exit 0" in foot and "PLACEHOLDER" not in term
            key = {1: "input tok", 2: "haze_appeared", 3: "gate 0.09", 4: "3/3 photos", 5: "max_tokens_exceeded"}[n]
            ok = ok and key in term
            card(pg, i).scroll_into_view_if_needed()
            record(f"M15 lab0{n} runs in the browser", ok, f"{foot.strip()} · {secs}s · found '{key}'",
                   [shot(pg, f"07_m15_lab0{n}", card(pg, i))])
        goto(pg, FI15, 9)
        i, foot, term, secs = run_lab(pg, FI15, "exercises/solutions/ex15_image_preflight.py")
        record("M15 exercise solution passes all checks", "exit 0" in foot and "✕" not in term.split("all offline tests pass")[0],
               f"{foot.strip()} · {secs}s", [shot(pg, "08_m15_exercise", card(pg, i))])
        i, foot, term, secs = run_lab(pg, FI15, "exercises/ex15_image_preflight.py")
        record("M15 exercise starter stops before any API call", "no API call was made" in term,
               "starter shows ✕ TODO lines and spends nothing", [shot(pg, "08b_m15_exercise_starter", card(pg, i))])

        # Thai
        pg.evaluate("setLang('th')")
        pg.wait_for_timeout(900)
        goto(pg, FI15, 0)
        th = pg.inner_text("#content h2") + pg.inner_text("#content .md")[:200]
        record("M15 Thai version renders (ไทย toggle)", any("฀" <= ch <= "๿" for ch in th), th[:70].replace("\n", " "),
               [shot(pg, "09_m15_thai")])
        goto(pg, FI16, 6)
        shot(pg, "09b_m16_thai_bakeoff")
        pg.evaluate("setLang('en')")
        pg.wait_for_timeout(900)

        # Module 16
        goto(pg, FI16, 0)
        record("M16 intro renders", "Laya" in pg.inner_text("#content"), "", [shot(pg, "10_m16_intro")])
        goto(pg, FI16, 2)
        shot(pg, "11_m16_landscape")
        goto(pg, FI16, 7)
        pg.wait_for_timeout(2500)
        svgs = pg.evaluate("document.querySelectorAll('#content svg').length")
        record("M16 📊 diagrams render", svgs >= 3, f"{svgs} SVG diagrams", [shot(pg, "12_m16_diagrams")])
        pg.evaluate("document.getElementById('content').scrollTop=document.getElementById('content').scrollHeight")
        pg.wait_for_timeout(3000)
        shot(pg, "12b_m16_charts")
        goto(pg, FI16, 8)
        for rel, key in [("labs/lab01_landscape.py", "Jev Decision Index"),
                         ("labs/lab02_laya_local.py", "LOCAL (this machine)"),
                         ("labs/lab03_gliner_extract_then_decide.py", "⟦1203⟧"),
                         ("labs/lab04_gliner_decide.py", "GLiNER2.5-multi-Decide"),
                         ("labs/lab05_bakeoff.py", "18/18")]:
            i, foot, term, secs = run_lab(pg, FI16, rel)
            ok = "exit 0" in foot and key in term
            card(pg, i).scroll_into_view_if_needed()
            local = "LOCAL models" in term
            record(f"M16 {rel.split('/')[1][:5]} runs in the browser", ok,
                   f"{foot.strip()} · {secs}s · {'local models ran on this Mac' if local else 'offline/recorded'} · found '{key}'",
                   [shot(pg, f"13_m16_{rel.split('/')[1][:5]}", card(pg, i))])
        goto(pg, FI16, 9)
        i, foot, term, secs = run_lab(pg, FI16, "exercises/solutions/ex16_local_or_hosted.py")
        record("M16 exercise solution passes all checks", "exit 0" in foot and "all offline tests pass" in term,
               f"{foot.strip()} · {secs}s", [shot(pg, "14_m16_exercise", card(pg, i))])

        # Regression: an older module still works with the updated client
        goto(pg, 0, 0)
        labs_sec = pg.evaluate("COURSE[0].sections.findIndex(s=>s.kind==='labs')")
        goto(pg, 0, labs_sec)
        first = pg.evaluate("COURSE[0].labs[0].file")
        i, foot, term, secs = run_lab(pg, 0, first)
        record("Regression: Module 01 lab still runs (updated Jev client)", "exit 0" in foot and "PLACEHOLDER" not in term,
               f"{first} · {foot.strip()}", [shot(pg, "15_m01_regression", card(pg, i))])
        record("No JavaScript errors in the page", not console_errors, "; ".join(console_errors)[:200])
        ctx.close()

        # ───────────────────────── video clips for GIFs ─────────────────────────
        def clip(name, fn):
            c = browser.new_context(viewport={"width": 1280, "height": 760}, record_video_dir=str(CLIPS / "_tmp"),
                                    record_video_size={"width": 1280, "height": 760})
            c.add_init_script(INIT)
            pp = c.new_page()
            ready(pp)
            fn(pp)
            pp.wait_for_timeout(1500)
            vid = pp.video.path()
            c.close()
            shutil.move(vid, CLIPS / f"{name}.webm")
            print(f"🎬 {name}.webm", flush=True)

        def tour(pp):
            for si in (0, 1, 2, 4, 7):
                pp.evaluate(f"gotoSec({FI15},{si})")
                pp.wait_for_timeout(1600)
                if si == 4:
                    im = pp.query_selector("#content img[src*='puddle_box_example']")
                    if im:
                        pp.evaluate("(i)=>i.scrollIntoView({behavior:'smooth',block:'center'})", im)
                        pp.wait_for_timeout(1800)

        def box_lab(pp):
            pp.evaluate(f"gotoSec({FI15},8)")
            pp.wait_for_timeout(800)
            i = fid(pp, FI15, "labs/lab03_box_and_crop.py")
            card(pp, i).scroll_into_view_if_needed()
            pp.wait_for_timeout(600)
            pp.click(f"#runbtn-{i}")
            pp.wait_for_selector(f"#foot-{i}.show", timeout=60_000)
            for _ in range(6):
                pp.mouse.wheel(0, 220)
                pp.wait_for_timeout(450)

        def ask_jev(pp):
            pp.evaluate(f"gotoSec({FI15},6)")
            pp.wait_for_timeout(700)
            pp.locator(".jevblk").first.scroll_into_view_if_needed()
            pp.wait_for_timeout(900)
            pp.click("#jb-ask-0")
            pp.wait_for_selector("#jb-out-0.show", timeout=30_000)
            pp.locator("#jb-out-0").scroll_into_view_if_needed()
            pp.wait_for_timeout(1500)

        def bakeoff(pp):
            pp.evaluate(f"gotoSec({FI16},8)")
            pp.wait_for_timeout(800)
            i = fid(pp, FI16, "labs/lab05_bakeoff.py")
            card(pp, i).scroll_into_view_if_needed()
            pp.wait_for_timeout(600)
            pp.click(f"#runbtn-{i}")
            pp.wait_for_selector(f"#foot-{i}.show", timeout=150_000)
            for _ in range(6):
                pp.mouse.wheel(0, 220)
                pp.wait_for_timeout(450)

        clip("tour_m15", tour)
        clip("m15_box_lab", box_lab)
        clip("m15_ask_jev", ask_jev)
        clip("m16_bakeoff", bakeoff)
        shutil.rmtree(CLIPS / "_tmp", ignore_errors=True)
        browser.close()

    (OUT / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = sum(r["pass"] for r in RESULTS)
    print(f"\n◆ {passed}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
