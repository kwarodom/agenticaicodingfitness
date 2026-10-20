#!/usr/bin/env python3
"""Lab 17-1 · Build the dataset — and prove the test set is not hiding in it.

Most fine-tuning failures are data failures, so this lab comes before any model:
  1. generate a training pool from ~70 hand-written situations (datagen.py)
  2. split it two ways and measure leakage: a RANDOM split puts re-worded copies on both sides,
     a split BY SITUATION does not
  3. check the hand-written test set against everything we will train on
  4. look at the label balance per language
  5. write the two training formats — Laya JSONL and GLiNER2 JSONL — from the SAME question set
  6. (optional) ask Jev to label a sample: a cheap second opinion on the generator's labels

Free and fast: no model is loaded (step 6 replays recorded Jev answers in DRY mode).
Run: .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab01_build_dataset.py
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ftkit as F  # noqa: E402  (first: may re-run this lab inside week24_typed_decisions_jev/.venv-local)
import datagen  # noqa: E402
import jevkit  # noqa: E402
from jevkit import banner, step, table  # noqa: E402

banner("Lab 17-1 · build the dataset", "generate · split · leak-check · balance · write both formats")

# ── 1 · generate ──────────────────────────────────────────────────────────────
step(1, "generate a pool from hand-written situations")
pool = datagen.generate(seed=17, per_situation=8)
langs = collections.Counter(t["lang"] for t in pool)
print(f"» {len(datagen.SITUATIONS)} situations × 8 variants = {len(pool)} tickets "
      f"({langs['en']} English, {langs['th']} Thai)")
rng = random.Random(3)
for t in rng.sample(pool, 4):
    print(f"   {t['queue']:<12} urgency {t['urgency']}  {t['text']}")
print("» Labels come from the situation, so 576 tickets needed ~70 labelling decisions, not 576.")

# ── 2 · two splits, one of them leaky ─────────────────────────────────────────
step(2, "random split vs split by situation — how many validation tickets have a near-copy in training?")
NEAR = 0.5                                                   # char-5-gram Jaccard, room numbers masked


def leak_share(train, val):
    sims = [F.nearest(t["text"], train)[0] for t in val]
    return sum(s >= NEAR for s in sims) / len(sims), max(sims)


tr_r, va_r = datagen.split_random(pool)
tr_s, va_s = datagen.split_by_situation(pool)
lr, mr = leak_share(tr_r, va_r)
ls, ms = leak_share(tr_s, va_s)
table([["random (shuffle tickets)", len(tr_r), len(va_r), f"{lr:.0%}", f"{mr:.2f}"],
       ["by situation (group split)", len(tr_s), len(va_s), f"{ls:.0%}", f"{ms:.2f}"]],
      ["split", "train", "val", f"val with a near-copy (≥{NEAR})", "closest pair"])
print("» A random split grades the model on re-worded copies of its own homework. The group split")
print("  asks the real question: does it handle a situation it has never seen? We use the group split.")
train, val = tr_s, va_s

# ── 3 · the hand-written test set ─────────────────────────────────────────────
step(3, "is any hand-written test ticket a near-copy of the training data?")
test, m16 = F.split("test"), F.split("m16")
rows = []
for name, tickets in (("test (48, hand-written)", test), ("Module 16 (18)", m16)):
    sims = [(F.nearest(t["text"], train + val), t) for t in tickets]
    worst = max(sims, key=lambda x: x[0][0])
    rows.append([name, f"{sum(s >= NEAR for (s, _), _t in sims)}/{len(sims)}", f"{worst[0][0]:.2f}",
                 worst[1]["text"][:46] + "…"])
table(rows, ["held-out set", f"near-copies (≥{NEAR})", "closest", "its closest ticket"])
(sim, who), t = max(((F.nearest(t["text"], train + val), t) for t in m16), key=lambda x: x[0][0])
print(f"» closest pair overall:\n   test  : {t['text']}\n   train : {who['text']}  (similarity {sim:.2f})")
print("» Same topic is fine — the test must cover the same jobs. Same SENTENCE is a leak. Anything ≥ 0.5")
print("  deserves a look: here it is one short Thai towel request, the same ask in near-identical words.")
print("  We keep it and say so. Neither held-out set is ever trained on; Lab 17-6 grades on both.")

# ── 4 · balance ───────────────────────────────────────────────────────────────
step(4, "label balance per split and language")


def bal(tickets):
    c = collections.Counter(t["queue"] for t in tickets)
    return [c[q] for q in F.QUEUES] + [sum(t["safety"] for t in tickets),
                                       "/".join(str(sum(t["urgency"] == u for t in tickets)) for u in range(3))]


table([[f"{name} · {lg}", *bal([t for t in ts if t["lang"] == lg])]
       for name, ts in (("train", train), ("val", val), ("test", test)) for lg in ("en", "th")],
      ["split", *F.QUEUES, "safety=yes", "urgency 0/1/2"])
print("» Thai is ~40 % of training, on purpose: the base models are weakest there (Module 16).")

# ── 5 · write both formats ────────────────────────────────────────────────────
step(5, "write data/ — one question set, two training formats")
F.DATA.mkdir(exist_ok=True)
for name, ts in (("train", train), ("val", val)):
    (F.DATA / f"{name}.json").write_text(json.dumps(ts, ensure_ascii=False, indent=0), encoding="utf-8")
    F.write_jsonl(F.DATA / f"laya_{name}.jsonl", F.to_laya_rows(ts))
    F.write_jsonl(F.DATA / f"gliner_{name}.jsonl", F.to_gliner_records(ts))
for f in ("train.json", "val.json", "laya_train.jsonl", "laya_val.jsonl", "gliner_train.jsonl", "gliner_val.jsonl"):
    p = F.DATA / f
    print(f"   ✓ data/{f:<20} {sum(1 for _ in p.open(encoding='utf-8')) if f.endswith('l') else len(json.loads(p.read_text())):>4} rows"
          f"  {p.stat().st_size / 1024:6.0f} KB")
lr0, gr0 = F.to_laya_rows(train[:1])[0], F.to_gliner_records(train[:1])[0]
print("» Laya row   :", json.dumps({"state": lr0["state"], "questions": "…the Q dict from ftkit…",
                                     "expected": lr0["expected"]}, ensure_ascii=False))
g = gr0["output"]["classifications"][0]
print("» GLiNER row :", json.dumps({"input": gr0["input"][:40] + "…", "task": g["task"], "true_label": g["true_label"],
                                     "prompt": g["prompt"], "label_descriptions": "{…same descriptions as Q…}"},
                                    ensure_ascii=False))
print("» Train the way you infer: both rows come from ftkit.Q through the same adapter Lab 16-4 uses.")

# ── 6 · a second opinion from Jev (optional) ──────────────────────────────────
step(6, "ask Jev to label 24 training tickets — does the teacher agree with the generator?")
sample = random.Random(6).sample(train, 24)
got, agree, disagree = 0, collections.Counter(), []
for t in sample:
    r = jevkit.ask({"message": t["text"]}, F.Q, quiet=True)
    if r.get("_source") == "placeholder":
        continue
    got += 1
    a = jevkit.validate(r, F.Q)
    jq, js, ju = a["queue"]["choice"], a["safety"]["noul"] >= 0.5, round(a["urgency"]["score"])
    agree["queue"] += jq == t["queue"]
    agree["safety"] += js == t["safety"]
    agree["urgency"] += ju == t["urgency"]
    if jq != t["queue"] or ju != t["urgency"]:
        disagree.append((t, jq, ju))
if not got:
    print("» no recorded Jev answers for this sample and no key — skipped (add TYPESAFE_API_KEY to try it).")
else:
    print(f"» Jev ({jevkit.mode().upper()}) agrees with the generator: queue {agree['queue']}/{got} · "
          f"safety {agree['safety']}/{got} · urgency {agree['urgency']}/{got}   cost ≈ ${got * 480 * jevkit.PRICE_PER_MTOK / 1e6:.4f}")
    for t, jq, ju in disagree[:5]:
        print(f"   ≠ gen {t['queue']}/{t['urgency']} · jev {jq}/{ju}  {t['text'][:70]}")
    print("» Disagreements are where a PERSON looks: sometimes the teacher is wrong, sometimes the")
    print("  generator's label is. With no generator, Jev-as-teacher labels a few hundred real tickets for")
    print("  well under a cent — then people check only the ones where Jev was unsure.")
