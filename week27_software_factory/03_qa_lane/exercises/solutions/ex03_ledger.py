import re
STOP = {"the", "with", "when", "that", "should", "returns", "return", "from", "into", "for"}
def keywords(ac):
    return [w for w in re.findall(r"[a-z_]+", ac.lower()) if len(w) > 3 and w not in STOP]
def ledger(acs, test_ids):
    rows = []
    for ac in acs:
        kws = keywords(ac); best, hits = None, 0
        for t in test_ids:
            name = t.split("::")[-1].lower()
            h = sum(1 for k in kws if k in name)
            if h > hits: best, hits = t, h
        covered = kws and (hits == len(kws) or hits >= 2)
        rows.append({"ac": ac, "unit": best if covered else "none", "gap": "—" if covered else "High"})
    return rows
