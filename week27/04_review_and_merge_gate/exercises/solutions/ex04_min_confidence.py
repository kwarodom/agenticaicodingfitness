def aggregate(findings, threshold=70):
    out = []
    for f in findings:
        f = dict(f)
        if f.get("kind") == "over-engineering" and f.get("severity") == "Blocker":
            f["severity"] = "Should fix"; f["reclassified"] = True
        out.append(f)
    conf = min((f["confidence"] for f in out), default=0)
    blockers = [f for f in out if f.get("severity") == "Blocker"]
    return {"confidence": conf, "approved": conf >= threshold and not blockers, "findings": out}
