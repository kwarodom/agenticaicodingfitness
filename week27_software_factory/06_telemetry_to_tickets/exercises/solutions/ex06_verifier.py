def verify(c, board, merged_prs, since):
    if not c.get("count") or not c.get("fingerprint"):
        return {"verdict": "needs-triage", "why": "no evidence to open", "ref": None}
    if c.get("deploy_window") and c["count"] <= 5:
        return {"verdict": "drop-noise", "why": f"{c['count']} events inside a deploy window", "ref": None}
    if c["last_seen"] < since:
        return {"verdict": "drop-stale", "why": f"last seen {c['last_seen']} before {since}", "ref": None}
    for pr in merged_prs:
        if pr["fingerprint"] == c["fingerprint"] and c["last_seen"] <= pr["merged_at"]:
            return {"verdict": "drop-fixed", "why": f"PR #{pr['number']} merged {pr['merged_at']}, no events after", "ref": f"#{pr['number']}"}
    for card in board:
        if card.get("fingerprint") == c["fingerprint"] or (card.get("route") and card.get("route") == c.get("route") and card["title"].split(":")[0] in c["title"]):
            return {"verdict": "duplicate", "why": f"same fingerprint/route as open #{card['number']}", "ref": f"#{card['number']}"}
    return {"verdict": "file", "why": "real, current, unfixed, not on the board", "ref": None}
