"""ex06 · The four-step verifier as a pure function.
verify(candidate, board, merged_prs, since) -> {"verdict": file|drop-fixed|drop-stale|drop-noise|duplicate|needs-triage, "why": str, "ref": str|None}
Order: real → still happening → already fixed → duplicate. "Unclear is not a drop" → needs-triage.
candidate: {id, title, fingerprint, route, count, first_seen, last_seen, deploy_window?}
board: list of open cards {number, title, fingerprint?, route?}
merged_prs: list {number, fingerprint, merged_at}
"""
def verify(candidate, board, merged_prs, since):  # TODO
    return {"verdict": "needs-triage", "why": "not implemented", "ref": None}
