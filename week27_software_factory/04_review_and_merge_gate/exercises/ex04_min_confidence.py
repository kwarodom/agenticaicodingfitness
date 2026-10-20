"""ex04 · Adversarial aggregation. aggregate(findings, threshold=70) -> dict(confidence, approved, findings)
findings: list of {agent, confidence, severity, kind, text}. Rules:
  1. confidence = MIN over agents' confidences
  2. any finding with kind == 'over-engineering' is reclassified severity 'Should fix' (never a blocker)
  3. approved iff confidence >= threshold and no remaining 'Blocker'
"""
def aggregate(findings, threshold=70):  # TODO
    return {}
