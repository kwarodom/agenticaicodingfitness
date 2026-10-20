"""An offline evaluator for `nat eval` (plugged in with `_type: langsmith_custom`, no LLM judge).

openevals calling convention: (inputs, outputs, reference_outputs) → {"key", "score", "comment"}.
The dataset's `answer` field lists the facts the reply must contain, separated by "|".
Score = fraction of those facts found in the agent's answer (case-insensitive).
"""


def mentions_expected(inputs=None, outputs=None, reference_outputs=None, **_):
    answer = str(outputs if not isinstance(outputs, dict) else outputs.get("output", outputs)).lower()
    ref = reference_outputs if not isinstance(reference_outputs, dict) else reference_outputs.get("output", "")
    facts = [f.strip().lower() for f in str(ref or "").split("|") if f.strip()]
    if not facts:
        return {"key": "mentions_expected", "score": 0.0, "comment": "no expected facts in the dataset row"}
    found = [f for f in facts if f in answer]
    missing = [f for f in facts if f not in answer]
    return {"key": "mentions_expected", "score": round(len(found) / len(facts), 3),
            "comment": f"found {found} · missing {missing}"}
