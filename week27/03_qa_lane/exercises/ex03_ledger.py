"""ex03 · Ledger skeleton. ledger(acs, test_ids) -> list of rows {ac, unit, gap}
acs: list of AC strings. test_ids: pytest `--collect-only -q` lines like tests/test_rollup.py::test_empty_is_zero
Match rule: an AC is covered when every keyword (len>3, lowercased, non-stopword) of the AC appears in a test id,
OR at least two keywords appear. Otherwise unit='none' and gap='High'.
"""
STOP = {"the", "with", "when", "that", "should", "returns", "return", "from", "into"}
def keywords(ac):  # TODO
    return []
def ledger(acs, test_ids):  # TODO
    return []
