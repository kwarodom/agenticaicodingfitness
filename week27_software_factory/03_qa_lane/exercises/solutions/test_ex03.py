import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex03_ledger import ledger
TESTS = ["tests/test_rollup.py::test_empty_list_is_zero", "tests/test_rollup.py::test_none_entries_ignored", "tests/test_api.py::test_health"]
def test_covered_and_gap():
    rows = ledger(["empty list returns zero", "none entries ignored", "default timezone used when tz omitted"], TESTS)
    assert rows[0]["unit"].endswith("test_empty_list_is_zero") and rows[1]["gap"] == "—"
    assert rows[2] == {"ac": "default timezone used when tz omitted", "unit": "none", "gap": "High"}
