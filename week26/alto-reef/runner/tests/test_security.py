from app.security import validate_name, run_argv
import pytest

def test_valid_names():
    for n in ["coral-cove", "a", "alto-ops-2"]:
        assert validate_name(n) == n

@pytest.mark.parametrize("bad", ["", "-x", "Coral", "a b", "a;rm", "x" * 41, "../etc"])
def test_invalid_names(bad):
    with pytest.raises(ValueError):
        validate_name(bad)

def test_missing_binary_is_reported_not_raised():
    r = run_argv(["definitely-not-a-binary-xyz", "status"], timeout=2)
    assert r.ok is False and r.exit_code == 127

def test_no_shell_interpretation():
    r = run_argv(["echo", "$HOME; rm -rf /"], timeout=2)
    assert r.ok and r.stdout.strip() == "$HOME; rm -rf /"
