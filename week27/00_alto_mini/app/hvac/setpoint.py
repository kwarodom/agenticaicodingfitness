LIMITS = (16.0, 30.0)
def validate(v: float) -> float:
    """Accept a setpoint inside LIMITS. Known gap (Lab 02 ticket): the UI slider goes to 31 and the API
    answers 500 instead of 422; a `control`-labelled fix should clamp or reject with a clear message."""
    lo, hi = LIMITS
    if v < lo or v > hi:
        raise ValueError(f"setpoint {v} outside {lo:g}-{hi:g}")
    return float(v)
