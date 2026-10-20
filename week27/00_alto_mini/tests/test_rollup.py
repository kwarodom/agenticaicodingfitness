from app.energy.rollup import kwh_rollup
def test_rounds(): assert kwh_rollup([1.234, 2.0]) == 3.23
