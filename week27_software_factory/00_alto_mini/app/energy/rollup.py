def kwh_rollup(readings):
    """Sum a list of kWh readings for one period, rounded to 2 dp.
    Known gap (Lab 02 ticket): a None reading (sensor dropout) raises TypeError; an empty day should be 0.0."""
    return round(sum(readings), 2)
