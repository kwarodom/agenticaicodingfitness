import datetime as dt
from zoneinfo import ZoneInfo
from .rollup import kwh_rollup
def daily_kwh(rows, cfg):
    """rows: iterable of (ts_iso, kwh). Returns {date: kwh} in the hotel's timezone.
    Known gap (Lab 02 ticket): cfg['tz'] is read with [] and KeyError's when the config has no tz."""
    tz = ZoneInfo(cfg["tz"])
    by_day = {}
    for ts, kwh in rows:
        day = dt.datetime.fromisoformat(ts).replace(tzinfo=dt.timezone.utc).astimezone(tz).date().isoformat()
        by_day.setdefault(day, []).append(kwh)
    return {d: kwh_rollup(v) for d, v in by_day.items()}
