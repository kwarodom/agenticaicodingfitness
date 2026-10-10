"""SQLite store for Alto Mini. One file, created by `make seed`."""
import os, sqlite3, datetime as dt, random, pathlib
DB = pathlib.Path(os.environ.get("ALTO_MINI_DB", pathlib.Path(__file__).resolve().parents[1] / "alto_mini.db"))
SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (id INTEGER PRIMARY KEY, name TEXT NOT NULL, floor INTEGER NOT NULL, setpoint REAL NOT NULL);
CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY AUTOINCREMENT, room_id INTEGER NOT NULL, ts TEXT NOT NULL, kwh REAL);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, room_id INTEGER, ts TEXT NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);
"""
def connect():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; c.executescript(SCHEMA); return c
def seed(days: int = 14, seed: int = 27):
    rnd = random.Random(seed)
    if DB.exists(): DB.unlink()
    c = connect()
    rooms = [(1201, "Deluxe 1201", 12, 24.0), (1205, "Deluxe 1205", 12, 23.0), (1410, "Suite 1410", 14, 22.5), (1502, "Suite 1502", 15, 25.0)]
    c.executemany("INSERT INTO rooms VALUES (?,?,?,?)", rooms)
    today = dt.date.today()
    for rid, _, _, _ in rooms:
        for d in range(days, 0, -1):
            day = today - dt.timedelta(days=d)
            if rid == 1205 and d == 1:   # yesterday has NO readings for 1205 → exercises the empty-day path
                continue
            for h in range(0, 24, 2):
                kwh = None if (rid == 1410 and d == 3 and h == 12) else round(rnd.uniform(0.2, 1.6), 3)  # one null reading
                c.execute("INSERT INTO readings(room_id, ts, kwh) VALUES (?,?,?)", (rid, f"{day.isoformat()}T{h:02d}:00:00", kwh))
    c.executemany("INSERT INTO alerts(room_id, ts, kind, message) VALUES (?,?,?,?)", [
        (1410, f"{(today - dt.timedelta(days=2)).isoformat()}T09:15:00", "high_kwh", "Suite 1410 used 31% above baseline"),
        (1502, f"{(today - dt.timedelta(days=1)).isoformat()}T18:40:00", "setpoint_drift", "Setpoint 25.0 above comfort band"),
        (1205, f"{today.isoformat()}T07:05:00", "sensor_gap", "Deluxe 1205: no kWh readings yesterday"),
    ])
    c.commit(); c.close()
    return DB
if __name__ == "__main__":
    print("seeded", seed())
