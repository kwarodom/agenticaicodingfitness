import argparse, datetime as dt, json, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
def parse_since(s: str, today: dt.date) -> dt.date:
    m = re.fullmatch(r"(\d+)d", s)
    if not m: raise SystemExit("--since must look like 14d")
    return today - dt.timedelta(days=int(m.group(1)))
def args():
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="14d"); p.add_argument("--file"); p.add_argument("--today")
    p.add_argument("--board", default=str(ROOT / "telemetry" / "board_snapshot.json"))
    return p.parse_args()
def load(path): return json.loads(pathlib.Path(path).read_text())
def today(a): return dt.date.fromisoformat(a.today) if a.today else dt.date.today()
