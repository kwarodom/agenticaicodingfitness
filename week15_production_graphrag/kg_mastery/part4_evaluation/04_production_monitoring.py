#!/usr/bin/env python3
"""Part 4.7 — Production monitoring for GraphRAG (kg_mastery.pdf §4.7).

KEY INSIGHT: Offline eval (RAGAS) tells you the system was good at build time.
Production MONITORING tells you it is still good RIGHT NOW. Wrap every graph
query in a decorator that records latency and outcome (success / error / empty),
then expose a rolling summary you can alert on.

The `@monitored_graph_query` decorator (using functools.wraps so the wrapped
function keeps its name/docstring) appends a record to a module-level
`metrics` store for each call. `get_metrics_summary()` reduces that store into
the operational numbers below.

MONITORING METRICS TABLE (PDF §4.7) — what to watch and when to page someone:

  Metric                 What it measures              Alert threshold
  ---------------------- ----------------------------- -----------------------
  Cypher success rate    % queries that ran w/o error  < 95%  -> alert
  Query latency p99      99th-percentile round trip    > 2000 ms -> alert
  Faithfulness (sampled) RAGAS faithfulness on a        < 0.85 -> alert
                         sampled % of live traffic
  Empty result rate      % queries returning 0 rows    > 10%  -> alert
  Token usage            LLM tokens per query / day     budget-dependent
  Graph node count       total nodes (growth/drift)    sudden drop -> alert

A spike in empty_rate usually means schema drift or a brittle generated Cypher
pattern; a latency spike with stable success usually means a missing index.

Requires:
  - pip install (none extra; uses stdlib + common.get_driver)
  - Neo4j optional: if reachable, real sample queries run through the wrapper.

Run:  python week15/kg_mastery/part4_evaluation/04_production_monitoring.py
"""
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection

# Module-level rolling store. In production this would be a time-series DB
# (Prometheus, TimescaleDB, ...) — the shape is the same.
metrics = defaultdict(list)


def monitored_graph_query(func):
    """Time a graph query and record success / error / empty outcome."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000.0
            metrics["latency_ms"].append(latency_ms)
            metrics["status"].append("success")
            # Treat a falsy / empty result set as an "empty" outcome too.
            is_empty = result is None or (
                hasattr(result, "__len__") and len(result) == 0
            )
            metrics["empty"].append(bool(is_empty))
            return result
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            metrics["latency_ms"].append(latency_ms)
            metrics["status"].append("error")
            metrics["empty"].append(False)
            raise

    return wrapper


@monitored_graph_query
def query_graph(cypher, **params):
    """Run one Cypher query through the monitoring wrapper."""
    driver = get_driver()
    try:
        with driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]
    finally:
        driver.close()


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def get_metrics_summary():
    """Reduce the rolling `metrics` store into operational numbers."""
    statuses = metrics["status"]
    total = len(statuses)
    if total == 0:
        return {
            "total_queries": 0,
            "success_rate": 0.0,
            "empty_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }
    successes = sum(1 for s in statuses if s == "success")
    empties = sum(1 for e in metrics["empty"] if e)
    latencies = metrics["latency_ms"]
    return {
        "total_queries": total,
        "success_rate": successes / total,
        "empty_rate": empties / total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p99_latency_ms": _percentile(latencies, 99),
    }


@monitored_graph_query
def _fake_query(rows):
    """Stand-in used when Neo4j is unreachable, to demo the decorator."""
    time.sleep(0.005)
    return list(range(rows))


def main():
    if check_connection():
        print("Neo4j reachable — running sample queries through the wrapper.\n")
        sample_queries = [
            "MATCH (r:Room) RETURN count(r) AS rooms",
            "MATCH (a:Alert) WHERE a.severity = 'HIGH' RETURN a LIMIT 5",
            "MATCH (d:Device {type:'NONEXISTENT'}) RETURN d",  # likely empty
        ]
        for cy in sample_queries:
            try:
                rows = query_graph(cy)
                print(f"  ok ({len(rows)} rows): {cy}")
            except Exception as e:  # noqa: BLE001
                print(f"  error: {cy}  -> {type(e).__name__}: {e}")
    else:
        print("Neo4j not reachable — demonstrating the decorator on a fake fn.\n")
        for n in (3, 0, 5, 0, 7):  # two empty results
            _fake_query(n)

    print("\n=== Monitoring summary ===")
    summary = get_metrics_summary()
    print(f"  total_queries : {summary['total_queries']}")
    print(f"  success_rate  : {summary['success_rate']:.1%}")
    print(f"  empty_rate    : {summary['empty_rate']:.1%}")
    print(f"  avg_latency_ms: {summary['avg_latency_ms']:.1f}")
    print(f"  p99_latency_ms: {summary['p99_latency_ms']:.1f}")


if __name__ == "__main__":
    main()
