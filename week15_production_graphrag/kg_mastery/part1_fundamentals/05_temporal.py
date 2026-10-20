#!/usr/bin/env python3
"""Part 1.8 — Temporal Knowledge Graphs, kg_mastery.pdf §1.8.

Two strategies for modelling how a node changes over time, both demonstrated on
temporary `:Demo` nodes that are removed at the end:

  Strategy 1 — Bi-temporal property-based:
      keep valid_from / valid_to / recorded_at on the node (or on versioned
      copies) and query the state as-of a point in time.

  Strategy 2 — Event sourcing:
      never mutate; append immutable :RoomEvent nodes linked by [:HAD_EVENT]
      and reconstruct history by querying the event stream.

┌──────────────────┬─────────────────────────────┬──────────────────────────┐
│ Strategy         │ Pros                        │ Cons                     │
├──────────────────┼─────────────────────────────┼──────────────────────────┤
│ Bi-temporal      │ simple "current" reads;     │ history needs versioned  │
│ properties       │ point-in-time with one MATCH│ copies; mutation risk    │
│ Event sourcing   │ full immutable audit trail; │ reads must replay/aggreg.│
│                  │ easy to add new event types │ more nodes/storage       │
└──────────────────┴─────────────────────────────┴──────────────────────────┘

Run:  python part1_fundamentals/05_temporal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection


def hdr(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def strategy1_bitemporal(s):
    hdr("Strategy 1: bi-temporal property-based")
    # NOTE: the live :Room label has an id-uniqueness constraint, so we can't
    # keep two same-id :Room nodes. Bi-temporal "as-of" history is modelled with
    # one logical room plus one :RoomVersion per validity interval (SCD type-2).
    # Here: status changed 3 days ago from OCCUPIED to VACANT.
    # valid_from/valid_to = when it was true in the world; recorded_at = when we learned it.
    s.run(
        """
        UNWIND [
          {status:'OCCUPIED', from_days:10, to_days:3,  rec_days:10},
          {status:'VACANT',   from_days:3,  to_days:null, rec_days:3}
        ] AS v
        CREATE (rv:Demo:RoomVersion {
            room_id:'DEMO-R301', status: v.status,
            valid_from: datetime() - duration({days: v.from_days}),
            valid_to:   CASE WHEN v.to_days IS NULL THEN null
                             ELSE datetime() - duration({days: v.to_days}) END,
            recorded_at: datetime() - duration({days: v.rec_days})
        })
        """
    )
    # Query the state "as of 5 days ago"
    rows = [r.data() for r in s.run(
        """
        WITH datetime() - duration({days:5}) AS asOf
        MATCH (rv:Demo:RoomVersion {room_id:'DEMO-R301'})
        WHERE rv.valid_from <= asOf AND (rv.valid_to IS NULL OR rv.valid_to > asOf)
        RETURN rv.status AS statusAsOf5DaysAgo
        """)]
    print("   state as-of 5 days ago:", rows)
    # Query the CURRENT state (valid_to IS NULL = still in effect)
    rows = [r.data() for r in s.run(
        """
        MATCH (rv:Demo:RoomVersion {room_id:'DEMO-R301'})
        WHERE rv.valid_to IS NULL
        RETURN rv.status AS currentStatus
        """)]
    print("   current state:", rows)


def strategy2_event_sourcing(s):
    hdr("Strategy 2: event sourcing (immutable :RoomEvent stream)")
    s.run(
        """
        MERGE (r:Demo:RoomEntity {id:'DEMO-R301'})
        WITH r
        UNWIND [
            {type:'CHECK_IN',     ago:30},
            {type:'TEMP_ALERT',   ago:5},
            {type:'STATUS_CHANGE', ago:2},
            {type:'CHECK_OUT',    ago:1}
        ] AS ev
        CREATE (e:Demo:RoomEvent {
            type: ev.type,
            ts: datetime() - duration({hours: ev.ago})
        })
        CREATE (r)-[:HAD_EVENT]->(e)
        """
    )
    # Reconstruct the last-24h history from the event stream
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Demo:RoomEntity {id:'DEMO-R301'})-[:HAD_EVENT]->(e:Demo:RoomEvent)
        WHERE e.ts >= datetime() - duration({hours:24})
        RETURN e.type AS type,
               duration.inSeconds(e.ts, datetime()).hours AS hoursAgo
        ORDER BY e.ts
        """)]
    print("   events in the last 24h:")
    for row in rows:
        print(f"     {row['type']:<14} ~{row['hoursAgo']}h ago")


def cleanup(s):
    s.run("MATCH (n:Demo) DETACH DELETE n")
    print("\n✅ Cleaned up all :Demo nodes.")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            s.run("MATCH (n:Demo) DETACH DELETE n")  # clear any leftovers from a prior run
            strategy1_bitemporal(s)
            strategy2_event_sourcing(s)
            cleanup(s)
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
