#!/usr/bin/env python3
"""Part 1.0 — Load the Hotel IoT dataset (RUN THIS FIRST).

Builds the knowledge graph every other script in this course queries. It is
idempotent (MERGE-based) — safe to re-run. Reflects the domain model from
kg_mastery.pdf §1.7 and §5.1.

Run:  python week15/kg_mastery/part1_fundamentals/00_load_hotel_dataset.py

Creates (roughly): 15 rooms across 5 floors, 2–3 devices each, ~24h of sensor
readings, a handful of alerts (some unresolved/HOT), staff, maintenance jobs,
guests and suppliers — enough for every example query to return real data.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection

random.seed(42)  # reproducible dataset

ROOM_TYPES = ["Standard", "Deluxe", "Suite", "Function Hall"]
DESCRIPTIONS = {
    "Standard": "Cozy standard room with city view and energy-efficient HVAC.",
    "Deluxe": "Spacious deluxe room with ocean view, smart lighting and climate control.",
    "Suite": "Luxury suite with panoramic ocean view, premium HVAC and quiet ambience.",
    "Function Hall": "Large function hall pre-cooled for events, high-capacity ventilation.",
}
DEVICE_TYPES = ["HVAC", "Lighting", "Sensor"]
HVAC_MODELS = [("Daikin VRV-IV", "Daikin"), ("Trane XR16", "Trane"), ("Carrier 38MA", "Carrier")]
STAFF = [
    ("S1", "Somchai", "HVAC Technician", "day"),
    ("S2", "Anong", "Electrician", "day"),
    ("S3", "Wichai", "Facilities Lead", "night"),
]
SUPPLIERS = [
    ("SUP1", "CoolTech Co", "TH", "HVAC"),
    ("SUP2", "BrightLite", "CN", "Lighting"),
    ("SUP3", "SensorWorks", "JP", "Sensor"),
]

CONSTRAINTS = [
    "CREATE CONSTRAINT room_id_unique IF NOT EXISTS FOR (r:Room) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:Alert) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT staff_id_unique IF NOT EXISTS FOR (s:Staff) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:MaintenanceJob) REQUIRE j.id IS UNIQUE",
    "CREATE CONSTRAINT guest_id_unique IF NOT EXISTS FOR (g:Guest) REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT supplier_id_unique IF NOT EXISTS FOR (sp:Supplier) REQUIRE sp.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
]
INDEXES = [
    "CREATE INDEX room_floor_idx IF NOT EXISTS FOR (r:Room) ON (r.floor)",
    "CREATE INDEX device_type_idx IF NOT EXISTS FOR (d:Device) ON (d.type)",
    "CREATE INDEX alert_ts_idx IF NOT EXISTS FOR (a:Alert) ON (a.ts)",
    "CREATE INDEX reading_ts_idx IF NOT EXISTS FOR (s:SensorReading) ON (s.ts)",
]


def build(session):
    # --- constraints + indexes first (Part 2.1 golden rule) ---
    for c in CONSTRAINTS + INDEXES:
        session.run(c)

    # --- rooms + devices + readings ---
    rooms = []
    for floor in range(1, 6):              # 5 floors
        for n in range(1, 6):              # 5 rooms per floor → 25 rooms (R101..R505)
            rid = f"R{floor}0{n}"
            rtype = random.choice(ROOM_TYPES)
            rooms.append(rid)
            session.run(
                """
                MERGE (r:Room {id:$id})
                SET r.floor=$floor, r.type=$type, r.capacity=$cap,
                    r.rate_thb=$rate, r.status=$status, r.description=$desc
                """,
                id=rid, floor=floor, type=rtype,
                cap=random.choice([2, 2, 4, 50]),
                rate=random.choice([1500, 2500, 4500, 12000]),
                status=random.choice(["OCCUPIED", "VACANT", "OCCUPIED"]),
                desc=DESCRIPTIONS[rtype],
            )

            # 2 devices per room: one HVAC, one Lighting/Sensor
            hvac_model, hvac_mfr = random.choice(HVAC_MODELS)
            devices = [
                (f"HVAC-{rid}", "HVAC", hvac_model, hvac_mfr),
                (f"{random.choice(['LIGHT','SENSOR'])}-{rid}",
                 random.choice(["Lighting", "Sensor"]), "Generic-100", "BrightLite"),
            ]
            for did, dtype, model, mfr in devices:
                session.run(
                    """
                    MATCH (r:Room {id:$rid})
                    MERGE (d:Device {id:$did})
                    SET d.type=$dtype, d.model=$model, d.manufacturer=$mfr,
                        d.installed_at=datetime() - duration({days:$age}), d.status='OK'
                    MERGE (r)-[:HAS_DEVICE]->(d)
                    """,
                    rid=rid, did=did, dtype=dtype, model=model, mfr=mfr,
                    age=random.randint(30, 900),
                )

            # ~12 hourly sensor readings over the last 24h
            for h in range(0, 24, 2):
                # make a few rooms run HOT (temp > 28/30) so HOT queries return data
                base = 24.0 + (4.0 if rid in ("R101", "R203", "R305") else 0.0)
                session.run(
                    """
                    MATCH (r:Room {id:$rid})
                    CREATE (s:SensorReading {
                        ts: datetime() - duration({hours:$h}),
                        temp_c: $temp, humidity_pct: $hum,
                        energy_kwh: $kwh, occupancy: $occ
                    })
                    CREATE (r)-[:HAS_READING]->(s)
                    """,
                    rid=rid, h=h,
                    temp=round(base + random.uniform(-1.5, 3.5), 1),
                    hum=random.randint(40, 70),
                    kwh=round(random.uniform(1.0, 6.0), 2),
                    occ=random.randint(0, 4),
                )

    # --- staff + suppliers ---
    for sid, name, role, shift in STAFF:
        session.run(
            "MERGE (s:Staff {id:$id}) SET s.name=$name, s.role=$role, s.shift=$shift",
            id=sid, name=name, role=role, shift=shift,
        )
    for spid, name, country, cat in SUPPLIERS:
        session.run(
            "MERGE (sp:Supplier {id:$id}) SET sp.name=$name, sp.country=$country, sp.category=$cat",
            id=spid, name=name, country=country, cat=cat,
        )
    # suppliers provide devices of their category
    session.run(
        """
        MATCH (sp:Supplier), (d:Device)
        WHERE sp.category = d.type
        MERGE (sp)-[:PROVIDES]->(d)
        """
    )

    # --- alerts on a few HVAC devices in the last 7 days (some unresolved) ---
    alert_specs = [
        ("AL1", "HVAC-R101", "HIGH_TEMP", "HIGH", "Room overheating", 1, False),
        ("AL2", "HVAC-R203", "HIGH_TEMP", "HIGH", "Setpoint not reached", 2, False),
        ("AL3", "HVAC-R305", "FILTER", "MEDIUM", "Dirty air filter", 3, True),
        ("AL4", "HVAC-R102", "VIBRATION", "LOW", "Compressor vibration", 5, True),
    ]
    for aid, did, atype, sev, msg, days_ago, resolved in alert_specs:
        session.run(
            """
            MATCH (d:Device {id:$did})
            MERGE (a:Alert {id:$aid})
            SET a.type=$atype, a.severity=$sev, a.message=$msg,
                a.ts=datetime() - duration({days:$days}), a.resolved=$resolved
            MERGE (d)-[:TRIGGERED]->(a)
            """,
            did=did, aid=aid, atype=atype, sev=sev, msg=msg,
            days=days_ago, resolved=resolved,
        )

    # --- maintenance jobs resolving the resolved alerts ---
    job_specs = [
        ("J1", "S1", "AL3", "R305", "HVAC_REPAIR", 3, 3),   # started 3d ago, done same day
        ("J2", "S1", "AL4", "R102", "HVAC_REPAIR", 5, 4),
    ]
    for jid, sid, aid, rid, jtype, start_days, done_days in job_specs:
        session.run(
            """
            MATCH (s:Staff {id:$sid}), (a:Alert {id:$aid}), (r:Room {id:$rid})
            MERGE (j:MaintenanceJob {id:$jid})
            SET j.type=$jtype, j.status='COMPLETED',
                j.started_at=datetime() - duration({days:$sd}),
                j.completed_at=datetime() - duration({days:$dd})
            MERGE (s)-[:PERFORMED]->(j)
            MERGE (j)-[:RESOLVES]->(a)
            MERGE (j)-[:FOR_ROOM]->(r)
            """,
            sid=sid, aid=aid, rid=rid, jid=jid, jtype=jtype, sd=start_days, dd=done_days,
        )
    # assign staff to the still-open HIGH alerts
    session.run(
        """
        MATCH (s:Staff {id:'S1'}), (a:Alert)
        WHERE a.resolved = false AND a.severity = 'HIGH'
        MERGE (s)-[:ASSIGNED_TO]->(a)
        """
    )

    # --- a couple of guests ---
    for gid, name, rid in [("G1", "Aom", "R102"), ("G2", "Ben", "R201")]:
        session.run(
            """
            MATCH (r:Room {id:$rid})
            MERGE (g:Guest {id:$gid})
            SET g.name=$name,
                g.check_in=datetime() - duration({days:2}),
                g.check_out=datetime() + duration({days:1})
            MERGE (g)-[:STAYED_IN]->(r)
            """,
            gid=gid, name=name, rid=rid,
        )


def summary(session):
    rows = session.run(
        """
        MATCH (n) WITH labels(n)[0] AS label, count(*) AS c
        RETURN label, c ORDER BY c DESC
        """
    )
    print("\nNodes by label:")
    for r in rows:
        print(f"  {r['label']:<16} {r['c']}")
    rels = session.run(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"
    )
    print("Relationships by type:")
    for r in rels:
        print(f"  {r['t']:<16} {r['c']}")


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    driver = get_driver()
    try:
        with driver.session() as s:
            print("Loading hotel IoT dataset (idempotent)...")
            build(s)
            summary(s)
        print("\n✅ Dataset ready. Explore it at http://localhost:7474")
        print("   Next: python part1_fundamentals/01_cypher_basics.py")
    finally:
        driver.close()
