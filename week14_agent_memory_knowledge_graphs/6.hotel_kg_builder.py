"""6.hotel_kg_builder.py — Smart Hotel Energy Optimization Knowledge Graph (Scenario 1).

300-room hotel · 12 function halls · 5 MAS agents · FIPA-ACL Contract Net Protocol
Run:  python week14/6.hotel_kg_builder.py
"""
import os
import sys
import random
from datetime import date, timedelta
from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(__file__))
from agent_memory import AgentMemory, URI, AUTH

load_dotenv()
random.seed(42)

DRIVER = GraphDatabase.driver(URI, auth=AUTH)
TODAY = date.today().isoformat()


# ── helpers ───────────────────────────────────────────────────────────────────

def run(cypher: str, **params):
    with DRIVER.session() as s:
        return [dict(r) for r in s.run(cypher, **params)]


# ── 0. clear ──────────────────────────────────────────────────────────────────

def clear_hotel_graph():
    labels = [
        "Hotel", "Floor", "Zone", "Room", "FunctionHall",
        "HVACUnit", "LightingSystem", "ElevatorSystem", "PoolHeating",
        "Sensor", "EnergyBudget", "WeatherForecast",
    ]
    for label in labels:
        run(f"MATCH (n:{label}) DETACH DELETE n")
    # Remove hotel-specific agent nodes (not the AFDD ones)
    run("""MATCH (a:Agent) WHERE a.id IN [
        'OccupancyAgent','HVACControlAgent','LightingAgent',
        'EnergyBudgetAgent','WeatherForecastAgent'
    ] DETACH DELETE a""")
    print("Cleared previous hotel nodes.")


# ── 1. hotel ──────────────────────────────────────────────────────────────────

def create_hotel():
    run("""
        MERGE (h:Hotel {id: 'hotel-grandvista'})
        ON CREATE SET
            h.name                    = 'Grand Vista Hotel',
            h.total_rooms             = 300,
            h.total_function_halls    = 12,
            h.location                = 'Bangkok, Thailand',
            h.energy_baseline_kwh_day = 10000.0,
            h.energy_target_kwh_day   = 7000.0,
            h.target_savings_thb_day  = 120000,
            h.comfort_score_min       = 4.5,
            h.response_time_s         = 60
    """)
    print("  Hotel: Grand Vista Hotel (300 rooms, 12 halls)")


# ── 2. floors ─────────────────────────────────────────────────────────────────

def create_floors():
    run("""
        MATCH (h:Hotel {id: 'hotel-grandvista'})
        MERGE (f:Floor {id: 'floor-G'})
        ON CREATE SET f.floor_number = 0, f.type = 'ground', f.room_count = 0, f.hall_count = 12
        MERGE (h)-[:HAS_FLOOR]->(f)
    """)
    for n in range(1, 16):
        run("""
            MATCH (h:Hotel {id: 'hotel-grandvista'})
            MERGE (f:Floor {id: $fid})
            ON CREATE SET f.floor_number = $n, f.type = 'guestroom', f.room_count = 20, f.hall_count = 0
            MERGE (h)-[:HAS_FLOOR]->(f)
        """, fid=f"floor-{n}", n=n)
    print("  Floors: Ground + 15 guestroom floors")


# ── 3. zones ──────────────────────────────────────────────────────────────────

def create_zones():
    # Ground-floor fixed zones
    for zone_id, zone_type, setpoint, area in [
        ("zone-LOBBY",      "LOBBY",    24.0, 400),
        ("zone-POOL",       "POOL",     26.0, 300),
        ("zone-CORRIDOR-G", "CORRIDOR", 25.0, 200),
    ]:
        run("""
            MATCH (f:Floor {id: 'floor-G'})
            MERGE (z:Zone {id: $zid})
            ON CREATE SET z.type = $zt, z.setpoint_celsius = $sp,
                          z.current_temp_c = $sp + 1.0, z.area_sqm = $area
            MERGE (f)-[:HAS_ZONE]->(z)
        """, zid=zone_id, zt=zone_type, sp=setpoint, area=area)

    # 12 function hall zones
    for i in range(1, 13):
        run("""
            MATCH (f:Floor {id: 'floor-G'})
            MERGE (z:Zone {id: $zid})
            ON CREATE SET z.type = 'FUNCTION_HALL', z.setpoint_celsius = 22.0,
                          z.current_temp_c = 23.5, z.area_sqm = 500
            MERGE (f)-[:HAS_ZONE]->(z)
        """, zid=f"zone-HALL-{i:02d}")

    # 15 guestroom-floor zones
    for n in range(1, 16):
        temp = round(23.0 + random.uniform(-0.5, 2.0), 1)
        run("""
            MATCH (f:Floor {id: $fid})
            MERGE (z:Zone {id: $zid})
            ON CREATE SET z.type = 'GUESTROOM', z.setpoint_celsius = 23.0,
                          z.current_temp_c = $temp, z.area_sqm = 1200
            MERGE (f)-[:HAS_ZONE]->(z)
        """, fid=f"floor-{n}", zid=f"zone-FLOOR-{n:02d}", temp=temp)

    print("  Zones: 3 ground + 12 function halls + 15 guestroom floors = 30 zones")


# ── 4. rooms ──────────────────────────────────────────────────────────────────

def create_rooms():
    count = 0
    for floor_n in range(1, 16):
        for room_pos in range(1, 21):
            room_num = floor_n * 100 + room_pos
            occ = random.randint(0, 2)
            room_type = "suite" if room_pos <= 2 else "standard"
            run("""
                MATCH (z:Zone {id: $zid})
                MERGE (r:Room {id: $rid})
                ON CREATE SET
                    r.room_number    = $rnum,
                    r.type           = $rtype,
                    r.floor_number   = $fnum,
                    r.occupancy_count= $occ,
                    r.status         = $status,
                    r.current_temp_c = 23.0
                MERGE (z)-[:CONTAINS_ROOM]->(r)
            """, zid=f"zone-FLOOR-{floor_n:02d}", rid=f"room-{room_num}",
                 rnum=room_num, rtype=room_type, fnum=floor_n,
                 occ=occ, status="occupied" if occ > 0 else "vacant")
            count += 1
    print(f"  Rooms: {count} across 15 floors (suites: rooms xx01–xx02)")


# ── 5. function halls ─────────────────────────────────────────────────────────

def create_function_halls():
    halls = [
        ("Orchid Ballroom",       800),
        ("Lotus Grand Hall",      600),
        ("Jasmine Conference A",  200),
        ("Jasmine Conference B",  200),
        ("Rose Meeting 1",         50),
        ("Rose Meeting 2",         50),
        ("Rose Meeting 3",         50),
        ("Emerald Banquet",       400),
        ("Sapphire Hall",         300),
        ("Pearl Boardroom",        30),
        ("Diamond Suite Hall",    150),
        ("Topaz Training Room",    80),
    ]
    for i, (name, cap) in enumerate(halls, 1):
        in_use = random.choice([True, False])
        run("""
            MATCH (z:Zone {id: $zid})
            MERGE (h:FunctionHall {id: $hid})
            ON CREATE SET
                h.name            = $name,
                h.capacity        = $cap,
                h.hall_number     = $hnum,
                h.status          = $status,
                h.occupancy_count = $occ
            MERGE (z)-[:CONTAINS_HALL]->(h)
        """, zid=f"zone-HALL-{i:02d}", hid=f"hall-{i:02d}",
             name=name, cap=cap, hnum=i,
             status="in-use" if in_use else "empty",
             occ=random.randint(10, cap // 2) if in_use else 0)
    print(f"  Function Halls: 12 (ballroom → boardroom)")


# ── 6. HVAC units ─────────────────────────────────────────────────────────────

def create_hvac_units():
    count = 0
    for zone_id, hvac_type, setpoint, power in [
        ("zone-LOBBY",      "AHU", 24.0, 15.0),
        ("zone-POOL",       "AHU", 26.0, 12.0),
        ("zone-CORRIDOR-G", "AHU", 25.0,  8.0),
    ]:
        run("""
            MATCH (z:Zone {id: $zid})
            MERGE (u:HVACUnit {id: $uid})
            ON CREATE SET u.type = $ht, u.setpoint_celsius = $sp,
                          u.current_temp_c = $sp + 1, u.status = 'running',
                          u.power_kw = $pw, u.efficiency_pct = 78.0
            MERGE (z)-[:HAS_HVAC]->(u)
        """, zid=zone_id, uid=f"hvac-{zone_id}", ht=hvac_type, sp=setpoint, pw=power)
        count += 1

    for i in range(1, 13):
        run("""
            MATCH (z:Zone {id: $zid})
            MERGE (u:HVACUnit {id: $uid})
            ON CREATE SET u.type = 'FCU', u.setpoint_celsius = 22.0,
                          u.current_temp_c = 23.5, u.status = 'running',
                          u.power_kw = 20.0, u.efficiency_pct = 82.0
            MERGE (z)-[:HAS_HVAC]->(u)
        """, zid=f"zone-HALL-{i:02d}", uid=f"hvac-zone-HALL-{i:02d}")
        count += 1

    for n in range(1, 16):
        run("""
            MATCH (z:Zone {id: $zid})
            MERGE (u:HVACUnit {id: $uid})
            ON CREATE SET u.type = 'AHU', u.setpoint_celsius = 23.0,
                          u.current_temp_c = 23.0, u.status = 'running',
                          u.power_kw = 30.0, u.efficiency_pct = 85.0
            MERGE (z)-[:HAS_HVAC]->(u)
        """, zid=f"zone-FLOOR-{n:02d}", uid=f"hvac-zone-FLOOR-{n:02d}")
        count += 1

    print(f"  HVAC Units: {count} (3 AHU ground + 12 FCU halls + 15 AHU floors)")


# ── 7. lighting ───────────────────────────────────────────────────────────────

def create_lighting():
    count = 0
    brightness_defaults = {"zone-LOBBY": 90, "zone-POOL": 70, "zone-CORRIDOR-G": 60}
    all_zones = (
        ["zone-LOBBY", "zone-POOL", "zone-CORRIDOR-G"]
        + [f"zone-HALL-{i:02d}" for i in range(1, 13)]
        + [f"zone-FLOOR-{n:02d}" for n in range(1, 16)]
    )
    for zone_id in all_zones:
        bright = brightness_defaults.get(zone_id, random.randint(30, 80))
        run("""
            MATCH (z:Zone {id: $zid})
            MERGE (l:LightingSystem {id: $lid})
            ON CREATE SET l.brightness_pct = $b, l.status = 'on',
                          l.type = 'LED', l.power_kw = 5.0
            MERGE (z)-[:HAS_LIGHTING]->(l)
        """, zid=zone_id, lid=f"light-{zone_id}", b=bright)
        count += 1
    print(f"  Lighting Systems: {count}")


# ── 8. elevators ──────────────────────────────────────────────────────────────

def create_elevators():
    for i in range(1, 7):
        run("""
            MATCH (h:Hotel {id: 'hotel-grandvista'})
            MERGE (e:ElevatorSystem {id: $eid})
            ON CREATE SET e.elevator_number = $n, e.current_floor = $floor,
                          e.status = $status, e.type = $etype, e.power_kw = 8.0
            MERGE (h)-[:HAS_ELEVATOR]->(e)
        """, eid=f"elevator-{i:02d}", n=i,
             floor=random.randint(1, 15),
             status=random.choice(["idle", "moving", "idle", "idle"]),
             etype="passenger" if i <= 4 else "service")
    print("  Elevators: 6 (4 passenger + 2 service)")


# ── 9. pool ───────────────────────────────────────────────────────────────────

def create_pool():
    run("""
        MATCH (h:Hotel {id: 'hotel-grandvista'})
        MATCH (z:Zone {id: 'zone-POOL'})
        MERGE (p:PoolHeating {id: 'pool-main'})
        ON CREATE SET p.name = 'Rooftop Infinity Pool', p.temp_c = 29.5,
                      p.setpoint_c = 30.0, p.status = 'heating',
                      p.power_kw = 45.0, p.volume_liters = 250000
        MERGE (h)-[:HAS_POOL]->(p)
        MERGE (z)-[:CONTAINS_POOL]->(p)
    """)
    print("  Pool: Rooftop Infinity Pool (250,000 L, 45 kW heater)")


# ── 10. sensors ───────────────────────────────────────────────────────────────

def create_sensors():
    count = 0

    for n in range(1, 16):
        zone_id = f"zone-FLOOR-{n:02d}"
        for stype, val, unit in [
            ("temperature", round(23.0 + random.uniform(-1, 2), 1), "degC"),
            ("occupancy",   random.randint(5, 18),                  "persons"),
            ("co2",         random.randint(600, 1100),              "ppm"),
        ]:
            run("""
                MATCH (z:Zone {id: $zid})
                MERGE (s:Sensor {id: $sid})
                ON CREATE SET s.type = $st, s.value = $v, s.unit = $u,
                              s.last_updated = datetime()
                MERGE (s)-[:MONITORS]->(z)
            """, zid=zone_id, sid=f"sensor-{zone_id}-{stype}", st=stype, v=val, u=unit)
            count += 1

    for zone_id, stype, val, unit in [
        ("zone-LOBBY", "temperature", 24.0, "degC"),
        ("zone-LOBBY", "occupancy",   random.randint(10, 80), "persons"),
        ("zone-POOL",  "temperature", 24.5, "degC"),
        ("zone-POOL",  "occupancy",   random.randint(5, 40),  "persons"),
    ]:
        run("""
            MATCH (z:Zone {id: $zid})
            MERGE (s:Sensor {id: $sid})
            ON CREATE SET s.type = $st, s.value = $v, s.unit = $u, s.last_updated = datetime()
            MERGE (s)-[:MONITORS]->(z)
        """, zid=zone_id, sid=f"sensor-{zone_id}-{stype}", st=stype, v=val, u=unit)
        count += 1

    for i in range(1, 13):
        zone_id = f"zone-HALL-{i:02d}"
        for stype, val, unit in [
            ("temperature", 23.5,                    "degC"),
            ("occupancy",   random.randint(0, 100),  "persons"),
        ]:
            run("""
                MATCH (z:Zone {id: $zid})
                MERGE (s:Sensor {id: $sid})
                ON CREATE SET s.type = $st, s.value = $v, s.unit = $u, s.last_updated = datetime()
                MERGE (s)-[:MONITORS]->(z)
            """, zid=zone_id, sid=f"sensor-{zone_id}-{stype}", st=stype, v=val, u=unit)
            count += 1

    print(f"  Sensors: {count} (temp + occupancy + CO2 across all zones)")


# ── 11. energy budget ─────────────────────────────────────────────────────────

def create_energy_budget():
    run("""
        MATCH (h:Hotel {id: 'hotel-grandvista'})
        MERGE (b:EnergyBudget {id: $bid})
        ON CREATE SET
            b.period         = 'daily',
            b.date           = $today,
            b.total_kwh      = 10000.0,
            b.consumed_kwh   = 6240.0,
            b.remaining_kwh  = 3760.0,
            b.target_kwh     = 7000.0,
            b.hvac_kwh       = 3800.0,
            b.lighting_kwh   = 900.0,
            b.elevators_kwh  = 480.0,
            b.pool_kwh       = 1060.0,
            b.on_track       = true
        MERGE (h)-[:HAS_BUDGET]->(b)
    """, bid=f"budget-daily-{TODAY}", today=TODAY)
    print(f"  Energy Budget: 10,000 kWh/day total | consumed 6,240 | remaining 3,760")


# ── 12. weather ───────────────────────────────────────────────────────────────

def create_weather():
    forecasts = [
        (TODAY,                                           34.0, 72, "hot_humid",     1.30),
        ((date.today() + timedelta(1)).isoformat(),       32.5, 68, "partly_cloudy", 1.15),
        ((date.today() + timedelta(2)).isoformat(),       30.0, 65, "overcast",      1.00),
    ]
    for fdate, temp, hum, ftype, clf in forecasts:
        run("""
            MATCH (h:Hotel {id: 'hotel-grandvista'})
            MERGE (w:WeatherForecast {id: $wid})
            ON CREATE SET w.date = $fdate, w.outdoor_temp_c = $temp,
                          w.humidity_pct = $hum, w.forecast_type = $ftype,
                          w.cooling_load_factor = $clf
            MERGE (w)-[:AFFECTS]->(h)
        """, wid=f"weather-{fdate}", fdate=fdate, temp=temp, hum=hum, ftype=ftype, clf=clf)
    print("  Weather: 3-day forecast (today 34 °C hot/humid, cooling factor 1.30)")


# ── 13. MAS agents + FIPA-ACL ─────────────────────────────────────────────────

def create_agents():
    agent_defs = [
        ("OccupancyAgent",      "Reads sensors, detects occupancy changes, broadcasts INFORM",        60),
        ("HVACControlAgent",    "Adjusts HVAC setpoints (occupancy + weather); Contract Net contractor", 60),
        ("LightingAgent",       "Adjusts brightness by zone occupancy; Contract Net contractor",      60),
        ("EnergyBudgetAgent",   "Tracks consumption, issues CFP when budget at-risk; CN initiator",  60),
        ("WeatherForecastAgent","Fetches 24 h forecast, pre-adjusts HVAC cooling load",            3600),
    ]
    for agent_id, role, interval in agent_defs:
        run("""
            MERGE (a:Agent {id: $aid})
            ON CREATE SET a.role = $role, a.status = 'active',
                          a.loop_interval_s = $interval, a.protocol = 'FIPA-ACL',
                          a.created = datetime()
        """, aid=agent_id, role=role, interval=interval)

    # Control edges: HVACControlAgent → all HVAC units
    run("""
        MATCH (a:Agent {id: 'HVACControlAgent'})
        MATCH (u:HVACUnit)
        MERGE (a)-[:CONTROLS]->(u)
    """)
    # LightingAgent → all lighting systems
    run("""
        MATCH (a:Agent {id: 'LightingAgent'})
        MATCH (l:LightingSystem)
        MERGE (a)-[:CONTROLS]->(l)
    """)
    # OccupancyAgent → all occupancy sensors
    run("""
        MATCH (a:Agent {id: 'OccupancyAgent'})
        MATCH (s:Sensor {type: 'occupancy'})
        MERGE (a)-[:MONITORS]->(s)
    """)
    # WeatherForecastAgent → hotel (weather affects whole building)
    run("""
        MATCH (a:Agent {id: 'WeatherForecastAgent'})
        MATCH (h:Hotel {id: 'hotel-grandvista'})
        MERGE (a)-[:MONITORS]->(h)
    """)
    # EnergyBudgetAgent → energy budget node
    run("""
        MATCH (a:Agent {id: 'EnergyBudgetAgent'})
        MATCH (b:EnergyBudget {date: $today})
        MERGE (a)-[:TRACKS]->(b)
    """, today=TODAY)

    # FIPA-ACL inter-agent message relationships
    messages = [
        ("OccupancyAgent",      "HVACControlAgent",    "INFORM",  "occupancy_change"),
        ("OccupancyAgent",      "LightingAgent",       "INFORM",  "occupancy_change"),
        ("OccupancyAgent",      "EnergyBudgetAgent",   "INFORM",  "occupancy_report"),
        ("WeatherForecastAgent","HVACControlAgent",    "INFORM",  "weather_forecast"),
        ("WeatherForecastAgent","EnergyBudgetAgent",   "INFORM",  "cooling_load_forecast"),
        ("EnergyBudgetAgent",   "HVACControlAgent",    "CFP",     "reduce_hvac_load"),
        ("EnergyBudgetAgent",   "LightingAgent",       "CFP",     "reduce_lighting_load"),
        ("HVACControlAgent",    "EnergyBudgetAgent",   "PROPOSE", "setpoint_adjustment"),
        ("LightingAgent",       "EnergyBudgetAgent",   "PROPOSE", "brightness_reduction"),
        ("EnergyBudgetAgent",   "HVACControlAgent",    "ACCEPT",  "contract_awarded"),
        ("EnergyBudgetAgent",   "LightingAgent",       "ACCEPT",  "contract_awarded"),
    ]
    for src, dst, perf, ctype in messages:
        run("""
            MATCH (a:Agent {id: $src})
            MATCH (b:Agent {id: $dst})
            MERGE (a)-[r:SENDS_MESSAGE {performative: $perf, content_type: $ctype}]->(b)
        """, src=src, dst=dst, perf=perf, ctype=ctype)

    print(f"  Agents: 5 with roles, control edges, and {len(messages)} FIPA-ACL message edges")


# ── 14. simulate one full Contract Net cycle via AgentMemory ──────────────────

def simulate_agent_events():
    agents = {
        "OccupancyAgent":       AgentMemory("OccupancyAgent"),
        "HVACControlAgent":     AgentMemory("HVACControlAgent"),
        "LightingAgent":        AgentMemory("LightingAgent"),
        "EnergyBudgetAgent":    AgentMemory("EnergyBudgetAgent"),
        "WeatherForecastAgent": AgentMemory("WeatherForecastAgent"),
    }

    # T=0: WeatherForecastAgent fetches hot-day forecast
    agents["WeatherForecastAgent"].store_event("WEATHER_FETCHED", {
        "outdoor_temp_c": 34.0, "humidity_pct": 72,
        "forecast_type": "hot_humid", "cooling_load_factor": 1.3,
    }, entities=[("Hotel", "Grand Vista Hotel")])

    # T=60: OccupancyAgent detects surge on floor 7
    agents["OccupancyAgent"].store_event("OCCUPANCY_CHANGE", {
        "zone": "zone-FLOOR-07", "previous_count": 8, "current_count": 15,
        "change_pct": 87.5,
    }, entities=[("Zone", "zone-FLOOR-07")])

    # T=60: HVACControlAgent reacts — lower setpoint (more cooling)
    agents["HVACControlAgent"].store_event("SETPOINT_ADJUSTED", {
        "zone": "zone-FLOOR-07", "old_setpoint": 23.0, "new_setpoint": 22.0,
        "reason": "occupancy_surge + hot_outdoor",
        "power_delta_kw": +2.5,
    }, entities=[("Zone", "zone-FLOOR-07"), ("HVACUnit", "hvac-zone-FLOOR-07")])

    # T=60: LightingAgent dims vacant floor 3
    agents["LightingAgent"].store_event("BRIGHTNESS_ADJUSTED", {
        "zone": "zone-FLOOR-03", "old_pct": 80, "new_pct": 40,
        "reason": "occupancy_zero", "power_saved_kw": 1.5,
    }, entities=[("Zone", "zone-FLOOR-03")])

    # T=120: EnergyBudgetAgent — budget at 85 %, triggers Contract Net
    agents["EnergyBudgetAgent"].store_event("BUDGET_ALERT", {
        "consumed_kwh": 8500, "total_kwh": 10000,
        "pct_consumed": 85.0, "remaining_kwh": 1500,
        "threshold_pct": 80.0,
    }, entities=[(f"EnergyBudget", f"budget-daily-{TODAY}")])

    agents["EnergyBudgetAgent"].store_event("CONTRACT_NET_CFP_SENT", {
        "cfp_to": ["HVACControlAgent", "LightingAgent"],
        "reduction_target_kwh": 500,
        "deadline_s": 30,
    })

    # T=150: Contractors submit proposals
    agents["HVACControlAgent"].store_event("PROPOSAL_SUBMITTED", {
        "in_response_to": "CONTRACT_NET_CFP",
        "proposed_reduction_kwh": 300,
        "method": "raise_setpoints_1C_vacant_zones",
        "comfort_impact": "minimal",
    })

    agents["LightingAgent"].store_event("PROPOSAL_SUBMITTED", {
        "in_response_to": "CONTRACT_NET_CFP",
        "proposed_reduction_kwh": 220,
        "method": "dim_corridors_50pct_vacant_floors",
        "comfort_impact": "none",
    })

    # T=180: EnergyBudgetAgent evaluates and accepts both
    agents["EnergyBudgetAgent"].store_event("PROPOSALS_EVALUATED", {
        "hvac_proposal_kwh": 300, "lighting_proposal_kwh": 220,
        "total_reduction_kwh": 520, "target_kwh": 500,
        "decision": "accept_both",
    })

    # T=240: Budget update after actions
    agents["EnergyBudgetAgent"].store_event("BUDGET_UPDATED", {
        "new_consumed_kwh": 8020, "remaining_kwh": 1980,
        "pct_consumed": 80.2, "status": "on_track",
    }, entities=[(f"EnergyBudget", f"budget-daily-{TODAY}")])

    for m in agents.values():
        m.close()

    print("  Simulated: weather fetch → occupancy surge → HVAC/lighting adjust")
    print("             → budget alert → Contract Net CFP cycle → proposals → accept")


# ── 15. verification ──────────────────────────────────────────────────────────

def verify():
    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)

    checks = [
        ("Total nodes",
         "MATCH (n) RETURN count(n) AS count"),
        ("Total relationships",
         "MATCH ()-[r]->() RETURN count(r) AS count"),
        ("Rooms (expect 300)",
         "MATCH (n:Room) RETURN count(n) AS count"),
        ("HVAC units (expect 30)",
         "MATCH (n:HVACUnit) RETURN count(n) AS count"),
        ("Sensors",
         "MATCH (n:Sensor) RETURN count(n) AS count"),
        ("Agents",
         "MATCH (n:Agent) RETURN n.id AS id, n.loop_interval_s AS interval_s"),
        ("FIPA-ACL message types",
         "MATCH ()-[r:SENDS_MESSAGE]->() RETURN DISTINCT r.performative AS perf, count(r) AS count"),
        ("Energy budget",
         "MATCH (b:EnergyBudget) RETURN b.date, b.consumed_kwh, b.remaining_kwh, b.on_track"),
        ("Today weather",
         "MATCH (w:WeatherForecast {date: $d}) RETURN w.outdoor_temp_c, w.forecast_type, w.cooling_load_factor",
         {"d": TODAY}),
        ("Zones above setpoint (cooling needed)",
         "MATCH (z:Zone) WHERE z.current_temp_c > z.setpoint_celsius "
         "RETURN z.id, z.type, z.current_temp_c AS temp, z.setpoint_celsius AS setpoint "
         "ORDER BY (z.current_temp_c - z.setpoint_celsius) DESC LIMIT 5"),
        ("Agent events (latest 8)",
         "MATCH (a:Agent)-[:PERFORMED]->(e:Event) "
         "RETURN a.id AS agent, e.type AS event ORDER BY e.ts DESC LIMIT 8"),
    ]

    for row in checks:
        label, query = row[0], row[1]
        params = row[2] if len(row) > 2 else {}
        results = run(query, **params)
        print(f"\n  {label}:")
        for r in results:
            print(f"    {r}")

    print("\n" + "=" * 60)
    print("Neo4j Browser — try these Cypher queries:")
    print("  MATCH (h:Hotel)-[*1..2]-(n) RETURN h, n LIMIT 80")
    print("  MATCH p=(a:Agent)-[:SENDS_MESSAGE]->(b:Agent) RETURN p")
    print("  MATCH p=(a:Agent)-[:PERFORMED]->(e:Event) RETURN p ORDER BY e.ts DESC LIMIT 20")
    print("  MATCH (z:Zone)-[:HAS_HVAC]->(u) RETURN z.id, z.type, u.setpoint_celsius, u.power_kw")
    print("=" * 60)


# ── 16. optional GraphRAG demo ────────────────────────────────────────────────

def graphrag_demo():
    print("\n" + "=" * 60)
    print("GRAPHRAG DEMO (NL → Cypher → Answer)")
    print("=" * 60)
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=512,
    )
    mem = AgentMemory("EnergyBudgetAgent")
    chain = mem.build_graphrag_chain(llm, verbose=True)

    questions = [
        "What is the current energy budget status?",
        "Which zones need more cooling right now?",
        "What actions did the HVAC agent take today?",
        "How many rooms are occupied?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        try:
            answer = chain.invoke({"query": q})
            print(f"A: {answer['result']}")
        except Exception as e:
            print(f"   (error: {e})")

    mem.close()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smart Hotel Energy Optimization — Knowledge Graph Builder")
    print("=" * 60)

    clear_hotel_graph()

    print("\n[1/4] Hotel structure")
    create_hotel()
    create_floors()
    create_zones()
    create_rooms()
    create_function_halls()

    print("\n[2/4] Building systems")
    create_hvac_units()
    create_lighting()
    create_elevators()
    create_pool()

    print("\n[3/4] IoT data & budgets")
    create_sensors()
    create_energy_budget()
    create_weather()

    print("\n[4/4] MAS agents & simulation")
    create_agents()
    simulate_agent_events()

    verify()

    DRIVER.close()

    print("\nKnowledge graph built successfully.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        graphrag_demo()
    else:
        print("Set ANTHROPIC_API_KEY to also run the GraphRAG demo.")
