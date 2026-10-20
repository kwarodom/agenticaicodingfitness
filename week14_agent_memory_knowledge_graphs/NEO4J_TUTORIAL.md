# Neo4j Cypher Tutorial — Smart Hotel Energy Optimization Graph

Learn Cypher by querying the knowledge graph built in `6.hotel_kg_builder.py`.
Open **Neo4j Browser** at `http://localhost:7474` (username: `neo4j`, password: `mas_memory_2024`)
and run every query as you read.

---

## Table of Contents

1. [What is Cypher?](#1-what-is-cypher)
2. [Explore the Graph](#2-explore-the-graph)
3. [MATCH — Find Nodes](#3-match--find-nodes)
4. [WHERE — Filter Results](#4-where--filter-results)
5. [RETURN — Shape the Output](#5-return--shape-the-output)
6. [Relationships](#6-relationships)
7. [Path Traversal](#7-path-traversal)
8. [Aggregation](#8-aggregation)
9. [CREATE and MERGE](#9-create-and-merge)
10. [SET — Update Properties](#10-set--update-properties)
11. [DELETE and DETACH DELETE](#11-delete-and-detach-delete)
12. [Advanced Patterns](#12-advanced-patterns)
13. [Hotel Scenario Queries](#13-hotel-scenario-queries)
14. [Exercises](#14-exercises)

---

## 1. What is Cypher?

Cypher is Neo4j's graph query language. It uses **ASCII art** to describe patterns.

| SQL concept  | Cypher equivalent           |
|--------------|-----------------------------|
| Table        | Node label (`:Hotel`)       |
| Row          | Node                        |
| Column       | Property (`n.name`)         |
| Foreign key  | Relationship (`-[:HAS_FLOOR]->`) |
| JOIN         | Pattern matching            |

**Core syntax:**
```
(node)                     — a node
(n:Label)                  — node with a label
(n:Label {prop: value})    — node with a label AND property
(a)-[:REL]->(b)            — directed relationship
(a)-[:REL]-(b)             — undirected relationship
(a)-[r:REL]->(b)           — relationship with a variable
```

---

## 2. Explore the Graph

### See everything (small sample)
```cypher
MATCH (n) RETURN n LIMIT 25
```

### Count all nodes
```cypher
MATCH (n) RETURN count(n) AS total_nodes
```

### Count all relationships
```cypher
MATCH ()-[r]->() RETURN count(r) AS total_rels
```

### See every node label and how many exist
```cypher
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS count', {})
  YIELD value
RETURN label, value.count AS count
ORDER BY count DESC
```

> **Shortcut without APOC:**
```cypher
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS count
ORDER BY count DESC
```

### See every relationship type
```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY count DESC
```

### See the full schema
```cypher
CALL db.schema.visualization()
```

---

## 3. MATCH — Find Nodes

### Find the hotel
```cypher
MATCH (h:Hotel)
RETURN h
```

### Find all floors
```cypher
MATCH (f:Floor)
RETURN f.id, f.floor_number, f.type, f.room_count
ORDER BY f.floor_number
```

### Find all agents
```cypher
MATCH (a:Agent)
RETURN a.id, a.role, a.loop_interval_s
```

### Find a specific room by number
```cypher
MATCH (r:Room {room_number: 701})
RETURN r
```

### Find all HVAC units that are running
```cypher
MATCH (u:HVACUnit {status: 'running'})
RETURN u.id, u.type, u.setpoint_celsius, u.power_kw
ORDER BY u.power_kw DESC
```

### Find all occupied function halls
```cypher
MATCH (h:FunctionHall {status: 'in-use'})
RETURN h.name, h.capacity, h.occupancy_count
ORDER BY h.occupancy_count DESC
```

---

## 4. WHERE — Filter Results

### Rooms on floor 7
```cypher
MATCH (r:Room)
WHERE r.floor_number = 7
RETURN r.room_number, r.type, r.status, r.occupancy_count
ORDER BY r.room_number
```

### Zones where temperature exceeds the setpoint (need cooling)
```cypher
MATCH (z:Zone)
WHERE z.current_temp_c > z.setpoint_celsius
RETURN z.id, z.type,
       z.current_temp_c   AS current_temp,
       z.setpoint_celsius AS setpoint,
       round(z.current_temp_c - z.setpoint_celsius, 2) AS over_by
ORDER BY over_by DESC
```

### HVAC units consuming more than 25 kW
```cypher
MATCH (u:HVACUnit)
WHERE u.power_kw > 25
RETURN u.id, u.type, u.power_kw, u.efficiency_pct
ORDER BY u.power_kw DESC
```

### Sensors with high CO2 readings (> 900 ppm)
```cypher
MATCH (s:Sensor {type: 'co2'})
WHERE s.value > 900
RETURN s.id, s.value AS co2_ppm, s.unit
ORDER BY s.value DESC
```

### Range filter — comfort temperature band [20, 26] °C
```cypher
MATCH (z:Zone)
WHERE z.current_temp_c >= 20 AND z.current_temp_c <= 26
RETURN z.id, z.type, z.current_temp_c
ORDER BY z.current_temp_c DESC
```

### String filter — find agents whose ID contains "Agent"
```cypher
MATCH (a:Agent)
WHERE a.id CONTAINS 'Agent'
RETURN a.id, a.status
```

### IN filter — find specific zones
```cypher
MATCH (z:Zone)
WHERE z.type IN ['LOBBY', 'POOL', 'CORRIDOR']
RETURN z.id, z.type, z.current_temp_c, z.setpoint_celsius
```

---

## 5. RETURN — Shape the Output

### Select specific properties (like SELECT columns)
```cypher
MATCH (r:Room)
RETURN r.room_number AS room, r.type, r.status, r.occupancy_count AS guests
LIMIT 10
```

### Computed columns
```cypher
MATCH (z:Zone)
RETURN z.id,
       z.current_temp_c - z.setpoint_celsius AS temp_deviation,
       CASE
         WHEN z.current_temp_c > z.setpoint_celsius + 1 THEN 'HOT'
         WHEN z.current_temp_c < z.setpoint_celsius - 1 THEN 'COLD'
         ELSE 'OK'
       END AS comfort_status
ORDER BY temp_deviation DESC
```

### Rename with AS
```cypher
MATCH (b:EnergyBudget)
RETURN b.date              AS date,
       b.total_kwh         AS budget_kwh,
       b.consumed_kwh      AS used_kwh,
       b.remaining_kwh     AS left_kwh,
       b.consumed_kwh / b.total_kwh * 100 AS pct_used
```

### DISTINCT — unique values
```cypher
MATCH (r:Room)
RETURN DISTINCT r.type AS room_type
```

### ORDER BY + LIMIT
```cypher
MATCH (r:Room)
WHERE r.status = 'occupied'
RETURN r.room_number, r.floor_number, r.occupancy_count
ORDER BY r.occupancy_count DESC, r.room_number
LIMIT 10
```

---

## 6. Relationships

### Hotel → floors (one hop)
```cypher
MATCH (h:Hotel)-[:HAS_FLOOR]->(f:Floor)
RETURN h.name, f.floor_number, f.type
ORDER BY f.floor_number
```

### Floor → zone → rooms (two hops)
```cypher
MATCH (f:Floor {floor_number: 5})-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
RETURN f.floor_number, z.id, r.room_number, r.status
ORDER BY r.room_number
```

### Zone → HVAC unit
```cypher
MATCH (z:Zone)-[:HAS_HVAC]->(u:HVACUnit)
RETURN z.id AS zone, z.type AS zone_type,
       u.id AS hvac_unit, u.type AS hvac_type,
       u.setpoint_celsius AS setpoint, u.power_kw
ORDER BY u.power_kw DESC
```

### Agent → what it controls
```cypher
MATCH (a:Agent {id: 'HVACControlAgent'})-[:CONTROLS]->(u)
RETURN a.id AS agent, labels(u)[0] AS controls_type, u.id AS unit_id
LIMIT 10
```

### Sensor → what it monitors
```cypher
MATCH (s:Sensor)-[:MONITORS]->(z:Zone)
RETURN s.id, s.type, s.value, s.unit, z.id AS zone
ORDER BY s.type, z.id
LIMIT 15
```

### Agent → events it performed (agent memory)
```cypher
MATCH (a:Agent {id: 'EnergyBudgetAgent'})-[:PERFORMED]->(e:Event)
RETURN e.type AS event_type, e.data, e.ts AS timestamp
ORDER BY e.ts DESC
```

### FIPA-ACL inter-agent messages
```cypher
MATCH (sender:Agent)-[m:SENDS_MESSAGE]->(receiver:Agent)
RETURN sender.id AS from,
       m.performative AS performative,
       m.content_type AS content,
       receiver.id    AS to
ORDER BY sender.id, m.performative
```

### Relationship variable — inspect relationship properties
```cypher
MATCH (a:Agent)-[m:SENDS_MESSAGE]->(b:Agent)
WHERE m.performative = 'CFP'
RETURN a.id AS initiator, m.content_type AS cfp_for, b.id AS contractor
```

---

## 7. Path Traversal

### Variable-length path — everything within 2 hops of the hotel
```cypher
MATCH (h:Hotel)-[*1..2]-(n)
RETURN DISTINCT labels(n)[0] AS node_type, count(n) AS count
ORDER BY count DESC
```

### Full hotel hierarchy (hotel → floor → zone → room)
```cypher
MATCH path = (h:Hotel)-[:HAS_FLOOR]->(f:Floor)-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WHERE f.floor_number = 3
RETURN path
```

### Shortest path between two agents
```cypher
MATCH p = shortestPath(
  (a:Agent {id: 'OccupancyAgent'})-[*]-(b:Agent {id: 'LightingAgent'})
)
RETURN p
```

### All paths between WeatherForecastAgent and a zone
```cypher
MATCH p = (a:Agent {id: 'WeatherForecastAgent'})-[*1..4]-(z:Zone)
RETURN p LIMIT 5
```

### Who controls HVAC on floor 7?
```cypher
MATCH (a:Agent)-[:CONTROLS]->(u:HVACUnit)<-[:HAS_HVAC]-(z:Zone {id: 'zone-FLOOR-07'})
RETURN a.id AS agent, u.id AS hvac_unit, z.id AS zone
```

---

## 8. Aggregation

### Occupied vs vacant room count
```cypher
MATCH (r:Room)
RETURN r.status AS status, count(r) AS count
ORDER BY count DESC
```

### Total occupancy per floor
```cypher
MATCH (f:Floor)-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
RETURN f.floor_number,
       count(r)                   AS total_rooms,
       sum(r.occupancy_count)     AS total_guests,
       sum(CASE r.status WHEN 'occupied' THEN 1 ELSE 0 END) AS occupied_rooms
ORDER BY f.floor_number
```

### Average temperature per zone type
```cypher
MATCH (z:Zone)
RETURN z.type,
       round(avg(z.current_temp_c), 2) AS avg_temp,
       round(min(z.current_temp_c), 2) AS min_temp,
       round(max(z.current_temp_c), 2) AS max_temp,
       count(z)                        AS zone_count
ORDER BY avg_temp DESC
```

### Total HVAC power by zone type
```cypher
MATCH (z:Zone)-[:HAS_HVAC]->(u:HVACUnit)
RETURN z.type,
       count(u)        AS unit_count,
       sum(u.power_kw) AS total_kw,
       avg(u.power_kw) AS avg_kw
ORDER BY total_kw DESC
```

### Event count per agent (agent activity)
```cypher
MATCH (a:Agent)-[:PERFORMED]->(e:Event)
RETURN a.id AS agent, count(e) AS event_count
ORDER BY event_count DESC
```

### Event breakdown per agent
```cypher
MATCH (a:Agent)-[:PERFORMED]->(e:Event)
RETURN a.id AS agent, e.type AS event_type, count(e) AS count
ORDER BY a.id, count DESC
```

### Energy budget breakdown
```cypher
MATCH (b:EnergyBudget)
WHERE b.total_kwh IS NOT NULL
RETURN b.date,
       b.hvac_kwh      AS hvac_kWh,
       b.lighting_kwh  AS lighting_kWh,
       b.elevators_kwh AS elevators_kWh,
       b.pool_kwh      AS pool_kWh,
       b.consumed_kwh  AS total_consumed,
       round(b.consumed_kwh / b.total_kwh * 100, 1) AS pct_used
```

### How many sensors monitor each zone type?
```cypher
MATCH (s:Sensor)-[:MONITORS]->(z:Zone)
RETURN z.type AS zone_type, s.type AS sensor_type, count(s) AS count
ORDER BY zone_type, sensor_type
```

---

## 9. CREATE and MERGE

### CREATE — always makes a new node
```cypher
CREATE (g:Guest {
  id:         'guest-001',
  name:       'Somchai Srisuk',
  room:       701,
  check_in:   date('2026-05-20'),
  vip:        true
})
RETURN g
```

> **Warning:** `CREATE` does not check if the node already exists. Run it twice and you get two nodes.

### MERGE — create only if not found (idempotent)
```cypher
MERGE (g:Guest {id: 'guest-001'})
ON CREATE SET
  g.name     = 'Somchai Srisuk',
  g.room     = 701,
  g.check_in = date('2026-05-20')
ON MATCH SET
  g.last_seen = datetime()
RETURN g
```

### MERGE a relationship
```cypher
MATCH (g:Guest {id: 'guest-001'})
MATCH (r:Room {room_number: 701})
MERGE (g)-[:STAYS_IN]->(r)
```

### Create an agent decision event
```cypher
MATCH (a:Agent {id: 'HVACControlAgent'})
CREATE (e:Event {
  id:    'HVACControlAgent:manual-test',
  type:  'SETPOINT_ADJUSTED',
  data:  '{"zone":"zone-FLOOR-10","new_setpoint":21.5,"reason":"manual_test"}',
  ts:    datetime()
})
CREATE (a)-[:PERFORMED]->(e)
RETURN a.id, e.type, e.ts
```

---

## 10. SET — Update Properties

### Change a room to occupied
```cypher
MATCH (r:Room {room_number: 501})
SET r.status          = 'occupied',
    r.occupancy_count = 2
RETURN r.room_number, r.status, r.occupancy_count
```

### Update HVAC setpoint when zone is over temperature
```cypher
MATCH (z:Zone {id: 'zone-FLOOR-07'})-[:HAS_HVAC]->(u:HVACUnit)
WHERE z.current_temp_c > z.setpoint_celsius + 1
SET u.setpoint_celsius = u.setpoint_celsius - 1.0
RETURN z.id, z.current_temp_c, u.id, u.setpoint_celsius AS new_setpoint
```

### Dim all lighting in vacant zones
```cypher
MATCH (z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH z, sum(r.occupancy_count) AS total_occ
WHERE total_occ = 0
MATCH (z)-[:HAS_LIGHTING]->(l:LightingSystem)
SET l.brightness_pct = 20,
    l.last_adjusted  = datetime()
RETURN z.id, l.id, l.brightness_pct
```

### Add a label to existing nodes
```cypher
MATCH (z:Zone)
WHERE z.current_temp_c > z.setpoint_celsius + 1
SET z:OverTemp
RETURN z.id, labels(z)
```

### Remove a label
```cypher
MATCH (z:OverTemp)
REMOVE z:OverTemp
RETURN z.id, labels(z)
```

---

## 11. DELETE and DETACH DELETE

### Delete a node with no relationships
```cypher
MATCH (g:Guest {id: 'guest-001'})
DELETE g
```

### DETACH DELETE — delete node AND all its relationships
```cypher
MATCH (e:Event {id: 'HVACControlAgent:manual-test'})
DETACH DELETE e
```

> **Rule:** Always use `DETACH DELETE` unless you are certain the node has no relationships.

### Delete all events of a specific type (use carefully)
```cypher
MATCH (e:Event {type: 'MANUAL_TEST'})
DETACH DELETE e
```

---

## 12. Advanced Patterns

### WITH — pipeline results between clauses
```cypher
MATCH (f:Floor)-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH f.floor_number AS floor, count(r) AS total, sum(r.occupancy_count) AS guests
WHERE guests > 10
RETURN floor, total, guests
ORDER BY guests DESC
```

### OPTIONAL MATCH — like a LEFT JOIN
```cypher
MATCH (z:Zone)
OPTIONAL MATCH (z)-[:HAS_HVAC]->(u:HVACUnit)
RETURN z.id, z.type,
       u.id AS hvac_unit,
       CASE WHEN u IS NULL THEN 'NO HVAC' ELSE u.status END AS hvac_status
```

### COLLECT — aggregate into a list
```cypher
MATCH (f:Floor)-[:HAS_ZONE]->(z:Zone)
RETURN f.floor_number,
       collect(z.id)   AS zones,
       collect(z.type) AS zone_types
ORDER BY f.floor_number
```

### UNWIND — expand a list back to rows
```cypher
WITH ['OccupancyAgent', 'HVACControlAgent', 'EnergyBudgetAgent'] AS agent_ids
UNWIND agent_ids AS aid
MATCH (a:Agent {id: aid})
RETURN a.id, a.role, a.loop_interval_s
```

### CASE expression — conditional logic
```cypher
MATCH (u:HVACUnit)
RETURN u.id,
       u.efficiency_pct,
       CASE
         WHEN u.efficiency_pct >= 85 THEN 'EFFICIENT'
         WHEN u.efficiency_pct >= 78 THEN 'ACCEPTABLE'
         ELSE 'NEEDS_SERVICE'
       END AS efficiency_grade
ORDER BY u.efficiency_pct DESC
```

### EXISTS — check if a pattern exists
```cypher
MATCH (z:Zone)
WHERE EXISTS {
  MATCH (z)-[:HAS_HVAC]->(:HVACUnit {status: 'running'})
}
RETURN z.id, z.type, z.current_temp_c
```

### NOT EXISTS — zones without sensors
```cypher
MATCH (z:Zone)
WHERE NOT EXISTS {
  MATCH (z)<-[:MONITORS]-(:Sensor)
}
RETURN z.id, z.type
```

### Subquery with CALL { }
```cypher
MATCH (a:Agent)
CALL {
  WITH a
  MATCH (a)-[:PERFORMED]->(e:Event)
  RETURN count(e) AS event_count
}
RETURN a.id, a.role, event_count
ORDER BY event_count DESC
```

---

## 13. Hotel Scenario Queries

These are realistic operational queries an energy management system would run every 60 seconds.

---

### Scenario A — Occupancy Snapshot (OccupancyAgent)

```cypher
// Which floors have more than 50% room occupancy?
MATCH (f:Floor)-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH f.floor_number AS floor,
     count(r) AS total_rooms,
     sum(CASE r.status WHEN 'occupied' THEN 1 ELSE 0 END) AS occupied
WHERE occupied * 1.0 / total_rooms > 0.5
RETURN floor,
       total_rooms,
       occupied,
       round(occupied * 100.0 / total_rooms, 1) AS occupancy_pct
ORDER BY occupancy_pct DESC
```

```cypher
// Sensor readings for floor 7 — full picture
MATCH (z:Zone {id: 'zone-FLOOR-07'})<-[:MONITORS]-(s:Sensor)
RETURN s.type AS sensor, s.value, s.unit
ORDER BY s.type
```

```cypher
// Which function halls are empty right now?
MATCH (h:FunctionHall {status: 'empty'})
RETURN h.name, h.capacity
ORDER BY h.capacity DESC
```

---

### Scenario B — HVAC Decision (HVACControlAgent)

```cypher
// Zones that need setpoint reduction (temp > setpoint + 1°C)
MATCH (z:Zone)-[:HAS_HVAC]->(u:HVACUnit)
WHERE z.current_temp_c > z.setpoint_celsius + 1.0
RETURN z.id          AS zone,
       z.type        AS zone_type,
       z.current_temp_c - z.setpoint_celsius AS over_by_C,
       u.id          AS hvac_unit,
       u.setpoint_celsius AS current_setpoint,
       u.setpoint_celsius - 1.0 AS recommended_setpoint
ORDER BY over_by_C DESC
```

```cypher
// Total HVAC power for occupied vs vacant zones
MATCH (z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH z, sum(r.occupancy_count) AS zone_guests
MATCH (z)-[:HAS_HVAC]->(u:HVACUnit)
RETURN CASE WHEN zone_guests > 0 THEN 'occupied' ELSE 'vacant' END AS zone_status,
       count(u)        AS hvac_units,
       sum(u.power_kw) AS total_power_kw
```

```cypher
// Pre-cool for tomorrow's weather
MATCH (w:WeatherForecast)
WHERE w.cooling_load_factor > 1.2
MATCH (u:HVACUnit {type: 'AHU'})
RETURN w.date, w.outdoor_temp_c, w.forecast_type, w.cooling_load_factor,
       count(u) AS ahu_count,
       'Lower setpoints by 1°C before peak hours' AS recommendation
```

---

### Scenario C — Energy Budget (EnergyBudgetAgent)

```cypher
// Current budget status
MATCH (h:Hotel)-[:HAS_BUDGET]->(b:EnergyBudget)
RETURN h.name,
       b.date,
       b.total_kwh                                  AS budget_kWh,
       b.consumed_kwh                               AS consumed_kWh,
       b.remaining_kwh                              AS remaining_kWh,
       round(b.consumed_kwh / b.total_kwh * 100, 1) AS pct_used,
       b.on_track
```

```cypher
// Energy breakdown: which system uses the most?
MATCH (b:EnergyBudget)
WHERE b.total_kwh IS NOT NULL
WITH b.hvac_kwh AS hvac, b.lighting_kwh AS lighting,
     b.elevators_kwh AS elevators, b.pool_kwh AS pool,
     b.total_kwh AS total
RETURN 'HVAC'      AS system, hvac      AS kwh, round(hvac*100.0/total,1)      AS pct
UNION ALL
MATCH (b:EnergyBudget) WHERE b.total_kwh IS NOT NULL
WITH b.lighting_kwh AS lighting, b.total_kwh AS total
RETURN 'Lighting'  AS system, lighting  AS kwh, round(lighting*100.0/total,1)  AS pct
UNION ALL
MATCH (b:EnergyBudget) WHERE b.total_kwh IS NOT NULL
WITH b.elevators_kwh AS elevators, b.total_kwh AS total
RETURN 'Elevators' AS system, elevators AS kwh, round(elevators*100.0/total,1) AS pct
UNION ALL
MATCH (b:EnergyBudget) WHERE b.total_kwh IS NOT NULL
WITH b.pool_kwh AS pool, b.total_kwh AS total
RETURN 'Pool'      AS system, pool      AS kwh, round(pool*100.0/total,1)      AS pct
```

```cypher
// Potential savings: power used by lighting in vacant zones
MATCH (z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH z, sum(r.occupancy_count) AS guests
WHERE guests = 0
MATCH (z)-[:HAS_LIGHTING]->(l:LightingSystem)
WHERE l.brightness_pct > 20
RETURN z.id           AS zone,
       l.brightness_pct AS current_pct,
       l.power_kw      AS current_kw,
       l.power_kw * (l.brightness_pct - 20) / 100.0 AS saveable_kw
ORDER BY saveable_kw DESC
```

---

### Scenario D — Contract Net Protocol Audit

```cypher
// Full FIPA-ACL message flow
MATCH (sender:Agent)-[m:SENDS_MESSAGE]->(receiver:Agent)
RETURN sender.id    AS from_agent,
       m.performative AS performative,
       m.content_type AS content_type,
       receiver.id  AS to_agent
ORDER BY
  CASE m.performative
    WHEN 'INFORM'  THEN 1
    WHEN 'CFP'     THEN 2
    WHEN 'PROPOSE' THEN 3
    WHEN 'ACCEPT'  THEN 4
    ELSE 5
  END
```

```cypher
// Replay today's Contract Net cycle from agent memory
MATCH (a:Agent)-[:PERFORMED]->(e:Event)
WHERE e.type IN ['CONTRACT_NET_CFP_SENT','PROPOSAL_SUBMITTED','PROPOSALS_EVALUATED','BUDGET_UPDATED']
RETURN a.id AS agent, e.type AS event, e.data
ORDER BY e.ts
```

```cypher
// What proposals were submitted and by whom?
MATCH (a:Agent)-[:PERFORMED]->(e:Event {type: 'PROPOSAL_SUBMITTED'})
RETURN a.id AS agent, e.data
```

---

### Scenario E — Comfort Compliance Check

```cypher
// Rooms breaching comfort temperature range [20, 26] °C
MATCH (r:Room)
WHERE r.current_temp_c < 20 OR r.current_temp_c > 26
RETURN r.room_number, r.floor_number, r.current_temp_c,
       CASE
         WHEN r.current_temp_c < 20 THEN 'TOO_COLD'
         ELSE 'TOO_HOT'
       END AS comfort_violation
ORDER BY abs(r.current_temp_c - 23) DESC
```

```cypher
// Zones with CO2 above 1000 ppm (air quality concern)
MATCH (s:Sensor {type: 'co2'})-[:MONITORS]->(z:Zone)
WHERE s.value > 1000
RETURN z.id AS zone, z.type, s.value AS co2_ppm
ORDER BY s.value DESC
```

```cypher
// Overall comfort dashboard
MATCH (z:Zone)
WITH count(z) AS total,
     sum(CASE WHEN z.current_temp_c BETWEEN 20 AND 26 THEN 1 ELSE 0 END) AS compliant
RETURN total AS total_zones,
       compliant AS compliant_zones,
       total - compliant AS non_compliant_zones,
       round(compliant * 100.0 / total, 1) AS compliance_pct
```

---

## 14. Exercises

Try writing these queries yourself before looking at the hints.

---

**Exercise 1** — Find the top 3 floors by total guest count.

<details>
<summary>Hint</summary>

```cypher
MATCH (f:Floor)-[:HAS_ZONE]->(z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH f.floor_number AS floor, sum(r.occupancy_count) AS guests
RETURN floor, guests
ORDER BY guests DESC
LIMIT 3
```
</details>

---

**Exercise 2** — Which agent has performed the most events?

<details>
<summary>Hint</summary>

```cypher
MATCH (a:Agent)-[:PERFORMED]->(e:Event)
RETURN a.id AS agent, count(e) AS events
ORDER BY events DESC
LIMIT 1
```
</details>

---

**Exercise 3** — List every function hall with its zone's current temperature.

<details>
<summary>Hint</summary>

```cypher
MATCH (z:Zone)-[:CONTAINS_HALL]->(h:FunctionHall)
RETURN h.name, h.status, h.occupancy_count,
       z.current_temp_c AS zone_temp, z.setpoint_celsius AS setpoint
ORDER BY h.name
```
</details>

---

**Exercise 4** — Find all agents that send a `CFP` message and who receives it.

<details>
<summary>Hint</summary>

```cypher
MATCH (a:Agent)-[m:SENDS_MESSAGE {performative: 'CFP'}]->(b:Agent)
RETURN a.id AS initiator, m.content_type, b.id AS contractor
```
</details>

---

**Exercise 5** — Calculate potential energy savings if all vacant-zone HVAC units raised their setpoint by 2°C.  
Assume every 1°C setpoint increase saves 3% power per unit.

<details>
<summary>Hint</summary>

```cypher
MATCH (z:Zone)-[:CONTAINS_ROOM]->(r:Room)
WITH z, sum(r.occupancy_count) AS guests
WHERE guests = 0
MATCH (z)-[:HAS_HVAC]->(u:HVACUnit)
WITH sum(u.power_kw) AS vacant_hvac_kw
RETURN vacant_hvac_kw AS current_kw,
       round(vacant_hvac_kw * 0.06, 2) AS saved_kw_with_2C_raise,
       round(vacant_hvac_kw * 0.94, 2) AS new_kw
```
</details>

---

**Exercise 6** — Write a MERGE statement that checks a guest into room 1205.  
Create the guest node and the `STAYS_IN` relationship to the room.

<details>
<summary>Hint</summary>

```cypher
MERGE (g:Guest {id: 'guest-checkout-test'})
ON CREATE SET g.name = 'Test Guest', g.check_in = date()
WITH g
MATCH (r:Room {room_number: 1205})
MERGE (g)-[:STAYS_IN]->(r)
SET r.status = 'occupied', r.occupancy_count = 1
RETURN g.name, r.room_number, r.status
```
</details>

---

**Exercise 7** — Find the shortest message path between `WeatherForecastAgent` and `LightingAgent`.

<details>
<summary>Hint</summary>

```cypher
MATCH p = shortestPath(
  (a:Agent {id: 'WeatherForecastAgent'})-[:SENDS_MESSAGE*]-(b:Agent {id: 'LightingAgent'})
)
RETURN [node IN nodes(p) | node.id] AS path,
       length(p) AS hops
```
</details>

---

## Quick Reference Card

```
// Read
MATCH (n:Label {prop:val})-[:REL]->(m) RETURN n, m LIMIT 25

// Filter
WHERE n.prop > 10 AND n.name CONTAINS 'text'

// Aggregate
count(n)   sum(n.kw)   avg(n.temp)   min()   max()   collect(n.id)

// Write — safe create/update
MERGE (n:Label {id: $id})
ON CREATE SET n.created = datetime()
ON MATCH  SET n.updated = datetime()

// Update property
SET n.prop = value

// Delete with relationships
DETACH DELETE n

// Pipeline
MATCH ... WITH ... WHERE ... RETURN

// Variable-length path  (1 to 3 hops)
MATCH (a)-[*1..3]->(b)

// Shortest path
shortestPath((a)-[*]-(b))
```

---

*Tutorial built against the graph in `6.hotel_kg_builder.py`. Re-run that script any time to reset the data.*
