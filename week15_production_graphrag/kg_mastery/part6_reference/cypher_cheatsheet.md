# Part 6.1 — Cypher Quick Reference

`kg_mastery.pdf` §6.1. A compact Cypher cheat-sheet, every example adapted to the
**hotel** domain (Room / Device / SensorReading / Alert / Staff / MaintenanceJob).

## Patterns

| Feature | Syntax | Hotel example |
|---------|--------|---------------|
| Node pattern | `(var:Label {prop: val})` | `(r:Room {id: 'R101'})` |
| Relationship | `(a)-[:REL]->(b)` | `(r:Room)-[:HAS_DEVICE]->(d:Device)` |
| Any direction | `(a)-[:REL]-(b)` | `(d:Device)-[:TRIGGERED]-(a:Alert)` |
| Named relationship | `(a)-[x:REL]->(b)` | `(d:Device)-[t:TRIGGERED]->(a:Alert)` then use `t.ts` |
| Rel with property | `(a)-[:REL {p: v}]->(b)` | `(m1)-[:INTERACTS_WITH {severity:'HIGH'}]->(m2)` |
| Variable-length path | `(a)-[:REL*min..max]->(b)` | `(d:Device)-[:HAS_READING*1..3]->(s)` |
| Shortest path | `shortestPath((a)-[:REL*]-(b))` | `MATCH p = shortestPath((r1:Room)-[*]-(r2:Room)) RETURN p` |

## Reading & writing

| Feature | Syntax | Hotel example |
|---------|--------|---------------|
| MATCH + WHERE | `MATCH (n) WHERE cond RETURN n` | `MATCH (s:SensorReading) WHERE s.temp_c > 28 RETURN s` |
| OPTIONAL MATCH | `OPTIONAL MATCH (a)-[:R]->(b)` | `MATCH (r:Room) OPTIONAL MATCH (r)-[:HAS_DEVICE]->(d) RETURN r, d` |
| CREATE | `CREATE (n:Label {props})` | `CREATE (j:MaintenanceJob {id: randomUUID(), status:'ASSIGNED'})` |
| MERGE (get-or-create) | `MERGE (n:Label {key}) SET n.p = v` | `MERGE (st:Staff {id:'S1'}) SET st.name='Lek'` |
| DETACH DELETE | `MATCH (n) DETACH DELETE n` | `MATCH (a:Alert {id:'AL1'}) DETACH DELETE a` |

## Pipelining & aggregation

| Feature | Syntax | Hotel example |
|---------|--------|---------------|
| WITH (chain/aggregate) | `MATCH ... WITH x, agg ... ` | `MATCH (r:Room)-[:HAS_DEVICE]->(d) WITH r, count(d) AS n WHERE n > 2 RETURN r` |
| UNWIND (list → rows) | `UNWIND list AS row` | `UNWIND ['R101','R203'] AS rid MATCH (r:Room {id: rid}) RETURN r` |
| COLLECT (rows → list) | `collect(x)` | `MATCH (r:Room)-[:HAS_DEVICE]->(d) RETURN r.id, collect(d.id) AS devices` |
| SIZE (list/pattern length) | `size(list)` | `MATCH (r:Room) RETURN r.id, size((r)-[:HAS_DEVICE]->()) AS device_count` |
| CASE (conditional) | `CASE WHEN cond THEN a ELSE b END` | `RETURN CASE WHEN s.temp_c > 28 THEN 'HOT' ELSE 'OK' END AS status` |
| FOREACH (per-item write) | `FOREACH (x IN list \| ...)` | `FOREACH (rid IN ['R101','R203'] \| MERGE (:Room {id: rid}))` |

## Subqueries & predicates

| Feature | Syntax | Hotel example |
|---------|--------|---------------|
| CALL { } (subquery) | `CALL { ... } ` | `CALL { MATCH (a:Alert) WHERE a.resolved=false RETURN count(a) AS open } RETURN open` |
| EXISTS { } (pattern test) | `WHERE EXISTS { (a)-[:R]->(b) }` | `MATCH (r:Room) WHERE EXISTS { (r)-[:HAS_DEVICE]->(:Device {status:'FAULT'}) } RETURN r` |
| Pattern comprehension | `[ (a)-[:R]->(b) \| b.prop ]` | `MATCH (r:Room) RETURN r.id, [ (r)-[:HAS_DEVICE]->(d) \| d.type ] AS device_types` |

## Functions

| Feature | Syntax | Hotel example |
|---------|--------|---------------|
| Duration arithmetic | `datetime() - duration({...})` | `WHERE s.ts >= datetime() - duration({hours: 24})` |
| Duration between | `duration.between(a, b)` | `RETURN duration.between(j.started_at, j.completed_at).minutes AS mins` |
| String functions | `toUpper / toLower / trim / split / replace / substring` | `RETURN toUpper(r.type), split(d.model, '-')[0]` |
| String contains/starts | `STARTS WITH / ENDS WITH / CONTAINS` | `WHERE d.manufacturer CONTAINS 'Honeywell'` |
| toInteger / toFloat | `toInteger(x)` / `toFloat(x)` | `RETURN toFloat(s.temp_c), toInteger(r.floor)` |
| randomUUID | `randomUUID()` | `CREATE (j:MaintenanceJob {id: randomUUID()})` |
| coalesce (first non-null) | `coalesce(a, b, ...)` | `RETURN coalesce(r.description, 'n/a')` |

## Handy full queries

```cypher
-- Hot rooms in the last 24h
MATCH (r:Room)-[:HAS_DEVICE]->(:Device)-[:HAS_READING]->(s:SensorReading)
WHERE s.temp_c > 28 AND s.ts >= datetime() - duration({hours: 24})
RETURN r.id AS room, r.floor AS floor, max(s.temp_c) AS max_temp
ORDER BY max_temp DESC;

-- Open HIGH alerts with their room
MATCH (room:Room)-[:HAS_DEVICE]->(dev:Device)-[:TRIGGERED]->(a:Alert)
WHERE a.resolved = false AND a.severity = 'HIGH'
RETURN a.id, a.message, dev.id AS device, room.id AS room;

-- Assign a maintenance job (write-back)
MATCH (room:Room {id: 'R101'}), (st:Staff {id: 'S1'})
CREATE (j:MaintenanceJob {id: randomUUID(), type: 'HVAC_REPAIR', status: 'ASSIGNED', started_at: datetime()})
CREATE (st)-[:PERFORMED]->(j)-[:FOR_ROOM]->(room)
RETURN j.id;
```
