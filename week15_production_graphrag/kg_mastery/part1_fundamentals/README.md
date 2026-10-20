# Part 1 — Fundamentals

Runnable companion scripts for **Part 1** of `kg_mastery.pdf` (Cypher
fundamentals, advanced features, schema validation, temporal & multi-label
modelling). Every script targets the shared **Hotel IoT** knowledge graph and
imports connection helpers from `../common.py`.

## Run order

Run the loader **first**, then the lessons in numeric order:

```bash
python part1_fundamentals/00_load_hotel_dataset.py   # build the graph (run first)
python part1_fundamentals/01_cypher_basics.py        # §1.3 Cypher basics (12 patterns)
python part1_fundamentals/02_cypher_intermediate.py  # §1.4 intermediate (10 patterns)
python part1_fundamentals/03_cypher_advanced_gds.py  # §1.5 APOC / full-text / GDS
python part1_fundamentals/04_schema_validation.py    # §1.7 schema audit checks
python part1_fundamentals/05_temporal.py             # §1.8 temporal KG strategies
python part1_fundamentals/06_multi_label.py          # §1.9 multi-label nodes
python part1_fundamentals/07_exercises.py            # §1.6 exercises (beginner→advanced)
```

Each script first calls `check_connection()` and exits cleanly if Neo4j is
down. Demo writes are tagged `:Demo` and cleaned up, so the dataset stays
pristine.

## Prerequisite

Everything needs Neo4j running (with the APOC + GDS plugins):

```bash
docker compose -f week15/kg_mastery/docker-compose.yml up -d
```
