# Smart Hotel MAS — Reference

Deep-dive reference compiling the design tables, mathematical models, and trace
examples from the workshop PDF. Companion to the runnable code in `checkpoints/`,
`patterns/`, `graphrag/`, and `production/`.

---

## Why 4 Memory Types

| Layer | Human Analogy | Agent Implementation | Hotel Use Case | Access Time |
|-------|---------------|----------------------|----------------|-------------|
| **L1 Working** | Short-term / scratchpad | In-process Python dict | "Which rooms am I processing in this batch right now" | ~microseconds (RAM) |
| **L2 Episodic** | Autobiographical memory | SQLite append-only event log | "What did the agents do today?" — durable timeline | ~milliseconds (local disk) |
| **L3 Semantic** | Conceptual / fuzzy recall | ChromaDB vector store + embeddings | "Find readings that *feel* like an HVAC failure" | ~10–50 ms (ANN search) |
| **L4 Knowledge Graph** | Structured world knowledge | Neo4j property graph (Cypher) | "Which staff can fix the HVAC alert on floor 3?" | ~10–100 ms (indexed query) |

**Rule of thumb:** L1 = now, L2 = recent history, L3 = meaning, L4 = relationships.

---

## Mathematical Models

### 1. HVAC Setpoint Optimization — Linear Program (PuLP)

Minimize total energy cost across rooms:

```
minimize    Σ_r  setpoint_r · tariff · area_r

subject to  comfort_lo ≤ setpoint_r ≤ comfort_hi      (occupied rooms)
            setpoint_r = eco_setpoint                 (empty rooms)
            |setpoint_r − setpoint_r_prev| ≤ ramp_max (ramp / comfort-shock limit)
            Σ_r demand_r ≤ peak_demand_cap            (grid demand cap)
```

Decision variables `setpoint_r` (continuous, °C); objective is linear in price ×
area × setpoint, so the problem is a standard LP solved to global optimum.

### 2. Occupancy / Energy Forecasting — ARIMA & Prophet decomposition

Prophet additive decomposition of a time series `y(t)`:

```
y(t) = g(t) + s(t) + h(t) + ε_t
       │      │      │      └─ noise
       │      │      └─ holiday / event effects
       │      └─ seasonality (daily + weekly Fourier terms)
       └─ trend (piecewise-linear growth)
```

ARIMA(p, d, q) baseline:

```
(1 − Σ φ_i Lⁱ)(1 − L)ᵈ y_t = (1 + Σ θ_j Lʲ) ε_t
```

where `L` is the lag operator, `d` the differencing order, `φ`/`θ` the AR/MA
coefficients.

### 3. HVAC Control Policy — Q-Learning / DQN (Bellman)

Bellman optimality update the DQN approximates:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

| Space | Definition | Size |
|-------|------------|------|
| **State** `s` | (temp bucket × occupancy × time-of-day × outside-temp × tariff-tier) | **240** discretized states |
| **Action** `a` | {−2°C, −1°C, hold, +1°C, +2°C} setpoint adjustment | **5** actions |
| **Reward** `r` | `(3.0 if comfort_ok else −1.0) − energy_kwh` | scalar |

Tabular Q-table size = **240 × 5 = 1200** entries. `α` = learning rate,
`γ` = discount factor.

### 4. Anomaly Detection — Isolation Forest

Anomaly score for a point `x` over `n` samples:

```
s(x, n) = 2 ^ ( − E[h(x)] / c(n) )
```

where `E[h(x)]` is the average path length to isolate `x` across the trees and
`c(n)` is the expected path length of an unsuccessful BST search on `n` points:

```
c(n) = 2·H(n−1) − 2(n−1)/n ,   H(i) ≈ ln(i) + 0.5772 (Euler–Mascheroni)
```

`s → 1` ⇒ anomaly (isolated quickly); `s → 0.5` ⇒ normal.

---

## Model Performance Comparison

| Model | Library | Training Time | Inference Time | Accuracy / Quality | Best For |
|-------|---------|---------------|----------------|--------------------|----------|
| **LP optimization** | PuLP | none (solve per call) | ~10–50 ms | Globally optimal (given constraints) | Real-time setpoint allocation under hard constraints |
| **Prophet / ARIMA** | Prophet | seconds–minutes | ~ms per horizon | MAPE ~5–15% on occupancy | Short-horizon occupancy & energy forecasts |
| **DQN** | Stable-Baselines3 | minutes–hours (sim) | ~1 ms | Converges to comfort/energy trade-off | Adaptive control where dynamics are learned |
| **Isolation Forest** | scikit-learn | seconds | ~ms per point | High recall on point anomalies | Unsupervised sensor anomaly flagging |

---

## Memory Layer Performance Summary

| Layer | Technology | Write Time | Read Time | Capacity | Persistence |
|-------|-----------|------------|-----------|----------|-------------|
| **L1 Working** | Python dict (RAM) | ~µs | ~µs | RAM-bound | None (process lifetime) |
| **L2 Episodic** | SQLite (local file) | ~1 ms | ~1–5 ms | GBs | Durable (disk) |
| **L3 Semantic** | ChromaDB + embeddings | ~10–50 ms (embed+add) | ~10–50 ms (ANN) | Millions of vectors | Durable (managed/local) |
| **L4 Knowledge Graph** | Neo4j (Bolt) | ~5–20 ms (MERGE) | ~10–100 ms (Cypher) | Billions of nodes/rels | Durable (managed/local) |

---

## Agent Design — Tool Design Best Practices

| Practice | What it means | Why it matters |
|----------|---------------|----------------|
| **Atomic Tools** | One tool = one job (`read_sensor`, not `read_and_alert`) | Composable, testable, predictable for the LLM |
| **Typed Outputs** | Return structured/typed dicts, not free prose | Downstream agents parse reliably; fewer hallucinations |
| **Bounded Results** | Always `LIMIT`/cap rows returned | Protects the prompt budget and latency |
| **Memory Write-Through** | Tools write their effects to L2/L4 as they run | Episodic + graph stay the source of truth |
| **Idempotent MERGE** | Use Cypher `MERGE` (not `CREATE`) for entities | Re-runs/retries don't duplicate nodes |

---

## GraphRAG Query Patterns

| Pattern | NL pattern | Cypher pattern | Notes |
|---------|-----------|----------------|-------|
| **Aggregation** | "average energy per floor" | `MATCH ... RETURN room.floor, avg(reading.kwh)` | use `avg/sum/count`; no LIMIT needed |
| **Existence** | "are there rooms over 28°C" | `MATCH ... WHERE reading.temp_c > 28 RETURN count(*) > 0` | boolean/exists check |
| **Path traversal** | "which staff can fix this alert" | `MATCH (alert)-[:AFFECTS]->(room)-[:HAS_DEVICE]->(d)<-...->(staff)` | multi-hop relationships |
| **Top-N** | "5 hottest rooms" | `MATCH ... RETURN room ORDER BY reading.temp_c DESC LIMIT 5` | always cap N |
| **Temporal** | "alerts in last 4 hours" | `WHERE alert.ts > datetime() - duration({hours: 4})` | use `duration({...})`, `toString()` on output |

---

## ReAct Loop Trace Example

```
Thought:  Guest in R301 reports it's too warm. I should check the latest reading.
Action:   read_sensor(room_id="R301")
Obs:      {temp_c: 28.5, humidity: 75, occupancy: true, kwh: 4.2}
Thought:  28.5°C exceeds the 27°C comfort threshold — this is a HIGH_TEMP alert.
Action:   create_alert(room="R301", type="HIGH_TEMP", value=28.5, threshold=27.0)
Obs:      {alert_id: "A-1183", severity: "high"}
Thought:  An alert needs an available HVAC technician on the current shift.
Action:   find_staff(skill="HVAC", shift="day")
Obs:      [{id: "S-07", name: "Anan", role: "technician"}]
Thought:  Assign the job and write it through to the graph (idempotent MERGE).
Action:   assign_job(alert_id="A-1183", staff_id="S-07", priority="high")
Obs:      {job_id: "J-552", status: "assigned"}
Thought:  Done — alert raised, technician assigned, all persisted to L4.
Final:    Raised HIGH_TEMP alert A-1183 for R301 (28.5°C) and assigned tech S-07.
```

---

## 5-Agent Interaction Timeline (0:00 → 0:05)

```
Time   Agent              Action
─────  ─────────────────  ───────────────────────────────────────────────────
0:00   SensorAgent        Reads batch for floor 3 (40 rooms) → writes L1 + L2
0:01   AnomalyAgent       Isolation Forest flags R301 (28.5°C) as anomalous
0:02   AlertAgent         Creates HIGH_TEMP alert in L4; AFFECTS R301
0:03   EnergyAgent        LP re-optimizes floor-3 setpoints around the alert
0:04   MaintenanceAgent   GraphRAG: "available HVAC staff?" → assigns S-07
0:05   OrchestratorAgent  Summarizes cycle, writes Event to L4, returns report
```

Each agent reads its injected memory context (L1 + L2 + L4 active alerts; see
`patterns/memory_injection.py`) before acting, and writes its effects through to
the durable tiers — so the next agent in the timeline sees a consistent world.
