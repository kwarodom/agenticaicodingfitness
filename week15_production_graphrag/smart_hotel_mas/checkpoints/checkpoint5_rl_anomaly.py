#!/usr/bin/env python3
"""CHECKPOINT 5 — RL Policy + Anomaly Detection.

Goal: give the energy layer a *learned* control policy and the sensor layer a
*statistical* anomaly detector, instead of hand-tuned thresholds.

  Part A — HotelHVACEnv: a Gymnasium environment modelling one room's HVAC.
  Part B — train_rl_agent: a DQN that learns a comfort-vs-energy setpoint policy.
  Part C — SensorAnomalyDetector: an Isolation Forest fit on real Neo4j readings.

(smart_hotel_mas.pdf §CP5)

Q-learning / DQN background (the value the DQN approximates):

    Bellman optimality update:
        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

    Per-step reward used by HotelHVACEnv:
        reward = (3.0 if comfort_ok else -1.0) - energy_kwh
    so the agent is paid for keeping guests comfortable and charged for the
    energy each setpoint draws — it must learn to trade the two off.

Run:  python week15/smart_hotel_mas/checkpoints/checkpoint5_rl_anomaly.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config.py
from config import check_neo4j, get_driver, MODEL  # noqa: F401  (MODEL kept per workshop convention)

# ── Guarded optional dependencies ────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("This checkpoint needs numpy. Install it with:")
    print("    pip install numpy")
    sys.exit(1)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Part A/B need gymnasium. Install it with:")
    print("    pip install gymnasium")
    sys.exit(1)

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    print("Part B needs stable-baselines3. Install it with:")
    print("    pip install stable-baselines3")
    sys.exit(1)

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    print("Part C needs scikit-learn. Install it with:")
    print("    pip install scikit-learn")
    sys.exit(1)


# ── Part A: the HVAC environment ─────────────────────────────────────────────
class HotelHVACEnv(gym.Env):
    """One hotel room's HVAC, as a Gymnasium environment.

    Observation: Box([temp_c, occupancy, hour_of_day, energy_kwh]).
    Action:      Discrete(5) → setpoint delta in {-2, -1, 0, +1, +2} °C.
    The agent nudges the setpoint each step; the room temperature drifts toward
    the setpoint with noise, and the reward balances guest comfort vs. energy.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([40, 1, 23, 5], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)
        self.action_deltas = [-2, -1, 0, +1, +2]

        self.temp = 22.0
        self.setpoint = 22.0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = float(self.np_random.uniform(20, 30))
        self.setpoint = 22.0
        self.step_count = 0

        occupancy = float(self.np_random.binomial(1, 0.6))
        hour = float(self.np_random.integers(0, 24))
        state = np.array([self.temp, occupancy, hour, 1.5], dtype=np.float32)
        return state, {}

    def step(self, action):
        delta = self.action_deltas[int(action)]
        self.setpoint = float(np.clip(self.setpoint + delta, 16, 30))

        # Room temperature relaxes toward the setpoint, with sensor/plant noise.
        self.temp += (self.setpoint - self.temp) * 0.1 + float(self.np_random.normal(0, 0.2))

        hour = float(self.np_random.integers(0, 24))
        occupancy = float(self.np_random.binomial(1, 0.6))
        energy_kwh = max(0.0, (30 - self.setpoint) * 0.08)

        comfort_ok = (20 <= self.temp <= 26) and bool(occupancy)
        reward = (3.0 if comfort_ok else -1.0) - energy_kwh

        self.step_count += 1
        done = self.step_count >= 200

        state = np.array([self.temp, occupancy, hour, energy_kwh], dtype=np.float32)
        return state, reward, done, False, {}


# ── Part B: train a DQN policy ───────────────────────────────────────────────
def train_rl_agent(timesteps=50_000):
    """Train a DQN HVAC controller and save it to /tmp/hotel_hvac_dqn."""
    env = make_vec_env(HotelHVACEnv, n_envs=4)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10_000,
        batch_size=64,
        gamma=0.90,
        exploration_fraction=0.2,
        verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    model.save("/tmp/hotel_hvac_dqn")
    print(f"✓ DQN trained for {timesteps:,} steps → /tmp/hotel_hvac_dqn.zip")
    return model


# ── Part C: anomaly detection over real sensor readings ──────────────────────
class SensorAnomalyDetector:
    """Isolation Forest over [temp_c, humidity, kwh, occupancy] sensor vectors.

    Key Insight: Isolation Forest isolates points with random splits. Anomalies
    sit far from the bulk of the data, so they get *isolated in fewer splits* —
    i.e. they have a SHORTER average path length in the random trees. The
    detector flags those short-path points as anomalous; it never needs labels.
    """

    def __init__(self, contamination=0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.fitted = False

    def fit_from_neo4j(self):
        """Fit on the most recent SensorReadings; synthesize data if too few."""
        rows = []
        if check_neo4j():
            try:
                driver = get_driver()
                with driver.session() as session:
                    result = session.run(
                        """
                        MATCH (sr:SensorReading)
                        RETURN sr.temp_c   AS temp_c,
                               sr.humidity  AS humidity,
                               sr.kwh       AS kwh,
                               CASE WHEN sr.occupancy THEN 1 ELSE 0 END AS occupancy
                        ORDER BY sr.ts DESC
                        LIMIT 5000
                        """
                    )
                    rows = [dict(r) for r in result]
                driver.close()
            except Exception as e:  # noqa: BLE001
                print(f"   (Neo4j query failed: {type(e).__name__}: {e}; synthesizing)")
                rows = []

        if len(rows) < 10:
            print("   Fewer than 10 readings — synthesizing 200 normal rows.")
            rng = np.random.default_rng(42)
            X = np.column_stack([
                rng.normal(22, 1.5, 200),          # temp_c
                rng.normal(55, 5, 200),            # humidity
                rng.normal(2.0, 0.4, 200),         # kwh
                rng.binomial(1, 0.6, 200),         # occupancy
            ])
        else:
            X = np.array(
                [[r["temp_c"], r["humidity"], r["kwh"], r["occupancy"]] for r in rows],
                dtype=float,
            )
            print(f"   Fit on {len(rows)} real SensorReading rows.")

        self.model.fit(X)
        self.fitted = True
        return self

    def predict(self, reading: dict) -> dict:
        if not self.fitted:
            self.fit_from_neo4j()

        occ = reading.get("occupancy", 0)
        occ = 1 if occ in (True, 1) else 0
        X = np.array([[
            reading.get("temp_c", 22),
            reading.get("humidity", 55),
            reading.get("kwh", 2),
            occ,
        ]], dtype=float)

        pred = int(self.model.predict(X)[0])
        score = float(self.model.decision_function(X)[0])
        return {
            "anomaly": pred == -1,
            "anomaly_score": round(score, 4),
            "interpretation": "ANOMALOUS" if pred == -1 else "NORMAL",
        }


if __name__ == "__main__":
    # ── Part C: anomaly detection ───────────────────────────────────────────
    print("── Part C: Sensor Anomaly Detection (Isolation Forest) ──")
    detector = SensorAnomalyDetector()
    detector.fit_from_neo4j()

    tests = [
        {"temp_c": 22, "humidity": 55, "kwh": 2.0, "occupancy": True},
        {"temp_c": 38.5, "humidity": 95, "kwh": 5.8, "occupancy": False},
    ]
    for r in tests:
        out = detector.predict(r)
        print(f"  {r} → {out['interpretation']} (score={out['anomaly_score']})")

    # ── Part B: train the DQN policy (short run for the workshop) ────────────
    print("\n── Part B: Training DQN HVAC policy (10k steps) ──")
    model = train_rl_agent(timesteps=10_000)

    # ── Key Insight ─────────────────────────────────────────────────────────
    # The DQN learns a setpoint policy from the comfort-vs-energy reward instead
    # of fixed thresholds; the Isolation Forest learns the *shape* of normal
    # readings and flags short-path outliers — both replace hand-tuned rules
    # with models that adapt to the hotel's actual operating data.
