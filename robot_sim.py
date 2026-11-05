"""
Simulated telemetry for a 6-axis pick-and-place robot, as 1 Hz feature streams.

Each joint j produces three streams (one value per 1-second window):
  I{j}  mean motor current      [A]
  V{j}  vibration RMS           [mm/s]
  T{j}  motor temperature       [degC]

Shared drivers (what makes healthy sensors co-vary):
  - the 12 s pick-and-place cycle (joints move in fixed segments)
  - production schedule: busy blocks with a speed override, idle gaps
  - payload per cycle (affects shoulder/elbow currents)

Scenarios mirror the deck's slide-6 table:
  normal  : winter ambient (22 degC), healthy
  summer  : healthy, ambient ramps +10 degC from FAULT_START over FAULT_RAMP
  bearing : winter, J3 bearing wear from FAULT_START (impulsive vibration whose amplitude is
            not explained by the load, plus some friction heat), ramping in over FAULT_RAMP
  gain    : winter, J3 lubrication loss = friction that grows in proportion to speed
            (a pure gain change). Included to show ODAC's blind spot.

This is a toy physical model for a proof of concept, not a validated robot model.
"""
from __future__ import annotations

import numpy as np

CYCLE = 12                 # seconds per pick-and-place cycle
DURATION = 12 * 3600       # 12 hours per scenario at 1 Hz
BURN_IN = 3600             # simulated but discarded, so temperatures start warmed up
FAULT_START = 6 * 3600     # fault / ambient shift onset (s after the start of the kept data)
FAULT_RAMP = 60 * 60       # grows to full size over 60 min

T_LIMIT = 70.0             # static rule: temperature limit [degC]
BEARING_AMP = 1.6          # peak extra vibration from bearing impacts at full severity [mm/s]
                           # (early-stage wear: stays under V_LIMIT)
V_LIMIT = 4.5              # static rule: vibration RMS limit [mm/s]

# Joint motion segments inside one cycle: (start_s, end_s, relative_speed)
SEGMENTS = {
    1: [(0, 3, 1.0), (6, 9, 1.0)],                 # base: swing to pick, swing to place
    2: [(0, 3, 0.8), (4, 6, 1.0), (10, 12, 0.4)],  # shoulder: reach down, lift, retract
    3: [(0, 3, 0.9), (4, 6, 1.0), (10, 12, 0.5)],  # elbow: moves with the shoulder
    4: [(6, 9, 0.7)],                              # wrist roll: re-orient during place swing
    5: [(0, 3, 0.3), (10, 12, 0.6)],               # wrist pitch: approach / retract
    6: [(6, 9, 0.9)],                              # flange: rotate part during place swing
}
GRAVITY = {1: 0.3, 2: 4.0, 3: 2.5, 4: 0.2, 5: 0.6, 6: 0.1}       # holding current [A]
K_SPEED = {1: 3.0, 2: 3.5, 3: 3.0, 4: 1.2, 5: 1.0, 6: 0.8}       # current per unit speed
K_PAYLOAD = {1: 0.2, 2: 0.6, 3: 0.4, 4: 0.05, 5: 0.1, 6: 0.05}   # current per kg
V_BASE = {1: 0.4, 2: 0.5, 3: 0.5, 4: 0.3, 5: 0.3, 6: 0.3}        # vibration floor [mm/s]
V_SPEED = {1: 1.6, 2: 1.8, 3: 1.7, 4: 1.0, 5: 0.9, 6: 0.8}       # vibration per unit speed
DT_FULL = {1: 28.0, 2: 46.0, 3: 41.0, 4: 18.0, 5: 18.0, 6: 15.0} # rise at full load [degC]
TAU_THERM = 600.0                                                 # thermal time constant [s]


def _schedule(rng: np.random.Generator, n: int) -> np.ndarray:
    """Busy blocks (6-16 min) with a speed override, separated by idle gaps (1-4 min)."""
    speed = np.zeros(n)
    t = 0
    while t < n:
        busy = int(rng.integers(6 * 60, 16 * 60))
        speed[t : t + busy] = float(rng.choice([0.6, 0.8, 1.0]))
        t += busy + int(rng.integers(60, 4 * 60))
    return speed


def _joint_speed(j: int, phase: np.ndarray, override: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    prof = np.zeros(phase.shape)
    for a, b, s in SEGMENTS[j]:
        inside = (phase >= a) & (phase < b)
        u = (phase + 0.5 - a) / (b - a)   # 1 s window centre, position inside the segment
        prof = np.where(inside, s * (np.sin(np.pi * u) ** 2 + 0.25), prof)
    return prof * override * jitter


def simulate(scenario: str, seed: int = 7, duration: int = DURATION, bearing_amp: float | None = None):
    assert scenario in ("normal", "summer", "bearing", "gain")
    rng = np.random.default_rng(seed)
    n = duration + BURN_IN
    t = np.arange(n) - BURN_IN          # negative during burn-in
    override = _schedule(rng, n)
    phase = np.arange(n) % CYCLE
    n_cycles = n // CYCLE + 1
    payload = np.repeat(rng.uniform(2.0, 5.0, size=n_cycles), CYCLE)[:n]

    t_amb = 22.0 + 0.5 * np.sin(2 * np.pi * t / (6 * 3600))
    if scenario == "summer":
        # ambient rises by 10 degC while the robot keeps running (hot afternoon / HVAC failure)
        t_amb = t_amb + 10.0 * np.clip((t - FAULT_START) / FAULT_RAMP, 0, 1)

    severity = np.zeros(n)
    if scenario in ("bearing", "gain"):
        severity = np.clip((t - FAULT_START) / FAULT_RAMP, 0, 1)

    streams: dict[str, np.ndarray] = {}
    for j in range(1, 7):
        jitter = np.repeat(rng.normal(1.0, 0.05, size=n_cycles), CYCLE)[:n]   # cycle-to-cycle variation
        speed = _joint_speed(j, phase, override, jitter)
        moving = speed > 0

        current = (
            GRAVITY[j] * (1 + 0.15 * (payload - 3.5) / 3.5)
            + K_SPEED[j] * speed
            + K_PAYLOAD[j] * payload * moving
            + rng.normal(0, 0.15, n)
        )
        vib = V_BASE[j] + V_SPEED[j] * speed + rng.normal(0, 0.08, n)
        heat = 0.35 * current**2 + 0.6 * speed          # copper losses + friction (arbitrary units)
        heat_ref = heat[override == 1.0].mean()

        if j == 3 and scenario == "bearing":
            amp = BEARING_AMP if bearing_amp is None else bearing_amp
            # Spalled bearing: impacts while rotating, with random amplitude -> not explained by load
            impacts = rng.uniform(0.0, 1.0, n) ** 2 * moving
            vib = vib + severity * amp * impacts
            heat = heat + severity * 0.8 * moving
        if j == 3 and scenario == "gain":
            # Lubrication loss: friction torque grows in proportion to speed -> a pure gain change
            current = current + severity * 1.5 * speed
            heat = heat + severity * 1.5 * speed

        target = t_amb + DT_FULL[j] * heat / heat_ref
        temp = np.empty(n)
        temp[0] = target[: 10 * 60].mean()
        a = 1.0 / TAU_THERM
        for k in range(1, n):
            temp[k] = temp[k - 1] + a * (target[k] - temp[k - 1])
        temp = np.round(temp + rng.normal(0, 0.15, n), 1)

        streams[f"I{j}"] = current[BURN_IN:]
        streams[f"V{j}"] = vib[BURN_IN:]
        streams[f"T{j}"] = temp[BURN_IN:]

    meta = {
        "override": override[BURN_IN:],
        "severity": severity[BURN_IN:],
    }
    return streams, meta


def static_rule(streams: dict[str, np.ndarray]) -> np.ndarray:
    """Per-second alarm flag of the fixed-threshold baseline (any T > T_LIMIT or V > V_LIMIT)."""
    n = len(next(iter(streams.values())))
    alarm = np.zeros(n, bool)
    for k, v in streams.items():
        if k.startswith("T"):
            alarm |= v > T_LIMIT
        if k.startswith("V"):
            alarm |= v > V_LIMIT
    return alarm
