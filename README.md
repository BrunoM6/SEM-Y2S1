# StrEdge – proof of concept (simulated)

ODAC (Rodrigues, Gama & Pedroso, *Hierarchical Clustering of Time-Series Data Streams*, IEEE TKDE 2008), as implemented in River's `cluster.ODAC`, run on telemetry from a **simulated** 6-axis pick-and-place robot and compared with a static-threshold rule.

```bash
pip install river numpy
python run_poc.py          # ~40 s; writes results/summary.json and results/trace_<scenario>.json
```

## What is simulated (`robot_sim.py`)

- 6 joints × {motor current, vibration RMS, temperature} = 18 streams, 1 Hz feature windows, 12 h per run.
- Shared drivers that make healthy sensors move together: a 12 s pick-and-place cycle (joints move in fixed segments), busy/idle production blocks with a speed override, and the payload of each cycle.
- Temperatures follow a first-order thermal model (time constant 10 min) driven by copper and friction losses.
- Scenarios (event at 6 h, growing over 60 min):

| scenario | what changes |
|---|---|
| `normal` | nothing (winter, 22 °C ambient) |
| `summer` | ambient rises by +10 °C (heat wave / HVAC failure); robot is healthy |
| `bearing` | J3 bearing wear: impacts with random amplitude while J3 moves (up to +1.6 mm/s, under the 4.5 mm/s limit) plus a little friction heat |
| `gain` | J3 lubrication loss: extra friction proportional to speed, a pure gain change |

Static rule: alarm if any temperature > 70 °C or any vibration RMS > 4.5 mm/s.

This is a toy model meant to show the mechanism, not a validated robot model.

## Edge pipeline (`run_poc.py`)

- **Two ODAC trees, one per time scale.** Fast tree: 12 current and vibration streams at 1 Hz, tested every 2 min. Slow tree: 6 temperatures as 30 s means, tested every 30 min. With one tree at 1 Hz, the temperature subtree kept splitting and merging every few minutes, because a freshly split child judged slow signals on only 2 minutes of data.
- **Scaling.** Each stream is standardized with the first 10 min (commissioning) and multiplied by 0.1. Correlation does not care about scale, but River uses a one-sensor leaf's variance as its "diameter", and that has to be comparable to rnomc distances.
- **Hoeffding bound.** River computes `ln(1/confidence_level)`, so `confidence_level=0.05` gives the paper's 95 % bound. The default of 0.9 gives a much tighter bound.
- **Alert rule.** A merge after warm-up: a split that used to hold is no longer supported by the data. The split that follows names the sensor that moved.

## Two changes to River's ODAC (`odac_patch.py`)

1. **Average distance.** The split heuristic uses the mean pairwise distance, as in the paper. River 0.26.1 divides the sum of distances by the number of observations. With that version, healthy runs had 9 false merges in 120 h; with the fix they had 2.
2. **10-minute grace period.** A new leaf can only trigger a merge after it has seen 600 observations. This removed the last 2 false merges. All 10 bearing runs were still detected, one of them 16 min later.

## Results (10 runs × 12 h per scenario)

| scenario | static rule fired | ODAC alerted | ODAC delay after onset | sensor isolated afterwards |
|---|---|---|---|---|
| healthy | 0/10 | 0/10 | – | – |
| ambient +10 °C | 10/10 (false alarm) | 0/10 | – | – |
| early bearing wear, J3 | 0/10 (missed) | 10/10 | median 81 min (64–100) | V3 (J3 vibration), 10/10 |
| lubrication loss, J3 | 0/10 | 0/10 | – | – |

- **Learned healthy structure (fast tree):** {I1, V1} base · {I2, I3, V2, V3} shoulder + elbow · {I4, I6, V4, V6} wrist roll + flange · {I5, V5} wrist pitch.
- **Speed:** about 20 µs per sample for the fast tree in plain Python.

## Known limits (shown by this PoC)

- **Correlation ignores offset and gain.** A fault that only scales a signal (`gain`) is invisible to ODAC. It needs a per-sensor level or residual test next to it, for example Page-Hinkley on each sensor's deviation from its cluster.
- **Statistics never forget.** ODAC's sufficient statistics accumulate from the moment a cluster is created. After weeks of healthy data, a new fault moves the correlation slowly. A product needs periodic re-baselining or fading factors.
- **The Hoeffding guarantee is approximate here.** It assumes i.i.d. samples, and sensor streams are autocorrelated.
- **Simulated data only.** Validating on real robot data is the next step.
