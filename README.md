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

### Model equations

Samples are 1 s apart. $k$ is the sample index and $t = k - 3600$ is the time in seconds, because the first hour is burn-in and is discarded.

Drivers shared by all joints:

- Cycle phase $\phi = k \bmod 12$.
- Speed override $u$: 0.6, 0.8 or 1.0 in busy blocks of 6–16 min, 0 in idle gaps of 1–4 min.
- Payload $m \sim \mathcal{U}(2, 5)$ kg, one value per cycle, which gives the factor $p = 1 + 0.15\cdot(m - 3.5)/3.5$.
- Ambient temperature $T_{\mathrm{amb}}$ and event ramp $r$, which is 0 until 6 h and 1 from 7 h:

$$T_{\mathrm{amb}}(t) = 22 + 0.5\cdot\sin\left(\frac{2\pi t}{21600}\right)$$

$$r(t) = \mathrm{clip}\left(\frac{t - 21600}{3600}, 0, 1\right)$$

Each joint $j$ moves in the segments $(a, b, A)$ listed in the table below, and $P_j$ is 0 outside them:

$$v_j = u\cdot\xi_j\cdot P_j(\phi), \qquad \xi_j \sim \mathcal{N}(1, 0.05^2) \text{ per cycle}$$

$$P_j(\phi) = A\cdot\left[\sin^2\left(\frac{\pi(\phi + 0.5 - a)}{b - a}\right) + 0.25\right] \quad \text{for } a \le \phi < b$$

$M_j = 1$ while the joint moves ($v_j > 0$), otherwise 0.

| Joint | Segments $(a, b, A)$ | Current $I_j$ [A] | Vibration $V_j$ [mm/s] | Target temperature $T^{\star}_j$ [°C] |
|---|---|---|---|---|
| J1 base | (0, 3, 1.0), (6, 9, 1.0) | $0.3\cdot p + 3.0\cdot v_1 + 0.20\cdot m\cdot M_1 + \varepsilon_I$ | $0.4 + 1.6\cdot v_1 + \varepsilon_V$ | $T_{\mathrm{amb}} + 28\cdot H_1/H_{\mathrm{ref},1}$ |
| J2 shoulder | (0, 3, 0.8), (4, 6, 1.0), (10, 12, 0.4) | $4.0\cdot p + 3.5\cdot v_2 + 0.60\cdot m\cdot M_2 + \varepsilon_I$ | $0.5 + 1.8\cdot v_2 + \varepsilon_V$ | $T_{\mathrm{amb}} + 46\cdot H_2/H_{\mathrm{ref},2}$ |
| J3 elbow | (0, 3, 0.9), (4, 6, 1.0), (10, 12, 0.5) | $2.5\cdot p + 3.0\cdot v_3 + 0.40\cdot m\cdot M_3 + \varepsilon_I$ | $0.5 + 1.7\cdot v_3 + \varepsilon_V$ | $T_{\mathrm{amb}} + 41\cdot H_3/H_{\mathrm{ref},3}$ |
| J4 wrist roll | (6, 9, 0.7) | $0.2\cdot p + 1.2\cdot v_4 + 0.05\cdot m\cdot M_4 + \varepsilon_I$ | $0.3 + 1.0\cdot v_4 + \varepsilon_V$ | $T_{\mathrm{amb}} + 18\cdot H_4/H_{\mathrm{ref},4}$ |
| J5 wrist pitch | (0, 3, 0.3), (10, 12, 0.6) | $0.6\cdot p + 1.0\cdot v_5 + 0.10\cdot m\cdot M_5 + \varepsilon_I$ | $0.3 + 0.9\cdot v_5 + \varepsilon_V$ | $T_{\mathrm{amb}} + 18\cdot H_5/H_{\mathrm{ref},5}$ |
| J6 flange | (6, 9, 0.9) | $0.1\cdot p + 0.8\cdot v_6 + 0.05\cdot m\cdot M_6 + \varepsilon_I$ | $0.3 + 0.8\cdot v_6 + \varepsilon_V$ | $T_{\mathrm{amb}} + 15\cdot H_6/H_{\mathrm{ref},6}$ |

The noise terms $\varepsilon_I \sim \mathcal{N}(0, 0.15^2)$ and $\varepsilon_V \sim \mathcal{N}(0, 0.08^2)$ are drawn every second.

Heat $H_j$ drives a first-order thermal model with a 600 s time constant, and the measured temperature is rounded to 0.1 °C:

$$H_j = 0.35\cdot I_j^2 + 0.6\cdot v_j$$

$$T_j[k] = T_j[k-1] + \frac{T^{\star}_j[k] - T_j[k-1]}{600}, \qquad T_j[0] = \frac{1}{600}\sum_{i=0}^{599} T^{\star}_j[i]$$

$$T^{\mathrm{meas}}_j = \mathrm{round}(T_j + \varepsilon_T, 1), \qquad \varepsilon_T \sim \mathcal{N}(0, 0.15^2)$$

$H_{\mathrm{ref},j}$ is the mean of $H_j$ over the seconds with $u = 1$, computed for each run.

What each scenario adds, with $w \sim \mathcal{U}(0, 1)$ drawn every second:

| scenario | added terms |
|---|---|
| `normal` | none |
| `summer` | $T_{\mathrm{amb}} \leftarrow T_{\mathrm{amb}} + 10\cdot r(t)$ |
| `bearing` | $V_3 \leftarrow V_3 + 1.6\cdot r(t)\cdot w^2\cdot M_3$ and $H_3 \leftarrow H_3 + 0.8\cdot r(t)\cdot M_3$ |
| `gain` | $I_3 \leftarrow I_3 + 1.5\cdot r(t)\cdot v_3$ and $H_3 \leftarrow H_3 + 1.5\cdot r(t)\cdot v_3$ |

The fault terms are added after $H_j$ and $H_{\mathrm{ref},j}$ are computed, so $H_{\mathrm{ref},3}$ does not change and the extra current in `gain` is not squared into the heat.

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
