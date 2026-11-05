"""
StrEdge proof of concept: River's ODAC on simulated robot telemetry vs. a static-threshold rule.

Edge pipeline (one robot, 18 sensors):
  fast tree : currents + vibration RMS (12 streams) at 1 Hz        -> ODAC, test every 2 min
  slow tree : temperatures (6 streams), 30 s means (1/30 Hz)       -> ODAC, test every 30 min
  Each stream is scaled with commissioning statistics (first 10 min). River's ODAC uses the
  series variance as the "diameter" of a one-sensor leaf, so scaling keeps it comparable to
  rnomc distances (correlation itself is scale-invariant).

Alert rule: a MERGE after warm-up (an earlier split is no longer supported by the data, i.e. the
correlation structure drifted). The re-split that follows localizes the sensor(s) that moved.

Usage: python run_poc.py   -> results/summary.json, results/trace_<scenario>.json
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
from river import cluster

import odac_patch
from robot_sim import FAULT_START, simulate, static_rule

# Two small changes to river.cluster.ODAC (see odac_patch.py):
#  - split heuristic uses the mean pairwise distance, as in Rodrigues et al. (2008)
#  - a new leaf must see GRACE observations before it can trigger a merge
GRACE = 600
odac_patch.apply()
odac_patch.apply_grace(GRACE)

CALIB = 10 * 60
CONF = 0.05          # River computes ln(1/confidence_level): 0.05 gives the paper's 95 % bound
TAU = 0.1
SCALE = 0.1
FAST = dict(rate_s=1, n_min=120, warmup_s=60 * 60)
SLOW = dict(rate_s=30, n_min=60, warmup_s=3 * 3600)
SCENARIOS = ["normal", "summer", "bearing", "gain"]
SEEDS = list(range(1, 11))
OUT = Path(__file__).parent / "results"


def leaves(model):
    return sorted(sorted(map(str, c.timeseries_names)) for c in model._find_all_active_clusters(model._root_node))


def mates(part):
    return {s: frozenset(g) - {s} for g in part for s in g}


def run_tree(Z: dict[str, np.ndarray], cfg: dict):
    """Feed one ODAC tree; return list of structure events with type, time and moved streams."""
    names = list(Z)
    n = len(Z[names[0]])
    model = cluster.ODAC(confidence_level=CONF, n_min=cfg["n_min"], tau=TAU)
    events, prev = [], None
    snap = None
    for k in range(n):
        model.learn_one({s: float(Z[s][k]) for s in names})
        if snap is None and k * cfg["rate_s"] >= cfg["warmup_s"]:
            snap = leaves(model)
            run_tree.warmup_ascii = model.render_ascii()
        if model.structure_changed:
            cur = leaves(model)
            t = k * cfg["rate_s"]
            if prev is not None:
                kind = "merge" if len(cur) < len(prev) else "split"
                mb, ma = mates(prev), mates(cur)
                moved = sorted(s for s in mb if mb[s] != ma.get(s))
                new_groups = [g for g in cur if g not in prev]
                isolated = sorted(g[0] for g in new_groups if len(g) == 1)
                events.append({"t_min": round(t / 60, 1), "type": kind, "moved": moved, "isolated": isolated,
                               "partition": cur, "after_warmup": t >= cfg["warmup_s"], "n_leaves": len(cur)})
            prev = cur
    run_tree.warmup_partition = snap
    return events, leaves(model), model.render_ascii()


def prepare(streams: dict[str, np.ndarray]):
    fast, slow = {}, {}
    for k, v in streams.items():
        if k[0] in "IV":
            x = v
            mu, sd = x[:CALIB].mean(), x[:CALIB].std() + 1e-9
            fast[k] = SCALE * (x - mu) / sd
        else:
            r = SLOW["rate_s"]
            x = v[: len(v) // r * r].reshape(-1, r).mean(1)
            c = CALIB // r
            mu, sd = x[:c].mean(), x[:c].std() + 1e-9
            slow[k] = SCALE * (x - mu) / sd
    return fast, slow


def evaluate(scenario: str, seed: int):
    streams, meta = simulate(scenario, seed=seed)
    fast, slow = prepare(streams)
    t0 = time.perf_counter()
    ev_f, part_f, ascii_f = run_tree(fast, FAST)
    us_fast = 1e6 * (time.perf_counter() - t0) / len(next(iter(fast.values())))
    warm_part_f, warm_ascii_f = run_tree.warmup_partition, run_tree.warmup_ascii
    t0 = time.perf_counter()
    ev_s, part_s, ascii_s = run_tree(slow, SLOW)
    us_slow = 1e6 * (time.perf_counter() - t0) / len(next(iter(slow.values())))
    warm_part_s = run_tree.warmup_partition
    alerts = sorted([dict(e, tree="fast") for e in ev_f if e["after_warmup"] and e["type"] == "merge"]
                    + [dict(e, tree="slow") for e in ev_s if e["after_warmup"] and e["type"] == "merge"],
                    key=lambda e: e["t_min"])
    st = static_rule(streams)
    onset_min = FAULT_START / 60
    first_static = round(int(np.argmax(st)) / 60, 1) if st.any() else None
    # localization: after the alert (merge), the next split(s) show which sensor left its group
    loc = None
    if alerts:
        a = alerts[0]
        pool = ev_f if a["tree"] == "fast" else ev_s
        after = [e for e in pool if e["type"] == "split" and e["t_min"] > a["t_min"]]
        if after:
            loc = {"t_min": after[0]["t_min"], "isolated": after[0].get("isolated"), "moved": after[0]["moved"]}
    return {
        "scenario": scenario, "seed": seed, "onset_min": onset_min,
        "alerts": alerts, "first_alert_min": alerts[0]["t_min"] if alerts else None,
        "localized": loc,
        "static_first_min": first_static, "static_alarm_s": int(st.sum()),
        "fast_events": ev_f, "slow_events": ev_s,
        "fast_healthy": warm_part_f, "fast_healthy_ascii": warm_ascii_f, "slow_healthy": warm_part_s,
        "us_per_sample_fast": round(us_fast, 1), "us_per_sample_slow": round(us_slow, 1),
        "fast_final": part_f, "slow_final": part_s, "fast_ascii": ascii_f, "slow_ascii": ascii_s,
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rows = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            r = evaluate(sc, seed)
            rows.append({k: r[k] for k in ("scenario", "seed", "onset_min", "first_alert_min", "localized",
                                           "static_first_min", "static_alarm_s", "us_per_sample_fast",
                                           "fast_healthy", "slow_healthy")}
                        | {"n_alerts": len(r["alerts"]),
                           "alerts_before_onset": sum(a["t_min"] < r["onset_min"] for a in r["alerts"])})
            if seed == SEEDS[0]:
                (OUT / f"trace_{sc}.json").write_text(json.dumps(r, indent=2))
            print(sc, seed, "alerts:", [(a["t_min"], a["tree"], a["moved"]) for a in r["alerts"]][:4],
                  "| loc:", r["localized"], "| static first:", r["static_first_min"])
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))

    print("\nScenario      | static rule fired | ODAC alerted | ODAC delay after onset (min) | isolated sensor")
    for sc in SCENARIOS:
        R = [r for r in rows if r["scenario"] == sc]
        delays = [r["first_alert_min"] - r["onset_min"] for r in R
                  if r["first_alert_min"] is not None and r["first_alert_min"] >= r["onset_min"]]
        iso = sorted({tuple((r["localized"] or {}).get("isolated") or []) for r in R if r["localized"]})
        print(f"{sc:13s} | {sum(r['static_first_min'] is not None for r in R):2d}/{len(R)}             "
              f"| {sum(r['n_alerts'] > 0 for r in R):2d}/{len(R)}        "
              f"| {('median %.0f (range %d-%d)' % (statistics.median(delays), min(delays), max(delays))) if delays else '-':28s} "
              f"| {iso if iso else '-'}")
