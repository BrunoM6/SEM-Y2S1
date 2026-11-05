"""
Exploratory analysis of the StrEdge PoC: runs the simulation through the same pipeline as
run_poc.py and shows what happens inside it.

Written to results/eda/<scenario>_seed<seed>/ for each run:
  report.txt             (also printed) run values, stream statistics, results, alerts with the
                         test values that fired them, ODAC trees with their values, top pairs,
                         feasibility figures (state size, speed, data sent, samples per decision)
  1_streams.png          the 18 sensor streams and the speed override
  2_trees.png            ODAC trees at the end of warm-up, just before each later event, at the end
  3_odac_tests.png       leaf diameters and how close each leaf is to a merge or a split
  4_rnomc_all_pairs.png  rnomc over time for all 153 pairs of streams
  5_rnomc_top_pairs.png  the same, larger, for the pairs that changed most (plus --pairs)
And results/eda/compare_seed<seed>.png: each scenario against the healthy run, same seed.

rnomc(a, b) = sqrt((1 - corr(a, b)) / 2) is the distance ODAC uses. Two versions are plotted:
  ODAC's own  what the tree sees: only pairs in the same leaf, cumulative since that leaf was
              created, restarted by every split or merge.
  rolling     the same distance on a sliding window, for every pair. An analysis aid, not part of
              the pipeline. Current/vibration pairs use the 1 Hz streams; temperature pairs use
              30 s means, like the slow tree; current/vibration vs temperature pairs, which no
              tree tracks, use 60 s means (5 whole cycles).

Usage:
  python eda.py                                  all scenarios, seed 1
  python eda.py --scenario bearing --seed 3      one run
  python eda.py --seed 1-10 --no-plots           results table over several seeds
  python eda.py --pairs V3-I3,T3-T2 --window 60  always plot these pairs, change the window
  python eda.py --show                           also open the figures in windows
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import statistics
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from river import cluster
from river.cluster import odac as river_odac

import run_poc  # importing it applies the odac_patch changes
from robot_sim import BURN_IN, DURATION, FAULT_RAMP, FAULT_START, T_LIMIT, V_LIMIT, simulate

# Assumptions behind the feasibility figures, the same ones the Feasibility slide uses
BYTES_PER_VALUE = 4          # 30 channels x 100 Hz x 4 B = the slide's 1.04 GB/day
HEARTBEAT_B = 1000           # a 1 kB tree heartbeat...
HEARTBEAT_EVERY_S = 600      # ...every 10 min
SNAPSHOT_S = 600             # 10 min of data sent per alert
STATE_BYTES_PER_VALUE = 8    # 495 values x 8 B = the slide's 4 kB for 30 channels

OUT = run_poc.OUT / "eda"
TREES = {"fast": run_poc.FAST, "slow": run_poc.SLOW}
ONSET = FAULT_START / 60
RAMP_END = (FAULT_START + FAULT_RAMP) / 60
END = DURATION / 60
ROLL_STEP_S = 120
UNITS = {"I": "A", "V": "mm/s", "T": "°C"}
ABOUT = {
    "normal": "healthy, 22 °C ambient, no event",
    "summer": "healthy, ambient rises 10 °C from 360 min",
    "bearing": "J3 bearing wear from 360 min",
    "gain": "J3 lubrication loss from 360 min (pure gain change)",
}

SURFACE, WASH = "#fcfcfb", "#f0efec"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = SERIES[:2]
LEAF_FILL = "#e6f0fc"
CRITICAL, SERIOUS = "#d03b3b", "#ec835a"
DOTTED = (0, (1, 2))


# ---------------------------------------------------------------- tracing the trees

def freeze(node) -> dict:
    """Plain-data copy of an ODAC node and everything below it."""
    sensors = sorted(map(str, node.timeseries_names))
    kids = [node.children.first, node.children.second] if node.children else []
    has_pairs = node.active and len(sensors) > 1 and node.n > 1
    return {
        "name": node.name, "sensors": sensors, "active": node.active, "n": node.n,
        "d0": node.d0, "d1": node.d1, "d2": node.d2, "avg": node.avg, "e": node.e,
        "pivot": getattr(node, "pivot_1", None),
        "pairs": node._calculate_rnomc_dict() if has_pairs else {},
        "children": [freeze(k) for k in kids],
    }


def deep_size(obj, seen: set | None = None) -> int:
    """Bytes of Python memory held by an object and everything it references."""
    seen = set() if seen is None else seen
    if id(obj) in seen or isinstance(obj, (type, types.ModuleType, types.FunctionType)):
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_size(x, seen) for x in obj)
    if hasattr(obj, "__dict__"):
        size += deep_size(vars(obj), seen)
    return size


def test_values(leaf) -> dict:
    """Everything a leaf's merge and split tests look at, read just before they run."""
    d0, d1, d2, avg, e = leaf.d0, leaf.d1, leaf.d2, leaf.avg, leaf.e
    rec = {
        "sensors": tuple(map(str, leaf.timeseries_names)), "n": leaf.n,
        "d0": d0, "d1": d1, "d2": d2, "avg": avg, "e": e, "pivot": getattr(leaf, "pivot_1", None),
        "pairs": leaf._calculate_rnomc_dict() if len(leaf.timeseries_names) > 1 else {},
        "merge_margin": None, "split_margin": None, "fired": None,
    }
    if leaf.parent is not None:
        p = leaf.parent
        sibling = p.children.second if p.children.first is leaf else p.children.first
        rec |= {"parent_d1": p.d1, "parent_e": p.e, "merge_bound": max(p.e, e),
                "parent_sensors": tuple(map(str, p.timeseries_names)),
                "sibling_sensors": tuple(map(str, sibling.timeseries_names)),
                "merge_margin": d1 - p.d1 - max(p.e, e)}
    if d2 is not None:
        het = (d1 - d0) * abs(d1 + d0 - 2 * avg)
        rec |= {"split_gap": d1 - d2, "split_het": het,
                "split_margin": min(max(d1 - d2, run_poc.TAU), het) - e}
    return rec


@contextlib.contextmanager
def tap_leaf_tests(before, fired):
    """Call back around River's leaf tests. The callbacks only read, so the tree is unchanged."""
    cls = river_odac.ODACCluster
    test_aggregate, test_split = cls.test_aggregate, cls.test_split

    def tapped_aggregate(self):
        before(self)
        merged = test_aggregate(self)
        if merged:
            fired(self, "merge")
        return merged

    def tapped_split(self, tau):
        split = test_split(self, tau)
        if split:
            fired(self, "split")
        return split

    cls.test_aggregate, cls.test_split = tapped_aggregate, tapped_split
    try:
        yield
    finally:
        cls.test_aggregate, cls.test_split = test_aggregate, test_split


def trace_tree(Z: dict[str, np.ndarray], cfg: dict) -> dict:
    """run_poc.run_tree with the same inputs and settings, keeping every value along the way."""
    names = list(Z)
    n = len(Z[names[0]])
    rate = cfg["rate_s"]
    model = cluster.ODAC(confidence_level=run_poc.CONF, n_min=cfg["n_min"], tau=run_poc.TAU)
    out = {"tests": [], "fired": [], "changes": [], "warmup": None, "final": None}
    k, rec, tree_before = 0, None, None

    def before(leaf):
        nonlocal rec, tree_before
        top = leaf
        while top.parent is not None:
            top = top.parent
        if top is not model._root_node:
            rec = None  # its branch merged away earlier in this test step, but River still visits it
            return
        rec = test_values(leaf) | {"t_s": k * rate}
        tree_before = freeze(model._root_node)
        out["tests"].append(rec)

    def fired(leaf, kind):
        if rec is None:
            return
        rec["fired"] = kind
        into = [tuple(map(str, c.timeseries_names)) for c in (leaf.children.first, leaf.children.second)] \
            if kind == "split" else None
        out["fired"].append(rec | {"kind": kind, "tree_before": tree_before, "into": into})

    with tap_leaf_tests(before, fired):
        for k in range(n):
            model.learn_one({s: float(Z[s][k]) for s in names})
            if k == 0:
                out["memory_start"] = deep_size(model)  # the root, tracking every pair: the largest it gets
            if out["warmup"] is None and k * rate >= cfg["warmup_s"]:
                out["warmup"] = {"t_s": k * rate, "tree": freeze(model._root_node)}
            if model.structure_changed:
                out["changes"].append((round(k * rate / 60, 1), run_poc.leaves(model)))
    out["final"] = {"t_s": (n - 1) * rate, "tree": freeze(model._root_node)}
    out["memory_end"] = deep_size(model)
    return out


def odac_pair_series(tests: list[dict]) -> dict:
    """pair -> segments of ODAC's own rnomc. A new segment starts when the statistics restart."""
    series: dict = {}
    for rec in tests:
        for pair, r in rec["pairs"].items():
            segments = series.setdefault(tuple(map(str, pair)), [])
            if not segments or rec["n"] <= segments[-1]["n"]:
                segments.append({"t": [], "r": [], "n": 0})
            segments[-1]["t"].append(rec["t_s"] / 60)
            segments[-1]["r"].append(r)
            segments[-1]["n"] = rec["n"]
    return series


def check_trace(official: dict, trace: dict) -> list[str]:
    """Confirm the traced run is the run_poc.py run, and that the margins explain every firing."""
    problems = []
    for tree, tr in trace.items():
        events = official[f"{tree}_events"]
        if [(e["t_min"], e["partition"]) for e in events] != tr["changes"][1:]:  # [0]: root created
            problems.append(f"{tree} tree: traced events differ from run_poc.evaluate()")
        for r in tr["tests"]:
            merge = r["merge_margin"] is not None and r["merge_margin"] > 0 and r["n"] >= run_poc.GRACE
            split = not merge and r["split_margin"] is not None and r["split_margin"] > 0
            if r["fired"] != ("merge" if merge else "split" if split else None):
                problems.append(f"{tree} tree, t={r['t_s'] / 60:.1f} min: margins disagree with the test outcome")
        for e in events:
            if e["type"] == "split" and any(f["kind"] == "merge" and minutes(f["t_s"]) == e["t_min"]
                                            for f in tr["fired"]):
                problems.append(f"{tree} tree, t={e['t_min']} min: a merge and a split fired in the same test "
                                f"step; run_poc.py counts this as a split, so it raises no alert")
    return problems


# ---------------------------------------------------------------- rolling rnomc, all pairs

def rolling_rnomc(block: dict[str, np.ndarray], rate_s: int, window_min: int) -> dict:
    names = sorted(block)
    x = np.array([block[s] for s in names])
    width = window_min * 60 // rate_s
    ends = np.arange(width, x.shape[1] + 1, max(1, ROLL_STEP_S // rate_s))
    corr = np.array([np.corrcoef(x[:, e - width : e]) for e in ends])
    dist = np.sqrt(np.abs(1 - corr) / 2)
    t = ends * rate_s / 60
    return {(a, b): {"t": t, "r": dist[:, i, j], "window": window_min}
            for (i, a), (j, b) in itertools.combinations(enumerate(names), 2)}


def all_pairs_rolling(streams, fast, slow, window, slow_window) -> dict:
    series = rolling_rnomc(fast, run_poc.FAST["rate_s"], window)
    series |= rolling_rnomc(slow, run_poc.SLOW["rate_s"], slow_window)
    minute = {k: v[: len(v) // 60 * 60].reshape(-1, 60).mean(1) for k, v in streams.items()}
    for pair, s in rolling_rnomc(minute, 60, slow_window).items():
        if (pair[0][0] == "T") != (pair[1][0] == "T"):
            series[pair] = s
    return series


def pair_changes(rolling: dict) -> list[dict]:
    """Mean rolling rnomc before, during and after the event, most changed pair first.

    Pairs are ranked by the change divided by how much that pair wandered before the event.
    Temperature pairs wander slowly and widely, so a raw ranking would fill up with them.
    """
    rows = []
    for pair, s in rolling.items():
        t, r = s["t"], s["r"]
        settled = RAMP_END + s["window"]
        quiet = r[t <= ONSET]
        before, during, after = quiet.mean(), r[(t > ONSET) & (t < settled)].mean(), r[t >= settled].mean()
        change = max(during - before, after - before, key=abs)
        rows.append({"pair": pair, "before": before, "during": during, "after": after, "change": change,
                     "wander": quiet.std(), "score": abs(change) / quiet.std()})
    return sorted(rows, key=lambda row: -row["score"])


# ---------------------------------------------------------------- one run

@dataclass
class Run:
    scenario: str
    seed: int
    streams: dict
    meta: dict
    official: dict   # what run_poc.evaluate() returns, the source of events, alerts and results
    trace: dict      # tree -> trace_tree() output
    rolling: dict    # pair -> rolling rnomc
    own: dict        # pair -> segments of ODAC's own rnomc
    changes: list    # pair_changes()
    problems: list   # check_trace()

    @property
    def label(self) -> str:
        return f"{self.scenario} · seed {self.seed}"

    @property
    def alerts(self) -> list[dict]:
        return self.official["alerts"]

    def healthy_pairs(self) -> set:
        groups = self.official["fast_healthy"] + self.official["slow_healthy"]
        return {pair for g in groups for pair in itertools.combinations(sorted(g), 2)}


def analyse(scenario: str, seed: int, window: int, slow_window: int) -> Run:
    official = run_poc.evaluate(scenario, seed)
    streams, meta = simulate(scenario, seed=seed)
    fast, slow = run_poc.prepare(streams)
    trace = {"fast": trace_tree(fast, run_poc.FAST), "slow": trace_tree(slow, run_poc.SLOW)}
    rolling = all_pairs_rolling(streams, fast, slow, window, slow_window)
    own = odac_pair_series(trace["fast"]["tests"]) | odac_pair_series(trace["slow"]["tests"])
    return Run(scenario, seed, streams, meta, official, trace, rolling, own,
               pair_changes(rolling), check_trace(official, trace))


# ---------------------------------------------------------------- text report

def minutes(t_s: float) -> float:
    return round(t_s / 60, 1)


def num(x, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def group(sensors) -> str:
    return "[" + " ".join(sensors) + "]"


def alert_firing(run: Run, alert: dict) -> dict | None:
    """The leaf test that raised this alert."""
    return next((f for f in run.trace[alert["tree"]]["fired"]
                 if f["kind"] == "merge" and minutes(f["t_s"]) == alert["t_min"]), None)


def compact_state(sensor_groups: list) -> int:
    """ODAC's state as the Feasibility slide counts it: one sum per pair and two per sensor."""
    values = sum(len(g) * (len(g) - 1) // 2 + 2 * len(g) for g in sensor_groups)
    return values * STATE_BYTES_PER_VALUE


def stream_per_day(n_streams: int) -> int:
    return n_streams * BYTES_PER_VALUE * 86400


def sent_per_day(n_streams: int, alerts: int) -> float:
    """Bytes sent upstream per day: the tree heartbeats plus one snapshot per alert."""
    return 86400 / HEARTBEAT_EVERY_S * HEARTBEAT_B + alerts * n_streams * BYTES_PER_VALUE * SNAPSHOT_S


def bound_at(n: int) -> float:
    """The Hoeffding bound e after n samples, as River computes it."""
    return math.sqrt(math.log(1 / run_poc.CONF) / (2 * n))


def after_onset(run: Run, t_min: float) -> str:
    if run.scenario == "normal":
        return "no event in this scenario"
    return f"{t_min - ONSET:+.1f} min from onset"


def merge_line(f: dict) -> str:
    gap = f["d1"] - f["parent_d1"]
    return (f"leaf {group(f['sensors'])} n={f['n']:,}: d1 {num(f['d1'])} - parent d1 {num(f['parent_d1'])} = {num(gap)}"
            f" > max(e, parent e) = {num(f['merge_bound'])}")


def split_line(f: dict) -> str:
    pivot = "-".join(map(str, f["pivot"])) if f["pivot"] else "–"
    return (f"leaf {group(f['sensors'])} n={f['n']:,}: d1 {num(f['d1'])} on {pivot}, d1-d2 {num(f['split_gap'])},"
            f" (d1-d0)|d1+d0-2avg| {num(f['split_het'])}, e {num(f['e'])}, tau {run_poc.TAU}")


def tree_lines(node: dict, prefix: str = "", branch: str = "") -> list[str]:
    head = f"{prefix}{branch}{node['name']}"
    if not node["active"]:
        values = f"frozen  n={node['n']:<6,} d1={num(node['d1'])} d2={num(node['d2'])} e={num(node['e'])}"
    elif node["d1"] is None:
        values = f"leaf    n={node['n']:<6,} not tested yet  {group(node['sensors'])}"
    elif len(node["sensors"]) == 1:
        values = f"leaf    n={node['n']:<6,} var={num(node['d1'])} e={num(node['e'])}  {group(node['sensors'])}"
    elif len(node["sensors"]) == 2:  # one pair: d0, avg and d1 are the same number and there is no d2
        values = f"leaf    n={node['n']:<6,} d1={num(node['d1'])} e={num(node['e'])}  {group(node['sensors'])}"
    else:
        pivot = "-".join(map(str, node["pivot"])) if node["pivot"] else "–"
        values = (f"leaf    n={node['n']:<6,} d0={num(node['d0'])} avg={num(node['avg'])} d1={num(node['d1'])}"
                  f" d2={num(node['d2'])} e={num(node['e'])}  {group(node['sensors'])}  farthest {pivot}")
    lines = [f"{head:<28} {values}"]
    inner = prefix + {"├── ": "│   ", "└── ": "    ", "": ""}[branch]
    pairs = [f"{a}-{b}={r:.3f}" for (a, b), r in sorted(node["pairs"].items())]
    if node["active"] and node["d1"] is not None and len(pairs) > 1:
        for i in range(0, len(pairs), 6):
            lines.append(f"{inner}{'    rnomc ' if i == 0 else ' ' * 10}{'  '.join(pairs[i : i + 6])}")
    for i, child in enumerate(node["children"]):
        lines += tree_lines(child, inner, "└── " if i == len(node["children"]) - 1 else "├── ")
    return lines


def tree_snapshots(run: Run, tree: str) -> list[dict]:
    """End of warm-up, just before each event after it, and the end of the run."""
    tr = run.trace[tree]
    warmup_s = TREES[tree]["warmup_s"]
    shots = [{"title": "end of warm-up", "t_s": tr["warmup"]["t_s"], "tree": tr["warmup"]["tree"], "fired": None}]
    alert_times = {a["t_min"] for a in run.alerts if a["tree"] == tree}
    for f in tr["fired"]:
        if f["t_s"] >= warmup_s:
            alert = f["kind"] == "merge" and minutes(f["t_s"]) in alert_times
            title = f"just before the {f['kind']}" + (" (▲ ALERT)" if alert else "")
            shots.append({"title": title, "t_s": f["t_s"], "tree": f["tree_before"], "fired": f})
    shots.append({"title": "end of run", "t_s": tr["final"]["t_s"], "tree": tr["final"]["tree"], "fired": None})
    return shots


def report(run: Run, say, top: int, chosen: list) -> None:
    def section(title):
        say()
        say(title)
        say("─" * len(title))

    o = run.official
    say("═" * 100)
    say(f"{run.label}: {ABOUT[run.scenario]}")
    say("═" * 100)

    section("RUN VALUES")
    say(f"  data         {DURATION:,} s at 1 Hz ({END / 60:.0f} h), after {BURN_IN / 3600:.0f} h of burn-in that is discarded")
    if run.scenario != "normal":
        say(f"  event        starts at {ONSET:.0f} min, reaches full size at {RAMP_END:.0f} min")
    for tree, cfg in TREES.items():
        names = run.trace[tree]["final"]["tree"]["sensors"]
        grace = run_poc.GRACE * cfg["rate_s"] / 60
        say(f"  {tree} tree    {len(names)} streams, one sample every {cfg['rate_s']} s, test every {cfg['n_min']} samples "
            f"({cfg['n_min'] * cfg['rate_s'] / 60:.0f} min), warm-up {cfg['warmup_s'] / 60:.0f} min, "
            f"merge grace {run_poc.GRACE} samples ({grace:.0f} min)")
    say(f"  ODAC         confidence_level={run_poc.CONF}, tau={run_poc.TAU}; "
        f"streams standardized on the first {run_poc.CALIB / 60:.0f} min, then × {run_poc.SCALE}")
    say(f"  static rule  any T > {T_LIMIT:.0f} °C or any V > {V_LIMIT} mm/s")
    say(f"  speed        {o['us_per_sample_fast']} µs/sample fast tree, {o['us_per_sample_slow']} µs/sample slow tree")

    section("STREAMS (each Δ is that period's mean minus the mean of the first period)")
    cuts = (0, FAULT_START, FAULT_START + FAULT_RAMP, DURATION)
    periods = list(zip(cuts, cuts[1:]))
    spans = [f"{a // 60}-{b // 60}" for a, b in periods]
    titles = [f"{span} min" for span in spans]
    if run.scenario != "normal":
        titles[1] += ", event ramp"
    deltas = "".join(f"{'Δ' + span:>10}" for span in spans[1:])
    say(f"  {'':10}| " + " | ".join(f"{title:<30}" for title in titles) + " |")
    say(f"  {'':10}| " + " | ".join(f"{'mean':>7}{'sd':>7}{'min':>8}{'max':>8}" for _ in periods)
        + f" |{deltas}  {'s over limit':>12}")
    for kind, j in itertools.product("IVT", range(1, 7)):
        name, x = f"{kind}{j}", run.streams[f"{kind}{j}"]
        parts = [x[a:b] for a, b in periods]
        stats = " | ".join(f"{p.mean():>7.2f}{p.std():>7.2f}{p.min():>8.2f}{p.max():>8.2f}" for p in parts)
        changes = "".join(f"{p.mean() - parts[0].mean():>+10.2f}" for p in parts[1:])
        limit = {"T": T_LIMIT, "V": V_LIMIT}.get(kind)
        over = f"{int((x > limit).sum()):,}" if limit else ""
        say(f"  {name:<3}{UNITS[kind]:<7}| {stats} |{changes}  {over:>12}")

    section("RESULT")
    first = o["first_alert_min"]
    say(f"  ODAC alerts   {len(run.alerts)}" + (f", first at {first} min ({after_onset(run, first)})" if run.alerts else ""))
    loc = o["localized"]
    if loc:
        isolated = "isolates " + " ".join(loc["isolated"]) if loc["isolated"] else "isolates no single sensor"
        say(f"  localization  split at {loc['t_min']} min {isolated}; sensors that changed group: {' '.join(loc['moved'])}")
    fired = o["static_first_min"]
    say("  static rule   " + (f"first alarm at {fired} min, {o['static_alarm_s']:,} s in alarm" if fired is not None else "no alarm"))
    say(f"  final leaves  fast tree {len(o['fast_final'])}, slow tree {len(o['slow_final'])}")
    for tree in TREES:
        if not o[f"{tree}_events"]:
            say(f"  note          the {tree} tree stayed a single root: nothing to merge, so it could not alert in this run")

    section("ALERTS")
    if not run.alerts:
        say("  no ODAC alert")
    for a in run.alerts:
        f = alert_firing(run, a)
        say(f"  ▲ ALERT  {a['t_min']} min ({after_onset(run, a['t_min'])}), {a['tree']} tree")
        if f:
            say(f"      merge  {group(f['sensors'])} + {group(f['sibling_sensors'])} → {group(f['parent_sensors'])}")
            say(f"      fired  {merge_line(f)}")
        say(f"      moved  {' '.join(a['moved'])}")
    limits = [(k, T_LIMIT if k[0] == "T" else V_LIMIT) for k in run.streams if k[0] in "TV"]
    alarms = [(int(np.argmax(run.streams[k] > v)) / 60, k, int((run.streams[k] > v).sum()))
              for k, v in limits if (run.streams[k] > v).any()]
    if not alarms:
        say("  no static alarm")
    for t, k, seconds in sorted(alarms):
        say(f"  ◆ STATIC {t:.1f} min ({after_onset(run, t)}), {k} over its limit, {seconds:,} s in total")

    section("ODAC EVENTS (every structure change; an alert is a merge after warm-up)")
    events = sorted([e | {"tree": t} for t in TREES for e in o[f"{t}_events"]], key=lambda e: e["t_min"])
    alert_keys = {(a["tree"], a["t_min"]) for a in run.alerts}
    if not events:
        say("  none")
    for e in events:
        tag = "▲ ALERT" if (e["tree"], e["t_min"]) in alert_keys else "" if e["after_warmup"] else "warm-up"
        say(f"  {e['t_min']:>6.1f} min  {e['tree']}  {e['type']:<5}  {e['n_leaves']} leaves  {tag}")
        for f in run.trace[e["tree"]]["fired"]:
            if minutes(f["t_s"]) == e["t_min"]:
                if f["kind"] == "split":
                    say(f"      {group(f['sensors'])} → {' + '.join(group(c) for c in f['into'])}")
                    say(f"      {split_line(f)}")
                else:
                    say(f"      {group(f['sensors'])} + {group(f['sibling_sensors'])} → {group(f['parent_sensors'])}")
                    say(f"      {merge_line(f)}")

    section("ODAC TREES (d1: largest rnomc in the leaf; a one-sensor leaf uses its variance; frozen: values at its split)")
    for tree in TREES:
        for shot in tree_snapshots(run, tree):
            say(f"{tree} tree, {shot['t_s'] / 60:.1f} min, {shot['title']}")
            for line in tree_lines(shot["tree"]):
                say("  " + line)
            say()

    section(f"RNOMC PAIRS THAT CHANGED MOST (rolling mean: up to {ONSET:.0f} min, windows touching the event, after it)")
    say("  Ranked by change / wander, where wander is the sd of the pair's rolling rnomc before the event.")
    say(f"  {'pair':<7}{'tracked by':<12}{'same leaf':<11}{'before':>7}{'during':>8}{'after':>8}{'change':>8}{'wander':>8}"
        f"{'ratio':>7}   ODAC's own")
    healthy = run.healthy_pairs()
    rows = {c["pair"]: c for c in run.changes}
    for pair in list(dict.fromkeys(chosen + [c["pair"] for c in run.changes[:top]])):
        c = rows[pair]
        kinds = {pair[0][0], pair[1][0]}
        by = "slow tree" if kinds == {"T"} else "no tree" if "T" in kinds else "fast tree"
        segments = run.own.get(pair)
        own = "–"
        if segments:
            last_t, last_r = segments[-1]["t"][-1], segments[-1]["r"][-1]
            own = f"{last_r:.3f} at the end" if last_t >= END - 1 else f"{last_r:.3f}, last seen at {last_t:.0f} min"
        same = "yes" if pair in healthy else "no"
        say(f"  {'-'.join(pair):<7}{by:<12}{same:<11}{c['before']:>7.3f}{c['during']:>8.3f}{c['after']:>8.3f}{c['change']:>+8.3f}"
            f"{c['wander']:>8.3f}{c['score']:>7.1f}   {own}")

    section("FEASIBILITY (data volumes are estimates: the PoC sends nothing)")
    sensors = {t: run.trace[t]["final"]["tree"]["sensors"] for t in TREES}
    worst = {t: compact_state([s]) for t, s in sensors.items()}
    now = compact_state(o["fast_final"] + o["slow_final"])
    say(f"  ODAC state    one sum per pair + two per sensor, {STATE_BYTES_PER_VALUE} B each: {sum(worst.values()):,} B at most "
        f"(fast {worst['fast']:,} B, slow {worst['slow']:,} B), {now:,} B at the end of this run")
    start = sum(run.trace[t]["memory_start"] for t in TREES)
    end = sum(run.trace[t]["memory_end"] for t in TREES)
    say(f"  memory        River's Python objects, measured: {start / 1e3:.0f} kB after the first sample, {end / 1e3:.0f} kB at the end of this run")
    fast_us, slow_us = o["us_per_sample_fast"], o["us_per_sample_slow"]
    budget = 100 * fast_us / 1e6 / run_poc.FAST["rate_s"]
    say(f"  speed         {fast_us} µs/sample fast tree ({len(sensors['fast'])} sensors), {slow_us} µs/sample slow tree "
        f"({len(sensors['slow'])} sensors); the fast tree uses {budget:.4f} % of its {run_poc.FAST['rate_s']} s budget")
    n = sum(len(s) for s in sensors.values())
    stream = stream_per_day(n)
    volumes = ", ".join(f"{label} {sent_per_day(n, a) / 1e3:.0f} kB ({stream / sent_per_day(n, a):.0f}× less)"
                        for a, label in ((0, "no alert"), (1, "1 alert"), (5, "5 alerts")))
    say(f"  data sent     stream {stream / 1e6:.1f} MB/day ({n} streams × 1 Hz × {BYTES_PER_VALUE} B); {volumes}")
    grace = {t: run_poc.GRACE * cfg["rate_s"] / 60 for t, cfg in TREES.items()}
    say(f"  decisions     a merge needs n ≥ {run_poc.GRACE} samples ({grace['fast']:.0f} min fast tree, {grace['slow']:.0f} min slow tree), "
        f"where e = {bound_at(run_poc.GRACE):.3f}")
    alert_keys = {(a["tree"], a["t_min"]) for a in run.alerts}
    late = [(t, f) for t, cfg in TREES.items() for f in run.trace[t]["fired"] if f["t_s"] >= cfg["warmup_s"]]
    for t, f in sorted(late, key=lambda item: item[1]["t_s"]):
        kind = f["kind"] + (" (▲ alert)" if (t, minutes(f["t_s"])) in alert_keys else "")
        say(f"                {minutes(f['t_s'])} min, {t} tree, {kind}: the leaf had n={f['n']:,}, e={f['e']:.3f}")
    if not late:
        say("                no split or merge after warm-up in this run")

    if run.problems:
        section("WARNINGS")
        for p in run.problems:
            say(f"  ! {p}")
    else:
        say()
        say("Checks passed: traced events match run_poc.evaluate(), and the margins explain every firing.")


def results_table(runs: list[Run], say) -> None:
    say()
    say("═" * 100)
    say(f"RESULTS over {len(runs)} run(s)")
    say("═" * 100)
    say(f"{'scenario':<9}{'runs':>5}  {'static rule fired':<19}{'ODAC alerted':<14}{'delay after onset (min)':<26}isolated afterwards")
    for sc in dict.fromkeys(r.scenario for r in runs):
        rs = [r for r in runs if r.scenario == sc]
        delays = [r.official["first_alert_min"] - ONSET for r in rs
                  if r.official["first_alert_min"] is not None and r.official["first_alert_min"] >= ONSET]
        spread = f"median {statistics.median(delays):.0f} ({min(delays):.0f}-{max(delays):.0f})" if delays else "–"
        isolated = sorted({s for r in rs for s in ((r.official["localized"] or {}).get("isolated") or [])})
        n_static = sum(r.official["static_first_min"] is not None for r in rs)
        n_alert = sum(bool(r.alerts) for r in rs)
        say(f"{sc:<9}{len(rs):>5}  {f'{n_static}/{len(rs)}':<19}{f'{n_alert}/{len(rs)}':<14}{spread:<26}{' '.join(isolated) or '–'}")

    say()
    say(f"FEASIBILITY over {len(runs)} run(s)")
    sensors = {t: runs[0].trace[t]["final"]["tree"]["sensors"] for t in TREES}
    n = sum(len(s) for s in sensors.values())
    speed = [r.official["us_per_sample_fast"] for r in runs]
    memory = [sum(r.trace[t]["memory_start"] for t in TREES) for r in runs]
    at_alert = [alert_firing(r, a)["n"] for r in runs for a in r.alerts]
    say(f"  ODAC state            {sum(compact_state([s]) for s in sensors.values()):,} B at most in compact form; "
        f"River's Python objects about {statistics.median(memory) / 1e3:.0f} kB at most")
    say(f"  speed, fast tree      median {statistics.median(speed):.1f} µs/sample (range {min(speed)}-{max(speed)})")
    one = sent_per_day(n, 1)
    say(f"  data sent, 1 alert    {one / 1e3:.0f} kB/day, {stream_per_day(n) / one:.0f}× less than the {stream_per_day(n) / 1e6:.1f} MB/day stream")
    if at_alert:
        say(f"  samples before alert  median {statistics.median(at_alert):,.0f} (range {min(at_alert):,}-{max(at_alert):,}) in the leaf that fired; "
            f"minimum allowed {run_poc.GRACE}")
    else:
        say(f"  samples before alert  no alerts in these runs; minimum allowed {run_poc.GRACE}")


# ---------------------------------------------------------------- figures

def set_style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 150,
        # a list, so a glyph missing from one font (the alert triangle, the arrow) comes from the next
        "font.family": [f for f in ("Helvetica Neue", "Arial", "DejaVu Sans") if f in installed],
        "font.size": 8, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 2, "ytick.major.size": 0, "xtick.major.width": 0.5,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
    })


def titled(fig, title: str, subtitle: str) -> None:
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.14 / height, title, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.012, 1 - 0.40 / height, subtitle, ha="left", va="top", fontsize=9, color=INK2)


def time_axis(ax, labelled: bool = True) -> None:
    ax.set_xlim(0, END)
    ax.set_xticks(np.arange(0, END + 1, 120))
    ax.tick_params(labelbottom=labelled)
    if labelled:
        ax.set_xlabel("time (min)")


def top_label(ax, x: float, text: str, ha: str = "left") -> None:
    ax.annotate(text, xy=(x, 1), xycoords=ax.get_xaxis_transform(), xytext=(2 if ha == "left" else -2, 2),
                textcoords="offset points", ha=ha, va="bottom", fontsize=7, color=INK2, annotation_clip=False)


def mark_run(ax, run: Run, tree: str | None = None, labels: bool = False) -> None:
    """The event ramp, its start, ODAC's alerts and the split that localized the first alert."""
    if run.scenario != "normal":
        ax.axvspan(ONSET, RAMP_END, color=WASH, lw=0, zorder=0)
        ax.axvline(ONSET, color=INK2, lw=0.7, ls=DOTTED, zorder=1)
        if labels:
            top_label(ax, ONSET, f"event ramp {ONSET:.0f}-{RAMP_END:.0f} min", ha="right")
    alerts = [a for a in run.alerts if tree in (None, a["tree"])]
    for a in alerts:
        ax.axvline(a["t_min"], color=CRITICAL, lw=1, zorder=3)
    if labels and alerts:
        more = f" (+{len(alerts) - 1} more)" if len(alerts) > 1 else ""
        top_label(ax, alerts[0]["t_min"], f"▲ alert {alerts[0]['t_min']:.0f} min{more}")
    loc = run.official["localized"]
    if loc and alerts and tree in (None, alerts[0]["tree"]) and loc["isolated"]:
        ax.axvline(loc["t_min"], color=INK, lw=0.7, zorder=3)
        if labels:
            top_label(ax, loc["t_min"], f"split isolates {' '.join(loc['isolated'])}")


def fig_streams(run: Run):
    height = 12.5
    fig = plt.figure(figsize=(14, height))
    over_limit = ", orange wash: over it" if run.official["static_first_min"] is not None else ""
    titled(fig, f"Sensor streams: {run.label}",
           f"{ABOUT[run.scenario]}. Line: 1-min mean; band: 1-min min to max. Dashed: static limit{over_limit}.")
    gs = fig.add_gridspec(7, 3, height_ratios=[0.6] + [1] * 6, left=0.045, right=0.99, top=1 - 1.0 / height,
                          bottom=0.045, hspace=0.2, wspace=0.13)
    t = np.arange(DURATION // 60) + 0.5

    top = fig.add_subplot(gs[0, :])
    severity = run.meta["severity"][::60]
    top.plot(t, run.meta["override"][::60], color=BLUE, lw=0.9, drawstyle="steps-post", label="speed override")
    if severity.any():
        top.plot(t, severity, color=ORANGE, lw=1.3, label="fault severity")
        top.legend(loc="lower left", bbox_to_anchor=(0, 0.98), ncols=2, fontsize=8, borderaxespad=0, handlelength=1.5)
    else:
        top.set_title("Speed override (0 = idle)", loc="left", fontsize=8.5, color=INK2, pad=11)
    mark_run(top, run, labels=True)
    top.set(ylim=(0, 1.05), yticks=[0, 0.5, 1])
    time_axis(top, labelled=False)

    first = {}
    for row, joint in enumerate(range(1, 7), start=1):
        for col, kind in enumerate("IVT"):
            name = f"{kind}{joint}"
            ax = fig.add_subplot(gs[row, col], sharey=first.get(kind))
            first.setdefault(kind, ax)
            m = run.streams[name][: DURATION // 60 * 60].reshape(-1, 60)
            mark_run(ax, run)
            ax.fill_between(t, m.min(1), m.max(1), color=BLUE, alpha=0.14, lw=0)
            ax.plot(t, m.mean(1), color=BLUE, lw=0.9)
            limit = {"T": T_LIMIT, "V": V_LIMIT}.get(kind)
            if limit is not None:
                ax.axhline(limit, color=INK2, lw=0.7, ls=(0, (4, 3)))
                ax.fill_between(t, 0, 1, where=m.max(1) > limit, transform=ax.get_xaxis_transform(),
                                color=SERIOUS, alpha=0.3, lw=0)
            ax.text(0.008, 0.93, name, transform=ax.transAxes, va="top", fontsize=8.5, fontweight="bold")
            time_axis(ax, labelled=row == 6)
            if row == 1:
                ax.set_title({"I": "Motor current (A)", "V": "Vibration RMS (mm/s)", "T": "Temperature (°C)"}[kind],
                             loc="left", fontsize=9, pad=6)
    return fig


def node_label(node: dict) -> str:
    sensors = node["sensors"]
    rows = [" ".join(sensors[i : i + 6]) for i in range(0, len(sensors), 6)]
    n = f"n={node['n']:,}  e={num(node['e'])}"
    if not node["active"]:
        return "\n".join([f"{node['name']} (frozen)", *rows, n, f"d1={num(node['d1'])}  d2={num(node['d2'])}"])
    if node["d1"] is None:
        return "\n".join([node["name"], *rows, f"n={node['n']:,}", "not tested yet"])
    if len(sensors) == 1:
        return "\n".join([node["name"], *rows, n, f"var={num(node['d1'])}"])
    if len(sensors) == 2:  # one pair: d0, avg and d1 are the same number and there is no d2
        return "\n".join([node["name"], *rows, n, f"d1={num(node['d1'])}"])
    return "\n".join([node["name"], *rows, n, f"d0={num(node['d0'])}  avg={num(node['avg'])}",
                      f"d1={num(node['d1'])}  d2={num(node['d2'])}"])


def layout(node: dict, depth: int = 0, leaves: list | None = None) -> list:
    leaves = [] if leaves is None else leaves
    node["y"] = -depth
    if node["children"]:
        for child in node["children"]:
            layout(child, depth + 1, leaves)
        node["x"] = sum(c["x"] for c in node["children"]) / 2
    else:
        node["x"] = len(leaves)
        leaves.append(node)
    return leaves


def walk(node: dict):
    yield node
    for child in node["children"]:
        yield from walk(child)


def fig_trees(run: Run):
    shots = [s | {"tree_name": tree} for tree in TREES for s in tree_snapshots(run, tree)]
    width_in_leaves = max(len(layout(s["tree"])) for s in shots)
    depths = [max(-n["y"] for n in walk(s["tree"])) for s in shots]
    ratios = [d + 1 for d in depths]
    level_in, gap_in = 1.0, 0.42
    height = sum(ratios) * level_in + gap_in * len(shots) + 1.1
    fig = plt.figure(figsize=(14, height))
    titled(fig, f"ODAC trees: {run.label}",
           "Blue: active leaf, tested every step. Gray: frozen at its split, its d1 and e are what children test against. "
           "Dark ring: the leaf whose test is about to fire.")
    gs = fig.add_gridspec(len(shots), 1, height_ratios=ratios, left=0.01, right=0.99, top=1 - 0.95 / height,
                          bottom=0.1 / height, hspace=gap_in * len(shots) / (sum(ratios) * level_in))
    for i, shot in enumerate(shots):
        ax = fig.add_subplot(gs[i])
        ax.axis("off")
        centre = (width_in_leaves - len(layout(shot["tree"]))) / 2
        ax.set_xlim(-0.55 - centre, width_in_leaves - 0.45 - centre)
        ax.set_ylim(-depths[i] - 0.5, 0.5)
        ax.set_title(f"{shot['tree_name']} tree, {shot['t_s'] / 60:.1f} min, {shot['title']}", loc="left",
                     fontsize=9.5, fontweight="bold", pad=3)
        nodes = list(walk(shot["tree"]))
        for node in nodes:
            for child in node["children"]:
                ax.plot([node["x"], child["x"]], [node["y"], child["y"]], color=AXIS, lw=0.9, zorder=1)
        for node in nodes:
            firing = shot["fired"] is not None and node["active"] and tuple(node["sensors"]) == shot["fired"]["sensors"]
            ax.text(node["x"], node["y"], node_label(node), ha="center", va="center", fontsize=7.6, zorder=2,
                    linespacing=1.25, color=INK,
                    bbox=dict(boxstyle="round,pad=0.45", fc=LEAF_FILL if node["active"] else WASH,
                              ec=INK if firing else "none", lw=1.3 if firing else 0))
    return fig


def leaf_series(tests: list[dict], key: str):
    times = sorted({r["t_s"] for r in tests})
    index = {t: i for i, t in enumerate(times)}
    series: dict = {}
    for r in tests:
        if r[key] is not None:
            series.setdefault(r["sensors"], np.full(len(times), np.nan))[index[r["t_s"]]] = r[key]
    return np.array(times) / 60, series


def fig_odac_tests(run: Run):
    height = 10.5
    fig = plt.figure(figsize=(15, height))
    titled(fig, f"Inside the ODAC tests: {run.label}",
           "Values read at every test step. Each line is one set of sensors; a break means that leaf did not exist then.")
    gs = fig.add_gridspec(4, 2, width_ratios=[2, 1], height_ratios=[0.55, 1, 1, 1], left=0.04, right=0.99,
                          top=1 - 1.2 / height, bottom=0.05, hspace=0.34, wspace=0.09)
    captions = ["Leaves in the tree",
                "Leaf diameter d1: largest rnomc in the leaf (one-sensor leaf: its variance)",
                f"Merge margin: d1 - parent d1 - max(e, parent e). A merge fires above 0 once n ≥ {run_poc.GRACE}",
                "Split margin: min(max(d1 - d2, tau), (d1 - d0)·|d1 + d0 - 2·avg|) - e. A split fires above 0"]
    for col, (tree, cfg) in enumerate(TREES.items()):
        tr = run.trace[tree]
        warmup = cfg["warmup_s"] / 60
        # healthy leaves first, so a leaf keeps its color across scenarios and seeds
        healthy = sorted(map(tuple, run.official[f"{tree}_healthy"]))
        late = dict.fromkeys(healthy + [r["sensors"] for r in tr["tests"] if r["t_s"] >= cfg["warmup_s"]])
        colors = {s: SERIES[i] if i < len(SERIES) else MUTED for i, s in enumerate(late)}
        for row in range(4):
            ax = fig.add_subplot(gs[row, col])
            mark_run(ax, run, tree=tree, labels=row == 0)
            time_axis(ax, labelled=row == 3)
            if col == 0:
                ax.set_title(captions[row], loc="left", fontsize=8.5, color=INK2, pad=11 if row == 0 else 4)
            else:
                ax.set_title(" ", loc="left", fontsize=8.5, pad=11 if row == 0 else 4)
            if row == 0:
                ts = [t for t, _ in tr["changes"]] + [END]
                ns = [len(p) for _, p in tr["changes"]]
                ax.step(ts, ns + ns[-1:], where="post", color=BLUE, lw=1)
                ax.axvline(warmup, color=MUTED, lw=0.7, zorder=1)
                top_label(ax, warmup, "warm-up ends", ha="left" if warmup < ONSET / 2 else "right")
                ax.set_ylim(0, max(ns) + 1)
                ax.yaxis.get_major_locator().set_params(integer=True)
                ax.text(0, 1.42, {"fast": "Fast tree: 12 current and vibration streams, tested every 2 min",
                                  "slow": "Slow tree: 6 temperatures, tested every 30 min"}[tree],
                        transform=ax.transAxes, fontsize=10, fontweight="bold")
                continue
            key = {1: "d1", 2: "merge_margin", 3: "split_margin"}[row]
            t, series = leaf_series(tr["tests"], key)
            for sensors, y in series.items():
                late_leaf = sensors in colors
                ax.plot(t, y, color=colors[sensors] if late_leaf else AXIS, lw=1 if late_leaf else 0.7,
                        zorder=2 if late_leaf else 1)
            if row > 1:
                ax.axhline(0, color=INK2, lw=0.8, zorder=1)
            if not series:
                reason = "the tree is only its root: no parent to merge into" if row == 2 else "no leaf with 3+ sensors"
                ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", color=INK2, zorder=4,
                        bbox=dict(fc=SURFACE, ec="none", pad=3))
                ax.set_yticks([])
            if row == 1:
                handles = [Line2D([], [], color=colors[s], lw=1.5, label=" ".join(s)) for s in late]
                if any(s not in colors for s in series):
                    handles.append(Line2D([], [], color=AXIS, lw=1, label="leaves gone by the end of warm-up"))
                ax.legend(handles=handles, ncols=3 if col == 0 else 1, loc="upper right", fontsize=7, borderaxespad=0.1,
                          handlelength=1.4, columnspacing=1.2)
    return fig


def fig_all_pairs(run: Run):
    order = [s for g in run.official["fast_healthy"] + run.official["slow_healthy"] for s in g]
    n = len(order)
    size = 15
    fig = plt.figure(figsize=(size, size))
    titled(fig, f"rnomc over time, all {n * (n - 1) // 2} pairs: {run.label}",
           f"Each panel is one pair: time 0 to {END:.0f} min across, rnomc 0 to 1 up. Streams are ordered by their leaf at the end of warm-up.")
    gs = fig.add_gridspec(n, n, left=0.012, right=0.985, top=1 - 0.9 / size, bottom=0.012, hspace=0.1, wspace=0.1)
    healthy = run.healthy_pairs()
    for i, a in enumerate(order):
        cell = fig.add_subplot(gs[i, i])
        cell.axis("off")
        cell.text(0.5, 0.5, a, ha="center", va="center", fontsize=9.5, fontweight="bold", transform=cell.transAxes)
        for j in range(i + 1, n):
            b = order[j]
            pair = tuple(sorted((a, b)))
            ax = fig.add_subplot(gs[i, j])
            if pair in healthy:
                ax.set_facecolor(LEAF_FILL)
            if run.scenario != "normal":
                ax.axvline(ONSET, color=INK2, lw=0.5, ls=DOTTED, zorder=1)
            for alert in run.alerts:
                ax.axvline(alert["t_min"], color=CRITICAL, lw=0.6, zorder=1)
            s = run.rolling[pair]
            ax.plot(s["t"], s["r"], color=BLUE, lw=0.6, zorder=2)
            for segment in run.own.get(pair, []):
                ax.plot(segment["t"], segment["r"], color=ORANGE, lw=0.8, zorder=3)
            ax.set(xlim=(0, END), ylim=(0, 1), xticks=[0, ONSET, END], yticks=[0, 0.5, 1],
                   xticklabels=["0", f"{ONSET:.0f}", f"{END:.0f}"], yticklabels=["0", ".5", "1"])
            ax.tick_params(labelbottom=False, labelleft=False, labeltop=i == 0, labelright=j == n - 1,
                           labelsize=6, length=0, pad=1.5)
            if i == 0:
                for label, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
                    label.set_ha(ha)
            if j == n - 1:
                for label, va in zip(ax.get_yticklabels(), ("bottom", "center", "top")):
                    label.set_va(va)
    handles = [Line2D([], [], color=BLUE, lw=1.5, label="rolling rnomc"),
               Line2D([], [], color=ORANGE, lw=1.5, label="ODAC's own rnomc, while both sensors share a leaf"),
               Line2D([], [], color=CRITICAL, lw=1.5, label="▲ ODAC alert"),
               Line2D([], [], color=INK2, lw=1, ls=DOTTED, label="event starts"),
               plt.Rectangle((0, 0), 1, 1, fc=LEAF_FILL, label="same leaf at the end of warm-up")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.03), fontsize=10, labelspacing=0.7)
    return fig


def fig_top_pairs(run: Run, pairs: list):
    cols = 3
    rows = math.ceil(len(pairs) / cols)
    height = 2.5 * rows + 1.5
    fig = plt.figure(figsize=(14, height))
    titled(fig, f"rnomc pairs that changed most: {run.label}",
           f"Ranked by change / wander. Numbers: rolling mean up to {ONSET:.0f} min → furthest level during or after the event "
           "(change, ratio to the pair's wander before the event).")
    gs = fig.add_gridspec(rows, cols, left=0.04, right=0.99, top=1 - 1.25 / height, bottom=0.55 / height,
                          hspace=0.3, wspace=0.08)
    info = {c["pair"]: c for c in run.changes}
    for i, pair in enumerate(pairs):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        s, c = run.rolling[pair], info[pair]
        first_row = i < cols
        mark_run(ax, run, labels=first_row)
        ax.plot(s["t"], s["r"], color=BLUE, lw=1, zorder=2)
        for segment in run.own.get(pair, []):
            ax.plot(segment["t"], segment["r"], color=ORANGE, lw=1.3, zorder=2.5)
        ax.set(ylim=(0, 1), yticks=np.arange(0, 1.01, 0.25))
        time_axis(ax, labelled=(i + cols) >= len(pairs))
        ax.tick_params(labelleft=i % cols == 0)
        if i % cols == 0:
            ax.set_ylabel("rnomc")
        pad = 13 if first_row else 4
        ax.set_title(f"{pair[0]} - {pair[1]}", loc="left", fontsize=10, fontweight="bold", pad=pad)
        ax.set_title(f"{c['before']:.2f} → {c['before'] + c['change']:.2f} ({c['change']:+.2f}, {c['score']:.1f}× wander)",
                     loc="right", fontsize=8, color=INK2, pad=pad)
    handles = [Line2D([], [], color=BLUE, lw=1.5, label="rolling rnomc"),
               Line2D([], [], color=ORANGE, lw=1.5, label="ODAC's own rnomc (cumulative since its leaf was created)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.62 / height), ncols=2, fontsize=8.5)
    return fig


def compare_pairs(runs: list[Run], chosen: list, per_scenario: int = 2) -> list:
    """Pairs that depart most from the healthy run after the event, per faulty scenario."""
    normal = next(r for r in runs if r.scenario == "normal")
    picked = list(chosen)
    for run in runs:
        if run is normal:
            continue
        def departure(pair):
            s, base = run.rolling[pair], normal.rolling[pair]
            return np.abs(s["r"] - base["r"])[s["t"] > ONSET].mean()
        picked += sorted(run.rolling, key=departure, reverse=True)[:per_scenario]
    return list(dict.fromkeys(picked))


def fig_compare(runs: list[Run], pairs: list):
    normal = next(r for r in runs if r.scenario == "normal")
    others = [r for r in runs if r is not normal]
    height = 2.0 * len(pairs) + 1.5
    fig = plt.figure(figsize=(14, height))
    titled(fig, f"Rolling rnomc against the healthy run, seed {normal.seed}",
           "Same seed means the same production schedule, so differences come from the scenario. "
           "Rows: the pairs that depart most from healthy after the event.")
    gs = fig.add_gridspec(len(pairs), len(others), left=0.05, right=0.99, top=1 - 1.4 / height,
                          bottom=0.55 / height, hspace=0.25, wspace=0.06)
    for row, pair in enumerate(pairs):
        for col, run in enumerate(others):
            ax = fig.add_subplot(gs[row, col])
            mark_run(ax, run, labels=row == 0)
            ax.plot(normal.rolling[pair]["t"], normal.rolling[pair]["r"], color=AXIS, lw=1, zorder=2)
            ax.plot(run.rolling[pair]["t"], run.rolling[pair]["r"], color=BLUE, lw=1, zorder=2.5)
            ax.set(ylim=(0, 1), yticks=[0, 0.5, 1])
            time_axis(ax, labelled=row == len(pairs) - 1)
            ax.tick_params(labelleft=col == 0)
            if col == 0:
                ax.set_ylabel(f"{pair[0]} - {pair[1]}", fontsize=9, fontweight="bold", color=INK)
            if row == 0:
                ax.text(0, 1.2, f"{run.scenario}: {ABOUT[run.scenario]}", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    handles = [Line2D([], [], color=BLUE, lw=1.5, label="this scenario"), Line2D([], [], color=AXIS, lw=1.5, label="healthy run")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.62 / height), ncols=2, fontsize=8.5)
    return fig


# ---------------------------------------------------------------- command line

def parse_seeds(text: str, ap: argparse.ArgumentParser) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        if not first.isdigit() or (last and not last.isdigit()) or (last and int(last) < int(first)):
            ap.error(f"--seed: '{part}' is not a seed or a range such as 3 or 1-10")
        seeds += range(int(first), int(last or first) + 1)
    return seeds


def parse_pairs(text: str, ap: argparse.ArgumentParser) -> list[tuple[str, str]]:
    valid = {f"{k}{j}" for k in "IVT" for j in range(1, 7)}
    pairs = []
    for part in filter(None, text.split(",")):
        names = part.strip().split("-")
        if len(names) != 2 or names[0] == names[1] or not set(names) <= valid:
            ap.error(f"'{part}' is not a pair of two different streams such as V3-I3")
        pairs.append(tuple(sorted(names)))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory analysis of the StrEdge PoC (see the top of this file).")
    ap.add_argument("--scenario", choices=[*run_poc.SCENARIOS, "all"], default="all")
    ap.add_argument("--seed", default="1", help="seeds to run: 3, 1,2,5 or 1-10 (default: 1)")
    ap.add_argument("--pairs", default="", help="pairs to always include in the plots, e.g. V3-I3,T3-T2")
    ap.add_argument("--top", type=int, default=6, help="how many top-changing pairs to report and plot (default: 6)")
    ap.add_argument("--window", type=int, default=30, help="rolling window for current/vibration pairs, in min (default: 30)")
    ap.add_argument("--slow-window", type=int, default=120, help="rolling window for pairs with a temperature, in min (default: 120)")
    ap.add_argument("--no-plots", action="store_true", help="only the text report")
    ap.add_argument("--show", action="store_true", help="also open the figures in windows")
    args = ap.parse_args()
    for w in (args.window, args.slow_window):
        if not 1 <= w <= END - RAMP_END:
            ap.error(f"rolling windows must be between 1 and {END - RAMP_END:.0f} min, to fit after the event")
    chosen = parse_pairs(args.pairs, ap)
    scenarios = run_poc.SCENARIOS if args.scenario == "all" else [args.scenario]
    seeds = parse_seeds(args.seed, ap)

    if not args.show:
        plt.switch_backend("Agg")
    set_style()

    def save(fig, path: Path) -> None:
        fig.savefig(path)
        print(f"  wrote {path}")
        if not args.show:
            plt.close(fig)

    runs = []
    for seed in seeds:
        of_seed = []
        for scenario in scenarios:
            run = analyse(scenario, seed, args.window, args.slow_window)
            lines: list[str] = []

            def say(text: str = "") -> None:
                print(text)
                lines.append(text)

            report(run, say, args.top, chosen)
            folder = OUT / f"{scenario}_seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\n  wrote {folder / 'report.txt'}")
            if not args.no_plots:
                top_pairs = list(dict.fromkeys(chosen + [c["pair"] for c in run.changes[: args.top]]))
                save(fig_streams(run), folder / "1_streams.png")
                save(fig_trees(run), folder / "2_trees.png")
                save(fig_odac_tests(run), folder / "3_odac_tests.png")
                save(fig_all_pairs(run), folder / "4_rnomc_all_pairs.png")
                if top_pairs:
                    save(fig_top_pairs(run, top_pairs), folder / "5_rnomc_top_pairs.png")
            print()
            of_seed.append(run)
        runs += of_seed
        if not args.no_plots and len(of_seed) > 1 and any(r.scenario == "normal" for r in of_seed):
            save(fig_compare(of_seed, compare_pairs(of_seed, chosen)), OUT / f"compare_seed{seed}.png")

    results_table(runs, print)
    if args.show and not args.no_plots:
        plt.show()


if __name__ == "__main__":
    main()
