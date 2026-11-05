"""
One pair of streams over one run: rnomc against time, with the event and the alarms marked.

Usage:
  python plot_pair.py                                     J2-J3 vibration, bearing scenario, seed 1
  python plot_pair.py --seed 3
  python plot_pair.py --scenario summer --pair T3-T5
  python plot_pair.py --slide                             the simple version for a presentation
Writes results/eda/<scenario>_seed<seed>/pair_<a>_<b>.png (with --slide: pair_<a>_<b>_slide.png)
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import eda
import run_poc

KINDS = {"I": "current", "V": "vibration", "T": "temperature"}


def describe(stream: str) -> str:
    return f"J{stream[1]} {KINDS[stream[0]]}"


def together(partition: list, a: str, b: str) -> bool:
    return any(a in group and b in group for group in partition)


def threshold_segments(run: eda.Run, pair: tuple) -> list:
    """The level the pair must pass to fire a merge: parent d1 + max(e, parent e).

    It exists only while the pair is a leaf of its own, so that the leaf's d1 is this pair's rnomc,
    and it only counts once the leaf has seen GRACE samples.
    """
    segments, previous_n = [], None
    for tree in run.trace.values():
        for r in tree["tests"]:
            if r["sensors"] == pair and r["n"] >= run_poc.GRACE:
                if previous_n is None or r["n"] <= previous_n:
                    segments.append(([], []))
                segments[-1][0].append(r["t_s"] / 60)
                segments[-1][1].append(r["parent_d1"] + r["merge_bound"])
                previous_n = r["n"]
    return segments


def plot_pair(run: eda.Run, pair: tuple):
    a, b = pair
    rolling = run.rolling[pair]
    own = run.own.get(pair, [])
    limits = threshold_segments(run, pair)
    kinds = {a[0], b[0]}
    tree_name = "slow" if kinds == {"T"} else None if "T" in kinds else "fast"  # no tree tracks current/vibration vs temperature
    static_first = run.official["static_first_min"]

    height = 6.4
    fig = plt.figure(figsize=(13, height))
    fig.text(0.012, 1 - 0.14 / height, f"{describe(a)} vs {describe(b)}: rnomc over one run", va="top",
             fontsize=14, fontweight="bold")
    static = f"first static alarm at {static_first} min" if static_first is not None else "no static alarm in this run"
    fig.text(0.012, 1 - 0.44 / height,
             f"{run.scenario} scenario, seed {run.seed}: {eda.ABOUT[run.scenario]}; {static}.\n"
             "rnomc = sqrt((1 - corr) / 2): 0 means the two signals move together, higher means they move less alike.",
             va="top", fontsize=9.5, color=eda.INK2, linespacing=1.4)
    ax = fig.add_axes([0.06, 0.1, 0.925, 1 - 0.1 - 1.5 / height])

    def above(x, text, ha, row=0):
        nudge = {"left": 4, "right": -4, "center": 0}[ha]
        ax.annotate(text, xy=(x, 1), xycoords=ax.get_xaxis_transform(), xytext=(nudge, 3 + 14 * row),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=9, color=eda.INK, annotation_clip=False)

    if run.scenario != "normal":
        ax.axvspan(eda.ONSET, eda.RAMP_END, color=eda.WASH, lw=0, zorder=0)
        ax.axvline(eda.ONSET, color=eda.INK, lw=1, ls=eda.DOTTED, zorder=1)
        above(eda.ONSET, f"event starts {eda.ONSET:.0f} min", "right")
        ramping = "ambient rises" if run.scenario == "summer" else "fault ramps up"
        above((eda.ONSET + eda.RAMP_END) / 2, f"{ramping} {eda.ONSET:.0f}-{eda.RAMP_END:.0f} min", "center", row=1)

    # Splits and merges that restart this pair's statistics, or take the pair apart. The alarm has its own line.
    if tree_name:
        alert_times = {x["t_min"] for x in run.alerts if x["tree"] == tree_name}
        partition = [run.trace[tree_name]["final"]["tree"]["sensors"]]  # the root: all sensors together
        for e in run.official[f"{tree_name}_events"]:
            was, now = together(partition, a, b), together(e["partition"], a, b)
            partition = e["partition"]
            if not e["after_warmup"] or e["t_min"] in alert_times or not (was or now) or not {a, b} & set(e["moved"]):
                continue
            ax.axvline(e["t_min"], color=eda.MUTED, lw=0.8, zorder=1)
            if now:
                ax.text(e["t_min"] - 2, 0.012, f"{e['type']} {e['t_min']:.0f} min", rotation=90, ha="right", va="bottom",
                        fontsize=8, color=eda.INK2, transform=ax.get_xaxis_transform())
            else:
                ax.text(e["t_min"] + 3, 0.02, f"{e['type']} {e['t_min']:.0f} min: ODAC stops tracking this pair",
                        ha="left", va="bottom", fontsize=8, color=eda.INK2, transform=ax.get_xaxis_transform())

    ax.plot(rolling["t"], rolling["r"], color=eda.BLUE, lw=1.3, zorder=2)
    for segment in own:
        ax.plot(segment["t"], segment["r"], color=eda.ORANGE, lw=1.9, zorder=3)
    for t, y in limits:
        ax.plot(t, y, color=eda.INK, lw=1.2, ls=(0, (3, 2)), zorder=4)

    top = max([rolling["r"].max()] + [max(s["r"]) for s in own] + [max(y) for _, y in limits])
    ymax = min(1.0, np.ceil(top * 1.35 * 20) / 20)

    for alert in run.alerts:
        ax.axvline(alert["t_min"], color=eda.CRITICAL, lw=1.6, zorder=5)
        delay = f" (+{alert['t_min'] - eda.ONSET:.0f} min)" if run.scenario != "normal" else ""
        above(alert["t_min"], f"▲ ODAC alarm {alert['t_min']:.0f} min{delay}", "left")
        fired = next(f for f in run.trace[alert["tree"]]["fired"]
                     if f["kind"] == "merge" and eda.minutes(f["t_s"]) == alert["t_min"])
        if fired["sensors"] == pair:
            level = fired["parent_d1"] + fired["merge_bound"]
            ax.plot([fired["t_s"] / 60], [fired["d1"]], "o", ms=7, color=eda.ORANGE, mec=eda.SURFACE, mew=1.5, zorder=6)
            ax.text(alert["t_min"] + 5, ymax * 0.93,
                    f"rnomc {fired['d1']:.3f} > threshold {level:.3f}\n"
                    f"(parent d1 {fired['parent_d1']:.3f} + bound {fired['merge_bound']:.3f})",
                    ha="left", va="top", fontsize=9, color=eda.INK, linespacing=1.35, zorder=6,
                    bbox=dict(fc=eda.SURFACE, ec="none", pad=2))
    if static_first is not None:
        ax.axvline(static_first, color=eda.SERIOUS, lw=1.6, zorder=5)
        above(static_first, f"◆ static alarm {static_first:g} min", "right" if run.alerts else "left", row=2 if run.alerts else 0)

    ax.set(xlim=(0, eda.END), ylim=(0, ymax), xlabel="time since the start of the run (min)",
           ylabel=f"rnomc({a}, {b})   (higher = less alike)")
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.set_xticks(np.arange(0, eda.END + 1, 60))
    ax.tick_params(labelsize=9)

    handles = [Line2D([], [], color=eda.BLUE, lw=1.6, label=f"rolling rnomc, {rolling['window']}-min window")]
    if own:
        handles.append(Line2D([], [], color=eda.ORANGE, lw=2, label="ODAC's own rnomc: cumulative, restarts at each split or merge"))
    if limits:
        handles.append(Line2D([], [], color=eda.INK, lw=1.2, ls=(0, (3, 2)), label="alarm threshold while this pair is its own leaf"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, borderaxespad=0.4)
    return fig


def plot_slide(run: eda.Run, pair: tuple):
    """The same run for a slide: one line, when the event starts, when the alarm fires.

    No title: the slide has its own. Sized to sit under a slide title on a 16:9 deck, so the fonts
    are their real size on the slide. The line is the rolling rnomc; ODAC's own values, which decide
    the alarm, restart at every split or merge and would need explaining.
    """
    a, b = pair
    s = run.rolling[pair]
    hours, r = s["t"] / 60, s["r"]
    event = {"summer": "ambient rise"}.get(run.scenario, "fault")

    fig = plt.figure(figsize=(12, 5.6), facecolor="white")
    ax = fig.add_axes([0.072, 0.15, 0.91, 0.775], facecolor="white")

    def marker_label(x, text, ha):
        ax.annotate(text, xy=(x, 1), xycoords=ax.get_xaxis_transform(), xytext=(6 if ha == "left" else -6, 5),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=15, color=eda.INK, annotation_clip=False)

    if run.scenario != "normal":
        ax.axvspan(eda.ONSET / 60, eda.RAMP_END / 60, color=eda.WASH, lw=0, zorder=0)
        ax.axvline(eda.ONSET / 60, color=eda.INK, lw=1.8, ls=eda.DOTTED, zorder=1)
        marker_label(eda.ONSET / 60, f"{event.capitalize()} starts", "right")

    ax.plot(hours, r, color=eda.BLUE, lw=2.6, zorder=2)

    for i, alert in enumerate(run.alerts):
        ax.axvline(alert["t_min"] / 60, color=eda.CRITICAL, lw=2.6, zorder=3)
        if i == 0:
            marker_label(alert["t_min"] / 60, "ODAC alarm", "left")

    ymax = np.ceil(r.max() * 1.25 * 10) / 10
    same_kind = a[0] == b[0]
    name = f"J{a[1]}-J{b[1]} {KINDS[a[0]]}" if same_kind else f"{a}-{b}"
    ax.set_ylabel(f"{name} distance (rnomc)", fontsize=14, color=eda.INK, labelpad=8)
    ax.set_xlabel("Time (hours)", fontsize=14, color=eda.INK, labelpad=6)
    ax.set(xlim=(0, eda.END / 60), ylim=(0, ymax))
    ax.set_xticks(np.arange(0, eda.END / 60 + 1, 2))
    ax.set_yticks(np.arange(0, ymax + 0.001, 0.1))
    ax.tick_params(labelsize=14, length=0, pad=6)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot rnomc for one pair of streams over one run.")
    ap.add_argument("--scenario", choices=run_poc.SCENARIOS, default="bearing")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pair", default="V2-V3", help="two streams, e.g. V2-V3 (default)")
    ap.add_argument("--slide", action="store_true", help="the simple version for a presentation")
    args = ap.parse_args()
    pairs = eda.parse_pairs(args.pair, ap)
    if len(pairs) != 1:
        ap.error("--pair: give exactly one pair, such as V2-V3")
    pair = pairs[0]

    plt.switch_backend("Agg")
    eda.set_style()
    run = eda.analyse(args.scenario, args.seed, 30, 120)
    for problem in run.problems:
        print(f"! {problem}")

    print(f"{run.label}: {eda.ABOUT[run.scenario]}")
    if run.scenario != "normal":
        print(f"  event starts        {eda.ONSET:.1f} min")
        print(f"  event at full size  {eda.RAMP_END:.1f} min")
    for alert in run.alerts:
        print(f"  ODAC alarm          {alert['t_min']:.1f} min, {eda.after_onset(run, alert['t_min'])}, {alert['tree']} tree")
    if not run.alerts:
        print("  ODAC alarm          none")
    static_first = run.official["static_first_min"]
    print(f"  static alarm        {f'{static_first} min' if static_first is not None else 'none'}")
    change = next(c for c in run.changes if c["pair"] == pair)
    print(f"  rolling rnomc({pair[0]}, {pair[1]})  mean {change['before']:.3f} before the event, "
          f"{change['during']:.3f} in windows touching it, {change['after']:.3f} after")

    folder = eda.OUT / f"{args.scenario}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if args.slide:
        path = folder / f"pair_{pair[0]}_{pair[1]}_slide.png"
        plot_slide(run, pair).savefig(path, dpi=200, facecolor="white")
    else:
        path = folder / f"pair_{pair[0]}_{pair[1]}.png"
        plot_pair(run, pair).savefig(path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
