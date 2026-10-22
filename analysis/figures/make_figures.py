"""Figures for the RQ3 re-analysis section (paper Section 7).

Every plotted value is read from files in ``analysis/results/``; nothing is
recomputed or typed in. The one constant defined here, SEPARABLE_THRESHOLD, is the
registered separability threshold (``results.md``, Definitions: "Separable pair:
flip probability <= 0.05"); the script asserts that it reproduces the
``gap_max_unstable_pp`` values stored in ``rank_inversion_summary.json``.

Outputs (vector PDF, one IEEE column wide, 3.5 in):
    paper/figures/fig_rank_intervals.pdf   per-submission mean with run-to-run and
                                           task-sampling 95% intervals, sorted
    paper/figures/fig_flip_vs_gap.pdf      pairwise flip probability against the
                                           observed gap, both resampling schemes

Usage: python make_figures.py [--results ../results] [--out ../../paper/figures]

Style and palette: ``viz_style.py`` (blue = run-to-run, orange = task-sampling in every figure).
Schemes are also separated by marker shape and line style, so the figures read in
greyscale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from viz_style import COL_W, FS, GRID, INK, INK2, MUTED, RUN, TASK, plt, save, style_axes, tint  # noqa: E402

HERE = Path(__file__).resolve().parent
SEPARABLE_THRESHOLD = 0.05  # results.md, Definitions
RUN_COLOR, TASK_COLOR = RUN, TASK  # run-to-run: blue; task-sampling: orange (same in every figure)


def flagged_submissions(results: Path) -> list[str]:
    """Analysis-set submissions flagged in the load step (results.md, Sensitivity)."""
    d = pd.read_csv(results / "tb2_load_counts.csv")
    d = d[d["included"] == 1]
    return d.loc[(d["duplicate_trial_names"] > 0) | (d["zero_within_task_variance"] > 0), "submission"].tolist()


def fig_rank_intervals(results: Path, out: Path) -> None:
    boot = pd.read_csv(results / "tb2_bootstrap.csv")
    ranks = pd.read_csv(results / "rank_intervals_run.csv")[["submission", "observed_rank"]]
    d = boot.merge(ranks, on="submission", how="inner").sort_values("observed_rank").reset_index(drop=True)
    assert len(d) == len(boot) == len(ranks), "analysis set mismatch between result files"
    x = d["observed_rank"].to_numpy()
    pct = lambda col: 100 * d[col].to_numpy()
    mean, rlo, rhi, tlo, thi = pct("mean_success"), pct("run_ci_low"), pct("run_ci_high"), pct("task_ci_low"), pct("task_ci_high")

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    style_axes(ax)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    # two sources, two line weights and a cap on the run interval, so they separate in grey scale too
    ax.vlines(x, tlo, thi, color=TASK_COLOR, linewidth=0.7, alpha=0.65, zorder=2, label="Task-sampling 95%")
    ax.errorbar(x, mean, yerr=[mean - rlo, rhi - mean], fmt="none", ecolor=RUN_COLOR, elinewidth=1.5, capsize=1.0,
                capthick=0.6, zorder=3, label="Run-to-run 95%")
    ax.scatter(x, mean, s=7, color="#262626", edgecolors="white", linewidths=0.2, zorder=6, label="Observed mean")

    flagged = d[d["submission"].isin(flagged_submissions(results))]
    for _, r in flagged.iterrows():
        xr, yr = r["observed_rank"], 100 * r["mean_success"]
        ax.plot([xr], [yr], marker="D", markersize=4.6, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=0.8, zorder=7)
        ax.annotate("flagged: identical\nruns on every task", xy=(xr, yr + 3), xytext=(xr + 1.5, yr + 26),
                    fontsize=FS, color=INK, ha="center", va="bottom", linespacing=1.05,
                    arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=0.5, mutation_scale=5, shrinkA=1,
                                    shrinkB=4))

    ax.set_xlim(0, len(d) + 1)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks([1] + list(range(20, len(d) + 1, 20)))
    ax.set_xlabel("Submission rank by observed mean")
    ax.set_ylabel("Mean task success (%)")
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(k) for k in ("Task-sampling 95%", "Run-to-run 95%", "Observed mean")]
    ax.legend([hs[i] for i in order], [ls[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.2,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, out, "fig_rank_intervals.pdf", "make_figures.py")


def fig_flip_vs_gap(results: Path, out: Path, xmax: float = 12.0) -> None:
    pairs = pd.read_csv(results / "rank_inversion_pairs.csv")
    bins = pd.read_csv(results / "rank_gap_bins.csv")
    summ = json.loads((results / "rank_inversion_summary.json").read_text(encoding="utf-8"))

    schemes = {
        "run": dict(col="p_flip_run", color=RUN_COLOR, marker="o", ls="-", label="Run-to-run"),
        "task": dict(col="p_flip_task", color=TASK_COLOR, marker="s", ls="--", label="Task sampling"),
    }
    # The stored minimum stable gap must follow from the pair table and the threshold.
    for key, s in schemes.items():
        recomputed = pairs.loc[pairs[s["col"]] > SEPARABLE_THRESHOLD, "gap_pp"].max()
        assert np.isclose(recomputed, summ[key]["gap_max_unstable_pp"]), (key, recomputed)
        beyond = pairs.loc[pairs["gap_pp"] > xmax, s["col"]]
        assert (beyond <= SEPARABLE_THRESHOLD).all(), "pairs beyond the plotted range are not all separable"

    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    style_axes(ax)
    gap = pairs["gap_pp"].clip(lower=0)  # exact ties sit at -1e-14 after floating-point differencing
    near = gap <= xmax
    for key, s in schemes.items():
        ax.scatter(gap[near], pairs.loc[near, s["col"]], s=1.5, color=s["color"], alpha=0.12,
                   marker=s["marker"], linewidths=0, zorder=2)
        b = bins[(bins["scheme"] == s["col"]) & (bins["bin"] >= 0) & (bins["bin"] < xmax)]
        ax.plot(b["bin"] + 0.5, b["mean_flip"], color=s["color"], linestyle=s["ls"], linewidth=1.5,
                marker=s["marker"], markersize=3.2, markeredgecolor="white", markeredgewidth=0.3,
                zorder=4, label=s["label"])
        g = summ[key]["gap_max_unstable_pp"]
        ax.axvline(g, color=s["color"], linestyle=":", linewidth=0.9, zorder=3)
        # threshold labels sit on the line, in the scheme's colour family: left of the 9.7 pp line, right of the 3.6 pp one
        ax.text(g + (0.18 if key == "run" else -0.18), 0.6, f"{g:.1f} pp:\n{s['label'].lower()}", fontsize=FS,
                color=INK, ha="left" if key == "run" else "right", va="center", linespacing=1.05)
    # direct curve labels (the line style and marker still differ, so the pair reads in grey scale)
    ax.text(3.8, 0.034, "Run-to-run", color=RUN_COLOR, fontsize=FS, fontweight="bold", ha="left", va="center")
    ax.text(5.2, 0.17, "Task sampling", color=TASK_COLOR, fontsize=FS, fontweight="bold", ha="left", va="center")
    ax.axhline(SEPARABLE_THRESHOLD, color=INK2, linewidth=0.6, zorder=3)
    ax.text(xmax - 0.1, SEPARABLE_THRESHOLD + 0.012, "5% flip probability", fontsize=FS, color=INK,
            ha="right", va="bottom")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 0.66)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Observed gap between the two submissions (pp)")
    ax.set_ylabel("Flip probability of the pair")
    fig.tight_layout(pad=0.3)
    save(fig, out, "fig_flip_vs_gap.pdf", "make_figures.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "paper" / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_rank_intervals(a.results, a.out)
    fig_flip_vs_gap(a.results, a.out)
    print(f"wrote figures to {a.out}")


if __name__ == "__main__":
    main()
