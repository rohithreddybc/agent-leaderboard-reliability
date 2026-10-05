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

Palette: Okabe-Ito blue and vermilion (checked with the dataviz validator:
adjacent CVD delta-E 21.9, normal-vision delta-E 31.2, both above 3:1 contrast).
Schemes are also separated by marker shape and line style, so the figures read in
greyscale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SEPARABLE_THRESHOLD = 0.05  # results.md, Definitions
COL_W = 3.5  # IEEE single-column width, inches

RUN_COLOR = "#0072B2"   # run-to-run (primary estimand)
TASK_COLOR = "#D55E00"  # task-sampling (secondary estimand)
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def style_axes(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


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
    ax.vlines(x, tlo, thi, color=TASK_COLOR, linewidth=0.9, alpha=0.9, zorder=2,
              label="Task-sampling 95% interval")
    ax.vlines(x, rlo, rhi, color=RUN_COLOR, linewidth=2.0, zorder=3, label="Run-to-run 95% interval")
    ax.plot(x, mean, linestyle="none", marker="o", markersize=1.6, color=INK, zorder=4, label="Observed mean")

    flagged = d[d["submission"].isin(flagged_submissions(results))]
    for _, r in flagged.iterrows():
        xr, yr = r["observed_rank"], 100 * r["mean_success"]
        ax.plot([xr], [yr], marker="D", markersize=4.2, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=0.7, zorder=5)
        ax.annotate("flagged: identical\nruns on every task", xy=(xr, yr), xytext=(xr + 3, yr + 22),
                    fontsize=6, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkA=0, shrinkB=3))

    ax.set_xlim(0, len(d) + 1)
    ax.set_ylim(0, 100)
    ax.set_xticks([1] + list(range(10, len(d) + 1, 10)))
    ax.set_xlabel("Submission, ordered by observed mean (best first)")
    ax.set_ylabel("Mean task success (%)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_rank_intervals.pdf", metadata={"CreationDate": None, "Creator": "make_figures.py"})
    plt.close(fig)


def fig_flip_vs_gap(results: Path, out: Path, xmax: float = 12.0) -> None:
    pairs = pd.read_csv(results / "rank_inversion_pairs.csv")
    bins = pd.read_csv(results / "rank_gap_bins.csv")
    summ = json.loads((results / "rank_inversion_summary.json").read_text(encoding="utf-8"))

    schemes = {
        "run": dict(col="p_flip_run", color=RUN_COLOR, marker="o", ls="-", label="Run-to-run"),
        "task": dict(col="p_flip_task", color=TASK_COLOR, marker="s", ls="--", label="Task-sampling"),
    }
    # The stored minimum stable gap must follow from the pair table and the threshold.
    for key, s in schemes.items():
        recomputed = pairs.loc[pairs[s["col"]] > SEPARABLE_THRESHOLD, "gap_pp"].max()
        assert np.isclose(recomputed, summ[key]["gap_max_unstable_pp"]), (key, recomputed)
        beyond = pairs.loc[pairs["gap_pp"] > xmax, s["col"]]
        assert (beyond <= SEPARABLE_THRESHOLD).all(), "pairs beyond the plotted range are not all separable"

    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    gap = pairs["gap_pp"].clip(lower=0)  # exact ties sit at -1e-14 after floating-point differencing
    near = gap <= xmax
    for key, s in schemes.items():
        ax.scatter(gap[near], pairs.loc[near, s["col"]], s=2.0, color=s["color"], alpha=0.22,
                   marker=s["marker"], linewidths=0, zorder=2)
        b = bins[(bins["scheme"] == s["col"]) & (bins["bin"] >= 0) & (bins["bin"] < xmax)]
        ax.plot(b["bin"] + 0.5, b["mean_flip"], color=s["color"], linestyle=s["ls"], linewidth=1.1,
                marker=s["marker"], markersize=2.8, markeredgecolor="white", markeredgewidth=0.3,
                zorder=4, label=f'{s["label"]}, mean per 1 pp bin')
        g = summ[key]["gap_max_unstable_pp"]
        ax.axvline(g, color=s["color"], linestyle=":", linewidth=0.9, zorder=3)
        ax.annotate(f"{g:.1f} pp", xy=(g, 0.62), xytext=(g + 0.15, 0.62), fontsize=6.5, color=INK,
                    ha="left", va="center")
    ax.axhline(SEPARABLE_THRESHOLD, color=INK, linewidth=0.7, zorder=3)
    ax.text(xmax - 0.1, SEPARABLE_THRESHOLD + 0.012, "5% flip probability", fontsize=6.3, color=INK,
            ha="right", va="bottom")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 0.66)
    ax.set_xlabel("Observed gap between the two submissions (pp)")
    ax.set_ylabel("Flip probability of the pair")
    ax.legend(loc="center right", frameon=False, handlelength=2.0, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_flip_vs_gap.pdf", metadata={"CreationDate": None, "Creator": "make_figures.py"})
    plt.close(fig)


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
