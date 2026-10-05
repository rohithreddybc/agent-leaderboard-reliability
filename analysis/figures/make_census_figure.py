# SPDX-License-Identifier: MIT
"""Figure for the RQ2 leaderboard census (paper Section 6).

A dot plot with one row per analysed leaderboard (22), sorted by task count n, largest at the top.
For each leaderboard it shows the share of adjacent score gaps that fall below
  (a) the paired item-sampling bound with the empirical task-level correlation rho (headline), with its
      Wilson 95% interval; boards with per-task data use their own median adjacent-pair rho (marked *),
      the others the pooled median of those;
  (a) the same bound at rho = 0 (unpaired, scores only; the conservative case),
  (b) the fixed-task-set rerun bound at f = the Terminal-Bench 2.0 median run-noise share, k-adjusted.
Every plotted value is read from ``analysis/leaderboards/results/census.csv`` (primary set);
nothing is recomputed here.

Output (vector PDF, one IEEE column wide, 3.5 in):
    paper/figures/fig_census.pdf

Usage: python make_census_figure.py [--census ../leaderboards/results/census.csv] [--out ../../paper/figures]

Palette: Okabe-Ito blue, bluish green and vermilion (checked with the dataviz validator:
all checks pass; worst adjacent CVD delta-E 11.0, normal-vision delta-E 25.8, contrast >= 3:1).
The three bounds also differ in marker shape, so the figure reads in greyscale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
COL_W = 3.5  # IEEE single-column width, inches

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
# (label, bound family, parameter selector, colour, marker); selector None = one row per board,
# "tb2" = the b_f_kadj row whose param_label is "f=TB2 median"
SERIES = [
    (r"item-sampling bound, empirical $\rho$", "a_emp", None, "#0072B2", "o"),
    (r"item-sampling bound, $\rho=0$", "a", 0.0, "#009E73", "s"),
    (r"rerun bound, $f$ = TB2 median", "b_f_kadj", "tb2", "#D55E00", "D"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.3,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def pick(c: pd.DataFrame, lb: str, fam: str, par) -> pd.Series:
    q = c[(c["leaderboard"] == lb) & (c["set"] == "primary") & (c["bound"] == fam)]
    if par == "tb2":
        q = q[q["param_label"] == "f=TB2 median"]
    elif par is not None:
        q = q[np.isclose(q["param"].astype(float), par)]
    assert len(q) == 1, (lb, fam, par, len(q))
    return q.iloc[0]


def make(census: Path, out: Path) -> Path:
    c = pd.read_csv(census)
    lbs = list(dict.fromkeys(c.loc[c["set"] == "primary", "leaderboard"]))
    base = {lb: pick(c, lb, "a", 0.0) for lb in lbs}
    own = {lb: str(pick(c, lb, "a_emp", None).param_label).endswith("own median)") for lb in lbs}
    lbs.sort(key=lambda lb: (-int(base[lb].n_tasks), lb))  # largest n at the top
    m = len(lbs)

    fig, ax = plt.subplots(figsize=(COL_W, 0.178 * m + 0.78))
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.grid(True, axis="y", color="#f1f1f1", linewidth=0.4)
    ax.set_axisbelow(True)
    y0 = np.arange(m)[::-1].astype(float)
    off = (0.2, 0.0, -0.2)
    for si, (lab, fam, par, col, mk) in enumerate(SERIES):
        rows = [pick(c, lb, fam, par) for lb in lbs]
        share = np.array([100 * r.share_adjacent_below for r in rows])
        y = y0 + off[si]
        if si == 0:  # Wilson 95% interval for the headline series only
            lo = np.array([100 * r.wilson_lo for r in rows])
            hi = np.array([100 * r.wilson_hi for r in rows])
            ax.errorbar(share, y, xerr=[np.clip(share - lo, 0, None), np.clip(hi - share, 0, None)], fmt="none",
                        ecolor=col, elinewidth=0.7, alpha=0.55, capsize=0, zorder=2)
        ax.scatter(share, y, s=9.5, marker=mk, color=col, edgecolor="white", linewidth=0.3, label=lab, zorder=3)

    ax.set_yticks(y0)
    ax.set_yticklabels([lb + (" *" if own[lb] else "") for lb in lbs])
    ax.set_ylim(-0.6, m - 0.4)
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(y0)
    ax2.set_yticklabels([f"{int(base[lb].n_tasks)}" for lb in lbs])
    ax2.tick_params(axis="y", length=0, labelsize=6.2, pad=2)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("task count $n$", fontsize=6.5, labelpad=4)
    ax.set_xlim(0, 101)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Adjacent score gaps below the 95% bound (%)\n"
                  r"* $\rho$ from per-task data; others: pooled median", linespacing=1.3)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.36, 1.0), frameon=False, ncol=1,
                    handletextpad=0.3, borderaxespad=0.2, labelspacing=0.2, markerscale=0.9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(pad=0.3)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fig_census.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "Creator": "make_census_figure.py"})
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(HERE.parent / "leaderboards/results/census.csv"))
    ap.add_argument("--out", default=str(HERE.parents[1] / "paper/figures"))
    a = ap.parse_args()
    print("wrote", make(Path(a.census), Path(a.out)))


if __name__ == "__main__":
    main()
