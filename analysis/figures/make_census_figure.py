# SPDX-License-Identifier: MIT
"""Figure for the RQ2 leaderboard census (paper Section 6).

A dot plot with one row per analysed leaderboard (22), sorted by task count n, largest at the top.
For each leaderboard it shows the share of adjacent score gaps that fall below
  (a) the paired item-sampling bound with the empirical task-level correlation rho (headline); boards with
      per-task data use their own median adjacent-pair rho (marked *), the others the pooled median of those
      (no interval is drawn: adjacent pairs share entries);
  (a) the same bound at rho = 0 (scores only; the widest bound for non-negative rho),
  (b) the fixed-task-set rerun bound at f = the Terminal-Bench 2.0 median run-noise share, k-adjusted.
Every plotted value is read from ``analysis/leaderboards/results/census.csv`` (primary set);
nothing is recomputed here.

Output (vector PDF, one IEEE column wide, 3.5 in):
    paper/figures/fig_census.pdf

Usage: python make_census_figure.py [--census ../leaderboards/results/census.csv] [--out ../../paper/figures]

Style and palette: ``viz_style.py``. Orange = task-sampling bound, blue = rerun bound; the three series
also differ in marker shape and fill, so the figure reads in greyscale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from viz_style import COL_W, FS, GRID, INK, MUTED, RUN, TASK, plt, save  # noqa: E402

HERE = Path(__file__).resolve().parent

# (label, bound family, parameter selector, colour, marker, filled); selector None = one row per board,
# "tb2" = the b_f_kadj row whose param_label is "f=TB2 median".
# Hue follows the entity: task-sampling (item-sampling) bounds are orange, the run-to-run (rerun) bound is blue.
SERIES = [
    (r"item-sampling bound, estimated $\rho$", "a_emp", None, TASK, "o", True),
    (r"item-sampling bound, $\rho=0$", "a", 0.0, TASK, "s", False),
    (r"rerun bound, $f$ = TB2 median", "b_f_kadj", "tb2", RUN, "D", True),
]


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

    fig, ax = plt.subplots(figsize=(COL_W, 0.2 * m + 0.95))
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.grid(True, axis="y", color="#f1efea", linewidth=0.4)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    y0 = np.arange(m)[::-1].astype(float)
    off = (0.2, 0.0, -0.2)
    for si, (lab, fam, par, col, mk, filled) in enumerate(SERIES):
        rows = [pick(c, lb, fam, par) for lb in lbs]
        share = np.array([100 * r.share_adjacent_below for r in rows])
        y = y0 + off[si]
        # No interval is drawn (2026-10-20, review round 4, Codex m5): adjacent pairs share entries, so a
        # binomial interval that treats them as independent would misstate the uncertainty of the share.
        if filled:
            ax.scatter(share, y, s=11, marker=mk, color=col, edgecolor="white", linewidth=0.3, label=lab, zorder=3)
        else:
            ax.scatter(share, y, s=11, marker=mk, facecolor="white", edgecolor=col, linewidth=0.9, label=lab, zorder=3)

    ax.set_yticks(y0)
    ax.set_yticklabels([lb.replace("banking_knowledge", "banking") + (" *" if own[lb] else "") for lb in lbs])
    ax.set_ylim(-0.6, m - 0.4)
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(y0)
    ax2.set_yticklabels([f"{int(base[lb].n_tasks)}" for lb in lbs])
    ax2.tick_params(axis="y", length=0, labelsize=FS, pad=2)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("task count $n$", fontsize=FS, labelpad=4)
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
    return save(fig, out, "fig_census.pdf", "make_census_figure.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(HERE.parent / "leaderboards/results/census.csv"))
    ap.add_argument("--out", default=str(HERE.parents[1] / "paper/figures"))
    a = ap.parse_args()
    print("wrote", make(Path(a.census), Path(a.out)))


if __name__ == "__main__":
    main()
