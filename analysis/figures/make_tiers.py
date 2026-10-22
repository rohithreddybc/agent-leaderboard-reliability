# SPDX-License-Identifier: MIT
"""Tier strips for six representative leaderboards of the census (paper Sec. 6).

One panel per board; one dot per kept entry, ordered by score (best first); point colour is constant; two pale
alternating fills behind the points mark successive tier bands, with thin lines at tier boundaries; the five best entries carry a thin ring. Tiers use the headline bound of the census, the
paired item-sampling bound with the board's task-level correlation rho, and the greedy rule of
``census_summary.md`` (Definitions). Inputs, read from ``analysis/leaderboards/results/``:
  census_entries.csv     score per kept entry
  census_parameters.json rho used per board, task count is read from census.csv
  census.csv             n_tasks, n_tiers (the recomputed tier count must equal n_tiers for every board)

Output (vector PDF, double column, 7.16 in): paper/figures/fig_tiers.pdf
Usage: python make_tiers.py [--out ../../paper/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from viz_style import (DBL_W, FS, FS_L, INK, INK2, LB_RESULTS, MUTED, PAPER_FIGS, TIER_A, TIER_B, WHITE,
                       greedy_tiers, plt, save, style_axes, tint)

BOARDS = ["SWE-bench Verified", "SWE-bench test (full)", "Terminal-Bench 2.0", "TheAgentCompany",
          "tau2-bench banking_knowledge", "HAL GAIA"]
TITLES = {"tau2-bench banking_knowledge": r"$\tau^2$-bench banking_knowledge", "HAL GAIA": "HAL GAIA"}


def load() -> tuple[dict, pd.DataFrame]:
    ent = pd.read_csv(LB_RESULTS / "census_entries.csv")
    ent = ent[ent["status"] == "kept"]
    cen = pd.read_csv(LB_RESULTS / "census.csv")
    cen = cen[(cen["set"] == "primary") & (cen["bound"] == "a_emp")].set_index("leaderboard")
    par = json.loads((LB_RESULTS / "census_parameters.json").read_text(encoding="utf-8"))
    out = {}
    for lb, g in ent.groupby("leaderboard"):
        if lb not in cen.index:  # excluded boards are not in the census
            continue
        g = g.sort_values(["score", "entry"], ascending=[False, True])
        sc = g["score"].to_numpy(float)
        n = int(cen.loc[lb, "n_tasks"])
        tier = greedy_tiers(sc, n, par["rho_used"][lb])
        assert tier.max() == int(cen.loc[lb, "n_tiers"]), (lb, tier.max(), cen.loc[lb, "n_tiers"])
        assert len(sc) == int(cen.loc[lb, "n_entries"]), lb
        out[lb] = dict(score=100 * sc, tier=tier, n=n)
    return out, cen


def make(out: Path) -> Path:
    data, _ = load()
    fig, axs = plt.subplots(2, 3, figsize=(DBL_W, 3.75))
    for ax, lb in zip(axs.ravel(), BOARDS):
        d = data[lb]
        sc, tier = d["score"], d["tier"]
        r = np.arange(1, len(sc) + 1)
        style_axes(ax)
        ax.grid(False, axis="x")
        # tier bands behind the points: two pale fills alternate between neighbouring tiers, with a thin
        # boundary line where a tier ends; the point colour is constant, so colour never reads as a variable
        for t in range(1, tier.max() + 1):
            idx = r[tier == t]
            ax.axvspan(idx.min() - 0.5, idx.max() + 0.5, color=tint(TIER_A if t % 2 else TIER_B, 0.10),
                       linewidth=0, zorder=0)
            if t > 1:
                ax.axvline(idx.min() - 0.5, color=tint(INK2, 0.5), linewidth=0.3, zorder=1)
        big = len(sc) > 60
        ax.scatter(r, sc, s=7 if big else 15, c=INK2, edgecolor=WHITE, linewidth=0.3, zorder=3)
        # thin ring on the five best entries
        ax.scatter(r[:5], sc[:5], s=30 if not big else 20, facecolor="none", edgecolor=INK, linewidth=0.6, zorder=4)
        ax.set_xlim(0.5 - 0.02 * len(sc), len(sc) + 0.5 + 0.02 * len(sc))
        ax.set_title(f"{TITLES.get(lb, lb)}", loc="left", fontsize=8, fontweight="bold", pad=11)
        ax.text(0, 1.03, f"n = {d['n']} tasks · {len(sc)} entries · {tier.max()} tiers",
                transform=ax.transAxes, fontsize=FS, color=INK2, ha="left", va="bottom")
        ax.margins(y=0.08)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        step = 50 if len(sc) > 100 else (20 if len(sc) > 40 else (10 if len(sc) > 30 else 5))
        ax.set_xticks([1] + list(range(step, len(sc) + 1, step)))
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)
    for ax in axs[:, 0]:
        ax.set_ylabel("Score (%)")
    for ax in axs[1, :]:
        ax.set_xlabel("Rank (best first)")
    fig.text(0.5, 0.995, "Shading marks tiers (greedy partitions at the bound): an entry joins the current tier while it is not separable from "
             "the tier's best entry.\nAlternating colours distinguish neighbouring tiers.",
             ha="center", va="top", fontsize=FS_L, color=INK, linespacing=1.2)
    handle = Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.6,
                    markersize=5.5, label="top five entries")
    fig.legend(handles=[handle], loc="lower center", frameon=False, handletextpad=0.5, bbox_to_anchor=(0.5, -0.005))
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout(pad=0.4, h_pad=1.3, w_pad=0.9, rect=(0, 0.05, 1, 0.9))
    return save(fig, out, "fig_tiers.pdf", "make_tiers.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PAPER_FIGS)
    print("wrote", make(ap.parse_args().out))


if __name__ == "__main__":
    main()
