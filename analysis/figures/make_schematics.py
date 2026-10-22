# SPDX-License-Identifier: MIT
"""Schematic figures: concept (Sec. 1), seven axes (Sec. 4), visual abstract (front page).

fig_concept.pdf   single column. ILLUSTRATIVE toy leaderboard (invented values, labelled as such on
                  the figure): eight entries, each a score with its item-sampling interval, and tier
                  brackets. No measured data.
fig_axes.pdf      double column. Axes A1-A7 of the systematization (paper/sections/04-systematization.tex,
                  Table "Systematization of reliability measurement"), grouped as in that table into
                  colour-coded families, with the statistic named in the table. No numbers.
fig_overview.pdf  double column visual abstract, four panels with takeaway headlines. Every number is read from result
                  files: census.csv (panel 3) and rank_inversion_summary.json (panel 4); panel 1's whiskers
                  and panel 3's toy chart are illustrative and labelled so on the figure.

Usage: python make_schematics.py [--out ../../paper/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from viz_style import (COL_W, DBL_W, FAM_CONSIST, FAM_FAULT, FAM_REPORT, FAM_ROBUST, FAM_TOOL, FS, FS_L, INK, INK2,
                       LB_RESULTS, PAPER_FIGS, RESULTS, RUN, TASK, TIER_A, TIER_B, WHITE, draw_toy, plt, save, tint)

SCRIPT = "make_schematics.py"


def rbox(ax, x, y, w, h, fc, ec="none", lw=0.0, r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                linewidth=lw, zorder=z))


# --------------------------------------------------------------------------------------- concept
def fig_concept(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    draw_toy(ax)
    ax.text(0.5 * (69 + 79), 7.55, "interval of #1", ha="center", va="bottom", fontsize=FS, color=INK2,
            clip_on=False)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.9, bottom=0.2)
    return save(fig, out, "fig_concept.pdf", SCRIPT)


# --------------------------------------------------------------------------------------- axes
FAMILY_COLOR = {"consistency": FAM_CONSIST, "robustness": FAM_ROBUST, "fault tolerance": FAM_FAULT,
                "tool-call validity": FAM_TOOL, "reporting conditions": FAM_REPORT}
AXES = [  # id, name, key statistic (from the Statistic / Minimum columns of the systematization table)
    ("A1", "Run-to-run\nconsistency", "$\\mathrm{pass}^k$;\nSD over runs"),
    ("A2", "Measurement\nreliability", "ICC;\nG-coefficient"),
    ("A3", "Uncertainty and\ncomparisons", "bootstrap CI;\nflip probability"),
    ("A4", "Perturbation\nrobustness", "degradation $\\Delta$\nwith CI"),
    ("A5", "Fault tolerance\nand recovery", "recovery rate;\ntime to recover"),
    ("A6", "Tool-call validity,\nincluding MCP", "invalid-call rate"),
    ("A7", "Reporting\nconditions", "model version,\nseeds, cost\n(no statistic)"),
]
GROUPS = [("consistency", 0, 2), ("robustness", 3, 3), ("fault tolerance", 4, 4), ("tool-call validity", 5, 5),
          ("reporting conditions", 6, 6)]
FAM_OF = {i: g for g, a, b in GROUPS for i in range(a, b + 1)}


def fig_axes(out: Path) -> Path:
    W, H = DBL_W, 1.8
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    pad, gl = 0.06, 0.5                    # outer padding; left gutter for the shared row label
    g_in, g_out = 0.035, 0.11              # gap inside a family, gap between families
    gaps = [g_in if FAM_OF[i] == FAM_OF[i + 1] else g_out for i in range(6)]
    cw = (W - 2 * pad - gl - sum(gaps)) / 7
    xs = [pad + gl]
    for g in gaps:
        xs.append(xs[-1] + cw + g)
    x0 = lambda i: xs[i]
    ytop = H - 0.06
    PH = 0.36                       # family band height
    for name, a, b in GROUPS:
        col = FAMILY_COLOR[name]
        xl, xr = x0(a), x0(b) + cw
        rbox(ax, xl, ytop - PH, xr - xl, PH, tint(col, 0.06), ec=col, lw=0.8, r=0.13)
        ax.add_patch(Circle((xl + 0.15, ytop - PH / 2), 0.05, fc=col, ec="none", zorder=3))
        label = name if b > a else name.replace(" ", "\n")   # narrow families break onto two lines
        if name == "consistency":
            label = "consistency, A1 to A3"
        ax.text(xl + 0.27, ytop - PH / 2 - 0.005, label, ha="left", va="center", color=INK, fontsize=FS_L,
                fontweight="bold", linespacing=1.05)
    yb, hc = 0.06, H - 0.06 - PH - 0.07 - 0.06
    ystat = yb + 0.62                                   # top of the statistic region, shared by every card
    for i, (aid, name, stat) in enumerate(AXES):
        col = FAMILY_COLOR[FAM_OF[i]]
        xl = x0(i)
        rbox(ax, xl, yb, cw, hc, tint(col, 0.06), ec=tint(col, 0.4), lw=0.5, z=1)
        ax.add_patch(FancyBboxPatch((xl, yb + hc - 0.05), cw, 0.05, boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=col, ec="none", zorder=2))
        ax.text(xl + 0.09, yb + hc - 0.23, aid, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(xl + 0.09, yb + hc - 0.5, name, ha="left", va="center", fontsize=FS, color=INK,
                linespacing=1.15)
        ax.plot([xl + 0.07, xl + cw - 0.07], [ystat, ystat], color=tint(col, 0.5), linewidth=0.5, zorder=3)
        ax.text(xl + 0.09, ystat - 0.07, stat, ha="left", va="top", fontsize=8.2, color=INK,
                linespacing=1.12, zorder=3)
    ax.text(pad + 0.02, ystat - 0.07, "Measure /\nreport", ha="left", va="top", fontsize=FS, color=INK2,
            linespacing=1.12)
    return save(fig, out, "fig_axes.pdf", SCRIPT)


# --------------------------------------------------------------------------------------- overview
def census_headline() -> dict:
    # Round 5 (2026-10-20, review/ra/adjudication-round5.md D1): the panel shows the primary inferential set,
    # the protocol-compatible subset P1 with single-run SEs (robustness_2026-10-20.md section F), not the
    # 22-board as-published diagnostic of census.csv.
    p = pd.read_csv(RESULTS / "robustness_2026-10-20_secF_boards.csv")
    p = p[(p["variant"] == "P1") & (p["se"] == "single")]
    assert len(p) == 21
    return dict(
        n_boards=len(p),
        not_sep_2=int((~p["sep12"].astype(bool)).sum()),
        not_sep_5=int((~p["sep15"].astype(bool)).sum()),
        median_tiers=float(p["n_tiers"].median()),
    )


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, fc=INK2, ec=INK2,
                                 linewidth=0.9, shrinkA=0, shrinkB=0, zorder=5))


def fig_overview(out: Path) -> Path:
    hl = census_headline()
    sh = json.loads((RESULTS / "rank_inversion_summary.json").read_text(encoding="utf-8"))
    n_sub, n_tasks = sh["n_submissions"], sh["n_tasks"]
    spearman = sh["split_half"]["spearman_mean"]
    flip_share = sh["run"]["share_adjacent_flip_gt_5pct"]
    n_flip, n_pairs = sh["run"]["n_adjacent_flip_gt_5pct"], sh["run"]["n_adjacent_pairs"]
    # Panel headlines are claims; each must follow from the values read above.
    assert hl["not_sep_2"] > hl["n_boards"] / 2 and hl["not_sep_5"] > 0   # "many leaders ..."
    assert spearman > 0.9 and flip_share > 0.8                                              # "stable overall; ..."

    W, H = DBL_W, 3.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ag = 0.24
    pw = (W - 0.12 - 3 * ag) / 4
    px = [0.06 + i * (pw + ag) for i in range(4)]
    pb, yt = 0.06, H - 0.06          # panel bottom and top
    T = lambda t: yt - t             # distance from the panel top -> y
    heads = ["One score hides\nuncertainty", "Seven reliability\naxes", "Many leaders remain\ninseparable",
             "Stable overall;\nneighbours flip"]
    subs = ["The question", "Framework", f"Census, {hl['n_boards']} comparable boards", "Terminal-Bench 2.0"]
    for i in range(4):
        rbox(ax, px[i], pb, pw, yt - pb, WHITE, ec=tint(INK2, 0.35), lw=0.5, r=0.09, z=0)
        ax.text(px[i] + 0.1, T(0.1), heads[i], ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK,
                linespacing=1.08, zorder=5)
        ax.text(px[i] + 0.1, T(0.5), subs[i], ha="left", va="top", fontsize=FS, color=INK2, zorder=5)
    for i in range(3):
        arrow(ax, px[i] + pw + 0.04, px[i + 1] - 0.04, H / 2)
    lead = dict(ha="left", va="top", fontsize=FS, color=INK, linespacing=1.15, zorder=5)

    # ---------------- panel 1: one score per entry, hidden uncertainty
    x = px[0]
    sc = [74.0, 72.0, 70.0, 68.0, 62.0]
    ys = [T(0.98 + 0.22 * k) for k in range(5)]
    xs0, xs1 = x + 0.4, x + pw - 0.1
    mp = lambda v: xs0 + (v - 50) / (84 - 50) * (xs1 - xs0)
    for k, (s, y) in enumerate(zip(sc, ys)):
        ax.text(x + 0.12, y, f"#{k + 1}", ha="left", va="center", fontsize=FS, color=INK, zorder=5)
        ax.plot([mp(s - 7.5), mp(s + 7.5)], [y, y], color=TASK, linewidth=1.1, solid_capstyle="butt",
                linestyle=(0, (2, 1.2)), zorder=3)
        ax.plot([mp(s - 3.2), mp(s + 3.2)], [y, y], color=RUN, linewidth=3.0, solid_capstyle="butt", zorder=4)
        ax.plot([mp(s)], [y], marker="o", markersize=3.8, color=INK, markeredgecolor=WHITE, markeredgewidth=0.4,
                zorder=5)
    ax.text(x + pw - 0.1, ys[-1] - 0.2, "illustrative values", ha="right", va="center", fontsize=FS, color=INK2,
            style="italic", zorder=5)
    ax.text(x + 0.1, T(2.28), "Hidden behind each score:", ha="left", va="center", fontsize=FS, color=INK,
            fontweight="bold", zorder=5)
    ax.plot([x + 0.12, x + 0.38], [T(2.5), T(2.5)], color=TASK, linewidth=1.1, linestyle=(0, (2, 1.2)), zorder=5)
    ax.text(x + 0.45, T(2.5), "which tasks were drawn", ha="left", va="center", fontsize=FS, color=INK, zorder=5)
    ax.plot([x + 0.12, x + 0.38], [T(2.7), T(2.7)], color=RUN, linewidth=3.0, solid_capstyle="butt", zorder=5)
    ax.text(x + 0.45, T(2.7), "which run was drawn", ha="left", va="center", fontsize=FS, color=INK, zorder=5)
    rbox(ax, x + 0.1, pb + 0.1, pw - 0.2, 0.3, WHITE, ec=INK2, lw=0.7, r=0.12, z=2)
    ax.text(x + pw / 2, pb + 0.25, "Is #1 better than #2?", ha="center", va="center", fontsize=FS_L,
            fontweight="bold", color=INK, zorder=5)

    # ---------------- panel 2: seven axes, coloured by family (same as fig_axes)
    x = px[1]
    ax.text(x + 0.1, T(0.72), "One measure and one\nminimum report per axis.", **lead)
    names = ["Run-to-run consistency", "Measurement reliability", "Uncertainty, comparisons",
             "Perturbation robustness", "Fault tolerance", "Tool-call validity", "Reporting conditions"]
    y0, dy = T(1.3), 0.25
    for i, nm in enumerate(names):
        y = y0 - i * dy
        col = FAMILY_COLOR[FAM_OF[i]]
        rbox(ax, x + 0.1, y - 0.095, 0.29, 0.19, tint(col, 0.28), ec=col, lw=0.8, r=0.04, z=3)
        ax.text(x + 0.245, y - 0.003, f"A{i + 1}", ha="center", va="center", color=INK, fontsize=FS,
                fontweight="bold", zorder=5)
        ax.text(x + 0.46, y, nm, ha="left", va="center", fontsize=FS, color=INK, zorder=5)
    ax.text(x + 0.1, pb + 0.2, "Colour marks the family;\nA1 to A3 form consistency.", ha="left", va="center",
            fontsize=FS, color=INK2, linespacing=1.15, zorder=5)

    # ---------------- panel 3: census
    x = px[2]
    ia = fig.add_axes([(x + 0.12) / W, T(1.95) / H, (pw - 0.55) / W, 0.95 / H])
    draw_toy(ia, label_size=FS, show_xlabel=False, n_rows=8)
    ia.set_facecolor("none")
    ia.set_xticks([])
    ia.tick_params(axis="x", length=0)
    ia.set_yticks([])
    ax.text(x + 0.1, T(2.08), "illustrative toy example", ha="left", va="center", fontsize=FS, color=INK2,
            style="italic", zorder=5)
    cols = [(f"{hl['not_sep_2']}/{hl['n_boards']}", "#1 not\nseparable\nfrom #2"),
            (f"{hl['not_sep_5']}/{hl['n_boards']}", "#1 not\nseparable\nfrom #5")]
    for i, (big, lab) in enumerate(cols):
        xx = x + 0.1 + i * (pw - 0.1) / 2
        ax.text(xx, T(2.4), big, ha="left", va="center", fontsize=16, fontweight="bold", color=TASK, zorder=5)
        ax.text(xx, T(2.57), lab, ha="left", va="top", fontsize=FS, color=INK, linespacing=1.1, zorder=5)
    ax.text(x + 0.1, pb + 0.2, f"Median leaderboard: {hl['median_tiers']:.0f} tiers", ha="left", va="center",
            fontsize=FS, color=INK, zorder=5)

    # ---------------- panel 4: Terminal-Bench 2.0 multi-run
    x = px[3]
    ax.text(x + 0.1, T(0.68), f"{n_sub} submissions, {n_tasks} tasks,\nrepeated runs.", **lead)
    bw = pw - 0.2

    def meter(y, frac, color, big, lab):
        ax.text(x + 0.1, y, big, ha="left", va="center", fontsize=16, fontweight="bold", color=color, zorder=5)
        ax.text(x + 0.1, y - 0.16, lab, ha="left", va="top", fontsize=FS, color=INK, linespacing=1.1, zorder=5)
        return y

    y1, y2 = T(1.1), T(1.85)
    meter(y1, spearman, INK2, f"{spearman:.2f}", "Spearman correlation of\nrankings from run halves")
    meter(y2, flip_share, RUN, f"{100 * flip_share:.1f}%",
          f"of adjacent pairs have a\n>5% flip probability\n({n_flip}/{n_pairs}, run-to-run)")
    for y, frac, color in ((y1 - 0.52, spearman, INK2), (y2 - 0.62, flip_share, RUN)):
        rbox(ax, x + 0.1, y, bw, 0.07, WHITE, ec=tint(color, 0.4), lw=0.5, r=0.03, z=2)
        rbox(ax, x + 0.1, y, bw * frac, 0.07, color, r=0.03, z=3)
    ry = pb + 0.1
    ax.text(x + 0.1, ry + 0.52, "Report", ha="left", va="center", fontsize=FS, fontweight="bold", color=INK,
            zorder=5)
    for k, txt in enumerate(["tasks $n$ and runs $k$", "the interval and its source", "gaps against the bound"]):
        yy = ry + 0.37 - k * 0.14
        ax.add_patch(Circle((x + 0.14, yy), 0.016, fc=INK2, ec="none", zorder=5))
        ax.text(x + 0.22, yy, txt, ha="left", va="center", fontsize=FS, color=INK, zorder=5)
    return save(fig, out, "fig_overview.pdf", SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PAPER_FIGS)
    a = ap.parse_args()
    for f in (fig_concept, fig_axes, fig_overview):
        print("wrote", f(a.out))


if __name__ == "__main__":
    main()
