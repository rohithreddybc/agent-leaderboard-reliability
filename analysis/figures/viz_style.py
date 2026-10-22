# SPDX-License-Identifier: MIT
"""Shared style for every paper figure: one palette, one set of font sizes, helpers.

Print / light surface only (white page). All figures are vector PDF at IEEE widths:
single column 3.5 in, double column 7.16 in. No text is smaller than 7 pt at print size.

Palette (hues from the dataviz reference categorical palette, light mode; validated with its
validator on a white surface, all-pairs, over the four hues below: worst CVD delta-E 9.2,
worst normal-vision delta-E 16.3; aqua is 2.82:1 on white, so every tier is also named by a direct
label and top-5 entries carry a ring):
    RUN   blue   #2a78d6  run-to-run variation (rerun scheme), and anything about runs
    TASK  orange #eb6834  task-sampling (item-sampling) variation, and anything about tasks
    TIER_A violet #4a3aa7 and TIER_B aqua #1baf7a: odd / even tier in a tier strip (hue carries only
                          the parity of the tier, never a quantity)
Same entity, same hue in every figure. Neutral ink and grey carry text, axes and structure.

Axis families (fig_axes.pdf, and the A1-A7 badges of fig_overview.pdf) reuse three further hues from
the same palette, in palette order, so adjacent families stay separable (validator, adjacent pairs:
violet, aqua, yellow, magenta; worst CVD delta-E 9.1, worst normal-vision delta-E 19.6): consistency
violet, robustness aqua, fault tolerance yellow, tool-call validity magenta, reporting conditions grey.
Families are always named in text; hue never carries the family alone.
Panels and cards use soft tints of the figure's hues (``tint``), never a new hue.
Type: one sans-serif family (Calibri, with Arial / Helvetica / DejaVu Sans fallbacks) in every figure; 7 pt minimum, 7.5 pt labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL_W = 3.5    # IEEE single column, inches
DBL_W = 7.16   # IEEE double column, inches

RUN = "#2a78d6"
TASK = "#eb6834"
TIER_A = "#4a3aa7"
TIER_B = "#1baf7a"
FAM_CONSIST = "#4a3aa7"   # violet
FAM_ROBUST = "#1baf7a"    # aqua
FAM_FAULT = "#eda100"     # yellow
FAM_TOOL = "#e87ba4"      # magenta
FAM_REPORT = "#898781"    # grey
INK = "#1c1c1b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PANEL = "#f3f2ee"
WHITE = "#ffffff"

FS = 7.5       # minimum and default text size (pt); Calibri has a small x-height, so 7.5 pt nominal
FS_L = 8.0     # axis labels, panel titles

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": FS,
    "axes.labelsize": FS_L,
    "axes.titlesize": FS_L,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis" / "results"
LB_RESULTS = REPO / "analysis" / "leaderboards" / "results"
PAPER_FIGS = REPO / "paper" / "figures"


def tint(hex_color: str, a: float) -> str:
    """Mix a colour with white: a = 0 gives white, a = 1 the colour itself (opaque, print-safe)."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(255 - a * (255 - c)):02x}" for c in rgb)


def style_axes(ax, x: bool = False, y: bool = True) -> None:
    if y:
        ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    if x:
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str, script: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, metadata={"CreationDate": None, "Creator": script})
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------
# Greedy tiers, as defined in census_summary.md (Definitions): the best entry opens tier 1, each
# later entry joins the tier while it is not separable from the tier's best entry.
# ---------------------------------------------------------------------------------------------
def pair_bound_pp(p1: float, p2: float, n: int, rho: float) -> float:
    """1.96 sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2) in percentage points (item-sampling, paired)."""
    se1 = np.sqrt(p1 * (1 - p1) / n)
    se2 = np.sqrt(p2 * (1 - p2) / n)
    return 100 * 1.96 * np.sqrt(max(se1**2 + se2**2 - 2 * rho * se1 * se2, 0.0))


def greedy_tiers(scores: np.ndarray, n: int, rho: float) -> np.ndarray:
    """scores in [0,1], sorted best first. A pair is 'below the bound' when gap < bound or tied."""
    tier = np.zeros(len(scores), dtype=int)
    t, best = 1, scores[0]
    for i, s in enumerate(scores):
        gap = 100 * (best - s)
        below = (gap < pair_bound_pp(best, s, n, rho)) or np.isclose(best, s)
        if not below:
            t += 1
            best = s
        tier[i] = t
    return tier


# ---------------------------------------------------------------------------------------------
# Illustrative toy leaderboard (not data). Eight entries with a common item-sampling half-width.
# ---------------------------------------------------------------------------------------------
TOY_SCORES = np.array([74.0, 72.0, 70.0, 68.0, 62.0, 59.0, 50.0, 47.0])
TOY_HALF = 5.0  # +/- half-width of every interval, in score points


def toy_tiers() -> np.ndarray:
    """Same greedy rule as the census with the difference bound sqrt(2) * half-width."""
    bound = np.sqrt(2) * TOY_HALF
    tier = np.zeros(len(TOY_SCORES), dtype=int)
    t, best = 1, TOY_SCORES[0]
    for i, s in enumerate(TOY_SCORES):
        if best - s >= bound:
            t += 1
            best = s
        tier[i] = t
    return tier


def draw_toy(ax, label_size: float = FS, show_xlabel: bool = True, n_rows: int = 8) -> None:
    """Draw the toy leaderboard: score with its interval per entry, tier brackets at the right."""
    sc, tier = TOY_SCORES[:n_rows], toy_tiers()[:n_rows]
    assert tier.max() == 3 and list(tier) == [1, 1, 1, 1, 2, 2, 3, 3][:n_rows]
    y = np.arange(len(sc))[::-1].astype(float)
    top = sc[0]
    ax.axvspan(top - TOY_HALF, top + TOY_HALF, color=PANEL, zorder=0, linewidth=0)
    ax.axvline(top - TOY_HALF, color=MUTED, linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(top + TOY_HALF, color=MUTED, linewidth=0.5, linestyle=":", zorder=1)
    for yi, s, t in zip(y, sc, tier):
        col = TIER_A if t % 2 else TIER_B
        ax.plot([s - TOY_HALF, s + TOY_HALF], [yi, yi], color=col, linewidth=1.6, solid_capstyle="butt", zorder=3)
        ax.plot([s], [yi], marker="o", markersize=3.6, color=col, markeredgecolor=WHITE,
                markeredgewidth=0.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{i + 1}" for i in range(len(sc))], fontsize=label_size)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(sc) - 0.3)
    lo, hi = 38, 84
    ax.set_xlim(lo, hi)
    ax.set_xticks([40, 50, 60, 70, 80])
    xb = hi + 1.4
    for t in sorted(set(tier)):
        idx = np.where(tier == t)[0]
        y0, y1 = y[idx].min() - 0.3, y[idx].max() + 0.3
        ax.plot([xb - 0.8, xb, xb, xb - 0.8], [y0, y0, y1, y1], color=INK2, linewidth=0.7, clip_on=False)
        ax.text(xb + 1.0, (y0 + y1) / 2, f"tier {t}", ha="left", va="center", fontsize=label_size, color=INK,
                clip_on=False)
    if show_xlabel:
        ax.set_xlabel("Score (%), illustrative values")
    ax.spines["left"].set_visible(False)
