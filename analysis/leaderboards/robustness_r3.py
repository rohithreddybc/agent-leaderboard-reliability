# SPDX-License-Identifier: MIT
"""Robustness analyses, round 3 (2026-10-19), for the leaderboard census (bound (a) headline).

New files only: nothing existing is modified. Inputs are the committed census outputs
(``results/census_entries.csv``, ``census_parameters.json``, ``rho_by_board.csv``, ``rho_adjacent_pairs.csv``,
``census.csv``), the per-task outcomes the census already reads (``estimate_rho.LOADERS``, local files only), and
``analysis/results/{tau2_consistency,tb2_within_task_variance,robustness_2026-10-06_mcnemar_pairs}.csv``.
No network access. Per-task outcomes are used in memory; only aggregates are written.

Headline (reproduced first; the script stops if it differs): bound (a) = 1.96 sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2),
SE = sqrt(P(1-P)/n), rho = board's adjacent-pair median (7 boards with per-task data) or the pooled median
0.5218 (15 boards); greedy tiers (Algorithm 1); top-entry checks #1 vs #2 and #1 vs #5.

Sections (numbering = the report ``robustness_2026-10-19.md``):
 1 order confidence Phi(gap / SE_D)          2 parametric bootstrap null (plug-in and empirical-Bayes truth)
 3 rho sweep (incl. 0.9)                     4 leave-one-family-out rho
 5 all-pairs rho instead of adjacent rho     6 k-adjusted bound (a)
 7 Agresti-Caffo / Wilson adjusted SE        8 cluster-robust bound (cluster = repository / task-category prefix)
 9 calibration extension                    10 tau2-bench run-noise share f and the rho <= 1 - f coupling
11 confidence set of possible leaders       12 agreement with prior SWE-bench analyses
13 Cohen's kappa, bound vs paired test      14 systems x tasks variance decomposition and Ep^2
15 pair-specific rho at the top             16 bound (b) at f = 0.369 and feasible f

Seeds (all derived from SEED = 20261019 by ``numpy.random.SeedSequence([SEED, section, ...])``):
section 2 [SEED, 2, board index, truth (0 plug-in, 1 shrinkage)]; section 9 [SEED, 9, n, 100 P, 100 true rho,
100 assumed rho, 0 null / 1 power].

Usage: python robustness_r3.py [--root <project root>] [--out <output dir>] [--boot-reps 2000] [--cal-reps 20000]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
for _p in (str(HERE), str(ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import census as C  # noqa: E402
import estimate_rho as E  # noqa: E402
import robustness as Rb  # noqa: E402

SEED = 20261019
DATE = "2026-10-19"
PREFIX = f"robustness_{DATE}"
Z = C.Z
F_SWEEP = (0.25, 0.369, 0.5, 0.75)
EXPECTED_HEADLINE = {"top1_vs_2_not_sep": 17, "top1_vs_5_not_sep": 9, "tiers_median": 6.0, "adjacent_below_median_pct": 93.9,
                     "adjacent_ge_90": 12}
FAMILIES = {
    "SWE-bench": ["SWE-bench Verified", "SWE-bench Lite", "SWE-bench test (full)", "SWE-bench Multilingual"],
    "Terminal-Bench 1.0": ["Terminal-Bench 1.0"],
    "Terminal-Bench 2.0": ["Terminal-Bench 2.0"],
    "TheAgentCompany": ["TheAgentCompany"],
}


# ============================================================================ boards and the generic evaluator

@dataclass
class Board:
    lb: str
    n: int
    names: list[str]
    scores: np.ndarray      # census order: best first, ties by name
    ks: np.ndarray
    rho: float              # headline rho of the board
    source: str             # "own" | "pooled"

    @property
    def m(self) -> int:
        return len(self.scores)


def load_boards(results_dir: Path) -> tuple[list[Board], dict]:
    """The 22 analysed boards from the committed census files (kept entries only), in census order."""
    ent = pd.read_csv(results_dir / "census_entries.csv", dtype={"entry": str}, keep_default_na=False)
    ent["score"] = pd.to_numeric(ent["score"], errors="coerce")
    ent["k_runs"] = pd.to_numeric(ent["k_runs"], errors="coerce")
    par = json.loads((results_dir / "census_parameters.json").read_text(encoding="utf-8"))
    par["_excluded"] = {lb: int(((ent["leaderboard"] == lb) & (ent["status"] == "excluded")).sum()) for lb in par["rho_used"]}
    boards = []
    for lb in par["rho_used"]:
        d = ent[(ent["leaderboard"] == lb) & (ent["status"] == "kept")]
        names, scores = list(d["entry"]), list(d["score"])
        idx = C.order(scores, names)
        boards.append(Board(lb, C.N_VERIFIED[lb], [names[i] for i in idx], np.array([scores[i] for i in idx], float),
                            np.array([float(d["k_runs"].iloc[i]) for i in idx]), float(par["rho_used"][lb]),
                            par["rho_source"][lb]))
    return boards, par


def evaluate(s, kk, fn) -> dict:
    """Census statistics of one board under one bound. s: scores sorted best first; kk: runs per entry;
    fn(p1, p2, k1, k2, ia, ib) -> bound array (ia, ib = positions in the sorted order). Same rule as
    ``census.census_config``: a pair is below the bound when gap < bound or gap == 0."""
    s, kk = np.asarray(s, float), np.asarray(kk, float)
    m = len(s)
    ar = np.arange(m)
    gaps = s[:-1] - s[1:]
    b_adj = np.asarray(fn(s[:-1], s[1:], kk[:-1], kk[1:], ar[:-1], ar[1:]), float)
    below = (gaps < b_adj) | (gaps == 0)
    ii, jj = np.triu_indices(m, 1)
    ga = s[ii] - s[jj]
    ba = np.asarray(fn(s[ii], s[jj], kk[ii], kk[jj], ii, jj), float)
    below_all = (ga < ba) | (ga == 0)
    out = {"n_entries": m, "n_adjacent_pairs": m - 1, "n_adjacent_below": int(below.sum()),
           "share_adjacent_below": float(below.mean()), "share_all_below": float(below_all.mean()),
           "n_tiers": C.greedy_tiers(s, lambda i, j: float(fn(s[i], s[j], kk[i], kk[j], i, j))),
           "gap12": float(gaps[0]), "bound12": float(b_adj[0]), "top1_vs_2_separable": bool(not below[0]),
           "adj_below": below, "adj_bound": b_adj}
    if m >= 5:
        g15 = s[0] - s[4]
        b15 = float(fn(s[0], s[4], kk[0], kk[4], 0, 4))
        out.update(gap15=float(g15), bound15=b15, top1_vs_5_separable=bool(not (g15 < b15 or g15 == 0)))
    else:
        out.update(gap15=float("nan"), bound15=float("nan"), top1_vs_5_separable=None)
    return out


def fn_a(n: float, rho: float):
    """Bound (a) at a scalar rho."""
    return lambda p1, p2, k1, k2, ia, ib: C.bound_a(p1, p2, n, rho)


def fn_a_pair(n: float, Rs: np.ndarray):
    """Bound (a) with a pair-specific rho matrix (positions in the sorted order)."""
    return lambda p1, p2, k1, k2, ia, ib: C.bound_a(p1, p2, n, Rs[ia, ib])


def fn_b(n: float, f: float, kadj: bool = True):
    """Fixed-task rerun bound (b) at run-noise share f (census.bound_f)."""
    return lambda p1, p2, k1, k2, ia, ib: C.bound_f(p1, p2, n, f, k1 if kadj else 1.0, k2 if kadj else 1.0)


def aggregate(rows: list[dict]) -> dict:
    """Across-board counts and medians as in census_summary.md (``robustness.aggregate_rows``)."""
    return Rb.aggregate_rows([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows])


def evaluate_all(boards: list[Board], fn_for) -> list[dict]:
    """Evaluate every board; fn_for(board) -> bound function."""
    return [{"leaderboard": b.lb, **evaluate(b.scores, b.ks, fn_for(b))} for b in boards]


def board_df(rows: list[dict], extra: dict | None = None) -> pd.DataFrame:
    keep = ["leaderboard", "n_entries", "n_tiers", "n_adjacent_below", "n_adjacent_pairs", "share_adjacent_below",
            "share_all_below", "gap12", "bound12", "top1_vs_2_separable", "gap15", "bound15", "top1_vs_5_separable"]
    df = pd.DataFrame([{k: r[k] for k in keep} for r in rows])
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def ref_rows(boards: list[Board]) -> list[dict]:
    """The headline (bound (a), board rho) for every board."""
    return evaluate_all(boards, lambda b: fn_a(b.n, b.rho))


# ============================================================================ reproduction gate

def reproduce(boards: list[Board], census_csv: Path) -> dict:
    """Headline aggregates and a board-by-board comparison with the committed census.csv (a_emp, primary)."""
    rows = ref_rows(boards)
    agg = aggregate(rows)
    cen = pd.read_csv(census_csv)
    cen = cen[(cen["bound"] == "a_emp") & (cen["set"] == "primary")].set_index("leaderboard")
    same, worst = 0, 0.0
    for r in rows:
        c = cen.loc[r["leaderboard"]]
        ok = (int(c.n_tiers) == r["n_tiers"] and int(c.n_adjacent_below) == r["n_adjacent_below"]
              and bool(c.top1_vs_2_separable) == r["top1_vs_2_separable"] and bool(c.top1_vs_5_separable) == r["top1_vs_5_separable"]
              and int(c.n_entries) == r["n_entries"])
        same += ok
        worst = max(worst, abs(float(c.share_all_below) - r["share_all_below"]))
    ok_all = (agg["top1_vs_2_not_sep"] == EXPECTED_HEADLINE["top1_vs_2_not_sep"]
              and agg["top1_vs_5_not_sep"] == EXPECTED_HEADLINE["top1_vs_5_not_sep"]
              and agg["tiers_median"] == EXPECTED_HEADLINE["tiers_median"]
              and round(100 * agg["adjacent_below_median"], 1) == EXPECTED_HEADLINE["adjacent_below_median_pct"]
              and agg["adjacent_ge_90"] == EXPECTED_HEADLINE["adjacent_ge_90"] and same == len(rows))
    return {"aggregate": agg, "boards_identical": same, "boards": len(rows), "max_abs_diff_share_all_below": worst,
            "reproduced": bool(ok_all)}


# ============================================================================ 1. order confidence

def confidence(gap, se_d):
    """c = Phi(gap / SE_D). SE_D = 0: c = 1 when gap > 0 and 0.5 when gap = 0 (a tie carries no order information)."""
    gap, se_d = np.asarray(gap, float), np.asarray(se_d, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_d > 0, gap / np.where(se_d > 0, se_d, 1.0), np.where(gap > 0, np.inf, 0.0))
    return stats.norm.cdf(z)


def sec1_order_confidence(boards: list[Board]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pair_rows, board_rows = [], []
    for b in boards:
        s, m = b.scores, b.m
        gaps = s[:-1] - s[1:]
        sed = np.asarray(C.bound_a(s[:-1], s[1:], b.n, b.rho)) / Z
        c = confidence(gaps, sed)
        for r_ in range(m - 1):
            pair_rows.append({"leaderboard": b.lb, "pair": "adjacent", "rank_hi": r_ + 1, "rank_lo": r_ + 2,
                              "gap_pp": 100 * gaps[r_], "se_d_pp": 100 * sed[r_], "c": c[r_]})
        g15 = s[0] - s[4]
        sed15 = float(C.bound_a(s[0], s[4], b.n, b.rho)) / Z
        c15 = float(confidence(g15, sed15))
        pair_rows.append({"leaderboard": b.lb, "pair": "#1 vs #5", "rank_hi": 1, "rank_lo": 5, "gap_pp": 100 * g15,
                          "se_d_pp": 100 * sed15, "c": c15})
        board_rows.append({"leaderboard": b.lb, "rho": b.rho, "n_adjacent": m - 1, "c_1v2": float(c[0]), "c_1v5": c15,
                           "share_c_lt_0_8": float((c < 0.8).mean()), "share_c_lt_0_95": float((c < 0.95).mean()),
                           "n_c_lt_0_8": int((c < 0.8).sum()), "n_c_lt_0_95": int((c < 0.95).sum())})
    pairs, bd = pd.DataFrame(pair_rows), pd.DataFrame(board_rows)
    adj = pairs[pairs["pair"] == "adjacent"]
    summ = {"median_c_1v2": float(bd["c_1v2"].median()), "min_c_1v2": float(bd["c_1v2"].min()), "max_c_1v2": float(bd["c_1v2"].max()),
            "median_c_1v5": float(bd["c_1v5"].median()),
            "pooled_share_lt_0.8": float((adj["c"] < 0.8).mean()), "pooled_share_lt_0.95": float((adj["c"] < 0.95).mean()),
            "median_board_share_lt_0.8": float(bd["share_c_lt_0_8"].median()),
            "median_board_share_lt_0.95": float(bd["share_c_lt_0_95"].median()),
            "n_adjacent_pairs": int(len(adj)), "n_1v2_lt_0.95": int((bd["c_1v2"] < 0.95).sum()),
            "n_1v2_lt_0.8": int((bd["c_1v2"] < 0.8).sum())}
    return pairs, bd, summ


# ============================================================================ 2. parametric bootstrap null

def simulate_scores(mu, n: float, rho: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Observed scores X = mu + se * (sqrt(rho) z0 + sqrt(1-rho) z_i): multivariate normal with SD
    se_i = sqrt(mu_i (1 - mu_i) / n) and pairwise correlation rho (0 <= rho < 1), clipped to [0, 1]."""
    mu = np.asarray(mu, float)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    se = np.sqrt(np.clip(mu * (1.0 - mu), 0.0, None) / n)
    z0 = rng.standard_normal((reps, 1))
    zi = rng.standard_normal((reps, len(mu)))
    return np.clip(mu + se * (math.sqrt(rho) * z0 + math.sqrt(1.0 - rho) * zi), 0.0, 1.0)


def eb_shrink(obs, n: float) -> tuple[np.ndarray, float]:
    """Normal-normal empirical-Bayes shrinkage toward the board mean: mu_i = mean + B_i (obs_i - mean),
    B_i = tau2 / (tau2 + SE_i^2), tau2 = max(0, var(obs) - mean(SE^2)), SE_i^2 = obs_i (1 - obs_i) / n."""
    obs = np.asarray(obs, float)
    se2 = obs * (1.0 - obs) / n
    tau2 = max(0.0, float(np.var(obs, ddof=1)) - float(se2.mean()))
    den = tau2 + se2
    B = np.where(den > 0, tau2 / np.where(den > 0, den, 1.0), 0.0)
    return obs.mean() + B * (obs - obs.mean()), tau2


def greedy_tiers_batch(S: np.ndarray, bound_fn) -> np.ndarray:
    """Algorithm 1 for many replicates at once. S: (R, m) scores, each row sorted best first.
    bound_fn(p_head, p_j) -> bound array. Equal to ``census.greedy_tiers`` row by row."""
    R, m = S.shape
    ar = np.arange(R)
    head = np.zeros(R, dtype=int)
    tiers = np.ones(R, dtype=int)
    for j in range(1, m):
        sh, sj = S[ar, head], S[:, j]
        gap = sh - sj
        sep = ~((gap < bound_fn(sh, sj)) | (gap == 0))
        tiers += sep
        head = np.where(sep, j, head)
    return tiers


def bootstrap_board(b: Board, truth: str, reps: int, rng: np.random.Generator) -> dict:
    """Metrics of the parametric bootstrap for one board. truth: 'plugin' (mu = observed) or 'shrink'."""
    tau2 = float("nan")
    if truth == "plugin":
        mu = b.scores.copy()
    else:
        mu, tau2 = eb_shrink(b.scores, b.n)
    X = simulate_scores(mu, b.n, b.rho, reps, rng)
    top = X.argmax(axis=1)                      # ties: lowest position (census order)
    p_set = float((mu[top] == mu.max()).mean())  # simulated #1 is an entry tied for the top true score
    p_strict = float((top == 0).mean())          # simulated #1 is the single entry at census position 1
    mis = (X[:, :-1] < X[:, 1:]).astype(float) + 0.5 * (X[:, :-1] == X[:, 1:])
    exp_mis = float(mis.sum(axis=1).mean())
    S = -np.sort(-X, axis=1)
    tiers = greedy_tiers_batch(S, lambda p1, p2: C.bound_a(p1, p2, b.n, b.rho))
    obs_tiers = evaluate(b.scores, b.ks, fn_a(b.n, b.rho))["n_tiers"]
    q = np.array([np.percentile(tiers, 2.5, method="nearest"), np.median(tiers), np.percentile(tiers, 97.5, method="nearest")])
    return {"leaderboard": b.lb, "truth": truth, "reps": reps, "rho": b.rho, "m": b.m, "tau2": tau2,
            "n_top_tied_true": int((mu == mu.max()).sum()), "p_top_correct": p_set, "p_top_correct_strict": p_strict,
            "exp_misordered_adjacent": exp_mis, "exp_misordered_share": exp_mis / (b.m - 1),
            "tiers_sim_median": float(q[1]), "tiers_sim_q025": float(q[0]), "tiers_sim_q975": float(q[2]),
            "tiers_observed": obs_tiers, "observed_in_95": bool(q[0] <= obs_tiers <= q[2])}


def sec2_bootstrap(boards: list[Board], reps: int) -> pd.DataFrame:
    rows = []
    for bi, b in enumerate(boards):
        for ti, truth in enumerate(("plugin", "shrink")):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, bi, ti]))
            rows.append(bootstrap_board(b, truth, reps, rng))
    return pd.DataFrame(rows)


# ============================================================================ 3. rho sweep, 4. LOFO, 5. all-pairs rho

def summarize_rows(rows: list[dict]) -> dict:
    a = aggregate(rows)
    return {"n12_not_sep": a["top1_vs_2_not_sep"], "n15_not_sep": a["top1_vs_5_not_sep"], "n_boards": a["n_boards"],
            "tiers_median": a["tiers_median"], "tiers_min": a["tiers_min"], "tiers_max": a["tiers_max"],
            "adj_share_median": a["adjacent_below_median"], "adj_share_min": a["adjacent_below_min"],
            "adj_ge_90": a["adjacent_ge_90"], "all_pairs_median": a["all_pairs_below_median"]}


def sec3_rho_sweep(boards: list[Board], rhos=(0.0, 0.25, 0.5, "headline", 0.75, 0.9)) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ, per = [], []
    for r in rhos:
        rows = evaluate_all(boards, (lambda b: fn_a(b.n, b.rho)) if r == "headline" else (lambda b, r=r: fn_a(b.n, r)))
        summ.append({"rho": "headline (board)" if r == "headline" else f"{r:g}", **summarize_rows(rows)})
        per.append(board_df(rows, {"rho_setting": "headline" if r == "headline" else r}))
    return pd.DataFrame(summ), pd.concat(per, ignore_index=True)


def family_rhos(rho_by_board: pd.DataFrame) -> dict:
    """Family value = median of the member boards' adjacent-pair medians."""
    med = rho_by_board.set_index("leaderboard")["rho_adjacent_median"]
    return {f: float(np.median([med[lb] for lb in lbs])) for f, lbs in FAMILIES.items()}


def sec4_lofo(rho_by_board: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    fam = family_rhos(rho_by_board)
    rows = []
    for f, v in fam.items():
        pred = float(np.median([x for g, x in fam.items() if g != f]))
        rows.append({"family": f, "n_boards": len(FAMILIES[f]), "rho_family": v, "rho_predicted_by_others": pred,
                     "abs_error": abs(pred - v)})
    fdf = pd.DataFrame(rows)
    med = rho_by_board.set_index("leaderboard")["rho_adjacent_median"]
    brow = []
    for f, lbs in FAMILIES.items():
        others = [med[lb] for g, gl in FAMILIES.items() if g != f for lb in gl]
        pred = float(np.median(others))
        for lb in lbs:
            brow.append({"leaderboard": lb, "family": f, "rho_board": float(med[lb]), "rho_predicted_by_other_boards": pred,
                         "abs_error": abs(pred - float(med[lb]))})
    return fdf, pd.DataFrame(brow), float(np.median(list(fam.values())))


def rho_with_pooled(boards: list[Board], pooled: float) -> dict:
    """Headline rho per board with the pooled value replaced (own-rho boards unchanged)."""
    return {b.lb: (b.rho if b.source == "own" else pooled) for b in boards}


def sec5_allpairs(boards: list[Board], rho_by_board: pd.DataFrame, pooled_adj: float) -> tuple[pd.DataFrame, dict]:
    ap = rho_by_board.set_index("leaderboard")["rho_all_pairs_median"].dropna()
    ap_pooled = float(np.median(ap.values))
    variants = {
        "all-pairs (own); pooled adjacent 0.5218 for the other 15": {b.lb: float(ap[b.lb]) if b.lb in ap.index else pooled_adj for b in boards},
        "all-pairs (own); pooled all-pairs median for the other 15": {b.lb: float(ap[b.lb]) if b.lb in ap.index else ap_pooled for b in boards},
    }
    base = ref_rows(boards)
    out_rows, summ = [], {"all_pairs_pooled": ap_pooled}
    for name, rmap in variants.items():
        rows = evaluate_all(boards, lambda b, rmap=rmap: fn_a(b.n, rmap[b.lb]))
        summ[name] = summarize_rows(rows)
        for b0, r in zip(base, rows):
            out_rows.append({"variant": name, "leaderboard": r["leaderboard"], "rho_used": rmap[r["leaderboard"]],
                             "tiers_headline": b0["n_tiers"], "tiers_variant": r["n_tiers"],
                             "tiers_change": r["n_tiers"] - b0["n_tiers"],
                             "adj_below_headline": b0["share_adjacent_below"], "adj_below_variant": r["share_adjacent_below"],
                             "sep12_headline": b0["top1_vs_2_separable"], "sep12_variant": r["top1_vs_2_separable"],
                             "sep15_headline": b0["top1_vs_5_separable"], "sep15_variant": r["top1_vs_5_separable"]})
    return pd.DataFrame(out_rows), summ


# ============================================================================ 6. k-adjusted bound (a)

def bound_a_kadj(p1, p2, k1, k2, n: float, rho: float, f: float, literal: bool = False):
    """Bound (a) with each entry's SE multiplied by a_j = sqrt(1 - f (1 - 1/k_j)) (variance of a k-run mean is
    P(1-P)(1 - f(1-1/k)) because v = f P(1-P)). rho is defined with the single-run variance, so by default the
    covariance term keeps the single-run SEs; ``literal=True`` scales the SEs in all three terms."""
    s1, s2 = C.se(p1, n), C.se(p2, n)
    a1 = np.sqrt(1.0 - f * (1.0 - 1.0 / np.maximum(np.asarray(k1, float), 1.0)))
    a2 = np.sqrt(1.0 - f * (1.0 - 1.0 / np.maximum(np.asarray(k2, float), 1.0)))
    cross = s1 * a1 * s2 * a2 if literal else s1 * s2
    var = (s1 * a1) ** 2 + (s2 * a2) ** 2 - 2.0 * rho * cross
    return Z * np.sqrt(np.maximum(var, 0.0))


def compare(boards: list[Board], alt_rows: list[dict], base_rows: list[dict] | None = None) -> pd.DataFrame:
    """Board-by-board comparison of an alternative with the headline."""
    base_rows = base_rows or ref_rows(boards)
    out = []
    for b, a0, a1 in zip(boards, base_rows, alt_rows):
        ch = a0["adj_below"] != a1["adj_below"]
        out.append({"leaderboard": b.lb, "tiers_headline": a0["n_tiers"], "tiers_alt": a1["n_tiers"],
                    "adj_below_headline": a0["n_adjacent_below"], "adj_below_alt": a1["n_adjacent_below"],
                    "adj_pairs": a0["n_adjacent_pairs"], "adj_verdicts_changed": int(ch.sum()),
                    "sep12_headline": a0["top1_vs_2_separable"], "sep12_alt": a1["top1_vs_2_separable"],
                    "sep15_headline": a0["top1_vs_5_separable"], "sep15_alt": a1["top1_vs_5_separable"],
                    "gap12_pp": 100 * a0["gap12"], "bound12_headline_pp": 100 * a0["bound12"], "bound12_alt_pp": 100 * a1["bound12"]})
    return pd.DataFrame(out)


def compare_summary(cmp: pd.DataFrame, alt_rows: list[dict]) -> dict:
    s = summarize_rows(alt_rows)
    s.update(boards_sep12_changed=int((cmp["sep12_headline"] != cmp["sep12_alt"]).sum()),
             boards_sep15_changed=int((cmp["sep15_headline"] != cmp["sep15_alt"]).sum()),
             boards_tiers_changed=int((cmp["tiers_headline"] != cmp["tiers_alt"]).sum()),
             adj_verdicts_changed=int(cmp["adj_verdicts_changed"].sum()), adj_pairs=int(cmp["adj_pairs"].sum()),
             boards_with_adj_change=int((cmp["adj_verdicts_changed"] > 0).sum()))
    return s


def sec6_kadj(boards: list[Board], f: float) -> tuple[pd.DataFrame, dict]:
    base = ref_rows(boards)
    out, summ = [], {"headline": summarize_rows(base)}
    for name, lit in (("covariance term single-run (consistent with rho)", False), ("SE scaled in all terms (literal)", True)):
        rows = evaluate_all(boards, lambda b, lit=lit: (lambda p1, p2, k1, k2, ia, ib: bound_a_kadj(p1, p2, k1, k2, b.n, b.rho, f, lit)))
        cmp = compare(boards, rows, base)
        cmp.insert(0, "variant", name)
        cmp["max_k"] = [float(b.ks.max()) for b in boards]
        out.append(cmp)
        summ[name] = compare_summary(cmp, rows)
    return pd.concat(out, ignore_index=True), summ


# ============================================================================ 7. adjusted standard errors

def adjusted(P, n: float, kind: str) -> tuple[np.ndarray, float]:
    """(P~, n~) for the Agresti-Caffo adjustment (one success and one failure added: (x+1)/(n+2), n+2) or the
    Wilson-centre adjustment (z^2/2 successes and failures added: (x + z^2/2)/(n + z^2), n + z^2), x = P n."""
    P = np.asarray(P, float)
    add = 2.0 if kind == "ac" else Z * Z
    return (P * n + add / 2.0) / (n + add), n + add


def fn_a_adjusted(n: float, rho: float, kind: str, adjust_gap: bool = False):
    """Bound (a) with SEs from the adjusted (P~, n~). The gap stays the observed gap unless ``adjust_gap`` (then the
    caller passes adjusted scores and the bound is evaluated at them)."""
    def fn(p1, p2, k1, k2, ia, ib):
        if adjust_gap:
            _, nt = adjusted(0.0, n, kind)
            return C.bound_a(p1, p2, nt, rho)
        a1, nt = adjusted(p1, n, kind)
        a2, _ = adjusted(p2, n, kind)
        return C.bound_a(a1, a2, nt, rho)
    return fn


def sec7_adjusted(boards: list[Board]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base = ref_rows(boards)
    out, summ = [], {"headline": summarize_rows(base)}
    variants = {"Agresti-Caffo, observed gap": ("ac", False), "Agresti-Caffo, adjusted gap": ("ac", True),
                "Wilson centre, observed gap": ("wilson", False), "Wilson centre, adjusted gap": ("wilson", True)}
    for name, (kind, adj_gap) in variants.items():
        rows = []
        for b in boards:
            s = adjusted(b.scores, b.n, kind)[0] if adj_gap else b.scores
            rows.append({"leaderboard": b.lb, **evaluate(s, b.ks, fn_a_adjusted(b.n, b.rho, kind, adj_gap))})
        cmp = compare(boards, rows, base)
        cmp.insert(0, "variant", name)
        out.append(cmp)
        summ[name] = compare_summary(cmp, rows)
    ext = pd.DataFrame([{"leaderboard": b.lb, "entries_at_0": int((b.scores == 0).sum()), "entries_at_1": int((b.scores == 1).sum()),
                         "top_score": float(b.scores[0])} for b in boards])
    return pd.concat(out, ignore_index=True), summ, ext


# ============================================================================ 8. cluster-robust bound

def cluster_se_diff(y1, y2, labels) -> float:
    """Cluster-robust SE of the mean per-task difference d = y1 - y2:
    sqrt( G/(G-1) * sum_g (sum_{t in g} (d_t - dbar))^2 ) / n. NaN for a single cluster."""
    d = np.asarray(y1, float) - np.asarray(y2, float)
    codes = pd.factorize(np.asarray(labels))[0]
    G, n = int(codes.max()) + 1, len(d)
    if G < 2:
        return float("nan")
    u = np.bincount(codes, weights=d - d.mean())
    return math.sqrt(G / (G - 1.0) * float((u ** 2).sum())) / n


def paired_iid_se(y1, y2) -> float:
    """Binomial-paired SE at the pair's own empirical rho: sqrt(Var_pop(d) / n)."""
    d = np.asarray(y1, float) - np.asarray(y2, float)
    return math.sqrt(float(np.mean((d - d.mean()) ** 2)) / len(d))


def swebench_ids(root: Path, split: str, n: int) -> list[str]:
    d = json.loads((root / "data/raw/leaderboards/swe-bench" / f"swebench_{split}.json").read_text(encoding="utf-8"))
    full = [frozenset(r["covered_ids"]) for r in d.values() if r["source"] and len(r["covered_ids"]) == n]
    return sorted(full[0])


def tac_ids(root: Path) -> list[str]:
    t = json.loads((root / "data/raw/leaderboards/theagentcompany/tac_results.json").read_text(encoding="utf-8"))["results"]["1.0.0"]
    for tasks in t.values():
        by = {x.replace("-image", "") for x in tasks}
        if len(by) == 175:
            return sorted(by)
    raise ValueError("no TheAgentCompany entry with 175 tasks")


def repo_prefix(instance_id: str) -> str:
    """Repository of a SWE-bench instance id: 'django__django-12345' -> 'django__django'."""
    return re.sub(r"-\d+$", "", instance_id)


CLUSTER_BOARDS = {
    "SWE-bench Verified": ("verified", "repository prefix of the instance id"),
    "SWE-bench Lite": ("lite", "repository prefix of the instance id"),
    "SWE-bench test (full)": ("test", "repository prefix of the instance id"),
    "SWE-bench Multilingual": ("multilingual", "repository prefix of the instance id"),
    "TheAgentCompany": (None, "task-name prefix before the first hyphen (department-like category; heuristic)"),
}


def cluster_labels(root: Path, lb: str) -> list[str]:
    split = CLUSTER_BOARDS[lb][0]
    if split is None:
        return [t.split("-")[0] for t in tac_ids(root)]
    return [repo_prefix(t) for t in swebench_ids(root, split, C.N_VERIFIED[lb])]


def sec8_cluster(boards: list[Board], outcomes: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prow, brow = [], []
    for b in boards:
        if b.lb not in CLUSTER_BOARDS:
            continue
        P = np.vstack([outcomes[b.lb][nm] for nm in b.names])
        labels = cluster_labels(root, b.lb)
        assert len(labels) == P.shape[1], f"{b.lb}: {len(labels)} labels for {P.shape[1]} tasks"
        G = len(set(labels))
        tcrit = float(stats.t.ppf(0.975, G - 1)) if G > 1 else float("nan")
        rows = []
        for i in range(b.m - 1):
            gap = b.scores[i] - b.scores[i + 1]
            se_cl = cluster_se_diff(P[i], P[i + 1], labels)
            se_iid = paired_iid_se(P[i], P[i + 1])
            se_hl = float(C.bound_a(b.scores[i], b.scores[i + 1], b.n, b.rho)) / Z
            rows.append({"leaderboard": b.lb, "rank_hi": i + 1, "gap_pp": 100 * gap, "se_cluster_pp": 100 * se_cl,
                         "se_paired_iid_pp": 100 * se_iid, "se_headline_pp": 100 * se_hl,
                         "ratio_cluster_over_iid": se_cl / se_iid if se_iid > 0 else float("nan"),
                         "ratio_cluster_over_headline": se_cl / se_hl if se_hl > 0 else float("nan"),
                         "headline_separable": bool(not (gap < Z * se_hl or gap == 0)),
                         "cluster_separable": bool(gap > 0 and gap >= Z * se_cl),
                         "cluster_separable_t": bool(gap > 0 and gap >= tcrit * se_cl)})
        pr = pd.DataFrame(rows)
        prow.append(pr)
        # tiers under the cluster-robust bound (head vs entry j)
        cache = {}

        def bound_ij(i, j):
            if (i, j) not in cache:
                cache[(i, j)] = Z * cluster_se_diff(P[i], P[j], labels)
            return cache[(i, j)]
        tiers_cl = C.greedy_tiers(b.scores, bound_ij)
        base = evaluate(b.scores, b.ks, fn_a(b.n, b.rho))
        g15 = b.scores[0] - b.scores[4]
        sep15_cl = bool(g15 > 0 and g15 >= bound_ij(0, 4))
        brow.append({"leaderboard": b.lb, "cluster_label": CLUSTER_BOARDS[b.lb][1], "n_clusters": G,
                     "min_cluster_size": int(pd.Series(labels).value_counts().min()),
                     "max_cluster_size": int(pd.Series(labels).value_counts().max()), "n_adjacent": len(pr),
                     "median_ratio_cluster_over_iid": float(pr["ratio_cluster_over_iid"].median()),
                     "median_ratio_cluster_over_headline": float(pr["ratio_cluster_over_headline"].median()),
                     "adj_separable_headline": int(pr["headline_separable"].sum()),
                     "adj_separable_cluster": int(pr["cluster_separable"].sum()),
                     "adj_separable_cluster_t": int(pr["cluster_separable_t"].sum()),
                     "adj_verdicts_changed": int((pr["headline_separable"] != pr["cluster_separable"]).sum()),
                     "sep12_headline": bool(pr["headline_separable"].iloc[0]), "sep12_cluster": bool(pr["cluster_separable"].iloc[0]),
                     "sep15_headline": base["top1_vs_5_separable"], "sep15_cluster": sep15_cl,
                     "tiers_headline": base["n_tiers"], "tiers_cluster": tiers_cl})
    return pd.concat(prow, ignore_index=True), pd.DataFrame(brow)


# ============================================================================ 9. calibration extension

CAL_NS = (30, 50, 100, 500, 2294)
CAL_PS = (0.05, 0.2, 0.5, 0.8, 0.95)
CAL_RHOS = (0.25, 0.5, 0.75)


def assumed_rhos(rho_true: float) -> list[tuple[str, float]]:
    """Assumed rho = true, true - 0.25, true + 0.25, clipped to [0, 0.95]."""
    clip = lambda x: float(min(max(x, 0.0), 0.95))
    return [("true", clip(rho_true)), ("true-0.25", clip(rho_true - 0.25)), ("true+0.25", clip(rho_true + 0.25))]


def calibration_cell(n: int, P: float, rho_true: float, rho_assumed: float, reps: int, seed: int) -> dict:
    """False-separation rate at a true difference of 0 and power at a true difference equal to the bound the analyst
    computes (delta = 1.96 sqrt(2 P (1-P) (1 - rho_assumed) / n)); outcomes follow rho_true, the rule uses rho_assumed."""
    key = [seed, 9, n, int(round(P * 100)), int(round(rho_true * 100)), int(round(rho_assumed * 100))]
    fs = Rb.simulate_cell(n, P, P, rho_true, reps, np.random.default_rng(np.random.SeedSequence(key + [0])), rho_assumed)
    delta = Rb.nominal_bound(n, P, rho_assumed)
    hi, lo = P + delta / 2, P - delta / 2
    pw = float("nan")
    if lo >= 0.0 and hi <= 1.0:
        try:
            pw = Rb.simulate_cell(n, hi, lo, rho_true, reps, np.random.default_rng(np.random.SeedSequence(key + [1])), rho_assumed)
        except ValueError:
            pw = float("nan")
    return {"n": n, "P": P, "rho_true": rho_true, "rho_assumed": rho_assumed, "reps": reps, "false_separation_rate": fs,
            "mc_se_null": math.sqrt(max(fs * (1 - fs), 1e-12) / reps), "delta_pp": 100 * delta, "power_at_bound": pw,
            "mc_se_power": (math.sqrt(pw * (1 - pw) / reps) if not math.isnan(pw) else float("nan"))}


def sec9_calibration(reps: int, seed: int = SEED, ns=CAL_NS, Ps=CAL_PS, rhos=CAL_RHOS) -> pd.DataFrame:
    rows = []
    for n in ns:
        for P in Ps:
            for rt in rhos:
                for rel, ra in assumed_rhos(rt):
                    rows.append({"relation": rel, **calibration_cell(n, P, rt, ra, reps, seed)})
    return pd.DataFrame(rows)


# ============================================================================ 10. tau2-bench run-noise share, coupling

def tau2_f(df: pd.DataFrame) -> pd.DataFrame:
    """f = (pass^1 - pass^2) / (pass^1 (1 - pass^1)) per submission x domain where pass^2 exists (pass values in
    percent). Rows with pass^1 at 0 or 100 are undefined and dropped. pass^1 - pass^2 = E[p(1-p)] = v."""
    d = df.copy()
    p1, p2 = d["pass_1"] / 100.0, d["pass_2"] / 100.0
    d["f"] = (p1 - p2) / (p1 * (1.0 - p1))
    d.loc[(p1 <= 0) | (p1 >= 1) | p2.isna(), "f"] = np.nan
    return d


def feasible(rho: float, f1: float, f2: float | None = None) -> bool:
    """Coupling bound rho <= sqrt((1 - f1)(1 - f2)); with one f for both entries, rho <= 1 - f."""
    f2 = f1 if f2 is None else f2
    return bool(rho <= math.sqrt((1.0 - f1) * (1.0 - f2)) + 1e-12)


def sec10_tau2(results_dir: Path, boards: list[Board]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t = tau2_f(pd.read_csv(results_dir / "tau2_consistency.csv"))
    ok = t.dropna(subset=["f"])
    by = ok.groupby("domain")["f"].agg(["count", "median", "min", "max"])
    q = lambda x, p: float(np.percentile(x, p))
    stated = t["trials_stated"].notna()
    summ = {"n_rows_total": int(len(t)), "n_rows_with_pass2": int(t["pass_2"].notna().sum()), "n_rows_f": int(len(ok)),
            "median_f": float(ok["f"].median()), "q25_f": q(ok["f"], 25), "q75_f": q(ok["f"], 75),
            "min_f": float(ok["f"].min()), "max_f": float(ok["f"].max()),
            "n_rows_trials_stated": int(stated.sum()),
            "n_rows_f_trials_stated": int((stated & t["f"].notna()).sum()),
            "trials_stated_values": sorted({str(x) for x in t.loc[stated, "trials_stated"]}),
            "median_f_stated": float(ok.loc[ok["trials_stated"].notna(), "f"].median()) if ok["trials_stated"].notna().any() else float("nan"),
            "median_f_not_stated": float(ok.loc[ok["trials_stated"].isna(), "f"].median()) if ok["trials_stated"].isna().any() else float("nan"),
            "by_domain": by.reset_index().to_dict("records")}
    w = pd.read_csv(results_dir / "tb2_within_task_variance.csv")["f_run_noise_share"].dropna()
    summ.update(tb2_median_f=float(w.median()), tb2_q25_f=q(w, 25), tb2_q75_f=q(w, 75), tb2_min_f=float(w.min()),
                tb2_max_f=float(w.max()), tb2_n=int(len(w)))
    rows = []
    for b in boards:
        for f in F_SWEEP:
            rows.append({"leaderboard": b.lb, "rho": b.rho, "rho_source": b.source, "f": f, "max_rho_allowed": 1.0 - f,
                         "feasible": feasible(b.rho, f)})
    return t, pd.DataFrame(rows), summ


# ============================================================================ 11. confidence set of possible leaders

def leader_set(s, n: float, rho: float, z: float):
    """Positions (excluding #1) whose gap to #1 is below z * SE_D (or zero)."""
    s = np.asarray(s, float)
    gap = s[0] - s[1:]
    bd = Rb.bound_a_z(s[0], s[1:], n, rho, z)
    return (gap < bd) | (gap == 0)


def sec11_leaders(boards: list[Board]) -> pd.DataFrame:
    rows = []
    for b in boards:
        zb = Rb.z_bonferroni(b.m - 1)
        row = {"leaderboard": b.lb, "m": b.m, "z_bonferroni": zb}
        for lab, z in (("uncorrected", Z), ("bonferroni", zb)):
            k = int(leader_set(b.scores, b.n, b.rho, z).sum())
            row[f"others_{lab}"] = k
            row[f"set_size_{lab}"] = k + 1
            row[f"set_share_{lab}"] = (k + 1) / b.m
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ 12. SWE-bench Verified vs prior analyses

def sec12_prior(boards: list[Board]) -> dict:
    b = next(x for x in boards if x.lb == "SWE-bench Verified")
    s, ks = b.scores, b.ks
    out = {}
    for lab, rho in (("headline", b.rho), ("rho0", 0.0)):
        ev = evaluate(s[:30], ks[:30], fn_a(b.n, rho))
        adj_sep = int((~ev["adj_below"]).sum())
        ev10 = evaluate(s[:10], ks[:10], fn_a(b.n, rho))
        out[lab] = {"rho": rho, "top30_adjacent_pairs": 29, "top30_adjacent_separable": adj_sep,
                    "top30_adjacent_below": 29 - adj_sep, "top10_pairs": 45,
                    "top10_all_below": int(round(ev10["share_all_below"] * 45)), "top30_tiers": ev["n_tiers"],
                    "top10_tiers": ev10["n_tiers"], "top1_score": float(s[0]), "top30_score": float(s[29]), "top10_score": float(s[9])}
    return out


# ============================================================================ 13. kappa

def cohen_kappa(a, b) -> dict:
    """Cohen's kappa between two boolean verdict vectors."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": n, "agree": po, "expected": pe, "kappa": kappa}


def sec13_kappa(pairs_csv: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(pairs_csv)
    out, rows = {}, []
    for lab, sub in (("adjacent pairs", df[df["pair"] == "adjacent"]), ("adjacent + #1 vs #5", df)):
        a, t = sub["bound_separable"].astype(bool).to_numpy(), sub["test_separable"].astype(bool).to_numpy()
        k = cohen_kappa(a, t)
        either = a | t
        k.update(n_both=int((a & t).sum()), n_bound_only=int((a & ~t).sum()), n_test_only=int((~a & t).sum()),
                 n_neither=int((~a & ~t).sum()), n_either=int(either.sum()),
                 agree_among_either=float((a & t).sum() / either.sum()) if either.any() else float("nan"))
        out[lab] = k
    adj = df[df["pair"] == "adjacent"]
    for lb, g in adj.groupby("leaderboard", sort=False):
        a, t = g["bound_separable"].astype(bool).to_numpy(), g["test_separable"].astype(bool).to_numpy()
        k = cohen_kappa(a, t)
        either = a | t
        rows.append({"leaderboard": lb, "pairs": k["n"], "agree": k["agree"], "kappa": k["kappa"],
                     "n_bound_sep": int(a.sum()), "n_test_sep": int(t.sum()), "n_both": int((a & t).sum()),
                     "agree_among_either": float((a & t).sum() / either.sum()) if either.any() else float("nan")})
    return pd.DataFrame(rows), out


# ============================================================================ 14. variance decomposition, 15. pair rho

def two_way_components(Y: np.ndarray) -> dict:
    """Two-way random-effects (systems x tasks, one observation per cell) variance components by ANOVA moments, and
    the relative generalizability coefficient for ranking systems with the n tasks of the design:
    Ep2 = var_s / (var_s + var_res / n) = (MS_s - MS_res) / MS_s. Negative components are set to 0."""
    Y = np.asarray(Y, float)
    m, n = Y.shape
    g, rm, cm = Y.mean(), Y.mean(axis=1), Y.mean(axis=0)
    ss_s, ss_t = n * float(((rm - g) ** 2).sum()), m * float(((cm - g) ** 2).sum())
    ss_r = float(((Y - rm[:, None] - cm[None, :] + g) ** 2).sum())
    ms_s, ms_t, ms_r = ss_s / (m - 1), ss_t / (n - 1), ss_r / ((m - 1) * (n - 1))
    vs, vt, vr = max(0.0, (ms_s - ms_r) / n), max(0.0, (ms_t - ms_r) / m), ms_r
    tot = vs + vt + vr
    ep2 = vs / (vs + vr / n) if (vs + vr / n) > 0 else float("nan")
    return {"m": m, "n": n, "var_systems": vs, "var_tasks": vt, "var_residual": vr, "share_systems": vs / tot,
            "share_tasks": vt / tot, "share_residual": vr / tot, "ep2": ep2, "ms_systems": ms_s, "ms_residual": ms_r}


def sec14_decomposition(boards: list[Board], outcomes: dict, base_rows: list[dict], conf: pd.DataFrame, excluded: dict | None = None) -> pd.DataFrame:
    rows = []
    for b, r in zip(boards, base_rows):
        if b.lb not in outcomes:
            continue
        Y = np.vstack([outcomes[b.lb][nm] for nm in b.names])
        assert not np.isnan(Y).any(), f"{b.lb}: missing cells"
        d = two_way_components(Y)
        top = two_way_components(Y[:10])
        rows.append({"leaderboard": b.lb, "systems": d["m"], "tasks": d["n"], "census_excluded_entries": (excluded or {}).get(b.lb, 0),
                     "mean_runs": float(b.ks.mean()),
                     "cell": "binary outcome" if b.ks.max() == 1 else "share of runs", **{k: d[k] for k in
                     ("var_systems", "var_tasks", "var_residual", "share_systems", "share_tasks", "share_residual", "ep2")},
                     "ep2_top10": top["ep2"], "top1_vs_2_separable": r["top1_vs_2_separable"],
                     "c_1v2": float(conf.set_index("leaderboard").loc[b.lb, "c_1v2"]),
                     "share_adjacent_below": r["share_adjacent_below"]})
    return pd.DataFrame(rows)


def sec15_pair_rho(boards: list[Board], matrices: dict, rho_by_board: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = ref_rows(boards)
    rows, new_rows = [], []
    allp = rho_by_board.set_index("leaderboard")["rho_all_pairs_median"]
    for b, a0 in zip(boards, base):
        if b.lb in matrices:
            R = matrices[b.lb]
            r12, r15 = float(R[0, 1]), float(R[0, 4])
            r12u, r15u = (b.rho if np.isnan(r12) else r12), (b.rho if np.isnan(r15) else r15)
        else:
            r12 = r15 = float("nan")
            r12u = r15u = b.rho
        s, kk = b.scores, b.ks
        b12 = float(C.bound_a(s[0], s[1], b.n, r12u))
        b15 = float(C.bound_a(s[0], s[4], b.n, r15u))
        g12, g15 = s[0] - s[1], s[0] - s[4]
        sep12 = bool(not (g12 < b12 or g12 == 0))
        sep15 = bool(not (g15 < b15 or g15 == 0))
        rows.append({"leaderboard": b.lb, "has_pair_rho": b.lb in matrices, "rho_board_adjacent_median": b.rho,
                     "rho_all_pairs_median": float(allp.get(b.lb, np.nan)), "rho_pair_1v2": r12, "rho_pair_1v5": r15,
                     "pair_1v2_above_board": bool(r12 > b.rho) if not np.isnan(r12) else None,
                     "pair_1v5_above_board": bool(r15 > b.rho) if not np.isnan(r15) else None,
                     "bound12_headline_pp": 100 * a0["bound12"], "bound12_pair_pp": 100 * b12,
                     "bound15_headline_pp": 100 * a0["bound15"], "bound15_pair_pp": 100 * b15,
                     "sep12_headline": a0["top1_vs_2_separable"], "sep12_pair": sep12,
                     "sep15_headline": a0["top1_vs_5_separable"], "sep15_pair": sep15})
    df = pd.DataFrame(rows)
    own = df[df["has_pair_rho"]]
    summ = {"n_boards_pair_rho": int(len(own)), "n_1v2_above_board": int(own["pair_1v2_above_board"].sum()),
            "n_1v5_above_board": int(own["pair_1v5_above_board"].sum()),
            "n12_not_sep_headline": int((~df["sep12_headline"]).sum()), "n12_not_sep_pair": int((~df["sep12_pair"]).sum()),
            "n15_not_sep_headline": int((~df["sep15_headline"]).sum()), "n15_not_sep_pair": int((~df["sep15_pair"]).sum()),
            "n_boards": int(len(df)), "median_diff_1v2": float((own["rho_pair_1v2"] - own["rho_board_adjacent_median"]).median()),
            "median_diff_1v5": float((own["rho_pair_1v5"] - own["rho_board_adjacent_median"]).median())}
    return df, summ


# ============================================================================ 16. bound (b)

def sec16_rerun(boards: list[Board], f_values: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """f_values: label -> f (float) or a callable board -> f (e.g. the largest feasible f, 1 - rho)."""
    rows, summ = [], []
    for lab, f_ in f_values.items():
        fb = {b.lb: (f_(b) if callable(f_) else f_) for b in boards}
        f = f_ if not callable(f_) else float("nan")
        for form, kadj in (("k-adjusted", True), ("single-run", False)):
            ev = evaluate_all(boards, lambda b, kadj=kadj: fn_b(b.n, fb[b.lb], kadj))
            feas = np.array([feasible(b.rho, fb[b.lb]) for b in boards])
            ns12 = np.array([not r["top1_vs_2_separable"] for r in ev])
            ns15 = np.array([not r["top1_vs_5_separable"] for r in ev])
            for b, r, fe in zip(boards, ev, feas):
                rows.append({"f_label": lab, "f": fb[b.lb], "form": form, "leaderboard": b.lb, "rho": b.rho, "feasible": bool(fe),
                             "n_tiers": r["n_tiers"], "sep12": r["top1_vs_2_separable"], "sep15": r["top1_vs_5_separable"],
                             "share_adjacent_below": r["share_adjacent_below"]})
            summ.append({"f_label": lab, "f": f, "form": form, "boards": len(boards), "n12_not_sep_all": int(ns12.sum()),
                         "n15_not_sep_all": int(ns15.sum()), "boards_feasible": int(feas.sum()),
                         "n12_not_sep_feasible": int(ns12[feas].sum()), "n15_not_sep_feasible": int(ns15[feas].sum()),
                         "tiers_median_all": float(np.median([r["n_tiers"] for r in ev])),
                         "adj_share_median_all": float(np.median([r["share_adjacent_below"] for r in ev]))})
    return pd.DataFrame(rows), pd.DataFrame(summ)


# ============================================================================ per-task data

def load_outcomes(root: Path, boards: list[Board]) -> tuple[dict, dict, dict]:
    """(outcomes, rho matrices in census order, check) for the boards with per-task data. outcomes[lb][name] is the
    per-task success-rate vector (a 0/1 outcome for a single-run entry)."""
    outcomes, mats, check = {}, {}, {}
    for b in boards:
        if b.lb not in E.LOADERS:
            continue
        o = E.LOADERS[b.lb](root)
        missing = [nm for nm in b.names if nm not in o]
        if missing:
            raise ValueError(f"{b.lb}: no per-task outcomes for {len(missing)} entries")
        outcomes[b.lb] = {nm: o[nm] for nm in b.names}
        P = np.vstack([outcomes[b.lb][nm] for nm in b.names])
        R = E.rho_matrix(P)
        mats[b.lb] = R
        adj = E.adjacent(R)
        check[b.lb] = {"adj_median": float(np.nanmedian(adj)), "all_median": float(np.nanmedian(R[np.triu_indices(b.m, 1)])),
                       "score_maxdiff": float(np.abs(P.mean(axis=1) - b.scores).max()), "n_tasks": P.shape[1]}
    return outcomes, mats, check


# ============================================================================ report

def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def sep(b):
    return "n/a" if b is None or (isinstance(b, float) and math.isnan(b)) else ("separable" if b else "not separable")


def mdt(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_report(R: dict) -> str:
    L: list[str] = []
    A = L.append
    boards: list[Board] = R["boards"]
    rep = R["repro"]
    agg = rep["aggregate"]
    A(f"# Robustness analyses, round 3 ({DATE})\n")
    A("Numbers only. Generated by `analysis/leaderboards/robustness_r3.py` (seed 20261019; tests in `analysis/leaderboards/tests/test_robustness_r3.py`). "
      f"New files only: no existing script or output was changed. CSVs beside this report are prefixed `{PREFIX}_`. Inputs are the committed census "
      "outputs and the local snapshots (no network access); per-task outcomes were used in memory and only aggregates are written. "
      "Census snapshot: 2026-10-04 (SWE-bench commit 40f164d5, 2026-09-03; census_summary.md lists the others).\n")
    A("## 0. Definitions and reproduction check\n")
    A("- Bound (a), headline: a pair with scores P1 >= P2 on a board of n tasks is *separable* when gap = P1 - P2 is not below 1.96 SE_D, "
      "SE_D = sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2), SE = sqrt(P(1-P)/n); a tie is never separable. rho = the board's adjacent-pair median "
      "on the 7 boards with per-task data, the pooled median 0.5218 on the other 15. Tiers: greedy Algorithm 1 (an entry joins the tier while it is not separable "
      "from the tier's best entry). Adjacent pair = ranks (i, i+1) in census order (score descending, ties by name). All 22 analysed boards, kept entries only.")
    A(f"- Reproduction of the headline from the committed `census_entries.csv` and `census_parameters.json` (gate: the script stops if any value differs): "
      f"#1 not separable from #2 on **{agg['top1_vs_2_not_sep']} of 22**; #1 not separable from #5 on **{agg['top1_vs_5_not_sep']} of 22**; "
      f"tiers median **{agg['tiers_median']:g}** (min {agg['tiers_min']}, max {agg['tiers_max']}); adjacent share below the bound, median over boards "
      f"**{pct(agg['adjacent_below_median'])}** (min {pct(agg['adjacent_below_min'])}); adjacent share >= 90% on {agg['adjacent_ge_90']} of 22 boards. "
      f"Board-by-board comparison with `census.csv` (bound a_emp, primary sets): {rep['boards_identical']} of {rep['boards']} boards identical in entries, tiers, adjacent-below count and both top-entry verdicts "
      f"(largest difference in the all-pairs share {rep['max_abs_diff_share_all_below']:.1e}). Reproduced: **{'yes' if rep['reproduced'] else 'NO'}**.")
    ck = R["rho_check"]
    wc = max(abs(ck[lb]["adj_median"] - bd.rho) for bd in boards for lb in [bd.lb] if lb in ck)
    A(f"- Per-task outcomes re-read for the 7 boards: the adjacent-pair median rho recomputed from them differs from `census_parameters.json` by at most {wc:.1e}; "
      f"the largest difference between an entry's mean per-task outcome and its census score is {max(v['score_maxdiff'] for v in ck.values()):.1e} "
      f"({max(ck, key=lambda k: ck[k]['score_maxdiff'])}; zero to rounding on the other six boards. A Terminal-Bench 2.0 score is the share of all trials, which differs from the mean of per-task shares when trials per task are unequal).\n")

    # ---- 1
    s1 = R["s1"]
    bd1 = R["s1_boards"]
    A("## 1. Order confidence c = Phi(gap / SE_D)\n")
    A("c is the normal-approximation confidence that the higher-scoring entry of a pair is the better one (one-sided), at the headline rho; SE_D = bound / 1.96. "
      "A tie has c = 0.5 (and SE_D = 0 with gap > 0 gives c = 1). Per-pair values: `" + PREFIX + "_sec01_confidence_pairs.csv`.\n")
    A(f"- Median over the 22 boards of c(#1 vs #2): **{num(s1['median_c_1v2'])}** (range {num(s1['min_c_1v2'])} to {num(s1['max_c_1v2'])}); "
      f"boards with c(#1 vs #2) < 0.95: {s1['n_1v2_lt_0.95']} of 22; < 0.8: {s1['n_1v2_lt_0.8']} of 22. Median over boards of c(#1 vs #5): {num(s1['median_c_1v5'])}.")
    A(f"- Adjacent pairs with c < 0.8: **{pct(s1['pooled_share_lt_0.8'])}** pooled over all {s1['n_adjacent_pairs']} adjacent pairs, {pct(s1['median_board_share_lt_0.8'])} median over boards. "
      f"With c < 0.95: **{pct(s1['pooled_share_lt_0.95'])}** pooled, {pct(s1['median_board_share_lt_0.95'])} median over boards.\n")
    A(mdt([[r.leaderboard, f"{r.rho:.3f}", num(r.c_1v2), num(r.c_1v5), pct(r.share_c_lt_0_8), pct(r.share_c_lt_0_95)] for r in bd1.itertuples()],
          ["leaderboard", "rho", "c (#1 vs #2)", "c (#1 vs #5)", "adjacent share c < 0.8", "adjacent share c < 0.95"]))
    A("")

    # ---- 2
    b2 = R["s2"]
    A("## 2. Parametric bootstrap null\n")
    A(f"Per board {R['boot_reps']} replicates. Simulated observed scores X ~ multivariate normal with mean mu (the assumed true scores), SD sqrt(mu(1-mu)/n) and a common pairwise correlation "
      "rho (the board's headline rho), clipped to [0, 1]. Truth (i) *plug-in*: mu = observed scores. Truth (ii) *empirical Bayes*: mu_i = mean + B_i (obs_i - mean), B_i = tau2/(tau2 + SE_i^2), "
      "tau2 = max(0, var(obs) - mean(SE^2)), SE_i^2 = obs_i(1-obs_i)/n, mean = unweighted board mean. Metrics: P(top) = probability that the simulated #1 is the true #1 "
      "(an entry tied for the top mu counts; 'strict' = the single entry at census position 1); expected number of adjacent pairs of the true order that are reversed in X (a tie counts 0.5), "
      "as a count and as a share of the m-1 pairs; distribution of Algorithm-1 tier counts on the simulated data (headline rho) against the observed tier count. "
      f"Per board: `{PREFIX}_sec02_bootstrap_boards.csv`. Seed 20261019 (stream per board and truth).\n")
    for truth, name in (("plugin", "(i) plug-in truth"), ("shrink", "(ii) empirical-Bayes truth")):
        t = b2[b2["truth"] == truth]
        A(f"### {name}\n")
        A(mdt([[r.leaderboard, r.m, f"{r.rho:.3f}", num(r.tau2, 5) if truth == "shrink" else "-", num(r.p_top_correct), num(r.p_top_correct_strict),
                f"{r.exp_misordered_adjacent:.2f}", pct(r.exp_misordered_share), r.tiers_observed,
                f"{r.tiers_sim_median:g} ({r.tiers_sim_q025:g} to {r.tiers_sim_q975:g})"] for r in t.itertuples()],
              ["leaderboard", "m", "rho", "tau2", "P(top correct)", "P(top correct), strict", "E[reversed adjacent pairs]", "as share of m-1",
               "observed tiers", "simulated tiers: median (2.5% to 97.5%)"]))
        A("")
        A(f"- Median over boards: P(top correct) **{num(t['p_top_correct'].median())}** (strict {num(t['p_top_correct_strict'].median())}); expected reversed adjacent pairs "
          f"**{t['exp_misordered_adjacent'].median():.2f}** (count), **{pct(t['exp_misordered_share'].median())}** (share); simulated tier median of medians {t['tiers_sim_median'].median():g}, "
          f"observed tier median {t['tiers_observed'].median():g}; boards whose observed tier count lies inside the simulated 95% range: {int(t['observed_in_95'].sum())} of {len(t)}; "
          f"boards with P(top correct) < 0.5: {int((t['p_top_correct'] < 0.5).sum())} of {len(t)}.\n")

    # ---- 3
    s3 = R["s3"]
    A("## 3. rho sweep (same rho on all 22 boards for the constant rows)\n")
    A("Counts of boards on which the check is not separable, tiers median (min to max over boards), median over boards of the adjacent share below the bound. The rho = 0.9 row is the addition requested with item 15. "
      f"Per board: `{PREFIX}_sec03_rho_sweep_boards.csv`.\n")
    A(mdt([[r.rho, f"{r.n12_not_sep} of {r.n_boards}", f"{r.n15_not_sep} of {r.n_boards}", f"{r.tiers_median:g} ({r.tiers_min} to {r.tiers_max})",
            pct(r.adj_share_median), f"{r.adj_ge_90} of {r.n_boards}"] for r in s3.itertuples()],
          ["rho", "#1 vs #2 not separable", "#1 vs #5 not separable", "tiers: median (min to max)", "adjacent share below: median", "adjacent share >= 90%"]))
    A("")

    # ---- 4
    f4, b4, rho4 = R["s4"]
    h4 = R["s4_head"]
    A("## 4. Leave-one-family-out rho\n")
    A("Families: SWE-bench (Verified, Lite, test (full), Multilingual), Terminal-Bench 1.0, Terminal-Bench 2.0, TheAgentCompany. A family's rho is the median of its boards' adjacent-pair medians; "
      "its prediction is the median of the other three families' values. Board-level version (the board's rho predicted by the median of the boards outside its family) in the CSV.\n")
    A(mdt([[r.family, r.n_boards, num(r.rho_family, 4), num(r.rho_predicted_by_others, 4), num(r.abs_error, 4)] for r in f4.itertuples()],
          ["family", "boards", "rho of the family", "predicted by the other 3 families", "absolute error"]))
    A("")
    A(f"- Absolute errors: median **{num(f4['abs_error'].median(), 4)}**, max **{num(f4['abs_error'].max(), 4)}**. Board-level absolute errors: median {num(b4['abs_error'].median(), 4)}, max {num(b4['abs_error'].max(), 4)}.")
    A(f"- Pooled median over the 4 families: **{num(rho4, 4)}** (the census value, the median of the 7 board medians, is {num(R['pooled'], 4)}).")
    A(f"- Headline recomputed with {num(rho4, 4)} for the 15 boards without per-task data (the 7 keep their own rho): #1 not separable from #2 on **{h4['n12_not_sep']} of 22**, "
      f"from #5 on **{h4['n15_not_sep']} of 22**, tiers median **{h4['tiers_median']:g}** ({h4['tiers_min']} to {h4['tiers_max']}), adjacent share median **{pct(h4['adj_share_median'])}**, "
      f"adjacent share >= 90% on {h4['adj_ge_90']} of 22 (headline: {agg['top1_vs_2_not_sep']}, {agg['top1_vs_5_not_sep']}, {agg['tiers_median']:g}, {pct(agg['adjacent_below_median'])}, {agg['adjacent_ge_90']}).\n")

    # ---- 5
    s5, summ5 = R["s5"]
    A("## 5. Tier-head comparisons with the all-pairs rho\n")
    A("rho = the board's median over ALL pairs (7 boards) in place of the adjacent-pair median. Two choices for the 15 boards without per-task data: the census pooled adjacent value "
      f"0.5218, or the median of the seven all-pairs medians ({num(summ5['all_pairs_pooled'], 4)}). Per board: `{PREFIX}_sec05_allpairs_rho_boards.csv`.\n")
    for name in [k for k in summ5 if k != "all_pairs_pooled"]:
        v = summ5[name]
        A(f"- {name}: tiers median **{v['tiers_median']:g}** ({v['tiers_min']} to {v['tiers_max']}; headline {agg['tiers_median']:g}), #1 vs #2 not separable {v['n12_not_sep']} of 22, "
          f"#1 vs #5 not separable {v['n15_not_sep']} of 22, adjacent share median {pct(v['adj_share_median'])}.")
    A("")
    t5 = s5[s5["variant"] == list(summ5)[1]]
    A(mdt([[r.leaderboard, num(r.rho_used, 3), r.tiers_headline, r.tiers_variant, f"{r.tiers_change:+d}", sep(r.sep12_headline) + " > " + sep(r.sep12_variant),
            sep(r.sep15_headline) + " > " + sep(r.sep15_variant)] for r in t5.itertuples()],
          ["leaderboard (first variant: 15 boards at pooled 0.5218)", "rho used", "tiers headline", "tiers all-pairs rho", "change", "#1 vs #2", "#1 vs #5"]))
    A("")

    # ---- 6
    s6, summ6 = R["s6"]
    A("## 6. k-adjusted bound (a)\n")
    A(f"Entries averaged over k > 1 runs have SE multiplied by sqrt(1 - f (1 - 1/k)), f = 0.369 (the variance of a k-run mean is P(1-P)(1 - f(1-1/k)), v = f P(1-P)). "
      "Two forms: (1) the covariance term keeps the single-run SEs, because rho is defined with the single-run variance (the across-task covariance of the per-task rates does not depend on k); "
      "(2) the literal form that scales the SEs in all three terms. Boards with k > 1 in some entry: "
      + ", ".join(sorted({b.lb for b in boards if b.ks.max() > 1})) + f". Per board: `{PREFIX}_sec06_kadj_boards.csv`.\n")
    rows = []
    for name, v in summ6.items():
        rows.append([name, f"{v['n12_not_sep']} of 22", f"{v['n15_not_sep']} of 22", f"{v['tiers_median']:g} ({v['tiers_min']} to {v['tiers_max']})", pct(v["adj_share_median"]),
                     "-" if name == "headline" else f"{v['boards_sep12_changed']} / {v['boards_sep15_changed']} / {v['boards_tiers_changed']}",
                     "-" if name == "headline" else f"{v['adj_verdicts_changed']} of {v['adj_pairs']} (on {v['boards_with_adj_change']} boards)"])
    A(mdt(rows, ["bound", "#1 vs #2 not separable", "#1 vs #5 not separable", "tiers: median (min to max)", "adjacent share: median",
                 "boards changed: #1v#2 / #1v#5 / tiers", "adjacent verdicts changed"]))
    A("")

    # ---- 7
    s7, summ7, ext7 = R["s7"]
    A("## 7. Adjusted standard errors (Agresti-Caffo and Wilson centre)\n")
    A("Agresti-Caffo style: P~ = (x+1)/(n+2), n~ = n+2, x = P n (one success and one failure added). Wilson centre: P~ = (x + z^2/2)/(n + z^2), n~ = n + z^2, z = 1.96. SE = sqrt(P~(1-P~)/n~), "
      "same rho. 'Observed gap': the gap stays P1 - P2; 'adjusted gap': the gap is P~1 - P~2. "
      f"Per board: `{PREFIX}_sec07_adjusted_se_boards.csv`.\n")
    rows = []
    for name, v in summ7.items():
        rows.append([name, f"{v['n12_not_sep']} of 22", f"{v['n15_not_sep']} of 22", f"{v['tiers_median']:g} ({v['tiers_min']} to {v['tiers_max']})", pct(v["adj_share_median"]),
                     "-" if name == "headline" else f"{v['boards_sep12_changed']} / {v['boards_sep15_changed']} / {v['boards_tiers_changed']}",
                     "-" if name == "headline" else f"{v['adj_verdicts_changed']} of {v['adj_pairs']} (on {v['boards_with_adj_change']} boards)"])
    A(mdt(rows, ["bound", "#1 vs #2 not separable", "#1 vs #5 not separable", "tiers: median (min to max)", "adjacent share: median",
                 "boards changed: #1v#2 / #1v#5 / tiers", "adjacent verdicts changed"]))
    A("")
    e0 = ext7[(ext7["entries_at_0"] > 0) | (ext7["entries_at_1"] > 0)]
    A(f"Boards with entries at 0% or 100%: {len(e0)} of 22.")
    if len(e0):
        cm = s7[s7["leaderboard"].isin(e0["leaderboard"])]
        A("")
        rows = []
        for r in e0.itertuples():
            sub = cm[cm["leaderboard"] == r.leaderboard]
            chg = "; ".join(f"{v.variant}: adj changed {v.adj_verdicts_changed}, tiers {v.tiers_headline}>{v.tiers_alt}" for v in sub.itertuples()
                            if v.adj_verdicts_changed or v.tiers_headline != v.tiers_alt or v.sep12_headline != v.sep12_alt or v.sep15_headline != v.sep15_alt)
            rows.append([r.leaderboard, r.entries_at_0, r.entries_at_1, chg or "no change in any variant"])
        A(mdt(rows, ["leaderboard", "entries at 0%", "entries at 100%", "verdict changes (variants with any change)"]))
    A("")

    # ---- 8
    c8p, c8 = R["s8"]
    A("## 8. Cluster-robust bound\n")
    A("Cluster-robust SE of the mean per-task difference d = y1 - y2: sqrt(G/(G-1) sum_g (sum_{t in g}(d_t - dbar))^2) / n, over the G clusters (z = 1.96; a column with the t critical value, G-1 df, is in the table). "
      "'iid' = the binomial-paired SE at the pair's own empirical rho (sqrt(Var(d)/n)); 'headline' = SE_D at the board's headline rho. Clusters: the repository prefix of the SWE-bench instance id "
      "('django__django-12345' -> 'django__django'); for TheAgentCompany the task-name prefix before the first hyphen (department-like category, a heuristic label, not a derived grouping). "
      "Terminal-Bench 1.0 and 2.0 task ids carry no cluster label and are not analysed; no other board has local per-task outcomes. "
      f"Per pair: `{PREFIX}_sec08_cluster_pairs.csv`; per board: `{PREFIX}_sec08_cluster_boards.csv`.\n")
    A(mdt([[r.leaderboard, r.n_clusters, f"{r.min_cluster_size} to {r.max_cluster_size}", num(r.median_ratio_cluster_over_iid), num(r.median_ratio_cluster_over_headline),
            f"{r.adj_separable_headline} > {r.adj_separable_cluster} (t: {r.adj_separable_cluster_t}) of {r.n_adjacent}", r.adj_verdicts_changed,
            sep(r.sep12_headline) + " > " + sep(r.sep12_cluster), sep(r.sep15_headline) + " > " + sep(r.sep15_cluster), f"{r.tiers_headline} > {r.tiers_cluster}"]
           for r in c8.itertuples()],
          ["leaderboard", "clusters G", "cluster size range", "median ratio SE_cluster / SE_iid", "median ratio SE_cluster / SE_headline",
           "adjacent separable: headline > cluster (t) of pairs", "adjacent verdicts changed", "#1 vs #2", "#1 vs #5", "tiers"]))
    A("")

    # ---- 9
    c9 = R["s9"]
    A("## 9. Calibration extension\n")
    reps9 = int(c9["reps"].iloc[0])
    A(f"Setup as in section 2 of `robustness_2026-10-06.md` (two Bernoulli outcomes per task, phi correlation rho_true, plug-in SEs, separated when |D| > 1.96 SE_D) with the rule using an assumed rho: "
      f"true, true - 0.25 and true + 0.25, clipped to [0, 0.95]. n in {{30, 50, 100, 500, 2294}}, P in {{0.05, 0.2, 0.5, 0.8, 0.95}}, rho_true in {{0.25, 0.5, 0.75}}: {len(c9)} cells. "
      f"**{reps9:,} replicates per cell** (fast enough; no reduction). Monte Carlo SE: {c9['mc_se_null'].median():.4f} for a null rate near 0.05 (largest {c9['mc_se_null'].max():.4f}), "
      f"{c9['mc_se_power'].median():.4f} for a power near 0.5. False-separation rate: share separated when the true difference is 0 (both entries at P; nominal 0.05). "
      "Power at the bound: share separated when the true difference equals the bound the analyst computes, delta = 1.96 sqrt(2P(1-P)(1-rho_assumed)/n) (entries at P +- delta/2); "
      "cells where P - delta/2 < 0 or P + delta/2 > 1, or where the joint distribution is infeasible, have no power value. "
      f"All cells: `{PREFIX}_sec09_calibration.csv`.\n")
    A(f"Cells without a power value (infeasible): {int(c9['power_at_bound'].isna().sum())} of {len(c9)}.\n")
    rows = []
    for rel in ("true", "true-0.25", "true+0.25"):
        g = c9[c9["relation"] == rel]
        pw = g["power_at_bound"].dropna()
        rows.append([rel, len(g), f"{g['false_separation_rate'].median():.3f}", f"{g['false_separation_rate'].min():.3f} to {g['false_separation_rate'].max():.3f}",
                     f"{(g['false_separation_rate'] > 0.06).sum()} / {(g['false_separation_rate'] < 0.04).sum()}",
                     f"{pw.median():.3f}", f"{pw.min():.3f} to {pw.max():.3f}", len(pw)])
    A(mdt(rows, ["assumed rho", "cells", "false-separation rate: median", "min to max", "cells > 0.06 / < 0.04", "power at bound: median", "min to max", "cells with power value"]))
    A("")
    A("Median false-separation rate by n and assumed rho (over P and rho_true):\n")
    piv = c9.pivot_table(index="n", columns="relation", values="false_separation_rate", aggfunc="median")[["true-0.25", "true", "true+0.25"]]
    A(mdt([[n_] + [f"{v:.3f}" for v in row] for n_, row in piv.iterrows()], ["n", "assumed true-0.25", "assumed true", "assumed true+0.25"]))
    A("")
    A("Median false-separation rate by P and assumed rho (over n and rho_true):\n")
    piv = c9.pivot_table(index="P", columns="relation", values="false_separation_rate", aggfunc="median")[["true-0.25", "true", "true+0.25"]]
    A(mdt([[p_] + [f"{v:.3f}" for v in row] for p_, row in piv.iterrows()], ["P", "assumed true-0.25", "assumed true", "assumed true+0.25"]))
    A("")
    A("Median power at the bound by n and assumed rho (over P and rho_true, cells with a value):\n")
    piv = c9.pivot_table(index="n", columns="relation", values="power_at_bound", aggfunc="median")[["true-0.25", "true", "true+0.25"]]
    A(mdt([[n_] + [f"{v:.3f}" for v in row] for n_, row in piv.iterrows()], ["n", "assumed true-0.25", "assumed true", "assumed true+0.25"]))
    A("")
    w = c9.sort_values("false_separation_rate", ascending=False).head(6)
    A("Largest false-separation rates:\n")
    A(mdt([[r.n, r.P, r.rho_true, f"{r.rho_assumed:.2f} ({r.relation})", f"{r.false_separation_rate:.3f}"] for r in w.itertuples()],
          ["n", "P", "rho_true", "rho_assumed", "false-separation rate"]))
    A("")

    # ---- 10
    s10, feas10, i10 = R["s10"]
    A("## 10. tau2-bench run-noise share f and the coupling bound\n")
    A("f = (pass^1 - pass^2) / (pass^1 (1 - pass^1)) per submission x domain row of `analysis/results/tau2_consistency.csv` where pass^2 exists (pass^1 - pass^2 = E[p(1-p)] = v; pass values are percentages). "
      "Rows with pass^1 at 0 or 100 would be undefined (none occur). Aggregates only; no intervals.\n")
    A(f"- Rows in the file: {i10['n_rows_total']}; rows with pass^2: {i10['n_rows_with_pass2']}; rows with a defined f: **{i10['n_rows_f']}**.")
    A(f"- f over all rows: median **{num(i10['median_f'])}**, IQR {num(i10['q25_f'])} to {num(i10['q75_f'])}, range {num(i10['min_f'])} to {num(i10['max_f'])}.")
    A("")
    A(mdt([[d["domain"], int(d["count"]), num(d["median"]), num(d["min"]), num(d["max"])] for d in i10["by_domain"]], ["domain", "rows", "median f", "min f", "max f"]))
    A("")
    A(f"- Rows that state a trial count (`trials_stated`): **{i10['n_rows_trials_stated']} of {i10['n_rows_total']}** ({i10['n_rows_f_trials_stated']} with a defined f); stated values: {', '.join(i10['trials_stated_values'])}. "
      f"Median f where stated {num(i10['median_f_stated'])}, where not stated {num(i10['median_f_not_stated'])}. For rows without a stated count the number of trials behind pass^2 is not documented in the file "
      "(the columns pass_1 to pass_4 imply at least 4).")
    A(f"- Terminal-Bench 2.0 reference (`tb2_within_task_variance.csv`, {i10['tb2_n']} submissions): median f {num(i10['tb2_median_f'])}, IQR {num(i10['tb2_q25_f'])} to {num(i10['tb2_q75_f'])}, range {num(i10['tb2_min_f'])} to {num(i10['tb2_max_f'])}.")
    A("")
    A("Coupling bound: two entries with run-noise shares f1, f2 have task-level (persistent) variance (1-f)P(1-P), so the correlation between their single-run outcomes is at most "
      "sqrt((1-f1)(1-f2)), about 1 - f for equal shares. A cell (rho, f) of the sweep is *feasible* when rho <= 1 - f. Boards' rho = headline rho. "
      f"Per board and f: `{PREFIX}_sec10_feasibility.csv`.\n")
    rows = []
    for b in boards:
        fe = feas10[feas10["leaderboard"] == b.lb]
        rows.append([b.lb, f"{b.rho:.3f} ({b.source})"] + ["feasible" if x else "infeasible" for x in fe["feasible"]])
    A(mdt(rows, ["leaderboard", "rho"] + [f"f = {f:g}" for f in F_SWEEP]))
    A("")
    cnt = feas10.groupby("f")["feasible"].sum()
    A("Feasible boards per f: " + ", ".join(f"f = {f:g}: {int(cnt[f])} of 22" for f in F_SWEEP) + ".\n")

    # ---- 11
    s11 = R["s11"]
    A("## 11. Confidence set of possible leaders\n")
    A("Entries (other than #1) whose gap to #1 is below z SE_D at the headline rho (or zero). Set size = those entries plus #1. Uncorrected z = 1.96; Bonferroni z = Phi^-1(1 - 0.025/(m-1)) over the m-1 comparisons with #1. "
      f"Per board: `{PREFIX}_sec11_leader_set.csv`.\n")
    rows = []
    for lab in ("uncorrected", "bonferroni"):
        c = s11[f"others_{lab}"]
        sz = s11[f"set_size_{lab}"]
        rows.append([lab, f"{c.median():g}", f"{c.min()} to {c.max()}", f"{sz.median():g}", f"{sz.min()} to {sz.max()}", pct(s11[f"set_share_{lab}"].median()),
                     int((c == 0).sum())])
    A(mdt(rows, ["critical value", "entries not separable from #1: median", "range", "set size: median", "range", "set share of m: median", "boards where #1 is the only possible leader"]))
    A("")
    A(mdt([[r.leaderboard, r.m, f"{r.z_bonferroni:.3f}", r.others_uncorrected, r.others_bonferroni] for r in s11.itertuples()],
          ["leaderboard", "m", "Bonferroni z", "entries not separable from #1 (uncorrected)", "entries not separable from #1 (Bonferroni)"]))
    A("")

    # ---- 12
    s12 = R["s12"]
    A("## 12. Agreement with prior SWE-bench analyses (our snapshot)\n")
    A("Snapshot: swe-bench/experiments commit 40f164d5 of 2026-09-03, census of 2026-10-04; SWE-bench Verified, n = 500, 171 kept entries, headline rho 0.5829.\n")
    for lab, name in (("headline", "headline rho 0.5829"), ("rho0", "rho = 0 (conservative)")):
        v = s12[lab]
        A(f"- {name}: of the 29 adjacent pairs among the top 30 entries (scores {pct(v['top1_score'])} down to {pct(v['top30_score'])}), **{v['top30_adjacent_separable']} are separable** by the bound and {v['top30_adjacent_below']} are below it; "
          f"Algorithm 1 gives {v['top30_tiers']} tiers on the top 30. Of the 45 pairs among the top 10 (scores down to {pct(v['top10_score'])}), **{v['top10_all_below']} are below the bound**; tiers on the top 10: {v['top10_tiers']}.")
    A("")

    # ---- 13
    k13b, k13 = R["s13"]
    A("## 13. Paired-test validation details (Cohen's kappa)\n")
    A("From `robustness_2026-10-06_mcnemar_pairs.csv` (McNemar exact for single-run pairs, paired t for pairs with an entry averaging k > 1 runs, alpha = 0.05, no multiplicity correction). "
      "Kappa between the bound verdict (separable / not) and the test verdict. 'Agreement among pairs that either method separates' = both separable / (both + bound only + test only). "
      f"Per board: `{PREFIX}_sec13_kappa_boards.csv`.\n")
    rows = []
    for lab, v in k13.items():
        rows.append([lab, v["n"], pct(v["agree"]), num(v["kappa"]), v["n_both"], v["n_bound_only"], v["n_test_only"], v["n_neither"], pct(v["agree_among_either"])])
    A(mdt(rows, ["pairs", "n", "raw agreement", "Cohen's kappa", "both separable", "bound only", "test only", "neither", "agreement among pairs either method separates"]))
    A("")
    A(mdt([[r.leaderboard, r.pairs, pct(r.agree), num(r.kappa) if not math.isnan(r.kappa) else "undefined (one verdict category)", r.n_bound_sep, r.n_test_sep, r.n_both,
            pct(r.agree_among_either) if not math.isnan(r.agree_among_either) else "n/a (no pair separated)"] for r in k13b.itertuples()],
          ["leaderboard (adjacent pairs)", "pairs", "raw agreement", "kappa", "bound separable", "test separable", "both", "agreement among either"]))
    A("")

    # ---- 14
    s14 = R["s14"]
    A("## 14. Systems x tasks variance decomposition and relative generalizability Ep^2\n")
    A("Two-way random-effects model on the systems x tasks matrix (one observation per cell, ANOVA moments; negative components set to 0): outcome = mu + system + task + residual, "
      "where the residual is the system x task interaction confounded with run noise. Ep^2 = var_systems / (var_systems + var_residual / n_tasks) = (MS_systems - MS_residual) / MS_systems "
      "is the reliability of the system means for ranking systems at the board's n tasks (relative decisions). Entries used: the kept entries of the census, every one of which has an outcome on every task "
      "(balanced by construction), so no row or column is dropped from the matrix; what is left out relative to the published leaderboard are the entries the census excluded (counted in the table; reasons in census_summary.md). "
      "For SWE-bench and TheAgentCompany a cell is a 0/1 outcome; for Terminal-Bench 1.0 (k = 5) and 2.0 (k = 1 to 10) a cell is the share of that entry's runs that solved the task, so the residual variance includes run noise divided by k. "
      "'top 10' repeats the decomposition on the ten best entries only. Variance shares are of the total of the three components. "
      f"CSV: `{PREFIX}_sec14_variance_decomposition.csv`.\n")
    A(mdt([[r.leaderboard, r.systems, r.tasks, r.census_excluded_entries, r.cell, pct(r.share_systems), pct(r.share_tasks), pct(r.share_residual), num(r.ep2, 4), num(r.ep2_top10, 3),
            sep(r.top1_vs_2_separable) + f" (c = {r.c_1v2:.3f})", pct(r.share_adjacent_below)] for r in s14.itertuples()],
          ["leaderboard", "systems", "tasks", "entries excluded by census", "cell", "variance share: systems", "tasks", "residual", "Ep^2 (all systems)", "Ep^2 (top 10)", "#1 vs #2 verdict (headline)", "adjacent share below bound"]))
    A("")
    A(f"- Ep^2 over all systems: {num(s14['ep2'].min(), 4)} to {num(s14['ep2'].max(), 4)} across the 7 boards (median {num(s14['ep2'].median(), 4)}); Ep^2 on the top ten: {num(s14['ep2_top10'].min(), 3)} to {num(s14['ep2_top10'].max(), 3)} "
      f"(median {num(s14['ep2_top10'].median(), 3)}). #1 is not separable from #2 on {int((~s14['top1_vs_2_separable']).sum())} of these 7 boards, with Ep^2 above 0.9 on {int((s14['ep2'] > 0.9).sum())} of the 7.\n")

    # ---- 15
    s15, summ15 = R["s15"]
    A("## 15. Pair-specific rho at the top, and rho = 0.9\n")
    A(f"rho = 0.9 is in the section 3 sweep. Here: the rho of the #1 vs #2 pair and of the #1 vs #5 pair (phi between the two entries' per-task outcomes; entries averaging k > 1 runs use the expected single-run phi of `estimate_rho.py`) "
      f"against the board's adjacent-pair median. Per board: `{PREFIX}_sec15_pair_rho_boards.csv`.\n")
    A(f"- Boards (of {summ15['n_boards_pair_rho']} with per-task data) where the #1 vs #2 pair rho is above the board median: **{summ15['n_1v2_above_board']}**; where the #1 vs #5 pair rho is above it: {summ15['n_1v5_above_board']}. "
      f"Median of (pair rho - board median): #1 vs #2 {num(summ15['median_diff_1v2'])}, #1 vs #5 {num(summ15['median_diff_1v5'])}.")
    A(f"- Headline counts with the pair-specific rho in the two top-entry checks (7 boards; the other 15 keep their board/pooled rho; tiers unchanged): #1 not separable from #2 on **{summ15['n12_not_sep_pair']} of 22** "
      f"(headline {summ15['n12_not_sep_headline']}), from #5 on **{summ15['n15_not_sep_pair']} of 22** (headline {summ15['n15_not_sep_headline']}).\n")
    A(mdt([[r.leaderboard, num(r.rho_board_adjacent_median), num(r.rho_all_pairs_median), num(r.rho_pair_1v2), num(r.rho_pair_1v5),
            sep(r.sep12_headline) + " > " + sep(r.sep12_pair), sep(r.sep15_headline) + " > " + sep(r.sep15_pair)] for r in s15[s15["has_pair_rho"]].itertuples()],
          ["leaderboard", "board adjacent median", "board all-pairs median", "rho (#1, #2)", "rho (#1, #5)", "#1 vs #2: headline > pair rho", "#1 vs #5: headline > pair rho"]))
    A("")

    # ---- 16
    s16, summ16 = R["s16"]
    A("## 16. Fixed-task rerun bound (b) at f = 0.369, feasible f, and the section 10 range\n")
    A("Bound (b) = 1.96 sqrt((f/n)(P1(1-P1)/k1 + P2(1-P2)/k2)); k-adjusted uses each entry's k, single-run sets k = 1 (census_summary.md). f = 0.369 is the Terminal-Bench 2.0 median (0.3689). "
      "*Feasible* boards for an f: those with headline rho <= 1 - f (the coupling bound of section 10). The 'feasible only' columns count over the feasible boards. "
      "The f range of section 10 is evaluated at the tau2-bench per-row quartiles and extremes. "
      f"Per board and f: `{PREFIX}_sec16_rerun_boards.csv`.\n")
    for form in ("k-adjusted", "single-run"):
        t = summ16[summ16["form"] == form]
        A(f"### {form} form\n")
        A(mdt([[r.f_label, ("per board" if math.isnan(r.f) else f"{r.f:.3f}"), f"{r.n12_not_sep_all} of {r.boards}", f"{r.n15_not_sep_all} of {r.boards}", r.boards_feasible,
                f"{r.n12_not_sep_feasible} of {r.boards_feasible}", f"{r.n15_not_sep_feasible} of {r.boards_feasible}", f"{r.tiers_median_all:g}", pct(r.adj_share_median_all)]
               for r in t.itertuples()],
              ["f", "value", "#1 vs #2 not separable (all 22)", "#1 vs #5 not separable (all 22)", "feasible boards", "#1 vs #2 (feasible only)", "#1 vs #5 (feasible only)",
               "tiers median", "adjacent share median"]))
        A("")
    kk = summ16[(summ16["form"] == "k-adjusted") & (summ16["f_label"] == "sweep 0.369")].iloc[0]
    A(f"- At f = 0.369 the number of feasible boards is {int(kk['boards_feasible'])} of 22 (largest headline rho {max(b.rho for b in boards):.3f} <= 1 - 0.369 = 0.631), so the feasible-only counts equal the all-board counts: "
      f"#1 not separable from #2 on {int(kk['n12_not_sep_feasible'])}, from #5 on {int(kk['n15_not_sep_feasible'])} (k-adjusted form).")
    A(f"- Section 10 f range: tau2-bench rows {num(i10['min_f'])} to {num(i10['max_f'])} (IQR {num(i10['q25_f'])} to {num(i10['q75_f'])}, median {num(i10['median_f'])}); "
      f"Terminal-Bench 2.0 {num(i10['tb2_min_f'])} to {num(i10['tb2_max_f'])} (IQR {num(i10['tb2_q25_f'])} to {num(i10['tb2_q75_f'])}).\n")
    return "\n".join(L) + "\n"


# ============================================================================ driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--out", default=str(ANALYSIS / "results"))
    ap.add_argument("--boot-reps", type=int, default=2000)
    ap.add_argument("--cal-reps", type=int, default=20_000)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = HERE / "results"

    boards, par = load_boards(res)
    rep = reproduce(boards, res / "census.csv")
    print("headline:", {k: rep["aggregate"][k] for k in ("top1_vs_2_not_sep", "top1_vs_5_not_sep", "tiers_median", "adjacent_below_median", "adjacent_ge_90")},
          "boards identical:", rep["boards_identical"], "/", rep["boards"])
    if not rep["reproduced"]:
        raise SystemExit("STOP: the headline numbers were not reproduced; no new analysis was computed.")
    rho_by_board = pd.read_csv(res / "rho_by_board.csv")
    pooled = float(par["pooled_median_rho"])
    f_tb2 = float(par["f_tb2_median"])
    base = ref_rows(boards)

    outcomes, mats, rho_check = load_outcomes(root, boards)
    R: dict = {"boards": boards, "repro": rep, "pooled": pooled, "rho_check": rho_check, "boot_reps": a.boot_reps}
    save = lambda df, name: df.to_csv(out / f"{PREFIX}_{name}.csv", index=False)

    s1p, s1b, R["s1"] = sec1_order_confidence(boards)
    R["s1_boards"] = s1b
    save(s1p, "sec01_confidence_pairs")
    save(s1b, "sec01_confidence_boards")
    R["s2"] = sec2_bootstrap(boards, a.boot_reps)
    save(R["s2"], "sec02_bootstrap_boards")
    sw, swb = sec3_rho_sweep(boards)
    R["s3"] = sw
    save(sw, "sec03_rho_sweep_summary")
    save(swb, "sec03_rho_sweep_boards")
    f4, b4, rho4 = sec4_lofo(rho_by_board)
    R["s4"] = (f4, b4, rho4)
    h4rows = evaluate_all(boards, lambda b: fn_a(b.n, rho_with_pooled(boards, rho4)[b.lb]))
    R["s4_head"] = summarize_rows(h4rows)
    save(f4, "sec04_lofo_families")
    save(b4, "sec04_lofo_boards")
    save(board_df(h4rows, {"rho_4family_pooled": rho4}), "sec04_headline_4family_boards")
    s5, summ5 = sec5_allpairs(boards, rho_by_board, pooled)
    R["s5"] = (s5, summ5)
    save(s5, "sec05_allpairs_rho_boards")
    s6, summ6 = sec6_kadj(boards, f_tb2)
    R["s6"] = (s6, summ6)
    save(s6, "sec06_kadj_boards")
    s7, summ7, ext7 = sec7_adjusted(boards)
    R["s7"] = (s7, summ7, ext7)
    save(s7, "sec07_adjusted_se_boards")
    c8p, c8 = sec8_cluster(boards, outcomes, root)
    R["s8"] = (c8p, c8)
    save(c8p, "sec08_cluster_pairs")
    save(c8, "sec08_cluster_boards")
    R["s9"] = sec9_calibration(a.cal_reps)
    save(R["s9"], "sec09_calibration")
    t10, feas10, i10 = sec10_tau2(ANALYSIS / "results", boards)
    R["s10"] = (t10, feas10, i10)
    save(t10, "sec10_tau2_f")
    save(feas10, "sec10_feasibility")
    R["s11"] = sec11_leaders(boards)
    save(R["s11"], "sec11_leader_set")
    R["s12"] = sec12_prior(boards)
    save(pd.DataFrame([{"setting": k, **v} for k, v in R["s12"].items()]), "sec12_verified_top_pairs")
    k13b, k13 = sec13_kappa(ANALYSIS / "results" / "robustness_2026-10-06_mcnemar_pairs.csv")
    R["s13"] = (k13b, k13)
    save(k13b, "sec13_kappa_boards")
    R["s14"] = sec14_decomposition(boards, outcomes, base, s1b, par["_excluded"])
    save(R["s14"], "sec14_variance_decomposition")
    s15, summ15 = sec15_pair_rho(boards, mats, rho_by_board)
    R["s15"] = (s15, summ15)
    save(s15, "sec15_pair_rho_boards")
    f_values = {**{f"sweep {f:g}": f for f in F_SWEEP}, "census f_TB2 (0.3689)": f_tb2,
                "tau2 min": i10["min_f"], "tau2 q25": i10["q25_f"], "tau2 median": i10["median_f"], "tau2 q75": i10["q75_f"],
                "tau2 max": i10["max_f"]}
    f_values = {k: min(max(v, 0.0), 1.0) for k, v in f_values.items()}
    f_values["board maximum feasible f = 1 - rho"] = lambda b: 1.0 - b.rho
    R["s16"] = sec16_rerun(boards, f_values)
    save(R["s16"][0], "sec16_rerun_boards")
    save(R["s16"][1], "sec16_rerun_summary")
    (out / f"{PREFIX.replace('robustness_', 'robustness_')}.md").write_text(build_report(R), encoding="utf-8")
    print("wrote", out / f"{PREFIX}.md")


if __name__ == "__main__":
    main()
