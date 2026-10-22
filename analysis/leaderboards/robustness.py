# SPDX-License-Identifier: MIT
"""Robustness analyses requested by a simulated IEEE Access review (added 2026-10-06).

Writes NEW files only (``analysis/results/robustness_2026-10-06*``); no existing output and no default
behaviour of the census or the Terminal-Bench 2.0 scripts is changed. Inputs are the local snapshots
already used by ``census.py``; there is no network access.

1. McNemar validation of the item-sampling bound (7 boards with per-task outcomes). For each adjacent
   pair (census order) and for the pair (#1, #5), a paired test of the two entries' per-task outcomes
   at alpha = 0.05 is compared with the verdict of the headline bound (a) at the board's rho.
   * both entries single-run (every per-task outcome 0 or 1): exact two-sided McNemar test, i.e. an exact
     binomial test of the b discordant tasks (A solves, B fails) against the c discordant tasks
     (B solves, A fails) at p = 0.5; p = 1 when b + c = 0.
   * otherwise (an entry averaging k > 1 runs, per-task success share in [0, 1]): paired t-test on the
     per-task differences of the success shares (n - 1 degrees of freedom). The Wilcoxon signed-rank test
     (zero differences dropped) and the exact sign test are recorded in the CSV as robustness checks.
   Bound verdict: separable = not (gap < bound or gap == 0), the census rule, bound = 1.96 sqrt(SE1^2 +
   SE2^2 - 2 rho SE1 SE2) with rho the board's headline rho (own median adjacent rho, ``estimate_rho``).
2. Calibration simulation of bound (a): pairs of correlated Bernoulli task outcomes, see ``simulate_cell``.
3. Eligibility sensitivity: the census headline on 26 boards (22 analysed + tau2-bench airline, tau2-bench
   retail, GAIA test, GAIA validation) with the pooled rho.
4. Terminal-Bench 2.0 missing-reward sensitivity: ``analysis/robustness_tb2.py``.

Usage: python robustness.py [--root <project root>] [--out <analysis/results>] [--reps 20000]
                            [--tb2-n-boot 10000] [--skip-tb2]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
for _p in (str(HERE), str(ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import census as C  # noqa: E402
import estimate_rho as E  # noqa: E402

ALPHA = 0.05
Z = C.Z
DATE = "2026-10-06"
PREFIX = f"robustness_{DATE}"


# ============================================================================ 1. paired tests

def mcnemar_exact(x, y) -> dict:
    """Exact two-sided McNemar test for two 0/1 outcome vectors over the same tasks.
    b = tasks solved by x and not by y, c = the reverse; p is the exact binomial two-sided p-value of
    min(b, c) out of b + c at probability 0.5 (1.0 when there is no discordant task)."""
    x, y = np.asarray(x), np.asarray(y)
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = b + c
    p = 1.0 if d == 0 else float(stats.binomtest(min(b, c), d, 0.5).pvalue)
    return {"b": b, "c": c, "n_discordant": d, "p": p}


def paired_tests(x, y) -> dict:
    """Paired tests on per-task success shares x, y in [0, 1] (entries averaging k > 1 runs): paired t-test
    (primary), Wilcoxon signed-rank and exact sign test. Degenerate cases: no difference at all gives
    p = 1; a constant non-zero difference (zero variance) gives p_t = 0."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if nz.size == 0:
        return {"mean_diff": 0.0, "p_t": 1.0, "p_wilcoxon": 1.0, "p_sign": 1.0, "n_nonzero": 0}
    if d.std(ddof=1) == 0:
        p_t = 0.0
    else:
        p_t = float(stats.ttest_1samp(d, 0.0).pvalue)
    pw = float(stats.wilcoxon(nz).pvalue)
    pos, neg = int((nz > 0).sum()), int((nz < 0).sum())
    ps = float(stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue)
    return {"mean_diff": float(d.mean()), "p_t": p_t, "p_wilcoxon": pw, "p_sign": ps, "n_nonzero": int(nz.size)}


def is_binary(v) -> bool:
    v = np.asarray(v)
    return bool(np.all((v == 0) | (v == 1)))


def bound_separable(p_hi: float, p_lo: float, n: int, rho: float) -> tuple[bool, float]:
    """(separable under bound (a), bound) with the census rule: below when gap < bound or gap == 0."""
    gap = p_hi - p_lo
    bd = float(C.bound_a(p_hi, p_lo, n, rho))
    return (not (gap < bd or gap == 0)), bd


def classify(bound_sep: bool, test_sep: bool) -> str:
    if bound_sep and test_sep:
        return "both separable"
    if not bound_sep and not test_sep:
        return "both not separable"
    return "bound only" if bound_sep else "test only"


CATS = ("both separable", "both not separable", "bound only", "test only")


def validate_board(lb: str, names: list[str], scores: list[float], outcomes: dict[str, np.ndarray],
                   n: int, rho: float, alpha: float = ALPHA) -> pd.DataFrame:
    """Per-pair table for one board. names/scores are in census order (best first)."""
    m = len(names)
    pairs = [(i, i + 1, "adjacent") for i in range(m - 1)]
    if m >= 5:
        pairs.append((0, 4, "#1 vs #5"))
    rows = []
    for i, j, kind in pairs:
        x, y = outcomes[names[i]], outcomes[names[j]]
        if is_binary(x) and is_binary(y):
            t = mcnemar_exact(x, y)
            test, p = "McNemar exact", t["p"]
            extra = {"b": t["b"], "c": t["c"], "n_discordant": t["n_discordant"]}
        else:
            t = paired_tests(x, y)
            test, p = "paired t", t["p_t"]
            extra = {"mean_diff_share": t["mean_diff"], "p_wilcoxon": t["p_wilcoxon"], "p_sign": t["p_sign"],
                     "n_nonzero_diff": t["n_nonzero"]}
        bsep, bd = bound_separable(scores[i], scores[j], n, rho)
        tsep = bool(p < alpha)
        rows.append({"leaderboard": lb, "pair": kind, "rank_hi": i + 1, "rank_lo": j + 1,
                     "entry_hi": names[i], "entry_lo": names[j], "n_tasks": n,
                     "score_hi": scores[i], "score_lo": scores[j], "gap_pp": 100 * (scores[i] - scores[j]),
                     "rho_used": rho, "bound_pp": 100 * bd, "bound_separable": bsep,
                     "test": test, "p_value": p, "test_separable": tsep, "category": classify(bsep, tsep), **extra})
    return pd.DataFrame(rows)


def agreement_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Counts of the four verdict categories per board and pair set, plus an overall row per pair set.
    Pair sets: 'adjacent' (all adjacent pairs; includes #1 vs #2), '#1 vs #2' (the first adjacent pair),
    '#1 vs #5'."""
    sets = {"adjacent (all)": pairs[pairs["pair"] == "adjacent"],
            "#1 vs #2": pairs[(pairs["pair"] == "adjacent") & (pairs["rank_hi"] == 1)],
            "#1 vs #5": pairs[pairs["pair"] == "#1 vs #5"]}
    rows = []
    for label, d in sets.items():
        groups = [(lb, g) for lb, g in d.groupby("leaderboard", sort=False)] + [("ALL BOARDS", d)]
        for lb, g in groups:
            cnt = g["category"].value_counts()
            n = len(g)
            agree = int(cnt.get("both separable", 0) + cnt.get("both not separable", 0))
            rows.append({"pair_set": label, "leaderboard": lb, "pairs": n,
                         **{c: int(cnt.get(c, 0)) for c in CATS},
                         "agree": agree, "agree_share": agree / n if n else float("nan"),
                         "bound_separable": int(g["bound_separable"].sum()), "test_separable": int(g["test_separable"].sum()),
                         "pairs_mcnemar": int((g["test"] == "McNemar exact").sum()),
                         "pairs_paired_t": int((g["test"] == "paired t").sum())})
    return pd.DataFrame(rows)


def run_validation(entries: list[C.Entry], est, root: Path, alpha: float = ALPHA) -> pd.DataFrame:
    frames = []
    for lb in E.LOADERS:
        names, _ = est.matrices[lb]
        kept = {e.name: e for e in entries if e.lb == lb and e.status == "kept"}
        outc = E.LOADERS[lb](root)
        frames.append(validate_board(lb, names, [kept[nm].score for nm in names], outc,
                                     C.N_VERIFIED[lb], est.rho_used[lb], alpha))
    return pd.concat(frames, ignore_index=True)


# ============================================================================ 2. calibration simulation

SIM_SEED = 20261006
NS = (50, 100, 300, 500)
PS = (0.2, 0.5, 0.8)
RHOS = (0.0, 0.25, 0.5, 0.75)


def joint_probs(p1: float, p2: float, rho: float) -> np.ndarray:
    """Cell probabilities (11, 10, 01, 00) of two Bernoulli outcomes with marginals p1, p2 and phi
    correlation rho. Raises ValueError when (p1, p2, rho) is not feasible."""
    p11 = p1 * p2 + rho * math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    pr = np.array([p11, p1 - p11, p2 - p11, 1 - p1 - p2 + p11])
    if (pr < -1e-12).any():
        raise ValueError(f"infeasible joint distribution for p1={p1}, p2={p2}, rho={rho}: {pr}")
    return np.clip(pr, 0.0, None) / np.clip(pr, 0.0, None).sum()


def separates(P1hat, P2hat, n: int, rho: float):
    """The census rule on plug-in estimates: |D| > 1.96 SE_D, SE_D^2 = SE1^2 + SE2^2 - 2 rho SE1 SE2,
    SE_j = sqrt(Pj_hat (1 - Pj_hat) / n), rho fixed at the supplied value."""
    P1hat, P2hat = np.asarray(P1hat, float), np.asarray(P2hat, float)
    D = P1hat - P2hat
    s1, s2 = C.se(P1hat, n), C.se(P2hat, n)
    sd = np.sqrt(np.maximum(s1 ** 2 + s2 ** 2 - 2.0 * rho * s1 * s2, 0.0))
    return np.abs(D) > Z * sd


def simulate_cell(n: int, p1: float, p2: float, rho: float, reps: int, rng: np.random.Generator,
                  rho_assumed: float | None = None) -> float:
    """Share of ``reps`` simulated task sets (n tasks, outcomes of two entries with marginals p1, p2 and phi
    correlation rho) in which the rule |D| > 1.96 SE_D separates the two entries."""
    pr = joint_probs(p1, p2, rho)
    cnt = rng.multinomial(n, pr, size=reps)  # (reps, 4): n11, n10, n01, n00
    P1hat, P2hat = (cnt[:, 0] + cnt[:, 1]) / n, (cnt[:, 0] + cnt[:, 2]) / n
    return float(separates(P1hat, P2hat, n, rho if rho_assumed is None else rho_assumed).mean())


def exact_rejection_prob(n: int, p1: float, p2: float, rho: float, rho_assumed: float | None = None) -> float:
    """Exact probability that the rule separates the pair, by enumerating the multinomial outcomes
    (feasible for n up to about 100). Used to validate the simulation."""
    pr = joint_probs(p1, p2, rho)
    a, b, c = np.meshgrid(np.arange(n + 1), np.arange(n + 1), np.arange(n + 1), indexing="ij")
    ok = (a + b + c) <= n
    a, b, c = a[ok], b[ok], c[ok]
    d = n - a - b - c
    logp = (math.lgamma(n + 1) - np.vectorize(math.lgamma)(a + 1) - np.vectorize(math.lgamma)(b + 1)
            - np.vectorize(math.lgamma)(c + 1) - np.vectorize(math.lgamma)(d + 1)
            + xlogy(a, pr[0]) + xlogy(b, pr[1]) + xlogy(c, pr[2]) + xlogy(d, pr[3]))
    w = np.exp(logp)
    sep = separates((a + b) / n, (a + c) / n, n, rho if rho_assumed is None else rho_assumed)
    return float(w[sep].sum())


def nominal_bound(n: int, P: float, rho: float) -> float:
    """1.96 sqrt(2 P (1 - P) (1 - rho) / n): the bound of the rule at two equal true scores P."""
    return Z * math.sqrt(2.0 * P * (1 - P) * (1 - rho) / n)


def calibration_grid(ns=NS, Ps=PS, rhos=RHOS, reps: int = 20_000, seed: int = SIM_SEED) -> pd.DataFrame:
    """False-separation rate (true difference 0, both entries at P) and power at a true difference equal to
    the nominal bound (entries at P + delta/2 and P - delta/2). One independent stream per cell."""
    rows = []
    for n in ns:
        for P in Ps:
            for rho in rhos:
                key = [seed, n, int(round(P * 100)), int(round(rho * 100))]
                rng0 = np.random.default_rng(np.random.SeedSequence(key + [0]))
                rng1 = np.random.default_rng(np.random.SeedSequence(key + [1]))
                fs = simulate_cell(n, P, P, rho, reps, rng0)
                delta = nominal_bound(n, P, rho)
                try:
                    pw = simulate_cell(n, P + delta / 2, P - delta / 2, rho, reps, rng1)
                except ValueError:
                    pw = float("nan")
                rows.append({"n": n, "P": P, "rho": rho, "reps": reps, "false_separation_rate": fs,
                             "mc_se_null": math.sqrt(0.05 * 0.95 / reps), "delta_pp": 100 * delta,
                             "power_at_bound": pw})
    return pd.DataFrame(rows)


# ============================================================================ 3. eligibility sensitivity

@dataclass
class Board:
    lb: str
    n: int
    names: list[str]
    scores: list[float]
    ks: list[float]
    rho: float
    rho_source: str
    group: str = "analysed"


def census_row(b: Board) -> dict:
    """Headline statistics of one board: ``census.census_config`` with bound (a) at the board's rho."""
    st = C.census_config(b.scores, b.names, b.ks, lambda p1, p2, k1, k2: C.bound_a(p1, p2, b.n, b.rho))
    return {"leaderboard": b.lb, "group": b.group, "n_tasks": b.n, "rho": b.rho, "rho_source": b.rho_source, **st}


def aggregate_rows(rows: list[dict]) -> dict:
    """Across-board counts and medians, as ``census.aggregates`` for the headline bound."""
    r = pd.DataFrame(rows)
    tiers = r["n_tiers"].to_numpy()
    has5 = r["top1_vs_5_separable"].notna()
    return {
        "n_boards": len(r),
        "top1_vs_2_not_sep": int((~r["top1_vs_2_separable"].astype(bool)).sum()),
        "top1_vs_5_not_sep": int((~r.loc[has5, "top1_vs_5_separable"].astype(bool)).sum()),
        "n_boards_with_5": int(has5.sum()),
        "tiers_median": float(np.median(tiers)), "tiers_min": int(tiers.min()), "tiers_max": int(tiers.max()),
        "tiers_per_entry_median": float(np.median(r["n_tiers"] / r["n_entries"])),
        "all_pairs_below_median": float(r["share_all_below"].median()),
        "all_pairs_below_min": float(r["share_all_below"].min()), "all_pairs_below_max": float(r["share_all_below"].max()),
        "adjacent_below_median": float(r["share_adjacent_below"].median()),
        "adjacent_below_min": float(r["share_adjacent_below"].min()),
        "adjacent_ge_90": int((r["share_adjacent_below"] >= 0.9).sum()),
    }


def load_gaia(root: Path, split: str, lb: str) -> list[C.Entry]:
    """One entry per row of the public GAIA results parquet; score = the row's ``score`` (share solved).
    No entry-level exclusion is applied (the file has no NaN and every score is in [0, 1])."""
    p = root / f"data/raw/leaderboards/gaia/results_public/2023/{split}-00000-of-00001.parquet"
    df = pd.read_parquet(p)
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        e = C.Entry(lb, f"{r.model} / {r.organisation or ''} ({r.date}) [row {i}]", float(r.score), 1.0,
                    extra={"organisation": (r.organisation or "").strip(), "model": r.model})
        out.append(e)
    return out


def dedupe_best_per_organisation(entries: list[C.Entry]) -> list[C.Entry]:
    """One entry per organisation (the row with the highest score; ties by name); a row with an empty
    organisation is keyed by its model name."""
    best: dict[str, C.Entry] = {}
    for e in sorted(entries, key=lambda x: (-x.score, x.name)):
        key = e.extra["organisation"] or f"model:{e.extra['model']}"
        best.setdefault(key, e)
    return list(best.values())


def board_from_entries(lb: str, n: int, es: list[C.Entry], rho: float, source: str, group: str) -> Board:
    kept = [e for e in es if e.status == "kept"]
    return Board(lb, n, [e.name for e in kept], [e.score for e in kept], [e.k for e in kept], rho, source, group)


def eligibility_variants(entries: list[C.Entry], est, root: Path) -> tuple[dict[str, list[Board]], pd.DataFrame]:
    """Boards of each sensitivity variant. Returns ({variant: boards}, table of the added boards)."""
    pooled = est.pooled_median
    base = []
    for lb in C.census_leaderboards(entries):
        kept = [e for e in entries if e.lb == lb and e.status == "kept"]
        base.append(board_from_entries(lb, C.N_VERIFIED[lb], kept, est.rho_used[lb], est.source[lb], "analysed"))
    air_e = C.load_tau2(root, "tau2-bench airline")
    ret_e = C.load_tau2(root, "tau2-bench retail")
    air = board_from_entries("tau2-bench airline", C.N_VERIFIED["tau2-bench airline"], air_e, pooled, "pooled", "added")
    ret114 = board_from_entries("tau2-bench retail", 114, ret_e, pooled, "pooled", "added")
    ret115 = board_from_entries("tau2-bench retail", 115, ret_e, pooled, "pooled", "added")
    gt_e, gv_e = load_gaia(root, "test", "GAIA test"), load_gaia(root, "validation", "GAIA validation")
    gt300 = board_from_entries("GAIA test", 300, gt_e, pooled, "pooled", "added")
    gt301 = board_from_entries("GAIA test", 301, gt_e, pooled, "pooled", "added")
    gtd = board_from_entries("GAIA test", 300, dedupe_best_per_organisation(gt_e), pooled, "pooled", "added")
    gv166 = board_from_entries("GAIA validation", 166, gv_e, pooled, "pooled", "added")
    gv165 = board_from_entries("GAIA validation", 165, gv_e, pooled, "pooled", "added")
    variants = {
        "22 analysed (census)": base,
        "23 = 22 + tau2 airline (10-entry rule relaxed to 8)": base + [air],
        "26 primary: retail n=114, GAIA test n=300 (all rows), GAIA validation n=166": base + [air, ret114, gt300, gv166],
        "26, GAIA validation n=165": base + [air, ret114, gt300, gv165],
        "26, GAIA test n=301": base + [air, ret114, gt301, gv166],
        "26, retail n=115 (paper count)": base + [air, ret115, gt300, gv166],
        "26, GAIA test one row per organisation": base + [air, ret114, gtd, gv166],
    }
    added = pd.DataFrame([census_row(b) for b in (air, ret114, gt300, gv166, gv165, gt301, ret115, gtd)])
    added["variant_note"] = ["airline", "retail n=114", "GAIA test n=300, all rows", "GAIA validation n=166",
                             "GAIA validation n=165", "GAIA test n=301, all rows", "retail n=115",
                             "GAIA test n=300, best row per organisation"]
    return variants, added


def check_against_census(base_rows: list[dict], census_csv: Path) -> dict:
    """Compare the re-computed 22-board headline (a_emp, primary) with the committed census.csv."""
    cen = pd.read_csv(census_csv)
    cen = cen[(cen["bound"] == "a_emp") & (cen["set"] == "primary")].set_index("leaderboard")
    worst = 0.0
    n_ok = 0
    for r in base_rows:
        c = cen.loc[r["leaderboard"]]
        same = (int(c.n_tiers) == r["n_tiers"] and int(c.n_adjacent_below) == r["n_adjacent_below"]
                and int(c.n_all_below) == r["n_all_below"] and bool(c.top1_vs_2_separable) == r["top1_vs_2_separable"]
                and bool(c.top1_vs_5_separable) == r["top1_vs_5_separable"])
        worst = max(worst, abs(float(c.share_all_below) - r["share_all_below"]))
        n_ok += same
    return {"boards_compared": len(base_rows), "boards_identical": int(n_ok), "max_abs_diff_share_all_below": worst}


# ============================================================================ 5-7. multiplicity, ties, TB2 run vs task SD (added 2026-10-07)

def z_bonferroni(k: int, alpha: float = ALPHA) -> float:
    """Two-sided normal critical value for k Bonferroni-adjusted comparisons: Phi^-1(1 - alpha / (2 k))."""
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * k)))


def bound_a_z(p1, p2, n: float, rho: float, z: float):
    """Bound (a) with critical value z in place of 1.96: z sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2)."""
    s1, s2 = C.se(p1, n), C.se(p2, n)
    return z * np.sqrt(np.maximum(s1 ** 2 + s2 ** 2 - 2.0 * rho * s1 * s2, 0.0))


def census_row_z(b: Board, z: float) -> dict:
    """``census_row`` with critical value z (rho fixed at the board's headline rho)."""
    st = C.census_config(b.scores, b.names, b.ks, lambda p1, p2, k1, k2: bound_a_z(p1, p2, b.n, b.rho, z))
    return {"leaderboard": b.lb, "group": b.group, "n_tasks": b.n, "rho": b.rho, "rho_source": b.rho_source, "z": z, **st}


def holm_reject_count(p, alpha: float = ALPHA) -> int:
    """Number of hypotheses rejected by Holm's step-down procedure (sorted p_(k) compared with
    alpha / (M - k + 1); stop at the first failure)."""
    p = np.sort(np.asarray(p, float))
    M = p.size
    if M == 0:
        return 0
    fails = p > alpha / (M - np.arange(M))
    return int(np.argmax(fails)) if fails.any() else M


def holm_all_pairs_share(b: Board, alpha: float = ALPHA) -> dict:
    """Holm over all m(m-1)/2 pairs of a board. Per pair the two-sided normal p-value of gap / SE_D
    (SE_D = bound (a) at z = 1: paired, rho fixed at the board's rho); gap = 0 gives p = 1 and SE_D = 0 with
    gap > 0 gives p = 0. Returns the share of pairs NOT rejected (the analogue of 'below the bound')."""
    idx = C.order(b.scores, b.names)
    s = np.array([b.scores[i] for i in idx])
    ii, jj = np.triu_indices(len(s), 1)
    gap = s[ii] - s[jj]
    sd = bound_a_z(s[ii], s[jj], b.n, b.rho, 1.0)
    pos = sd > 0
    p = np.where(pos, 2.0 * stats.norm.sf(np.divide(gap, sd, out=np.zeros_like(gap), where=pos)), np.where(gap > 0, 0.0, 1.0))
    p = np.where(gap == 0, 1.0, p)
    rej = holm_reject_count(p, alpha)
    return {"n_pairs": int(p.size), "n_rejected": rej, "share_not_rejected": 1.0 - rej / p.size,
            "n_unadjusted_rejected": int((p < alpha).sum())}


def multiplicity_table(boards: list[Board]) -> pd.DataFrame:
    """Per board: unadjusted headline, (i) Bonferroni over the m-1 adjacent pairs, (ii) Bonferroni over the
    2 top-entry comparisons (#1 vs #2, #1 vs #5), (iii) Holm over all pairs."""
    z2 = z_bonferroni(2)
    rows = []
    for b in boards:
        m = len(b.names)
        z_adj = z_bonferroni(m - 1)
        base, adj, top = census_row_z(b, Z), census_row_z(b, z_adj), census_row_z(b, z2)
        h = holm_all_pairs_share(b)
        rows.append({
            "leaderboard": b.lb, "n_entries": m, "z_adjacent_bonferroni": z_adj, "z_top2_bonferroni": z2,
            "unadj_tiers": base["n_tiers"], "unadj_n_adjacent_below": base["n_adjacent_below"],
            "unadj_adjacent_share_below": base["share_adjacent_below"],
            "unadj_all_pairs_share_below": base["share_all_below"],
            "unadj_top1_vs_2_separable": base["top1_vs_2_separable"], "unadj_top1_vs_5_separable": base["top1_vs_5_separable"],
            "adjbonf_tiers": adj["n_tiers"], "adjbonf_adjacent_share_below": adj["share_adjacent_below"],
            "adjbonf_all_pairs_share_below": adj["share_all_below"],
            "top2bonf_top1_vs_2_separable": top["top1_vs_2_separable"], "top2bonf_top1_vs_5_separable": top["top1_vs_5_separable"],
            "holm_all_pairs_share_not_rejected": h["share_not_rejected"], "holm_pairs_rejected": h["n_rejected"],
            "all_pairs": h["n_pairs"], "unadj_pairs_rejected": h["n_unadjusted_rejected"]})
    return pd.DataFrame(rows)


def multiplicity_summary(t: pd.DataFrame) -> dict:
    """Across-board numbers of ``multiplicity_table``."""
    return {
        "n_boards": len(t),
        "unadj_tiers_median": float(t["unadj_tiers"].median()), "unadj_tiers_min": int(t["unadj_tiers"].min()), "unadj_tiers_max": int(t["unadj_tiers"].max()),
        "unadj_adjacent_median": float(t["unadj_adjacent_share_below"].median()),
        "unadj_adjacent_min": float(t["unadj_adjacent_share_below"].min()),
        "unadj_all_pairs_median": float(t["unadj_all_pairs_share_below"].median()),
        "unadj_1v2_not_sep": int((~t["unadj_top1_vs_2_separable"].astype(bool)).sum()),
        "unadj_1v5_not_sep": int((~t["unadj_top1_vs_5_separable"].astype(bool)).sum()),
        "adjbonf_tiers_median": float(t["adjbonf_tiers"].median()), "adjbonf_tiers_min": int(t["adjbonf_tiers"].min()), "adjbonf_tiers_max": int(t["adjbonf_tiers"].max()),
        "adjbonf_adjacent_median": float(t["adjbonf_adjacent_share_below"].median()),
        "adjbonf_adjacent_min": float(t["adjbonf_adjacent_share_below"].min()),
        "adjbonf_all_pairs_median": float(t["adjbonf_all_pairs_share_below"].median()),
        "adjbonf_all_pairs_min": float(t["adjbonf_all_pairs_share_below"].min()),
        "adjbonf_all_pairs_max": float(t["adjbonf_all_pairs_share_below"].max()),
        "top2bonf_1v2_not_sep": int((~t["top2bonf_top1_vs_2_separable"].astype(bool)).sum()),
        "top2bonf_1v5_not_sep": int((~t["top2bonf_top1_vs_5_separable"].astype(bool)).sum()),
        "holm_all_pairs_median": float(t["holm_all_pairs_share_not_rejected"].median()),
        "holm_all_pairs_min": float(t["holm_all_pairs_share_not_rejected"].min()),
        "holm_all_pairs_max": float(t["holm_all_pairs_share_not_rejected"].max()),
    }


def tie_summary(boards: list[Board]) -> dict:
    """Boards whose entries ranked 1 and 2 have exactly equal scores, and the #1 vs #2 verdict (headline
    bound (a)) among the boards without such a tie."""
    tied, untied_not_sep, untied = [], 0, 0
    for b in boards:
        s = sorted(b.scores, reverse=True)
        if s[0] == s[1]:
            tied.append(b.lb)
            continue
        untied += 1
        untied_not_sep += int(not census_row(b)["top1_vs_2_separable"])
    return {"n_boards": len(boards), "tied_boards": tied, "n_tied": len(tied),
            "untied_boards": untied, "untied_1v2_not_sep": untied_not_sep}


def tb2_run_vs_task(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Terminal-Bench 2.0 analysis set: (run_sd / task_sd)^2 from ``tb2_bootstrap.csv`` and the SD of a
    single-run score implied by v, sqrt(v / T), with v = ``mean_p1mp_unbiased`` of
    ``tb2_within_task_variance.csv`` (mean within-task Bernoulli variance) and T = number of tasks."""
    boot = pd.read_csv(results_dir / "tb2_bootstrap.csv")
    wt = pd.read_csv(results_dir / "tb2_within_task_variance.csv")
    d = boot[["submission", "n_trials", "n_tasks", "run_sd", "task_sd"]].merge(
        wt[["submission", "mean_p1mp_unbiased"]].rename(columns={"mean_p1mp_unbiased": "v"}), on="submission")
    d["ratio_sq"] = (d["run_sd"] / d["task_sd"]) ** 2
    d["single_run_sd_from_v"] = np.sqrt(d["v"] / d["n_tasks"])
    out = {"n_submissions": len(d), "ratio_sq_median": float(d["ratio_sq"].median()),
           "ratio_sq_min": float(d["ratio_sq"].min()), "ratio_sq_max": float(d["ratio_sq"].max()),
           "single_run_sd_pp_median": float(100 * d["single_run_sd_from_v"].median()),
           "single_run_sd_pp_min": float(100 * d["single_run_sd_from_v"].min()),
           "single_run_sd_pp_max": float(100 * d["single_run_sd_from_v"].max()),
           "task_sd_pp_median": float(100 * d["task_sd"].median()), "run_sd_pp_median": float(100 * d["run_sd"].median()),
           "v_median": float(d["v"].median()), "n_tasks": int(d["n_tasks"].iloc[0])}
    return d, out


# ============================================================================ report

def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def mdt(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def sep_word(ok) -> str:
    if ok is None or (isinstance(ok, float) and math.isnan(ok)):
        return "n/a"
    return "separable" if ok else "not separable"


def build_report(val: pd.DataFrame, agree: pd.DataFrame, cal: pd.DataFrame, elig_rows: dict[str, list[dict]],
                 added: pd.DataFrame, chk: dict, tb2: dict | None, info: dict, extra: dict | None = None) -> str:
    L: list[str] = []
    w = L.append
    w(f"# Robustness analyses requested by the simulated review ({DATE})")
    w("")
    w("Numbers only. Generated by `analysis/leaderboards/robustness.py` (sections 1 to 3) and `analysis/robustness_tb2.py` (section 4). "
      "New files only: no existing output was changed. Inputs are the local snapshots used by `census.py` (no network access). "
      f"CSV files beside this report: `{PREFIX}_mcnemar_pairs.csv`, `{PREFIX}_mcnemar_agreement.csv`, `{PREFIX}_calibration.csv`, "
      f"`{PREFIX}_eligibility_boards.csv`, `{PREFIX}_eligibility_summary.csv`, `{PREFIX}_tb2_missing_reward.csv` (and `.json`).")
    w("")
    w("## Definitions")
    w("")
    w("- Bound (a) (headline of the census): a pair of entries with scores P1 >= P2 on a board of n tasks is *separable* when gap = P1 - P2 is not below "
      "1.96 sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2), SE = sqrt(P(1-P)/n), and *not separable* otherwise; a tie (gap = 0) is always not separable. "
      f"rho is the board's headline rho: the median over adjacent pairs of the phi correlation of per-task outcomes for the 7 boards with per-task data, "
      f"the pooled median {info['pooled']:.4f} for every other board (`census_summary.md`).")
    w("- Adjacent pair: ranks (i, i+1) in the census order (score descending, ties by entry name). #1 vs #2 is the first adjacent pair; #1 vs #5 is the pair of ranks 1 and 5.")
    w("")
    # ---------------------------------------------------------------- 1
    w("## 1. McNemar validation of the item-sampling bound")
    w("")
    w(f"Test, alpha = {ALPHA}, two-sided, on the entries' per-task outcomes. *Test separable* = p < {ALPHA}. A single-run pair (both outcome vectors 0/1: the four SWE-bench boards and "
      "TheAgentCompany, and Terminal-Bench 2.0 pairs of two one-run entries) uses the exact McNemar test (exact binomial test of the discordant tasks, b = x solves and y fails vs c = the reverse, "
      "p = 0.5; p = 1 when there is no discordant task). A pair with an entry that averages k > 1 runs (Terminal-Bench 1.0 with k = 5; Terminal-Bench 2.0 with k up to 10) has per-task success shares "
      "in [0, 1], not outcomes; for those the **paired t-test on the per-task differences of the shares** is used (n - 1 degrees of freedom); the Wilcoxon signed-rank and exact sign p-values are in the pairs CSV. "
      "Agreement categories: *both separable* (bound and test both separate the pair), *both not separable*, *bound only* (the bound separates, the test does not), *test only* (the test separates, the bound does not). "
      "Agree = both separable + both not separable. The bound is evaluated on the census scores (for Terminal-Bench 2.0 the share of all trials solved, which differs slightly from the mean of the per-task shares "
      "when an entry has unequal trial counts per task); no multiplicity correction is applied to the pair tests.")
    w("")
    for label in ("adjacent (all)", "#1 vs #2", "#1 vs #5"):
        d = agree[agree["pair_set"] == label]
        t = d[["leaderboard", "pairs", "both separable", "both not separable", "bound only", "test only", "agree"]].copy()
        t["agree"] = [f"{a} ({pct(s)})" for a, s in zip(d["agree"], d["agree_share"])]
        t["tests used (McNemar / t)"] = [f"{a} / {b}" for a, b in zip(d["pairs_mcnemar"], d["pairs_paired_t"])]
        w(f"### {label}" + (" (every adjacent pair of the board; #1 vs #2 is one of them)" if label == "adjacent (all)" else ""))
        w("")
        w(mdt(t))
        w("")
    # pooled sentence
    allrow = agree[(agree["pair_set"] == "adjacent (all)") & (agree["leaderboard"] == "ALL BOARDS")].iloc[0]
    sep_both = int(allrow["both separable"])
    w(f"Over all {int(allrow['pairs'])} adjacent pairs of the 7 boards, the bound separates {int(allrow['bound_separable'])} pairs and the test separates {int(allrow['test_separable'])}; "
      f"they agree on {int(allrow['agree'])} ({pct(allrow['agree_share'])}): {sep_both} both separable, {int(allrow['both not separable'])} both not separable, "
      f"{int(allrow['bound only'])} bound only, {int(allrow['test only'])} test only. The pooled count is dominated by SWE-bench Verified (170 of the {int(allrow['pairs'])} pairs); the per-board rows above are not weighted.")
    w("")
    # ---------------------------------------------------------------- 2
    w("## 2. Calibration simulation of bound (a)")
    w("")
    reps = int(cal["reps"].iloc[0])
    w(f"Per task, a pair of Bernoulli outcomes with marginal success probabilities P1, P2 and phi correlation rho (joint cell probability P(1,1) = P1 P2 + rho sqrt(P1(1-P1)P2(1-P2))). "
      f"An entry's score is the share of n tasks solved; each cell uses {reps:,} independent replications and a fixed seed ({SIM_SEED}, one stream per cell). "
      "Rule: *separated* when |D| > 1.96 SE_D, D = P1_hat - P2_hat, SE_D^2 = SE1^2 + SE2^2 - 2 rho SE1 SE2 with SE_j = sqrt(P_j_hat(1 - P_j_hat)/n) from the OBSERVED scores (plug-in, as in the census) and rho fixed at the TRUE value. "
      f"**False-separation rate**: share of replications that are separated when the true difference is 0 (P1 = P2 = P); the nominal value is 0.05; Monte Carlo SE of a rate near 0.05 is {cal['mc_se_null'].iloc[0]:.4f}. "
      "**Power at the bound**: share separated when the true difference equals the nominal bound delta = 1.96 sqrt(2 P(1-P)(1-rho)/n) (entries at P + delta/2 and P - delta/2); for a z-test whose critical value equals the true difference the power is 0.50.")
    w("")
    w(f"Validation of the simulation: for n = 50 and n = 100 the exact false-separation probability (enumeration of every multinomial outcome) differs from the simulated rate by at most "
      f"{info['sim_exact_maxdiff']:.4f} over the 12 (P, rho) cells at n = 50 and {info['sim_exact_maxdiff_100']:.4f} at n = 100 (consistent with the Monte Carlo SE above; differences of about 0.005 between cells are noise).")
    w("")
    for what, col, fmt in (("False-separation rate (true difference 0)", "false_separation_rate", "{:.3f}"),
                           ("Power at a true difference equal to the bound", "power_at_bound", "{:.3f}")):
        w(f"### {what}")
        w("")
        hdr = ["n"] + [f"P={P:g}, rho={r:g}" for P in PS for r in RHOS] + ["min to max over cells"]
        rows = []
        for n in NS:
            d = cal[cal["n"] == n]
            vals = [d[(d["P"] == P) & (d["rho"] == r)][col].iloc[0] for P in PS for r in RHOS]
            rows.append([n] + [fmt.format(v) for v in vals] + [f"{np.nanmin(vals):.3f} to {np.nanmax(vals):.3f}"])
        w("| " + " | ".join(hdr) + " |")
        w("|" + "|".join("---" for _ in hdr) + "|")
        for r_ in rows:
            w("| " + " | ".join(str(x) for x in r_) + " |")
        w("")
        w(f"Over all {len(cal)} cells: min {cal[col].min():.3f}, median {cal[col].median():.3f}, max {cal[col].max():.3f}.")
        w("")
    fs = cal["false_separation_rate"]
    w(f"Cells with a false-separation rate above 0.06: {int((fs > 0.06).sum())} of {len(fs)}; below 0.04: {int((fs < 0.04).sum())} of {len(fs)}. "
      f"The maximum is {fs.max():.3f} (n = {int(cal.loc[fs.idxmax(), 'n'])}, P = {cal.loc[fs.idxmax(), 'P']:g}, rho = {cal.loc[fs.idxmax(), 'rho']:g}).")
    w("")
    # ---------------------------------------------------------------- 3
    w("## 3. Eligibility sensitivity: 26 boards instead of 22")
    w("")
    w(f"The census headline (bound (a), rho = the board's own median for the 7 per-task boards, pooled median {info['pooled']:.4f} for the other 15 and for every added board) recomputed with the four boards "
      "that the written criteria exclude. Added boards: tau2-bench airline (n = 50; 8 kept entries, so the rule of at least 10 entries is relaxed), tau2-bench retail (8 kept entries, n = 114 as in `census.py`; the paper's Table 1 count is 115, variant below), "
      "GAIA test (n = 300 stated in the paper; all 3,817 rows of the results file are entries, k = 1; the scores are multiples of 1/301) and GAIA validation (n = 166 stated in the paper, variant 165; scores are multiples of 1/165). "
      "No entry-level exclusion is applied to the GAIA rows. The results file has repeated submissions by the same system, so a variant keeps the best row per organisation "
      "(empty organisation: keyed by the model name). Quantities are defined as in `census_summary.md` (\"Across the 22 leaderboards\").")
    w("")
    w(f"Check: the 22-board headline recomputed by this script equals the committed `census.csv` (bound a_emp, primary sets) on {chk['boards_identical']} of {chk['boards_compared']} boards "
      f"(tiers, adjacent and all-pair counts and both top-entry verdicts; largest difference in the all-pairs share {chk['max_abs_diff_share_all_below']:.1e}).")
    w("")
    hdr = ["variant", "boards", "#1 not sep. from #2", "#1 not sep. from #5", "tiers: median (min to max)", "median tiers per entry",
           "all pairs below: median (min to max)", "adjacent below: median (min)", "adjacent share >= 90%"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join("---" for _ in hdr) + "|")
    summ = []
    for name, rows in elig_rows.items():
        a = aggregate_rows(rows)
        summ.append({"variant": name, **a})
        w(f"| {name} | {a['n_boards']} | {a['top1_vs_2_not_sep']} of {a['n_boards']} | {a['top1_vs_5_not_sep']} of {a['n_boards_with_5']} | "
          f"{a['tiers_median']:g} ({a['tiers_min']} to {a['tiers_max']}) | {a['tiers_per_entry_median']:.3f} | "
          f"{pct(a['all_pairs_below_median'])} ({pct(a['all_pairs_below_min'])} to {pct(a['all_pairs_below_max'])}) | "
          f"{pct(a['adjacent_below_median'])} ({pct(a['adjacent_below_min'])}) | {a['adjacent_ge_90']} of {a['n_boards']} |")
    w("")
    w("The added boards (rho = pooled median; gap and bound in percentage points in brackets):")
    w("")
    hdr = ["board (variant)", "n", "entries", "adjacent below", "all pairs below", "tiers", "#1 vs #2", "#1 vs #5"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join("---" for _ in hdr) + "|")
    for r in added.itertuples():
        w(f"| {r.leaderboard} ({r.variant_note}) | {r.n_tasks} | {r.n_entries} | {r.n_adjacent_below}/{r.n_adjacent_pairs} ({pct(r.share_adjacent_below)}) | "
          f"{pct(r.share_all_below)} | {r.n_tiers} | {sep_word(r.top1_vs_2_separable)} ({r.gap_1_2_pp:.2f} / {r.bound_1_2_pp:.2f}) | "
          f"{sep_word(r.top1_vs_5_separable)} ({r.gap_1_5_pp:.2f} / {r.bound_1_5_pp:.2f}) |")
    w("")
    # ---------------------------------------------------------------- 4
    w("## 4. Terminal-Bench 2.0: trials with no verifier reward treated as missing")
    w("")
    if tb2 is None:
        w("Not run in this invocation (`--skip-tb2`).")
    else:
        b, m = tb2["baseline"], tb2["missing"]
        w(f"Baseline = the existing analysis: a trial with no verifier reward counts as a failure. Sensitivity = those trials are dropped ({tb2['n_no_reward_trials']} of {tb2['n_trials_total']:,} trials in the snapshot; "
          f"{tb2['n_no_reward_trials_in_analysis_set']} of them belong to the 72 analysis-set submissions) and a task's rate is K/N over the valid trials. "
          f"The 72 submissions and 89 tasks of the existing analysis set are kept, because re-applying the rule of at least 5 trials per task after the drop leaves only {tb2['strict_rule_submissions']} of the 72; "
          f"after the drop {m['cells_N0']} (submission, task) cells have no valid trial (that task is skipped for the submission), {m['cells_N1']} have one, {m['cells_below_5']} have fewer than 5 "
          f"(of {m['n_submissions'] * m['n_tasks']:,} cells). A cell with one valid trial has no resampling variance, so run-to-run SDs of the sensitivity are, if anything, understated. "
          "Missing-aware task-sampling and split-half versions are used (`analysis/robustness_tb2.py` docstring); on complete data they reproduce the existing ones. "
          f"B = {m['n_boot']:,} bootstrap replicates (the full number; the run took about 20 s, so no reduction was needed), {m['n_splits']:,} split-half splits, the seed scheme of `bootstrap_core`. "
          "The baseline column is recomputed by this script's path and equals the committed `rank_inversion_summary.json`, `tb2_bootstrap.csv` and `tb2_within_task_variance.csv` values.")
        w("")
        rows = [
            ("trials in the 72 submissions", f"{b['trials']:,}", f"{m['trials']:,}"),
            ("median run-to-run SD of a submission mean (pp)", f"{b['run_sd_median_pp']:.2f}", f"{m['run_sd_median_pp']:.2f}"),
            ("median task-sampling SD of a submission mean (pp)", f"{b['task_sd_median_pp']:.2f}", f"{m['task_sd_median_pp']:.2f}"),
            ("adjacent pairs with flip probability > 5%, run-to-run (of 71)", f"{b['run_n_adjacent_flip_gt_5pct']} ({pct(b['run_share_adjacent_flip_gt_5pct'])})", f"{m['run_n_adjacent_flip_gt_5pct']} ({pct(m['run_share_adjacent_flip_gt_5pct'])})"),
            ("adjacent pairs with flip probability > 5%, task-sampling (of 71)", f"{b['task_n_adjacent_flip_gt_5pct']} ({pct(b['task_share_adjacent_flip_gt_5pct'])})", f"{m['task_n_adjacent_flip_gt_5pct']} ({pct(m['task_share_adjacent_flip_gt_5pct'])})"),
            ("adjacent pairs above 5% under BOTH schemes (of 71)", f"{b['both_n_adjacent_flip_gt_5pct']}", f"{m['both_n_adjacent_flip_gt_5pct']}"),
            ("all pairs with flip probability > 5%, run-to-run", pct(b["run_share_all_pairs_flip_gt_5pct"]), pct(m["run_share_all_pairs_flip_gt_5pct"])),
            ("all pairs with flip probability > 5%, task-sampling", pct(b["task_share_all_pairs_flip_gt_5pct"]), pct(m["task_share_all_pairs_flip_gt_5pct"])),
            ("separable rank positions (exact), run-to-run / task-sampling", f"{b['run_separable_positions_exact']} / {b['task_separable_positions_exact']}", f"{m['run_separable_positions_exact']} / {m['task_separable_positions_exact']}"),
            ("split-half Spearman, mean over splits", f"{b['split_half_spearman_mean']:.4f}", f"{m['split_half_spearman_mean']:.4f}"),
            ("split-half Spearman-Brown corrected Spearman, mean", f"{b['split_half_spearman_sb_mean']:.4f}", f"{m['split_half_spearman_sb_mean']:.4f}"),
            ("median v (unbiased mean within-task variance), cells with N >= 2", f"{b['v_n2_median']:.4f}", f"{m['v_n2_median']:.4f}"),
            ("median f = v / (P(1-P))", f"{b['f_n2_median']:.4f}", f"{m['f_n2_median']:.4f}"),
        ]
        w(mdt(pd.DataFrame(rows, columns=["quantity (72 submissions)", "no reward = failure (baseline)", "no reward = missing (dropped)"])))
        w("")
        w(f"Definitions: flip probability = share of bootstrap replicates in which the observed lower-ranked submission of a pair scores above the higher-ranked one (ties half); "
          "an adjacent pair is *not separable* when this exceeds 5%; v = mean over tasks of p(1-p) N/(N-1) with p the task success rate over the N valid trials of the submission, "
          "P = the submission's mean over the tasks it covers. "
          f"With `tb2_bootstrap.within_task_summary` as is (a cell with N = 1 counted as a covered task with zero variance) the medians are v = {m['v_existing_median']:.4f} and f = {m['f_existing_median']:.4f}. "
          f"The order of the 72 submissions changes (Spearman correlation of the two orders {tb2['spearman_rank_baseline_vs_missing']:.4f}, largest shift {tb2['max_abs_rank_shift']} places), so a pair is compared by rank position, not by the two submissions.")
        w("")
        if "strict_rule" in tb2:
            s = tb2["strict_rule"]
            w(f"Literal application of the existing pipeline to the reduced table (the 5-trial rule re-applied): {s['n_submissions']} submissions remain, so {s['run_n_adjacent_pairs']} adjacent pairs; "
              f"{s['run_n_adjacent_flip_gt_5pct']} are above 5% run-to-run and {s['task_n_adjacent_flip_gt_5pct']} task-sampling; median run SD {s['run_sd_median_pp']:.2f} pp, median task SD {s['task_sd_median_pp']:.2f} pp, "
          f"split-half Spearman {s['split_half_spearman_mean']:.4f}. These 13 are the submissions with at least 5 valid trials on every task and are not comparable with the 71 pairs above.")
            w("")
    if extra is not None:
        L += build_extra_sections(extra)
    return "\n".join(L) + "\n"


def build_extra_sections(x: dict) -> list[str]:
    """Sections 5 to 7 of the report (added 2026-10-07)."""
    L: list[str] = []
    w = L.append
    ms, mt = x["mult_summary"], x["mult_table"]
    n = ms["n_boards"]
    w("")
    w("## 5. Multiplicity: Bonferroni and Holm versions of the headline")
    w("")
    w(f"The census headline (bound (a), empirical rho, the {n} analysed boards) with the critical value 1.96 replaced by an adjusted one. "
      "(i) Bonferroni over the m - 1 adjacent pairs of a board (m = entries): z = Phi^-1(1 - 0.025 / (m - 1)); this z is used for the adjacent share, "
      "the tiers and the all-pairs share. (ii) Bonferroni over the 2 top-entry comparisons (#1 vs #2, #1 vs #5) of a board: z = Phi^-1(1 - 0.0125) = "
      f"{z_bonferroni(2):.4f}. (iii) Holm over all m(m-1)/2 pairs of a board at familywise alpha = 0.05: each pair's two-sided normal p-value of gap / SE_D "
      "(SE_D = bound (a) at z = 1, rho fixed at the board's rho; gap = 0 gives p = 1), step-down thresholds alpha / (M - k + 1); the share reported is the share of pairs not rejected. "
      f"Everything else is as in `census_summary.md`. Per-board values: `{PREFIX}_multiplicity_boards.csv`.")
    w("")
    zs = mt["z_adjacent_bonferroni"]
    w(f"The adjacent-pair z ranges from {zs.min():.3f} (board with {int(mt.loc[zs.idxmin(), 'n_entries'])} entries) to {zs.max():.3f} (board with {int(mt.loc[zs.idxmax(), 'n_entries'])} entries).")
    w("")
    hdr = ["scheme", "tiers: median (min to max)", "adjacent below: median (min)", "all pairs below: median (min to max)", "#1 not sep. from #2", "#1 not sep. from #5"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join("---" for _ in hdr) + "|")
    na = "(not computed)"
    w(f"| unadjusted, z = 1.96 (headline) | {ms['unadj_tiers_median']:g} ({ms['unadj_tiers_min']} to {ms['unadj_tiers_max']}) | {pct(ms['unadj_adjacent_median'])} ({pct(ms['unadj_adjacent_min'])}) | "
      f"{pct(ms['unadj_all_pairs_median'])} | {ms['unadj_1v2_not_sep']} of {n} | {ms['unadj_1v5_not_sep']} of {n} |")
    w(f"| (i) Bonferroni over the m - 1 adjacent pairs | {ms['adjbonf_tiers_median']:g} ({ms['adjbonf_tiers_min']} to {ms['adjbonf_tiers_max']}) | "
      f"{pct(ms['adjbonf_adjacent_median'])} ({pct(ms['adjbonf_adjacent_min'])}) | {pct(ms['adjbonf_all_pairs_median'])} ({pct(ms['adjbonf_all_pairs_min'])} to {pct(ms['adjbonf_all_pairs_max'])}) | "
      f"{na} | {na} |")
    w(f"| (ii) Bonferroni over the 2 top-entry comparisons, z = {z_bonferroni(2):.4f} | {na} | {na} | {na} | "
      f"{ms['top2bonf_1v2_not_sep']} of {n} | {ms['top2bonf_1v5_not_sep']} of {n} |")
    w(f"| (iii) Holm over all pairs | {na} | {na} | {pct(ms['holm_all_pairs_median'])} ({pct(ms['holm_all_pairs_min'])} to {pct(ms['holm_all_pairs_max'])}) | {na} | {na} |")
    w("")
    w(f"Check: the unadjusted row equals the headline of section 3 and of `census_summary.md` ({x['mult_check']}).")
    w("")
    ts = x["ties"]
    w("## 6. Ties between entries 1 and 2")
    w("")
    w(f"Boards on which the entries ranked 1 and 2 have exactly equal scores: {ts['n_tied']} of {ts['n_boards']}"
      + (": " + ", ".join(ts["tied_boards"]) if ts["tied_boards"] else "") + ". "
      f"A tie is always 'not separable' under the census rule. Among the {ts['untied_boards']} boards without such a tie, #1 is not separable from #2 on "
      f"{ts['untied_1v2_not_sep']} of {ts['untied_boards']} (headline bound (a), empirical rho).")
    w("")
    t7 = x["tb2_rt"]
    w("## 7. Terminal-Bench 2.0: run-to-run versus task-sampling spread")
    w("")
    w(f"Over the {t7['n_submissions']} analysis-set submissions (`tb2_bootstrap.csv`, `tb2_within_task_variance.csv`; per-submission values in `{PREFIX}_tb2_run_vs_task.csv`). "
      "run_sd = SD of the corrected run-to-run bootstrap of the submission mean; task_sd = SD of the task-sampling bootstrap. "
      f"(run_sd / task_sd)^2: median {t7['ratio_sq_median']:.4f} (min {t7['ratio_sq_min']:.4f}, max {t7['ratio_sq_max']:.4f}). "
      f"Median run_sd {t7['run_sd_pp_median']:.2f} pp and median task_sd {t7['task_sd_pp_median']:.2f} pp.")
    w("")
    w(f"SD of a SINGLE-run score implied by v: sqrt(v / T), v = mean over tasks of the unbiased within-task variance p(1-p) N/(N-1) (median v {t7['v_median']:.4f}), T = {t7['n_tasks']} tasks "
      f"(a score from one run per task has variance equal to the sum of the within-task variances divided by T^2, i.e. v / T). Median over the submissions: {t7['single_run_sd_pp_median']:.2f} pp "
      f"(min {t7['single_run_sd_pp_min']:.2f}, max {t7['single_run_sd_pp_max']:.2f}), against the median task-sampling SD of {t7['task_sd_pp_median']:.2f} pp. "
      "The run_sd of the table above is for the mean over the submission's 5 or more runs per task; the single-run SD from v is what one run per task would give.")
    w("")
    if x.get("calib") is not None:
        L += build_section8(x["calib"])
    return L


def build_section8(cal: dict) -> list[str]:
    """Section 8: calibration of the Terminal-Bench 2.0 run-to-run bootstrap (``robustness_tb2_calibration.py``)."""
    import robustness_tb2_calibration as K
    L: list[str] = []
    w = L.append
    sm = K.summarise_calibration(cal)
    ft = K.flip_table(cal)
    pre = "robustness_2026-10-06_tb2_calibration"
    w("## 8. Calibration of the Terminal-Bench 2.0 run-to-run bootstrap (Eq. 6)")
    w("")
    w(f"Truth: the per-task probabilities p_t are the observed per-task success shares of a real submission (the 72 analysis-set submissions of `tb2_long.csv`; "
      f"{cal['n_zero_variance']} of them, {', '.join(cal['zero_variance_submissions'])}, has every task share at 0 or 1, so its true SD is 0 and it is left out of the ratios and the coverage summaries). "
      f"Each simulated dataset has r runs per task, Bernoulli(p_t), for r = 5 and r = 10. The corrected bootstrap "
      f"(p*_t = phat_t + sqrt(r/(r-1)) (ptilde_t - phat_t), B = {cal['n_boot']:,} replicates) is applied to every simulated dataset. {cal['reps']:,} simulated datasets per submission and setting "
      f"for the 10 representative submissions (evenly spaced ranks over the score range) and for each of the {sm['r5']['all_n']} non-degenerate submissions ({cal['reps_all']:,} datasets each). Fixed seed {cal['seed']}. "
      "The bootstrap sum over mixed tasks is drawn as a sum of Binomial(m_k r, k/r) over the counts k of successes, which has exactly the distribution of `bootstrap_core.run_to_run` (checked in the tests) "
      "with fresh random streams. An uncorrected bootstrap (scale 1) is evaluated on the same draws for comparison. "
      "Runtime of the calibration: about 6 minutes at these settings, so no reduction was needed (B = 2,000 as requested). Files: "
      f"`{pre}_submissions.csv`, `{pre}_pairs.csv`, `{pre}_flip_table.csv`, `{pre}_bins.csv`.")
    w("")
    w("### (a) Coverage and (b) SD ratio")
    w("")
    w("Coverage = share of simulated datasets whose 95% percentile interval of the bootstrap mean contains the true mean P = mean_t p_t (target 0.95; "
      f"Monte Carlo SE about 0.007 for {cal['reps']:,} datasets and 0.0008 for the pooled {cal['reps_all'] * sm['r5']['all_n']:,}). SD ratio = mean over datasets of the bootstrap SD divided by the true SD sqrt(sum_t p_t(1-p_t)/r)/T.")
    w("")
    hdr = ["set", "r", "coverage, corrected: mean (min to max)", "coverage, uncorrected: mean", "SD ratio, corrected: median (min to max)", "SD ratio, uncorrected: median"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join("---" for _ in hdr) + "|")
    for r in K.RS:
        s = sm[f"r{r}"]
        w(f"| 10 representative | {r} | {s['rep_cov_mean']:.3f} ({s['rep_cov_min']:.3f} to {s['rep_cov_max']:.3f}) | {s['rep_cov_unc_mean']:.3f} | "
          f"{s['rep_sd_ratio_median']:.3f} ({s['rep_sd_ratio_min']:.3f} to {s['rep_sd_ratio_max']:.3f}) | {s['rep_sd_ratio_unc_median']:.3f} |")
        w(f"| all {s['all_n']} non-degenerate | {r} | {s['all_cov_mean']:.3f} ({s['all_cov_min']:.3f} to {s['all_cov_max']:.3f}) | {s['all_cov_unc_mean']:.3f} | "
          f"{s['all_sd_ratio_median']:.3f} ({s['all_sd_ratio_min']:.3f} to {s['all_sd_ratio_max']:.3f}) | {s['all_sd_ratio_unc_median']:.3f} |")
    w("")
    rep = cal["representative"]
    t = rep[["submission", "r", "P", "coverage_corrected", "coverage_uncorrected", "sd_ratio_corrected", "mean_interval_width_pp_corrected"]].copy()
    t["P"] = (100 * t["P"]).map(lambda v: f"{v:.1f}%")
    t["coverage_corrected"] = t["coverage_corrected"].map(lambda v: f"{v:.3f}")
    t["coverage_uncorrected"] = t["coverage_uncorrected"].map(lambda v: f"{v:.3f}")
    t["sd_ratio_corrected"] = t["sd_ratio_corrected"].map(lambda v: f"{v:.3f}")
    t["mean_interval_width_pp_corrected"] = t["mean_interval_width_pp_corrected"].map(lambda v: f"{v:.2f}")
    t.columns = ["submission", "r", "true mean P", "coverage (corrected)", "coverage (uncorrected)", "SD ratio (corrected)", "mean interval width (pp)"]
    w("Representative submissions:")
    w("")
    w(mdt(t))
    w("")
    w("### (c) Flip-probability calibration")
    w("")
    w("Pairs: each representative submission A and its shifted copy B with p_B = p_A (1 - g / P_A), so A is better by exactly g = 0.5, 1, 2, 3 pp (40 pairs per r). "
      "Per simulated dataset the observed order is the observed one (a tie goes to A) and the bootstrap flip probability is the share of the B bootstrap replicates (paired with the A replicates) in which the observed lower-ranked entry scores above the higher-ranked one, ties counted half. "
      "TRUE flip probability = probability that an independent rerun (fresh r runs per task for both entries) reverses the observed order, ties half, averaged over datasets: "
      "(pi+ + pi0)(pi- + pi0/2) + pi-(pi+ + pi0/2) with pi+, pi0, pi- the exact probabilities that the success-count difference of the two entries is positive, zero, negative (convolution, no simulation). "
      "Averaged over datasets, this true value contains the sampling noise of the observed order as well as of the rerun, whereas the bootstrap flip probability is centred on the observed gap; the table therefore also gives the conditional calibration below.")
    w("")
    hdr = ["r", "true gap (pp)", "observed order = true order (share of datasets)", "true flip probability (mean over 10 pairs)", "mean bootstrap flip, corrected", "mean bootstrap flip, uncorrected",
           "corrected minus true: mean (min to max over pairs)"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join("---" for _ in hdr) + "|")
    for q in ft.itertuples():
        w(f"| {q.r} | {q.true_gap_pp:g} | {q.observed_order_correct:.3f} | {q.true_flip:.3f} | {q.boot_flip_corrected:.3f} | {q.boot_flip_uncorrected:.3f} | "
          f"{q.diff_corrected:+.3f} ({q.diff_corrected_min:+.3f} to {q.diff_corrected_max:+.3f}) |")
    w("")
    w(f"Conditional calibration, pooled over the 40 pairs and the {cal['reps']:,} datasets of each (so {40 * cal['reps']:,} datasets per r): datasets are binned by their bootstrap flip probability f (corrected); "
      "per bin, the mean f and the mean TRUE probability that an independent rerun reverses that dataset's observed order. A calibrated flip probability has mean true reversal equal to mean f. "
      "The last row of each r is the set of datasets declared separable by the 5% rule (f <= 0.05).")
    w("")
    bins = cal["bins"]
    hdr = ["r", "bin of f", "datasets", "share of datasets", "mean f", "mean true reversal probability"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join("---" for _ in hdr) + "|")
    for q in bins.itertuples():
        lab = (f"f <= {q.f_hi:g}" if q.f_lo == 0 and isinstance(q.note, str) else
               (f"0 to {q.f_hi:g}" if q.f_lo == 0 else (f"above {q.f_lo:g}" if q.f_hi == 1.0 else f"{q.f_lo:g} to {q.f_hi:g}")))
        w(f"| {q.r} | {lab}{' (5% rule)' if isinstance(q.note, str) else ''} | {q.datasets:,} | {q.share_of_datasets:.3f} | {q.mean_boot_flip:.3f} | {q.mean_true_reversal:.3f} |")
    w("")
    return L


def tb2_csv(tb2: dict) -> pd.DataFrame:
    rows = []
    keys = [k for k in tb2["baseline"] if k not in ("order", "means")]
    for k in keys:
        row = {"quantity": k, "baseline_no_reward_is_failure": tb2["baseline"][k], "missing_dropped": tb2["missing"][k]}
        if "strict_rule" in tb2:
            row["strict_rule_reapplied"] = tb2["strict_rule"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--out", default=str(ANALYSIS / "results"))
    ap.add_argument("--reps", type=int, default=20_000)
    ap.add_argument("--tb2-n-boot", type=int, default=10_000)
    ap.add_argument("--tb2-n-splits", type=int, default=1000)
    ap.add_argument("--skip-tb2", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    entries = C.load_all(root)
    est = E.estimate(entries, root)
    saved = json.loads((HERE / "results" / "rho_summary.json").read_text(encoding="utf-8"))
    assert abs(saved["pooled_median_of_board_medians"] - est.pooled_median) < 1e-12, "pooled rho differs from rho_summary.json"
    info = {"pooled": est.pooled_median}

    # 1
    val = run_validation(entries, est, root)
    agree = agreement_table(val)
    val.to_csv(out / f"{PREFIX}_mcnemar_pairs.csv", index=False)
    agree.to_csv(out / f"{PREFIX}_mcnemar_agreement.csv", index=False)
    # cross-check the bound verdicts with the committed census_pairs.csv
    cp = pd.read_csv(HERE / "results" / "census_pairs.csv")
    cp = cp[cp["set"] == "primary"]
    adj = val[val["pair"] == "adjacent"].merge(cp, left_on=["leaderboard", "rank_hi", "entry_hi", "entry_lo"],
                                               right_on=["leaderboard", "rank_hi", "entry_hi", "entry_lo"])
    assert len(adj) == int((val["pair"] == "adjacent").sum()) and np.allclose(adj["bound_pp"], adj["bound_a_emp_pp"]), "bound differs from census_pairs.csv"

    # 2
    cal = calibration_grid(reps=a.reps)
    cal.to_csv(out / f"{PREFIX}_calibration.csv", index=False)
    for n in (50, 100):
        ex = [exact_rejection_prob(n, P, P, r) for P in PS for r in RHOS]
        sm = [cal[(cal["n"] == n) & (cal["P"] == P) & (cal["rho"] == r)]["false_separation_rate"].iloc[0] for P in PS for r in RHOS]
        info["sim_exact_maxdiff" if n == 50 else "sim_exact_maxdiff_100"] = float(np.max(np.abs(np.array(ex) - np.array(sm))))

    # 3
    variants, added = eligibility_variants(entries, est, root)
    elig_rows = {name: [census_row(b) for b in boards] for name, boards in variants.items()}
    chk = check_against_census(elig_rows["22 analysed (census)"], HERE / "results" / "census.csv")
    allb = pd.concat([pd.DataFrame(rows).assign(variant=name) for name, rows in elig_rows.items()], ignore_index=True)
    allb.to_csv(out / f"{PREFIX}_eligibility_boards.csv", index=False)
    pd.DataFrame([{"variant": n, **aggregate_rows(r)} for n, r in elig_rows.items()]).to_csv(out / f"{PREFIX}_eligibility_summary.csv", index=False)

    # 4
    tb2 = None
    if not a.skip_tb2:
        import robustness_tb2 as T
        long = pd.read_csv(ANALYSIS / "results" / "tb2_long.csv")
        tb2 = T.run_sensitivity(long, a.tb2_n_boot, a.tb2_n_splits)
        (out / f"{PREFIX}_tb2_missing_reward.json").write_text(json.dumps(tb2, indent=2), encoding="utf-8")
        tb2_csv(tb2).to_csv(out / f"{PREFIX}_tb2_missing_reward.csv", index=False)

    # 5-7 (added 2026-10-07)
    base_boards = variants["22 analysed (census)"]
    mt = multiplicity_table(base_boards)
    mt.to_csv(out / f"{PREFIX}_multiplicity_boards.csv", index=False)
    unadj_ok = all(
        int(t_.unadj_tiers) == r["n_tiers"] and int(t_.unadj_n_adjacent_below) == r["n_adjacent_below"]
        for t_, r in zip(mt.itertuples(), elig_rows["22 analysed (census)"]))
    d7, t7 = tb2_run_vs_task(ANALYSIS / "results")
    d7.to_csv(out / f"{PREFIX}_tb2_run_vs_task.csv", index=False)
    extra = {"mult_table": mt, "mult_summary": multiplicity_summary(mt),
             "mult_check": "tiers and adjacent counts identical on all 22 boards" if unadj_ok else "MISMATCH",
             "ties": tie_summary(base_boards), "tb2_rt": t7}
    import robustness_tb2_calibration as KAL
    extra["calib"] = KAL.load_calibration(out)   # written by robustness_tb2_calibration.py (about 6 min); section 8 is omitted if absent
    (out / f"{PREFIX}.md").write_text(build_report(val, agree, cal, elig_rows, added, chk, tb2, info, extra), encoding="utf-8")
    print(f"wrote {out / (PREFIX + '.md')}")


if __name__ == "__main__":
    main()
