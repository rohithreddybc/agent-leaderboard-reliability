# SPDX-License-Identifier: MIT
"""Robustness analyses, round 4 (2026-10-20), for the leaderboard census (bound (a) headline).

New file; nothing existing is modified. Reuses ``robustness_r3`` (boards, evaluator, k-adjusted bound,
reproduction gate) and writes only files whose names start with ``robustness_2026-10-20``. Inputs: the
committed census outputs (``results/census_entries.csv``, ``census_parameters.json``, ``census.csv``) and
the committed ``analysis/results/robustness_2026-10-19_sec11_leader_set.csv`` (read only). No network access,
no per-task outcomes.

Sections of the report ``robustness_2026-10-20.md``
  B  protocol-compatible subset of the census. Every kept entry is classified from the existing columns of
     ``census_entries.csv``:
       selection-based : ``x_attempts`` is '2+' or a number > 1 (SWE-bench metadata: the submission used
                         several attempts per instance and selected one; the census sensitivity set
                         'attempts = 1 or unset' drops the same entries)
       unknown         : the number of runs is not stated and the census set k = 1 by default, i.e.
                         ``x_k_assumed`` is filled (MLE-bench: seed count not stated), or the board reads a trial
                         count from the source (``x_trials_stated`` / ``x_trials_used`` filled for some entry) and this
                         entry has neither (AndroidWorld: 'Number of trials' blank)
       averaged        : k_runs > 1 (score = mean over k runs: Terminal-Bench, HAL multi-run entries, tau2-bench)
       single-run      : k_runs == 1 and none of the above (``x_attempts`` is '1' or unset)
     Recomputed on single-run + averaged entries, dropping boards left with fewer than 2 entries, with
     the k-adjusted bound of ``robustness_r3.bound_a_kadj`` (each SE times sqrt(1 - f (1 - 1/k)), f = f_TB2 =
     0.369, covariance term at the single-run SEs) at the board's headline rho, and at rho = 0.
     rho is NOT re-estimated on the subset (the per-task estimates belong to the full boards).
  C  item-sampling bound lookup 1.96 sqrt(2 P (1 - P) (1 - rho) / n) and the 80 %-power minimum detectable
     difference (1.43 x bound).
  D  leader-set check: whether the section-11 counts of the round-3 report exclude entry 1, and the set size
     including entry 1.
  E  agreement between the item-sampling bound and the paired test on the 401 adjacent pairs of the seven boards with
     per-task outcomes, with the bound at each board's estimated rho (reproduces round 3, section 13), at rho = 0 and at
     the pooled median rho (0.5218). Input: the committed ``robustness_2026-10-06_mcnemar_pairs.csv`` (read only).

  F  full primary analysis of the protocol-compatible subset (single-run + averaged; selection-based, unknown and MLE-bench
     dropped). Variants: P1 = the section-B subset (AppWorld kept, one run per entry assumed as in the census methods);
     P2 = P1 without the two AppWorld boards. Standard-error choices: (i) single-run binomial SE for every entry, (ii) k-adjusted
     SE for averaged entries (f = 0.369, covariance at the single-run SEs, as section B). Reports #1 vs #2 / #1 vs #5 across rho
     (headline, 0, 0.25, 0.5, 0.75, 0.9), tiers, order confidence, the flat-prior plug-in probability that the observed leader
     has the highest true score, leader-set size and bound/test agreement on the adjacent pairs of the seven per-task boards.

Seed 20261020: only section F is random (the plug-in simulation of F4; stream SeedSequence([SEED, 4, board index in the 22-board
census order, SE choice]), so a board has the same stream in P1 and P2). Sections B to E have no random step.

Usage: python robustness_r4.py [--root <project root>] [--out <output dir>]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
for _p in (str(HERE), str(ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import census as C  # noqa: E402
import robustness as Rb  # noqa: E402
import robustness_r3 as R3  # noqa: E402

SEED = 20261020
DATE = "2026-10-20"
PREFIX = f"robustness_{DATE}"
KEEP = ("single-run", "averaged")
PROTOCOLS = ("single-run", "averaged", "selection-based", "unknown")
POWER80_FACTOR = 1.43          # (1.96 + 0.8416) / 1.96 = 1.429, rounded as in the task statement
LOOKUP_N = (50, 89, 100, 200, 300, 500, 1000, 2294)
LOOKUP_P = (0.3, 0.5, 0.7)
LOOKUP_RHO = (0.0, 0.5)


# ============================================================================ B. protocol classification

def _blank(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""


def _attempts_gt1(x) -> bool:
    """True for an attempts tag that means more than one attempt: '2+', '2', '3', ..., 'best-of-3', 'pass@5'."""
    s = str(x).strip().lower()
    if s in ("", "none", "nan", "1"):
        return False
    if s.endswith("+"):
        return True
    import re
    m = re.search(r"(\d+)", s)
    return bool(m and int(m.group(1)) > 1)


def classify_entries(ent: pd.DataFrame, assumed_boards: tuple[str, ...] = ()) -> pd.DataFrame:
    """Protocol class of every kept entry of ``census_entries.csv`` (dtype str, no NaN fill).

    ``assumed_boards``: boards whose single run is an assumption of the census (the source states no run
    count); their k = 1 entries are classed 'unknown' (used by the strict sensitivity variant)."""
    d = ent[ent["status"] == "kept"].copy()
    d["k_runs"] = pd.to_numeric(d["k_runs"], errors="coerce")
    has_trials_field = {lb: bool(((g["x_trials_stated"].map(lambda v: not _blank(v))) | (g["x_trials_used"].map(lambda v: not _blank(v)))).any())
                        for lb, g in d.groupby("leaderboard")}
    cls, why = [], []
    for r in d.itertuples(index=False):
        k = r.k_runs
        if _attempts_gt1(r.x_attempts):
            c, w = "selection-based", f"x_attempts = {r.x_attempts}"
        elif not _blank(r.x_k_assumed):
            c, w = "unknown", "run count not stated; census sets k = 1 (x_k_assumed)"
        elif has_trials_field[r.leaderboard] and _blank(r.x_trials_stated) and _blank(r.x_trials_used):
            c, w = "unknown", "trial count blank in a board that reads it from the source; census defaults to k = 1"
        elif r.leaderboard in assumed_boards and k == 1:
            c, w = "unknown", "run count not stated by the source; census assumes k = 1 (strict variant)"
        elif not np.isfinite(k) or k < 1:
            c, w = "unknown", f"k_runs = {r.k_runs}"
        elif k > 1:
            c, w = "averaged", f"k_runs = {k:g} (mean over runs)"
        else:
            c, w = "single-run", "k_runs = 1, no multi-attempt tag"
        cls.append(c)
        why.append(w)
    d["protocol"] = cls
    d["protocol_reason"] = why
    return d


def board_protocol_counts(cls: pd.DataFrame, lbs: list[str]) -> pd.DataFrame:
    rows = []
    for lb in lbs:
        g = cls[cls["leaderboard"] == lb]
        row = {"leaderboard": lb, "entries": len(g)}
        for p in PROTOCOLS:
            row[p] = int((g["protocol"] == p).sum())
        row["dropped"] = row["selection-based"] + row["unknown"]
        row["retained"] = row["single-run"] + row["averaged"]
        row["board_kept"] = bool(row["retained"] >= 2)
        rows.append(row)
    return pd.DataFrame(rows)


def subset_boards(boards: list[R3.Board], cls: pd.DataFrame, keep=KEEP, min_entries: int = 2) -> list[R3.Board]:
    """Boards restricted to entries of the kept protocols (census order preserved); boards with fewer than
    ``min_entries`` retained entries are dropped. rho, n and rho source are those of the full board."""
    out = []
    for b in boards:
        g = cls[cls["leaderboard"] == b.lb]
        prot = dict(zip(g["entry"], g["protocol"]))
        if len(prot) != len(g):
            raise ValueError(f"duplicate entry names on {b.lb}")
        mask = np.array([prot[n] in keep for n in b.names])
        if mask.sum() < min_entries:
            continue
        out.append(dataclasses.replace(b, names=[n for n, m in zip(b.names, mask) if m],
                                       scores=b.scores[mask], ks=b.ks[mask]))
    return out


def kadj_rows(boards: list[R3.Board], f: float, rho_zero: bool = False) -> list[dict]:
    """Census statistics under the k-adjusted bound (a) at the board's rho (or rho = 0)."""
    def fn_for(b):
        rho = 0.0 if rho_zero else b.rho
        return lambda p1, p2, k1, k2, ia, ib: R3.bound_a_kadj(p1, p2, k1, k2, b.n, rho, f)
    return R3.evaluate_all(boards, fn_for)


def headline_line(rows: list[dict]) -> dict:
    a = R3.aggregate(rows)
    t = np.array([r["n_tiers"] for r in rows])
    return {"boards": a["n_boards"], "sep12_not": a["top1_vs_2_not_sep"], "sep15_not": a["top1_vs_5_not_sep"],
            "boards_with_5": a["n_boards_with_5"], "tiers_median": a["tiers_median"], "tiers_min": int(t.min()),
            "tiers_max": int(t.max()), "adj_share_median": a["adjacent_below_median"],
            "adj_share_min": a["adjacent_below_min"], "entries": int(sum(r["n_entries"] for r in rows)),
            "adj_pairs": int(sum(r["n_adjacent_pairs"] for r in rows)),
            "adj_below": int(sum(r["n_adjacent_below"] for r in rows))}


def sectionB(boards: list[R3.Board], ent: pd.DataFrame, f: float, assumed_boards: tuple[str, ...] = ()) -> dict:
    lbs = [b.lb for b in boards]
    cls = classify_entries(ent, assumed_boards)
    cls = cls[cls["leaderboard"].isin(lbs)]
    counts = board_protocol_counts(cls, lbs)
    sub = subset_boards(boards, cls)
    out = {"classified": cls, "counts": counts, "subset": sub,
           "dropped_boards": [b.lb for b in boards if b.lb not in {s.lb for s in sub}]}
    out["rows_all_headline"] = R3.ref_rows(boards)
    out["rows_all_kadj"] = kadj_rows(boards, f)
    out["rows_all_kadj_rho0"] = kadj_rows(boards, f, True)
    out["rows_sub_kadj"] = kadj_rows(sub, f)
    out["rows_sub_kadj_rho0"] = kadj_rows(sub, f, True)
    out["rows_sub_headline_nok"] = R3.evaluate_all(sub, lambda b: R3.fn_a(b.n, b.rho))
    out["summary"] = {k: headline_line(out[k]) for k in ("rows_all_headline", "rows_all_kadj", "rows_all_kadj_rho0",
                                                         "rows_sub_kadj", "rows_sub_kadj_rho0", "rows_sub_headline_nok")}
    out["protocol_totals"] = {p: int((cls["protocol"] == p).sum()) for p in PROTOCOLS}
    return out


def board_table(B: dict) -> pd.DataFrame:
    """One row per board: protocol counts, then subset statistics (k-adjusted, headline rho and rho = 0)."""
    by = {r["leaderboard"]: r for r in B["rows_sub_kadj"]}
    by0 = {r["leaderboard"]: r for r in B["rows_sub_kadj_rho0"]}
    full = {r["leaderboard"]: r for r in B["rows_all_kadj"]}
    out = []
    for _, c in B["counts"].iterrows():
        row = c.to_dict()
        lb = row["leaderboard"]
        row["tiers_full_kadj"] = full[lb]["n_tiers"]
        for tag, d in (("", by), ("_rho0", by0)):
            r = d.get(lb)
            row[f"tiers_subset{tag}"] = None if r is None else r["n_tiers"]
            row[f"adj_below_subset{tag}"] = None if r is None else r["n_adjacent_below"]
            row[f"adj_pairs_subset{tag}"] = None if r is None else r["n_adjacent_pairs"]
            row[f"share_adj_below_subset{tag}"] = None if r is None else r["share_adjacent_below"]
            row[f"sep12_subset{tag}"] = None if r is None else bool(r["top1_vs_2_separable"])
            row[f"sep15_subset{tag}"] = None if (r is None or r["top1_vs_5_separable"] is None) else bool(r["top1_vs_5_separable"])
        out.append(row)
    return pd.DataFrame(out)


# ============================================================================ C. item-sampling lookup

def item_bound_pp(n, P, rho):
    """1.96 sqrt(2 P (1 - P) (1 - rho) / n), in percentage points: the paired item-sampling bound for two
    entries with the same score P."""
    return 100.0 * C.Z * np.sqrt(2.0 * np.asarray(P, float) * (1.0 - np.asarray(P, float)) * (1.0 - rho) / np.asarray(n, float))


def lookup_table(ns=LOOKUP_N, Ps=LOOKUP_P, rhos=LOOKUP_RHO) -> pd.DataFrame:
    rows = []
    for rho in rhos:
        for P in Ps:
            for n in ns:
                b = float(item_bound_pp(n, P, rho))
                rows.append({"rho": rho, "P": P, "n": n, "bound_pp": b, "mdd_80pct_power_pp": POWER80_FACTOR * b})
    return pd.DataFrame(rows)


# ============================================================================ D. leader set

def sectionD(boards: list[R3.Board], committed_csv: Path | None) -> dict:
    """Re-derive the section-11 leader sets and confirm what the counts include.

    ``leader_set`` compares entry 1 with entries 2..m (``s[0] - s[1:]``), so 'entries not separable from #1'
    excludes entry 1; the set size is that count plus one."""
    df = R3.sec11_leaders(boards)
    s = np.arange(1, 4, dtype=float)[::-1] / 4.0
    excl = bool(len(R3.leader_set(s, 100, 0.5, C.Z)) == len(s) - 1)
    out = {"table": df, "excludes_entry1": excl}
    for lab in ("uncorrected", "bonferroni"):
        k, sz = df[f"others_{lab}"], df[f"set_size_{lab}"]
        out[lab] = {"others_median": float(k.median()), "others_min": int(k.min()), "others_max": int(k.max()),
                    "size_median": float(sz.median()), "size_min": int(sz.min()), "size_max": int(sz.max()),
                    "size_equals_others_plus_1": bool((sz == k + 1).all()),
                    "only_leader": int((sz == 1).sum()), "share_median": float(df[f"set_share_{lab}"].median())}
    if committed_csv is not None and committed_csv.exists():
        com = pd.read_csv(committed_csv)
        out["matches_committed"] = bool(all(
            (com[c].to_numpy() == df[c].to_numpy()).all() for c in ("others_uncorrected", "others_bonferroni",
                                                                  "set_size_uncorrected", "set_size_bonferroni")))
    else:
        out["matches_committed"] = None
    return out


# ============================================================================ E. bound versus paired test

def bound_verdicts(pairs: pd.DataFrame, rho: float | None = None) -> np.ndarray:
    """Census verdict 'separable' (not (gap < bound or gap == 0)) of the item-sampling bound for every row of a
    pairs table (columns score_hi, score_lo, n_tasks, rho_used), with a common ``rho`` or, when None, each row's
    own ``rho_used``. Bound from ``census.bound_a``; a gap of exactly zero is never separable."""
    hi, lo = pairs["score_hi"].to_numpy(float), pairs["score_lo"].to_numpy(float)
    n = pairs["n_tasks"].to_numpy(float)
    r = pairs["rho_used"].to_numpy(float) if rho is None else float(rho)
    b = np.asarray(C.bound_a(hi, lo, n, r), float)
    gap = hi - lo
    return ~((gap < b) | (gap == 0.0))


def agreement_stats(bound_sep, test_sep) -> dict:
    """Raw agreement, Cohen's kappa and the agreement among pairs that either method separates."""
    a, t = np.asarray(bound_sep, bool), np.asarray(test_sep, bool)
    k = R3.cohen_kappa(a, t)
    either = a | t
    both = int((a & t).sum())
    k.update(n_bound_sep=int(a.sum()), n_test_sep=int(t.sum()), n_both=both, n_bound_only=int((a & ~t).sum()),
             n_test_only=int((~a & t).sum()), n_neither=int((~a & ~t).sum()), n_either=int(either.sum()),
             agree_among_either=float(both / either.sum()) if either.any() else float("nan"))
    return k


def sectionE(pairs_csv: Path, pooled_rho: float) -> dict:
    """Bound/test agreement on the adjacent pairs, with the bound at the board's estimated rho (the round-3 value),
    at rho = 0 and at the pooled median rho. The paired-test verdicts are those of the committed file."""
    df = pd.read_csv(pairs_csv)
    adj = df[df["pair"] == "adjacent"].reset_index(drop=True)
    test = adj["test_separable"].astype(bool).to_numpy()
    out = {"n_pairs": int(len(adj)), "reproduces_file": bool((bound_verdicts(adj) == adj["bound_separable"].astype(bool).to_numpy()).all())}
    for key, rho in (("estimated", None), ("rho0", 0.0), ("pooled", pooled_rho)):
        out[key] = agreement_stats(bound_verdicts(adj, rho), test)
    return out


# ============================================================================ F. full primary analysis of the subset

F_RHOS = (0.0, 0.25, 0.5, 0.75, 0.9)      # fixed-rho settings besides the board's headline rho
SE_CHOICES = ("single", "kadj")
SE_LABEL = {"single": "(i) single-run SE for every entry", "kadj": "(ii) k-adjusted SE (f = 0.369)"}
VARIANT_LABEL = {"P1": "P1 (AppWorld kept, one run assumed)", "P2": "P2 (AppWorld excluded)"}
SIM_REPS = 2000
C_THRESHOLD = 0.8


def primary_variants(sub: list[R3.Board]) -> dict[str, list[R3.Board]]:
    """P1 = the section-B subset (AppWorld kept); P2 = P1 without the AppWorld boards."""
    return {"P1": list(sub), "P2": [b for b in sub if not b.lb.startswith("AppWorld")]}


def bound_fn_se(b: R3.Board, se: str, f: float, rho: float | None = None):
    """Bound function for board ``b`` under SE choice ``se`` ('single' or 'kadj') at ``rho`` (None = headline rho)."""
    r = b.rho if rho is None else float(rho)
    if se == "single":
        return R3.fn_a(b.n, r)
    if se == "kadj":
        return lambda p1, p2, k1, k2, ia, ib: R3.bound_a_kadj(p1, p2, k1, k2, b.n, r, f)
    raise ValueError(se)


def rows_se(boards: list[R3.Board], se: str, f: float, rho: float | None = None) -> list[dict]:
    return R3.evaluate_all(boards, lambda b: bound_fn_se(b, se, f, rho))


def top_counts(rows: list[dict]) -> dict:
    """#1 vs #2 and #1 vs #5 not separable, and how many of them are exact ties (gap = 0)."""
    n12 = [(not r["top1_vs_2_separable"]) for r in rows]
    t12 = [(not r["top1_vs_2_separable"]) and r["gap12"] == 0 for r in rows]
    has5 = [r["top1_vs_5_separable"] is not None for r in rows]
    n15 = [h and (not r["top1_vs_5_separable"]) for r, h in zip(rows, has5)]
    t15 = [h and (not r["top1_vs_5_separable"]) and r["gap15"] == 0 for r, h in zip(rows, has5)]
    return {"boards": len(rows), "boards_with_5": int(sum(has5)), "sep12_not": int(sum(n12)), "tied12": int(sum(t12)),
            "sep15_not": int(sum(n15)), "tied15": int(sum(t15))}


def kadj_variance(p1, p2, k1, k2, n: float, rho: float, f: float):
    """Unclipped variance of the difference in ``robustness_r3.bound_a_kadj`` (default form). Non-positive when
    rho is large enough to cancel the two (scaled) variances."""
    s1, s2 = C.se(p1, n), C.se(p2, n)
    a1 = np.sqrt(1.0 - f * (1.0 - 1.0 / np.maximum(np.asarray(k1, float), 1.0)))
    a2 = np.sqrt(1.0 - f * (1.0 - 1.0 / np.maximum(np.asarray(k2, float), 1.0)))
    return (s1 * a1) ** 2 + (s2 * a2) ** 2 - 2.0 * rho * s1 * s2


def nonpositive_variance_pairs(boards: list[R3.Board], rho: float, f: float) -> dict:
    """Adjacent pairs whose k-adjusted difference variance is <= 0 at a common rho although both entries have a positive SE (the bound is
    then 0: any positive gap is separable, a tie is below the bound), and the adjacent pairs in total. Pairs where an SE is 0 (a score
    of 0 or 1) are not counted: their variance does not depend on rho."""
    bad = tot = 0
    for b in boards:
        v = kadj_variance(b.scores[:-1], b.scores[1:], b.ks[:-1], b.ks[1:], b.n, rho, f)
        both = (C.se(b.scores[:-1], b.n) > 0) & (C.se(b.scores[1:], b.n) > 0)
        bad += int(((v <= 0) & both).sum())
        tot += len(v)
    return {"pairs": tot, "nonpositive": bad}


def order_confidence_F(boards: list[R3.Board], se: str, f: float) -> dict:
    """c = Phi(gap / SE_D) of the adjacent pairs at the headline rho, SE_D = bound / 1.96 under the SE choice."""
    c12, shares, allc = [], [], []
    for b in boards:
        s = b.scores
        gaps = s[:-1] - s[1:]
        bd = bound_fn_se(b, se, f)(s[:-1], s[1:], b.ks[:-1], b.ks[1:], np.arange(b.m - 1), np.arange(1, b.m))
        c = R3.confidence(gaps, np.asarray(bd, float) / C.Z)
        c12.append(float(c[0]))
        shares.append(float((c < C_THRESHOLD).mean()))
        allc.append(c)
    allc = np.concatenate(allc)
    c12 = np.array(c12)
    return {"c12": c12, "board_shares": np.array(shares), "median_c12": float(np.median(c12)), "min_c12": float(c12.min()),
            "max_c12": float(c12.max()), "n_c12_lt": int((c12 < C_THRESHOLD).sum()),
            "pooled_share_lt": float((allc < C_THRESHOLD).mean()), "median_board_share_lt": float(np.median(shares)),
            "pairs": int(len(allc)), "pairs_lt": int((allc < C_THRESHOLD).sum())}


def se_scales(b: R3.Board, se: str, f: float) -> np.ndarray:
    """Per-entry SE scale factor a_j (1 for the single-run SE choice)."""
    if se == "single":
        return np.ones(b.m)
    return np.sqrt(1.0 - f * (1.0 - 1.0 / np.maximum(b.ks, 1.0)))


def simulate_posterior_draws(mu, n: float, rho: float, a, reps: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Draws clip(mu + noise, 0, 1) with noise ~ N(0, Sigma), Sigma_ii = (a_i s_i)^2, Sigma_ij = rho s_i s_j (i != j),
    s_i = sqrt(mu_i (1 - mu_i) / n). With every a_i = 1 this is ``robustness_r3.simulate_scores`` (same random stream). Sigma is
    positive semi-definite iff a_i^2 >= rho for all i and is then sampled with a one-factor representation; otherwise its
    negative eigenvalues are set to zero (second return value False)."""
    mu = np.asarray(mu, float)
    a = np.asarray(a, float)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    se = np.sqrt(np.clip(mu * (1.0 - mu), 0.0, None) / n)
    d = a ** 2 - rho
    if np.all(d >= -1e-12):
        z0 = rng.standard_normal((reps, 1))
        zi = rng.standard_normal((reps, len(mu)))
        noise = se * (math.sqrt(rho) * z0 + np.sqrt(np.clip(d, 0.0, None)) * zi)
        return np.clip(mu + noise, 0.0, 1.0), True
    cov = rho * np.outer(se, se)
    np.fill_diagonal(cov, (a * se) ** 2)
    w, V = np.linalg.eigh(cov)
    z = rng.standard_normal((reps, len(mu)))
    noise = (z * np.sqrt(np.clip(w, 0.0, None))) @ V.T
    return np.clip(mu + noise, 0.0, 1.0), False


def p_leader_highest(b: R3.Board, se: str, f: float, reps: int, rng: np.random.Generator) -> dict:
    """Flat-prior posterior probability that the observed leader has the highest true score, by the plug-in simulation: true
    scores ~ N(observed, Sigma) (flat prior, normal likelihood, SEs evaluated at the observed scores, Sigma as in
    ``simulate_posterior_draws``); P = share of draws whose highest true score belongs to the observed leader. Entries tied for the
    highest observed score form the leader set: a draw counts when its highest true score belongs to any of them."""
    draws, psd = simulate_posterior_draws(b.scores, b.n, b.rho, se_scales(b, se, f), reps, rng)
    top = draws.argmax(axis=1)
    tied = b.scores == b.scores.max()
    return {"p": float(tied[top].mean()), "n_tied_top": int(tied.sum()), "psd": psd}


def leader_set_F(b: R3.Board, se: str, f: float, z: float) -> np.ndarray:
    """Entries 2..m not separable from entry 1 (gap < z * SE_D or gap = 0) under the SE choice, at the headline rho."""
    s = b.scores
    gap = s[0] - s[1:]
    if se == "single":
        bd = Rb.bound_a_z(s[0], s[1:], b.n, b.rho, z)
    else:
        bd = np.asarray(R3.bound_a_kadj(s[0], s[1:], b.ks[0], b.ks[1:], b.n, b.rho, f), float) / C.Z * z
    return (gap < bd) | (gap == 0)


def sectionF_core(variant_boards: dict[str, list[R3.Board]], all_boards: list[R3.Board], f: float, reps: int = SIM_REPS) -> dict:
    """Items 1 to 5 for every variant and SE choice: per-board rows, the rho sweep and the summaries."""
    idx = {b.lb: i for i, b in enumerate(all_boards)}
    sweep, boards_rows, summ = [], [], {}
    for v, bl in variant_boards.items():
        for se in SE_CHOICES:
            head = rows_se(bl, se, f)
            hl = headline_line(head)
            oc = order_confidence_F(bl, se, f)
            pt, ls_u, ls_b, rows_here = [], [], [], []
            for b, r, c12 in zip(bl, head, oc["c12"]):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 4, idx[b.lb], SE_CHOICES.index(se)]))
                pp = p_leader_highest(b, se, f, reps, rng)
                zb = Rb.z_bonferroni(b.m - 1)
                ku, kb = int(leader_set_F(b, se, f, C.Z).sum()), int(leader_set_F(b, se, f, zb).sum())
                pt.append(pp["p"])
                ls_u.append(ku + 1)
                ls_b.append(kb + 1)
                row = {"variant": v, "se": se, "leaderboard": b.lb, "m": b.m, "n_tasks": b.n, "rho": b.rho,
                       "rho_source": b.source, "n_tiers": r["n_tiers"], "n_adjacent_below": r["n_adjacent_below"],
                       "n_adjacent_pairs": r["n_adjacent_pairs"], "share_adjacent_below": r["share_adjacent_below"],
                       "gap12_pp": 100 * r["gap12"], "bound12_pp": 100 * r["bound12"],
                       "sep12": bool(r["top1_vs_2_separable"]), "tied12": bool(r["gap12"] == 0),
                       "sep15": r["top1_vs_5_separable"], "c_1v2": float(c12), "n_tied_top": pp["n_tied_top"],
                       "p_leader_highest": pp["p"], "sigma_psd": pp["psd"], "leader_set_size_uncorrected": ku + 1,
                       "leader_set_size_bonferroni": kb + 1}
                rows_here.append(row)
            boards_rows.extend(rows_here)
            for lab, rho in (("headline", None),) + tuple((f"{x:g}", x) for x in F_RHOS):
                rr = head if rho is None else rows_se(bl, se, f, rho)
                sweep.append({"variant": v, "se": se, "rho_setting": lab, **top_counts(rr)})
            q = np.array(pt)
            summ[(v, se)] = {
                "boards": len(bl), "entries": hl["entries"], "head": hl, "conf": oc,
                "p_median": float(np.median(q)), "p_min": float(q.min()), "p_max": float(q.max()),
                "boards_tied_top": int(sum(r["n_tied_top"] > 1 for r in rows_here)),
                "boards_not_psd": int(sum(not r["sigma_psd"] for r in rows_here)),
                "ls_unc": (float(np.median(ls_u)), int(min(ls_u)), int(max(ls_u)), int(sum(x == 1 for x in ls_u))),
                "ls_bon": (float(np.median(ls_b)), int(min(ls_b)), int(max(ls_b)), int(sum(x == 1 for x in ls_b))),
                "nonpos": {x: nonpositive_variance_pairs(bl, x, f) for x in (0.75, 0.9)} if se == "kadj" else None,
            }
    return {"sweep": pd.DataFrame(sweep), "boards": pd.DataFrame(boards_rows), "summary": summ}


def subset_test_pairs(sub: list[R3.Board], outcomes: dict, f: float) -> pd.DataFrame:
    """Adjacent pairs of the retained entries of every board with per-task outcomes, in the subset's own order. The paired
    test is that of ``robustness.validate_board`` (exact McNemar for two 0/1 vectors, paired t on success shares otherwise;
    two-sided 5 %, no correction); ``bound_separable`` is the single-run bound (a) at the board's rho, ``bound_kadj_separable``
    the k-adjusted bound (f), both with the census rule."""
    out = []
    for b in sub:
        if b.lb not in outcomes:
            continue
        v = Rb.validate_board(b.lb, b.names, list(b.scores), outcomes[b.lb], b.n, b.rho)
        v = v[v["pair"] == "adjacent"].copy()
        hi, lo = v["rank_hi"].to_numpy() - 1, v["rank_lo"].to_numpy() - 1
        sh, sl = v["score_hi"].to_numpy(float), v["score_lo"].to_numpy(float)
        bk = np.asarray(R3.bound_a_kadj(sh, sl, b.ks[hi], b.ks[lo], b.n, b.rho, f), float)
        v["k_hi"], v["k_lo"] = b.ks[hi], b.ks[lo]
        v["bound_kadj_pp"] = 100 * bk
        v["bound_kadj_separable"] = ~(((sh - sl) < bk) | ((sh - sl) == 0))
        out.append(v)
    return pd.concat(out, ignore_index=True)


def agreement_F(pairs: pd.DataFrame) -> dict:
    """Agreement of the bound with the paired test on the table of ``subset_test_pairs``, for each SE choice."""
    test = pairs["test_separable"].astype(bool).to_numpy()
    return {"pairs": int(len(pairs)), "boards": int(pairs["leaderboard"].nunique()),
            "single": agreement_stats(pairs["bound_separable"].astype(bool).to_numpy(), test),
            "kadj": agreement_stats(pairs["bound_kadj_separable"].astype(bool).to_numpy(), test)}


def gate_validate_full(boards: list[R3.Board], outcomes: dict, pairs_csv: Path) -> dict:
    """Recompute the paired-test and single-run bound verdicts for the adjacent pairs of the FULL seven boards and compare
    them with the committed ``robustness_2026-10-06_mcnemar_pairs.csv`` (guards the pair-test code path used for the subset)."""
    com = pd.read_csv(pairs_csv)
    com = com[com["pair"] == "adjacent"].reset_index(drop=True)
    new = []
    for b in boards:
        if b.lb in outcomes:
            v = Rb.validate_board(b.lb, b.names, list(b.scores), outcomes[b.lb], b.n, b.rho)
            new.append(v[v["pair"] == "adjacent"])
    new = pd.concat(new, ignore_index=True)
    key = ["leaderboard", "rank_hi"]
    mg = new.merge(com, on=key, suffixes=("", "_com"))
    same = (len(new) == len(com) == len(mg) and (mg["entry_hi"] == mg["entry_hi_com"]).all()
            and (mg["test_separable"].astype(bool) == mg["test_separable_com"].astype(bool)).all()
            and (mg["bound_separable"].astype(bool) == mg["bound_separable_com"].astype(bool)).all())
    return {"pairs": int(len(new)), "reproduced": bool(same)}


def sectionF(sub: list[R3.Board], all_boards: list[R3.Board], f: float, outcomes: dict | None = None, reps: int = SIM_REPS,
             gate: dict | None = None) -> dict:
    vb = primary_variants(sub)
    out = sectionF_core(vb, all_boards, f, reps)
    out["variants"] = {v: [b.lb for b in bl] for v, bl in vb.items()}
    out["f"] = f
    out["reps"] = reps
    if outcomes is not None:
        pr = subset_test_pairs(sub, outcomes, f)
        out["test_pairs"] = pr
        out["agreement"] = agreement_F(pr)
    out["gate"] = gate
    return out


# ============================================================================ report

def _sumrow(label: str, s: dict) -> list:
    return [label, f"{s['sep12_not']} of {s['boards']}", f"{s['sep15_not']} of {s['boards_with_5']}",
            f"{s['tiers_median']:g} ({s['tiers_min']} to {s['tiers_max']})", R3.pct(s["adj_share_median"]),
            f"{s['adj_below']} of {s['adj_pairs']}", s["entries"]]


SUM_HEADER = ["set / bound", "#1 vs #2 not separable", "#1 vs #5 not separable", "tiers: median (min to max)",
              "adjacent share below the bound: median", "adjacent pairs below the bound (pooled)", "entries"]


def build_report(R: dict) -> str:
    L: list[str] = []
    A = L.append
    B, Bs, D = R["B"], R["B_strict"], R["D"]
    f = R["f"]
    A(f"# Robustness analyses, round 4 ({DATE})\n")
    A("Numbers only. Generated by `analysis/leaderboards/robustness_r4.py` (tests in `analysis/leaderboards/tests/test_robustness_r4.py`; "
      f"seed {SEED}; sections A to E have no random step). New files only: no existing script or output was changed. CSVs beside this report are prefixed `{PREFIX}_`. "
      "Inputs: committed census outputs and the committed `robustness_2026-10-19_sec11_leader_set.csv`.\n")
    A("## 0. Reproduction check\n")
    rep = R["repro"]
    a = rep["aggregate"]
    A(f"Headline recomputed from `census_entries.csv` and `census_parameters.json` (gate: the script stops if it differs): #1 not separable from #2 on {a['top1_vs_2_not_sep']} of 22; "
      f"#1 not separable from #5 on {a['top1_vs_5_not_sep']} of 22; tiers median {a['tiers_median']:g}; adjacent share below the bound, median {R3.pct(a['adjacent_below_median'])}. "
      f"Boards identical to `census.csv`: {rep['boards_identical']} of {rep['boards']}. Reproduced: {'yes' if rep['reproduced'] else 'NO'}.\n")
    A("## B. Protocol-compatible census subset\n")
    A("### B1. Classification rules\n")
    A("Each kept entry of the 22 analysed boards is classified from existing columns of `census_entries.csv` (`k_runs`, `x_attempts`, `x_k_assumed`, `x_trials_stated`, `x_trials_used`) in this order:\n")
    A("1. **selection-based**: `x_attempts` is `2+` or a number above 1 (SWE-bench metadata: several attempts per instance, one selected). `x_attempts` = `1` or unset (`None`) is not a multi-attempt tag.")
    A("2. **unknown**: the number of runs is not stated and the census sets k = 1 by default: `x_k_assumed` filled (MLE-bench: seed count not stated), or, on a board whose source carries a trial count, an entry with neither `x_trials_stated` nor `x_trials_used` (AndroidWorld: 'Number of trials' blank).")
    A("3. **averaged**: `k_runs` > 1 (score is the mean over k runs: Terminal-Bench 1.0 and 2.0, HAL multi-run entries, tau2-bench).")
    A("4. **single-run**: `k_runs` = 1 and none of the above.\n")
    A("No entry name carries a best-of or pass@x tag; the only multi-attempt tag in the data is SWE-bench `x_attempts`. "
      "AppWorld, WebArena, TheAgentCompany, WorkArena and the other single-evaluation boards are single-run: their census k = 1 follows the benchmark's protocol, and for AppWorld only the census note says 'one run per entry is assumed (the file states none)' (strict variant in B5).\n")
    t = B["protocol_totals"]
    A(f"Kept entries on the 22 boards: {sum(t.values())}; single-run {t['single-run']}, averaged {t['averaged']}, selection-based {t['selection-based']}, unknown {t['unknown']}. "
      f"Protocol-compatible (single-run + averaged): {t['single-run'] + t['averaged']}. Per entry: `{PREFIX}_secB_entry_protocol.csv`.\n")
    A("### B2. Entries and boards dropped\n")
    A(f"Boards left with fewer than 2 entries are dropped: **{len(B['dropped_boards'])}** ({', '.join(B['dropped_boards']) or 'none'}). Boards retained: **{len(B['subset'])}** of 22.\n")
    rows = []
    for _, c in B["counts"].iterrows():
        rows.append([c["leaderboard"], c["entries"], c["single-run"], c["averaged"], c["selection-based"], c["unknown"], c["dropped"], c["retained"],
                     "retained" if c["board_kept"] else "board dropped"])
    A(R3.mdt(rows, ["leaderboard", "kept entries", "single-run", "averaged", "selection-based", "unknown", "dropped", "retained", "board"]))
    A("")
    A("### B3. Headline on the protocol-compatible subset\n")
    A(f"k-adjusted bound (a): each SE multiplied by sqrt(1 - f (1 - 1/k)), f = {f:.3f}, covariance term at the single-run SEs (`robustness_r3.bound_a_kadj`, as section 6 of the round-3 report), "
      "at the board's headline rho (not re-estimated on the subset). Single-run entries have k = 1 and are unaffected by the adjustment. "
      "'#1 vs #5' is counted over boards with at least 5 retained entries. Per board: `" + f"{PREFIX}_secB_boards.csv`.\n")
    S = B["summary"]
    A(R3.mdt([_sumrow("all entries, headline bound (a) (round 3)", S["rows_all_headline"]),
              _sumrow("all entries, k-adjusted (round 3, section 6)", S["rows_all_kadj"]),
              _sumrow("**protocol-compatible subset, k-adjusted**", S["rows_sub_kadj"]),
              _sumrow("protocol-compatible subset, no k-adjustment (headline bound)", S["rows_sub_headline_nok"])], SUM_HEADER))
    A("")
    s = S["rows_sub_kadj"]
    A(f"- Subset, k-adjusted: #1 vs #2 not separable on **{s['sep12_not']} of {s['boards']}** retained boards; #1 vs #5 not separable on **{s['sep15_not']} of {s['boards_with_5']}** "
      f"(boards with at least 5 entries); tiers median **{s['tiers_median']:g}** (min {s['tiers_min']}, max {s['tiers_max']}); adjacent share below the bound, median **{R3.pct(s['adj_share_median'])}** (min {R3.pct(s['adj_share_min'])}).\n")
    A("### B4. Counts at rho = 0 on the subset\n")
    A("Same subset and k-adjusted bound with rho = 0 (the unpaired, conservative case).\n")
    A(R3.mdt([_sumrow("all entries, k-adjusted, rho = 0", S["rows_all_kadj_rho0"]),
              _sumrow("**protocol-compatible subset, k-adjusted, rho = 0**", S["rows_sub_kadj_rho0"])], SUM_HEADER))
    A("")
    s0 = S["rows_sub_kadj_rho0"]
    A(f"- Subset at rho = 0: #1 vs #2 not separable on **{s0['sep12_not']} of {s0['boards']}**; #1 vs #5 not separable on **{s0['sep15_not']} of {s0['boards_with_5']}**; "
      f"tiers median **{s0['tiers_median']:g}** (min {s0['tiers_min']}, max {s0['tiers_max']}); adjacent share below the bound, median **{R3.pct(s0['adj_share_median'])}**.\n")
    A("### B5. Strict variant (AppWorld also unknown)\n")
    Ss = Bs["summary"]
    A(f"AppWorld test_normal and test_challenge also classed unknown (the census note says the run count is assumed, not stated). "
      f"Entries: single-run {Bs['protocol_totals']['single-run']}, averaged {Bs['protocol_totals']['averaged']}, selection-based {Bs['protocol_totals']['selection-based']}, unknown {Bs['protocol_totals']['unknown']}. "
      f"Boards retained: {len(Bs['subset'])} (dropped: {', '.join(Bs['dropped_boards']) or 'none'}).\n")
    A(R3.mdt([_sumrow("strict subset, k-adjusted", Ss["rows_sub_kadj"]), _sumrow("strict subset, k-adjusted, rho = 0", Ss["rows_sub_kadj_rho0"])], SUM_HEADER))
    A("")
    A("## C. Item-sampling bound lookup\n")
    A("Bound = 1.96 sqrt(2 P (1 - P) (1 - rho) / n) in percentage points: the 95 % bound on the difference of two entries that both score P on n tasks, with task-level correlation rho. "
      f"Minimum detectable difference at 80 % power (two-sided 5 %) = {POWER80_FACTOR} x bound ((1.96 + 0.84) / 1.96 = 1.43). CSV: `{PREFIX}_secC_lookup.csv` (columns rho, P, n, bound_pp, mdd_80pct_power_pp).\n")
    lk = R["lookup"]
    for title, col in (("Bound (pp)", "bound_pp"), ("80 %-power minimum detectable difference (pp)", "mdd_80pct_power_pp")):
        A(f"**{title}**\n")
        rows = []
        for (rho, P), g in lk.groupby(["rho", "P"], sort=False):
            rows.append([f"{rho:g}", f"{P:g}"] + [f"{v:.2f}" for v in g[col]])
        A(R3.mdt(rows, ["rho", "P"] + [f"n = {n}" for n in LOOKUP_N]))
        A("")
    A("## D. Leader-set check (round 3, section 11)\n")
    A(f"- `leader_set` compares entry 1 with entries 2 to m (`s[0] - s[1:]`), so the section-11 count 'entries not separable from #1' **excludes entry 1**: {'confirmed' if D['excludes_entry1'] else 'NOT confirmed'}. "
      f"Set size = count + 1 on every board: uncorrected {'yes' if D['uncorrected']['size_equals_others_plus_1'] else 'NO'}, Bonferroni {'yes' if D['bonferroni']['size_equals_others_plus_1'] else 'NO'}. "
      f"Recomputed table equals the committed `robustness_2026-10-19_sec11_leader_set.csv`: {D['matches_committed']}.\n")
    rows = []
    for lab, name in (("uncorrected", "uncorrected (z = 1.96)"), ("bonferroni", "Bonferroni")):
        d = D[lab]
        rows.append([name, f"{d['others_median']:g}", f"{d['others_min']} to {d['others_max']}", f"{d['size_median']:g}",
                     f"{d['size_min']} to {d['size_max']}", R3.pct(d["share_median"]), d["only_leader"]])
    A(R3.mdt(rows, ["critical value", "count excluding #1: median", "range", "set size including #1: median", "range",
                    "set share of m: median", "boards where #1 is the only possible leader"]))
    A("")
    A("## E. Bound versus paired test, with the bound at rho = 0 and at the pooled rho\n")
    E = R["E"]
    A(f"The {E['n_pairs']} adjacent pairs of the seven boards with per-task outcomes (`robustness_2026-10-06_mcnemar_pairs.csv`). The paired-test verdict is the file's (exact McNemar, or a paired t-test where an entry averages several runs; two-sided 5 %, no correction); "
      f"the bound verdict is recomputed with `census.bound_a` (separable = not (gap < bound or gap = 0)). With each board's own estimated rho it reproduces the file's bound verdicts: {'yes' if E['reproduces_file'] else 'NO'}. "
      f"Pooled rho = {R['pooled_rho']:.4f} (median of the seven board medians).\n")
    rows = []
    for lab, key in (("bound at the board's estimated rho (headline; round 3, section 13)", "estimated"), ("bound at rho = 0", "rho0"),
                     (f"bound at the pooled rho = {R['pooled_rho']:.4f}", "pooled")):
        v = E[key]
        rows.append([lab, v["n"], v["n_bound_sep"], v["n_test_sep"], v["n_both"], v["n_bound_only"], v["n_test_only"], v["n_neither"],
                     R3.pct(v["agree"]), R3.num(v["kappa"]), f"{v['n_both']} of {v['n_either']} ({R3.pct(v['agree_among_either'])})"])
    A(R3.mdt(rows, ["bound used", "pairs", "bound separable", "test separable", "both separable", "bound only", "test only", "neither",
                    "raw agreement", "Cohen's kappa", "agreement among pairs either method separates"]))
    A("")
    return "\n".join(L) + "\n"


def build_report_F(R: dict) -> str:
    """Section F of the report (appended after section E; the earlier sections are unchanged)."""
    F = R["F"]
    S = F["summary"]
    f = F["f"]
    sw, bd = F["sweep"], F["boards"]
    L: list[str] = []
    A = L.append
    A("## F. Primary analysis on the protocol-compatible subset\n")
    A("### F0. Definitions\n")
    P1b, P2b = F["variants"]["P1"], F["variants"]["P2"]
    dropped2 = [b for b in P1b if b not in P2b]
    A(f"- Subset: single-run and averaged entries (section B1); selection-based and unknown entries and the MLE-bench board dropped. Boards: P1 **{len(P1b)}**, P2 **{len(P2b)}**; "
      f"entries: P1 **{S[('P1', 'single')]['entries']}**, P2 **{S[('P2', 'single')]['entries']}**.")
    A(f"- **P1**: AppWorld kept, its run count assumed to be 1 as stated in the census methods. **P2**: AppWorld excluded (dropped: {', '.join(dropped2)}). "
      "The seven boards with per-task outcomes contain no AppWorld entry, so P1 and P2 coincide on them.")
    A("- **SE (i)**: single-run binomial SE for every entry, bound = 1.96 sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2), SE = sqrt(P(1-P)/n) (`census.bound_a`; no f needed). "
      f"**SE (ii)**: each entry's SE multiplied by sqrt(1 - f (1 - 1/k)), f = {f:.3f}, covariance term at the single-run SEs (`robustness_r3.bound_a_kadj`, as section B); single-run entries are unaffected.")
    A("- rho: the **headline** value is the board's own adjacent-pair median where per-task outcomes exist (7 boards), else the pooled median 0.5218; it is not re-estimated on the subset. "
      "Fixed settings 0, 0.25, 0.5, 0.75 and 0.9 are applied to every board. Items F2 to F6 use the headline rho.")
    A("- Separable: not (gap < bound or gap = 0); a tie is never separable. Tiers: Algorithm 1 (`census.greedy_tiers`). All medians are over boards unless stated.")
    if S[("P1", "kadj")]["nonpos"]:
        n75, n90 = S[("P1", "kadj")]["nonpos"][0.75], S[("P1", "kadj")]["nonpos"][0.9]
        A(f"- Under SE (ii) the variance SE1'^2 + SE2'^2 - 2 rho SE1 SE2 can be zero or negative when rho exceeds the product of the scale factors (about 0.63 at the limit k to infinity, 0.705 at k = 5): "
          f"it is then set to 0 (`np.maximum(var, 0)`), so the bound is 0 and any positive gap is separable. P1, adjacent pairs with non-positive variance (both SEs positive): {n75['nonpositive']} of {n75['pairs']} at rho = 0.75, "
          f"{n90['nonpositive']} of {n90['pairs']} at rho = 0.9; none at the headline rho or below 0.63 (see the 0.75 and 0.9 rows of F1 for the effect). The SE (ii) rows at rho = 0.75 and 0.9 should be read with this in mind.")
    A(f"- Files: `{PREFIX}_secF_boards.csv` (per board, variant and SE choice), `{PREFIX}_secF_rho_sweep.csv`, `{PREFIX}_secF_test_pairs.csv` (adjacent pairs of the subset with test and bound verdicts). Seed {SEED}; only F4 is random.\n")

    A("### F1. #1 vs #2 and #1 vs #5 not separable, by rho\n")
    rows = []
    for v in ("P1", "P2"):
        for se in SE_CHOICES:
            for _, r in sw[(sw["variant"] == v) & (sw["se"] == se)].iterrows():
                lab = "headline" if r["rho_setting"] == "headline" else r["rho_setting"]
                rows.append([v, SE_LABEL[se], lab, f"{r['sep12_not']} of {r['boards']}", r["tied12"], f"{r['sep15_not']} of {r['boards_with_5']}", r["tied15"]])
    A(R3.mdt(rows, ["variant", "SE", "rho", "#1 vs #2 not separable (boards)", "of which tied (gap = 0)", "#1 vs #5 not separable (boards)", "of which tied (gap = 0)"]))
    A("")
    t12 = bd[(bd["variant"] == "P1") & (bd["se"] == "single") & (bd["tied12"])]["leaderboard"].tolist()
    A(f"- Boards where #1 and #2 have identical scores (counted not separable at every rho): {', '.join(t12) if t12 else 'none'}. No board has #1 and #5 tied. "
      "'Of which tied' does not depend on rho or on the SE choice.\n")

    A("### F2. Tiers and adjacent pairs below the bound (headline rho)\n")
    rows = []
    for v in ("P1", "P2"):
        for se in SE_CHOICES:
            h = S[(v, se)]["head"]
            rows.append([v, SE_LABEL[se], h["boards"], h["entries"], f"{h['tiers_median']:g} ({h['tiers_min']} to {h['tiers_max']})",
                         R3.pct(h["adj_share_median"]), R3.pct(h["adj_share_min"]), f"{h['adj_below']} of {h['adj_pairs']}"])
    A(R3.mdt(rows, ["variant", "SE", "boards", "entries", "tiers: median (min to max)", "adjacent share below the bound: median",
                    "min", "adjacent pairs below the bound (pooled)"]))
    A("")

    A("### F3. Order confidence c = Phi(gap / SE_D) (headline rho)\n")
    A("SE_D = bound / 1.96 under the SE choice; a tie has c = 0.5. Share of adjacent pairs with c < 0.8: pooled over all adjacent pairs, and median over boards.\n")
    rows = []
    for v in ("P1", "P2"):
        for se in SE_CHOICES:
            c = S[(v, se)]["conf"]
            rows.append([v, SE_LABEL[se], R3.num(c["median_c12"]), f"{R3.num(c['min_c12'])} to {R3.num(c['max_c12'])}", f"{c['n_c12_lt']} of {S[(v, se)]['boards']}",
                         f"{R3.pct(c['pooled_share_lt'])} ({c['pairs_lt']} of {c['pairs']})", R3.pct(c["median_board_share_lt"])])
    A(R3.mdt(rows, ["variant", "SE", "median c(#1 vs #2)", "range", "boards with c(#1 vs #2) < 0.8", "adjacent pairs with c < 0.8: pooled",
                    "adjacent pairs with c < 0.8: median over boards"]))
    A("")

    A("### F4. Flat-prior posterior probability that the observed leader has the highest true score\n")
    A(f"Plug-in simulation, {F['reps']} replicates per board: with a flat prior and the normal approximation, the true scores are distributed N(observed, Sigma), Sigma_ii = (a_i SE_i)^2, "
      "Sigma_ij = rho SE_i SE_j, SE_i evaluated at the observed score, a_i = 1 under SE (i) and sqrt(1 - f (1 - 1/k_i)) under SE (ii); draws clipped to [0, 1]. "
      "The probability is the share of draws in which the highest true score belongs to the observed leader. **Ties at the top: the tied set counts as the leader** "
      "(a draw is a success when its highest true score belongs to any entry tied for the highest observed score). "
      "With a_i = 1 the simulator is the one of `robustness_r3.simulate_scores`, so this is the round-3 section-2 'P(top correct), plug-in truth' on the subset; "
      "under SE (ii) it is the same quantity with the k-adjusted variances. Headline rho. Seeds: `SeedSequence([20261020, 4, board index, SE choice])`.\n")
    rows = []
    for v in ("P1", "P2"):
        for se in SE_CHOICES:
            s = S[(v, se)]
            rows.append([v, SE_LABEL[se], R3.num(s["p_median"]), f"{R3.num(s['p_min'])} to {R3.num(s['p_max'])}", s["boards_tied_top"], s["boards_not_psd"]])
    A(R3.mdt(rows, ["variant", "SE", "median over boards", "range", "boards with a tied top set", "boards where Sigma was not positive semi-definite"]))
    A("")

    A("### F5. Leader set size including entry 1 (headline rho)\n")
    A("Entries 2 to m whose gap to #1 is below z SE_D (or zero), plus entry 1. Uncorrected z = 1.96; Bonferroni z = Phi^-1(1 - 0.05 / (2 (m - 1))). Median over boards (min to max); last column: boards where #1 is the only possible leader.\n")
    rows = []
    for v in ("P1", "P2"):
        for se in SE_CHOICES:
            u, b_ = S[(v, se)]["ls_unc"], S[(v, se)]["ls_bon"]
            rows.append([v, SE_LABEL[se], f"{u[0]:g} ({u[1]} to {u[2]})", u[3], f"{b_[0]:g} ({b_[1]} to {b_[2]})", b_[3]])
    A(R3.mdt(rows, ["variant", "SE", "uncorrected: median (range)", "uncorrected: #1 only", "Bonferroni: median (range)", "Bonferroni: #1 only"]))
    A("")

    A("### F6. Adjacent-pair bound/test agreement on the seven per-task boards, restricted to subset entries (headline rho)\n")
    if "agreement" in F:
        ag = F["agreement"]
        g = F["gate"]
        A(f"Adjacent pairs are formed among the retained entries in the subset's own order (an entry dropped between two others makes them adjacent), {ag['pairs']} pairs on {ag['boards']} boards "
          "(SWE-bench Verified, Lite, test (full), Multilingual, Terminal-Bench 1.0 and 2.0, TheAgentCompany). The paired test is recomputed from the per-task outcomes with the code of the 2026-10-06 validation "
          "(`robustness.validate_board`: exact McNemar for two 0/1 vectors, paired t-test on success shares where an entry averages several runs; two-sided 5 %, no correction). "
          f"Gate: on the full boards the same code reproduces the committed `robustness_2026-10-06_mcnemar_pairs.csv` adjacent-pair test and bound verdicts ({g['pairs']} pairs): {'yes' if g['reproduced'] else 'NO'}. "
          "P1 and P2 are identical here (no AppWorld board has per-task outcomes). Agreement among pairs either method separates = both separable / (bound separable or test separable). Per pair: `" + f"{PREFIX}_secF_test_pairs.csv`.\n")
        rows = []
        for key, lab in (("single", SE_LABEL["single"]), ("kadj", SE_LABEL["kadj"])):
            v = ag[key]
            rows.append([lab, v["n"], v["n_bound_sep"], v["n_test_sep"], v["n_both"], v["n_bound_only"], v["n_test_only"], v["n_neither"], R3.pct(v["agree"]),
                         R3.num(v["kappa"]), f"{v['n_both']} of {v['n_either']} ({R3.pct(v['agree_among_either'])})"])
        A(R3.mdt(rows, ["SE", "pairs", "bound separable", "test separable", "both", "bound only", "test only", "neither", "raw agreement", "Cohen's kappa",
                        "agreement among pairs either separates"]))
        A("")
    A("### F7. Documentation: how rho and the empirical-Bayes shrinkage are computed\n")
    A(DOC_NOTES)
    return "\n".join(L) + "\n"


DOC_NOTES = """**(a) rho on boards whose entries average k > 1 runs** (`estimate_rho.py`). Neither the Pearson correlation of the per-task success averages nor the correlation of single-run outcomes. The per-task vector of an entry is its share of runs that solved each task (`tb2_outcomes`: `rate = df.groupby(["submission", "task_id"])["success"].mean().unstack("task_id")`; `tb1_outcomes`: resolved trials / total trials per task id; a single-run entry gives 0/1). `rho_matrix` then takes the across-task covariance of these rates and divides by the single-run variance P(1-P) of the entries' mean scores:

    cov = P @ P.T / n - np.outer(mean, mean)
    sd = np.sqrt(mean * (1.0 - mean))
    r = cov / np.outer(sd, sd)

The docstring states the target: "For entries with k > 1 runs it is the expected phi between one randomly chosen run of each entry: the numerator is the across-task covariance of the success rates (runs of different entries are independent given the task) and the denominator is the single-run outcome variance P(1-P), which is the variance the bound uses in SE." Pearson's correlation of the rates themselves is computed (`pearson_matrix`, column `rho_pearson_rates_adjacent_median`) but only recorded; it is larger for k > 1 and is not used. The methods text of `census_summary.md` (written by `census.py`) and `robustness_r3.py` use the single-run-equivalent value together with single-run SEs.

**(b) Empirical-Bayes shrinkage, round 3 section 2** (`robustness_r3.eb_shrink`, `simulate_scores`, `bootstrap_board`). Normal-normal shrinkage of each entry's observed score toward the unweighted board mean:

    mu_i = mean + B_i (obs_i - mean),   B_i = tau2 / (tau2 + SE_i^2),   SE_i^2 = obs_i (1 - obs_i) / n
    tau2 = max(0, var(obs, ddof = 1) - mean(SE_i^2))

tau2 is a method-of-moments estimate: the sample variance of the observed scores (ddof = 1) minus the mean sampling variance, floored at 0 (tau2 = 0 gives B_i = 0 and every mu_i equal to the board mean). The SEs are binomial single-run SEs at the observed scores; no k adjustment, and the correlation rho is not used in the estimate. Simulation (`simulate_scores`, called with the shrunk mu and the board's headline rho): X = mu + se (sqrt(rho) z0 + sqrt(1 - rho) z_i), se_i = sqrt(mu_i (1 - mu_i) / n) at the shrunk scores, z0 and z_i independent N(0, 1), clipped to [0, 1]. The implied covariance of X is Sigma_ii = se_i^2, Sigma_ij = rho se_i se_j: equicorrelated, single-run SEs, one common rho per board (own median or pooled 0.5218). Not verified numerically: with equicorrelated noise the expected sample variance of the observed scores is about tau2 + (1 - rho) mean(SE^2), so subtracting mean(SE^2) rather than (1 - rho) mean(SE^2) would bias tau2 downward by about rho mean(SE^2), i.e. toward more shrinkage.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--out", default=str(ANALYSIS / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = HERE / "results"
    boards, par = R3.load_boards(res)
    rep = R3.reproduce(boards, res / "census.csv")
    if not rep["reproduced"]:
        raise SystemExit("STOP: the headline numbers were not reproduced; no new analysis was computed.")
    ent = pd.read_csv(res / "census_entries.csv", dtype=str, keep_default_na=False)
    f = float(par["f_tb2_median"])
    B = sectionB(boards, ent, f)
    Bs = sectionB(boards, ent, f, assumed_boards=("AppWorld test_normal", "AppWorld test_challenge"))
    lookup = lookup_table()
    D = sectionD(boards, ANALYSIS / "results" / "robustness_2026-10-19_sec11_leader_set.csv")
    pooled = float(par["pooled_median_rho"])
    pairs_csv = ANALYSIS / "results" / "robustness_2026-10-06_mcnemar_pairs.csv"
    E = sectionE(pairs_csv, pooled)
    outcomes, _, _ = R3.load_outcomes(Path(a.root), boards)
    gate = gate_validate_full(boards, outcomes, pairs_csv)
    if not gate["reproduced"]:
        raise SystemExit("STOP: the pair-test code did not reproduce the committed McNemar verdicts; section F was not computed.")
    F = sectionF(B["subset"], boards, f, outcomes, gate=gate)
    R = {"repro": rep, "B": B, "B_strict": Bs, "lookup": lookup, "D": D, "E": E, "F": F, "f": f, "pooled_rho": pooled}
    cols = ["leaderboard", "entry", "score", "k_runs", "x_attempts", "x_k_assumed", "x_trials_stated", "x_trials_used", "protocol", "protocol_reason"]
    B["classified"][cols].to_csv(out / f"{PREFIX}_secB_entry_protocol.csv", index=False)
    board_table(B).to_csv(out / f"{PREFIX}_secB_boards.csv", index=False)
    lookup.to_csv(out / f"{PREFIX}_secC_lookup.csv", index=False)
    D["table"].to_csv(out / f"{PREFIX}_secD_leader_set_incl_entry1.csv", index=False)
    F["boards"].to_csv(out / f"{PREFIX}_secF_boards.csv", index=False)
    F["sweep"].to_csv(out / f"{PREFIX}_secF_rho_sweep.csv", index=False)
    F["test_pairs"].to_csv(out / f"{PREFIX}_secF_test_pairs.csv", index=False)
    (out / f"{PREFIX}.md").write_text(build_report(R) + build_report_F(R), encoding="utf-8")
    summ = {k: v for k, v in B["summary"].items()}
    print(json.dumps({"protocol_totals": B["protocol_totals"], "dropped_boards": B["dropped_boards"], "summary": summ,
                      "strict": Bs["summary"]["rows_sub_kadj"], "D": {k: D[k] for k in ("uncorrected", "bonferroni", "excludes_entry1", "matches_committed")},
                      "E": E, "F_sweep": F["sweep"].to_dict("records"),
                      "F_summary": {f"{k[0]}/{k[1]}": {a_: b_ for a_, b_ in v.items() if a_ != "conf"} for k, v in F["summary"].items()},
                      "F_agreement": F["agreement"]},
                     indent=1, default=str))
    print("wrote", out / f"{PREFIX}.md")


if __name__ == "__main__":
    main()
