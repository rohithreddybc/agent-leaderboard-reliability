"""Calibration of the Terminal-Bench 2.0 run-to-run bootstrap (Eq. 6 of the paper), added 2026-10-07.

The corrected run-to-run bootstrap (``bootstrap_core.run_to_run``) resamples the r_t runs of every task
with replacement and rescales each task's resampled rate about its observed rate,
    p*_t = phat_t + sqrt(r_t / (r_t - 1)) (ptilde_t - phat_t).
This module checks it on simulated data whose truth is known.

Simulation. True per-task probabilities p_t are the observed per-task success shares of a real submission
(the 72 analysis-set submissions of ``tb2_long.csv``, as in the baseline analysis). A dataset has r runs on
every task (r = 5 and r = 10), outcomes Bernoulli(p_t); the submission score is the mean of the task rates.
For each of ``reps`` simulated datasets the corrected bootstrap (B replicates) is applied to the simulated
data. Measured:
(a) coverage of the 95% percentile interval for the true mean P = mean_t p_t;
(b) the ratio of the mean bootstrap SD to the true SD sqrt(sum_t p_t (1 - p_t) / r) / T;
(c) for a pair (A, B) with A better by a true gap g: the mean bootstrap flip probability (share of bootstrap
    replicates in which the observed lower-ranked entry scores above the higher-ranked one, ties half,
    as in ``bootstrap_core.flip_matrix``; the order is the observed one, a tie goes to A) against the TRUE
    flip probability, the probability that an independent rerun (fresh r runs per task for both entries,
    same p_t) reverses the observed order, with the same half credit for a tie. Averaged over datasets,
        true = (pi+ + pi0)(pi- + pi0/2) + pi-(pi+ + pi0/2),
    where pi+, pi0, pi- = P(X_A - X_B > 0, = 0, < 0) for the success counts X = sum of independent
    binomials, computed exactly by convolution (``poisson_binomial_pmf``); no Monte Carlo in the truth.
    The partner B of a representative submission A is the shifted copy p_B = p_A (1 - g / P_A), so the true
    gap is exactly g.

Speed. With r runs per task the corrected bootstrap mean depends on the bootstrap draws only through
S = sum over mixed tasks of Binomial(r, phat_t); tasks with the same count k of successes are exchangeable, so
S = sum_k Binomial(m_k r, k / r) with m_k the number of tasks with k successes: r - 1 binomial draws per
replicate instead of one per task, with exactly the same distribution as ``bootstrap_core.run_to_run``
(checked in the tests). The bootstrap RNG streams differ from those of ``bootstrap_core`` (fresh streams, fixed
seed). An uncorrected bootstrap (scale 1) is evaluated on the same draws for comparison.

Usage: python robustness_tb2_calibration.py [--results results] [--reps 1000] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap_core as bc
from tb2_load import select_analysis_set

SEED_CAL = 20261007
RS = (5, 10)
GAPS_PP = (0.5, 1.0, 2.0, 3.0)
N_REPRESENTATIVE = 10


def poisson_binomial_pmf(probs) -> np.ndarray:
    """pmf of the number of successes among independent Bernoulli(probs)."""
    pmf = np.array([1.0])
    for p in np.asarray(probs, float):
        new = np.zeros(pmf.size + 1)
        new[:-1] += pmf * (1.0 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf


def count_pmf(p_task: np.ndarray, r: int) -> np.ndarray:
    """pmf of X = total successes over r runs of every task (r Bernoulli(p_t) trials per task)."""
    return poisson_binomial_pmf(np.repeat(np.asarray(p_task, float), r))


def true_flip_probability(pA: np.ndarray, pB: np.ndarray, r: int) -> dict:
    """Exact true flip probability (see module docstring) for entries with per-task probabilities pA, pB."""
    fa, fb = count_pmf(pA, r), count_pmf(pB, r)
    d = np.convolve(fa, fb[::-1])              # pmf of X_A - X_B, index i <-> difference i - len(fb) + 1
    off = fb.size - 1
    pi_plus, pi0, pi_minus = float(d[off + 1:].sum()), float(d[off]), float(d[:off].sum())
    true = (pi_plus + pi0) * (pi_minus + 0.5 * pi0) + pi_minus * (pi_plus + 0.5 * pi0)
    return {"pi_plus": pi_plus, "pi_zero": pi0, "pi_minus": pi_minus, "true_flip": float(true)}


def bootstrap_means(K: np.ndarray, r: int, B: int, rng: np.random.Generator, scale: float | None = None):
    """Corrected run-to-run bootstrap means for R simulated datasets at once. K: (R, T) success counts out
    of r runs. Returns (phat_mean (R,), dev (R, B)) with bootstrap mean = phat_mean + scale * dev, where
    scale = sqrt(r / (r - 1)) is the correction; dev is the uncorrected deviation of the resampled mean."""
    R, T = K.shape
    S = np.zeros((R, B))
    kmix = np.zeros(R)
    for k in range(1, r):
        m = (K == k).sum(axis=1)
        S += rng.binomial((m * r)[:, None], k / r, size=(R, B))
        kmix += m * k
    dev = (S - kmix[:, None]) / (r * T)
    return K.sum(axis=1) / (r * T), dev


def true_sd(p_task: np.ndarray, r: int) -> float:
    return float(np.sqrt((p_task * (1 - p_task)).sum() / r) / p_task.size)


def simulate_submission(p_task: np.ndarray, r: int, reps: int, B: int, seed) -> dict:
    """Coverage and SD ratio of the bootstrap on ``reps`` simulated datasets of one submission."""
    rng = np.random.default_rng(seed)
    T = p_task.size
    P = float(p_task.mean())
    K = rng.binomial(r, p_task, size=(reps, T))
    ph, dev = bootstrap_means(K, r, B, rng)
    sc = float(np.sqrt(r / (r - 1.0)))
    out = {"P": P, "true_sd": true_sd(p_task, r)}
    for name, s in (("corrected", sc), ("uncorrected", 1.0)):
        S = ph[:, None] + s * dev
        lo, hi = np.quantile(S, [0.025, 0.975], axis=1)
        out[f"coverage_{name}"] = float(((lo <= P) & (P <= hi)).mean())
        out[f"mean_interval_width_pp_{name}"] = float(100 * (hi - lo).mean())
        sd = S.std(axis=1, ddof=1).mean()
        out[f"mean_boot_sd_{name}"] = float(sd)
        out[f"sd_ratio_{name}"] = float(sd / out["true_sd"]) if out["true_sd"] > 0 else float("nan")
    return out


def _int_scores(ph, dev, scale):
    return np.rint((ph[:, None] + scale * dev) * 1e9).astype(np.int64)


def simulate_pair(pA: np.ndarray, pB: np.ndarray, r: int, reps: int, B: int, seed) -> dict:
    """Mean bootstrap flip probability (corrected and uncorrected) over ``reps`` simulated datasets of the
    pair, and the exact true flip probability. A is the better entry; the observed order breaks a tie for A."""
    rng = np.random.default_rng(seed)
    T = pA.size
    KA, KB = rng.binomial(r, pA, size=(reps, T)), rng.binomial(r, pB, size=(reps, T))
    phA, devA = bootstrap_means(KA, r, B, rng)
    phB, devB = bootstrap_means(KB, r, B, rng)
    a_hi = (KA.sum(axis=1) >= KB.sum(axis=1))
    sc = float(np.sqrt(r / (r - 1.0)))
    tf = true_flip_probability(pA, pB, r)
    q_a_hi = tf["pi_minus"] + 0.5 * tf["pi_zero"]   # true P(rerun reverses the order) when A is observed on top
    q_b_hi = tf["pi_plus"] + 0.5 * tf["pi_zero"]    # ... when B is observed on top
    true_cond = np.where(a_hi, q_a_hi, q_b_hi)
    out = {"gap_pp": 100 * float(pA.mean() - pB.mean()), "share_observed_order_correct": float(a_hi.mean()),
           "mean_observed_gap_pp": float(100 * ((KA.sum(axis=1) - KB.sum(axis=1)) / (r * T)).mean())}
    for name, s in (("corrected", sc), ("uncorrected", 1.0)):
        a, b = _int_scores(phA, devA, s), _int_scores(phB, devB, s)
        lo_gt_hi = np.where(a_hi[:, None], b > a, a > b).mean(axis=1)
        tie = (a == b).mean(axis=1)
        f = lo_gt_hi + 0.5 * tie
        out[f"mean_boot_flip_{name}"] = float(f.mean())
        out[f"_f_{name}"] = f
    out["_true_cond"] = true_cond
    out.update(tf)
    return out


def shifted_copy(p_task: np.ndarray, gap: float) -> np.ndarray:
    """p_B = p_A (1 - gap / P_A): mean lower by exactly ``gap`` (a fraction, not pp)."""
    P = float(p_task.mean())
    if gap >= P:
        raise ValueError("gap must be smaller than the mean success rate")
    return p_task * (1.0 - gap / P)


def representative(means: np.ndarray, zero_var: np.ndarray, n: int = N_REPRESENTATIVE) -> list[int]:
    """Indices of n submissions spread over the score range (evenly spaced ranks), skipping submissions
    whose observed task shares are all 0 or 1 (no within-task variance)."""
    order = [i for i in np.argsort(means) if not zero_var[i]]
    pos = np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))
    return [int(order[p]) for p in pos]


def load_truths(long: pd.DataFrame):
    subs, tasks, _ = select_analysis_set(long)
    c = bc.build_counts(long, subs, tasks)
    p = c.K / c.N
    zero_var = ((p > 0) & (p < 1)).sum(axis=1) == 0
    return subs, p, p.mean(axis=1), zero_var


def run_calibration(long: pd.DataFrame, reps: int = 1000, n_boot: int = 2000, reps_all: int | None = None) -> dict:
    subs, p, means, zero_var = load_truths(long)
    rep = representative(means, zero_var)
    reps_all = reps if reps_all is None else reps_all
    rows_sub, rows_pair, rows_all, ds = [], [], [], []
    for r in RS:
        for idx in rep:
            ss = np.random.SeedSequence([SEED_CAL, r, idx, 0])
            res = simulate_submission(p[idx], r, reps, n_boot, ss)
            rows_sub.append({"submission": subs[idx], "r": r, "set": "representative", "reps": reps, **res})
            for g in GAPS_PP:
                pB = shifted_copy(p[idx], g / 100.0)
                ss = np.random.SeedSequence([SEED_CAL, r, idx, 1, int(round(g * 10))])
                rp = simulate_pair(p[idx], pB, r, reps, n_boot, ss)
                ds.append(pd.DataFrame({"r": r, "true_gap_pp": g, "f_corrected": rp.pop("_f_corrected"),
                                        "f_uncorrected": rp.pop("_f_uncorrected"), "true_reversal_given_order": rp.pop("_true_cond")}))
                rows_pair.append({"submission_A": subs[idx], "r": r, "true_gap_pp": g, "reps": reps, **rp})
        for idx in range(len(subs)):
            if zero_var[idx]:
                continue
            ss = np.random.SeedSequence([SEED_CAL, r, idx, 2])
            res = simulate_submission(p[idx], r, reps_all, n_boot, ss)
            rows_all.append({"submission": subs[idx], "r": r, "set": "all", "reps": reps_all, **res})
    return {"representative": pd.DataFrame(rows_sub), "pairs": pd.DataFrame(rows_pair), "all": pd.DataFrame(rows_all),
            "bins": calibration_bins(pd.concat(ds, ignore_index=True)),
            "n_submissions": len(subs), "n_zero_variance": int(zero_var.sum()),
            "zero_variance_submissions": [subs[i] for i in np.where(zero_var)[0]],
            "reps": reps, "reps_all": reps_all, "n_boot": n_boot, "seed": SEED_CAL}


def summarise_calibration(cal: dict) -> dict:
    """Across-submission summaries used in the report."""
    rep, allp, pairs = cal["representative"], cal["all"], cal["pairs"]
    out = {}
    for r in RS:
        a, b = rep[rep["r"] == r], allp[allp["r"] == r]
        out[f"r{r}"] = {
            "rep_cov_min": float(a["coverage_corrected"].min()), "rep_cov_median": float(a["coverage_corrected"].median()),
            "rep_cov_max": float(a["coverage_corrected"].max()), "rep_cov_mean": float(a["coverage_corrected"].mean()),
            "rep_cov_unc_mean": float(a["coverage_uncorrected"].mean()),
            "rep_sd_ratio_min": float(a["sd_ratio_corrected"].min()), "rep_sd_ratio_median": float(a["sd_ratio_corrected"].median()),
            "rep_sd_ratio_max": float(a["sd_ratio_corrected"].max()), "rep_sd_ratio_unc_median": float(a["sd_ratio_uncorrected"].median()),
            "all_n": int(len(b)), "all_cov_mean": float(b["coverage_corrected"].mean()),
            "all_cov_min": float(b["coverage_corrected"].min()), "all_cov_median": float(b["coverage_corrected"].median()),
            "all_cov_max": float(b["coverage_corrected"].max()), "all_cov_unc_mean": float(b["coverage_uncorrected"].mean()),
            "all_sd_ratio_median": float(b["sd_ratio_corrected"].median()), "all_sd_ratio_min": float(b["sd_ratio_corrected"].min()),
            "all_sd_ratio_max": float(b["sd_ratio_corrected"].max()), "all_sd_ratio_unc_median": float(b["sd_ratio_uncorrected"].median()),
            "all_reps_total": int(b["reps"].sum()),
        }
    return out


def flip_table(cal: dict) -> pd.DataFrame:
    """Per (r, true gap): across the representative submissions, mean true flip probability, mean bootstrap flip
    probability (corrected, uncorrected), their difference and the range of the difference."""
    d = cal["pairs"].copy()
    d["diff_corrected"] = d["mean_boot_flip_corrected"] - d["true_flip"]
    d["diff_uncorrected"] = d["mean_boot_flip_uncorrected"] - d["true_flip"]
    g = d.groupby(["r", "true_gap_pp"])
    return g.agg(n_pairs=("true_flip", "size"), true_flip=("true_flip", "mean"),
                 boot_flip_corrected=("mean_boot_flip_corrected", "mean"), boot_flip_uncorrected=("mean_boot_flip_uncorrected", "mean"),
                 diff_corrected=("diff_corrected", "mean"), diff_corrected_min=("diff_corrected", "min"), diff_corrected_max=("diff_corrected", "max"),
                 diff_uncorrected=("diff_uncorrected", "mean"), observed_order_correct=("share_observed_order_correct", "mean"),
                 ).reset_index()


BIN_EDGES = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50)


def calibration_bins(ds: pd.DataFrame, edges=BIN_EDGES) -> pd.DataFrame:
    """Conditional calibration of the bootstrap flip probability f: datasets (pairs x replications) are binned by
    f (corrected, bins (lo, hi], the first bin includes 0); per bin the mean f and the mean TRUE probability that an
    independent rerun reverses that dataset's observed order. Pooled over the representative pairs and gaps, per r."""
    rows = []
    for r, g in ds.groupby("r"):
        f = g["f_corrected"].to_numpy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (f > lo) & (f <= hi) if lo > 0 else (f >= 0) & (f <= hi)
            if m.sum() == 0:
                continue
            rows.append({"r": r, "f_lo": lo, "f_hi": hi, "datasets": int(m.sum()), "share_of_datasets": float(m.mean()),
                         "mean_boot_flip": float(f[m].mean()),
                         "mean_true_reversal": float(g["true_reversal_given_order"].to_numpy()[m].mean())})
        m = f > edges[-1]
        rows.append({"r": r, "f_lo": edges[-1], "f_hi": 1.0, "datasets": int(m.sum()), "share_of_datasets": float(m.mean()),
                     "mean_boot_flip": float(f[m].mean()) if m.any() else float("nan"),
                     "mean_true_reversal": float(g["true_reversal_given_order"].to_numpy()[m].mean()) if m.any() else float("nan")})
        m = f <= 0.05  # the 5% rule: declared separable
        rows.append({"r": r, "f_lo": 0.0, "f_hi": 0.05, "datasets": int(m.sum()), "share_of_datasets": float(m.mean()),
                     "mean_boot_flip": float(f[m].mean()), "mean_true_reversal": float(g["true_reversal_given_order"].to_numpy()[m].mean()),
                     "note": "all datasets with f <= 0.05 (separable under the 5% rule)"})
    return pd.DataFrame(rows)


def load_calibration(results: Path) -> dict | None:
    """Rebuild the ``run_calibration`` result from the CSV files written by ``main`` (None if absent)."""
    pre = results / "robustness_2026-10-06_tb2_calibration"
    files = [Path(f"{pre}_submissions.csv"), Path(f"{pre}_pairs.csv"), Path(f"{pre}_bins.csv")]
    if not all(f.exists() for f in files):
        return None
    sub = pd.read_csv(files[0])
    meta = json.loads(Path(f"{pre}_meta.json").read_text(encoding="utf-8"))
    return {"representative": sub[sub["set"] == "representative"].reset_index(drop=True), "all": sub[sub["set"] == "all"].reset_index(drop=True),
            "pairs": pd.read_csv(files[1]), "bins": pd.read_csv(files[2]), **meta}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--reps-all", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <results>)")
    a = ap.parse_args()
    out = a.out or a.results
    long = pd.read_csv(a.results / "tb2_long.csv")
    cal = run_calibration(long, a.reps, a.n_boot, a.reps_all)
    pre = "robustness_2026-10-06_tb2_calibration"
    pd.concat([cal["representative"], cal["all"]]).to_csv(out / f"{pre}_submissions.csv", index=False)
    cal["pairs"].to_csv(out / f"{pre}_pairs.csv", index=False)
    flip_table(cal).to_csv(out / f"{pre}_flip_table.csv", index=False)
    cal["bins"].to_csv(out / f"{pre}_bins.csv", index=False)
    meta = {k: cal[k] for k in ("n_submissions", "n_zero_variance", "zero_variance_submissions", "reps", "reps_all", "n_boot", "seed")}
    (out / f"{pre}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(summarise_calibration(cal), indent=2))


if __name__ == "__main__":
    main()
