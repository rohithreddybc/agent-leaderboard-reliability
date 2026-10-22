"""Terminal-Bench 2.0 sensitivity: trials with no verifier reward treated as MISSING, not as failures.

Added 2026-10-06 (simulated review, robustness analysis 4). Existing scripts and outputs are not
changed; this module only reuses ``bootstrap_core``, ``rank_inversion``, ``tb2_bootstrap`` and
``tb2_load``, and writes new files.

Baseline (existing analysis): a trial with no verifier reward is a failure (``success = 0``).
Sensitivity: those trials (926 of 32,803 in the snapshot) are dropped, so a (submission, task) cell has
N = valid trials and the task rate is K/N over the valid trials.

Analysis set. The existing rule (at least 5 trials on every task) is applied to the FULL table to
choose the 72 submissions and is then HELD FIXED, because re-applying it after the drop leaves only a
handful of submissions (``strict_rule_submissions`` below); the 71 adjacent pairs of the baseline
stay comparable. A cell can therefore have N < 5, N = 1 or N = 0 (task missing for that submission).
Missing-aware versions of the three pieces that assumed N >= 5 / full coverage are used; each equals
the existing function when there is no missing cell (checked by the unit tests):

* run-to-run bootstrap: ``bootstrap_core.run_to_run`` as is (cells with N = 0 are skipped and the
  submission mean is over the tasks it covers; a cell with N = 1 has no resampling variance).
* task-sampling bootstrap: tasks are resampled with one draw shared by all submissions; a submission's
  resampled score is the weighted mean of its task rates over the covered resampled tasks
  (``task_sampling_missing``; equal to ``bootstrap_core.task_sampling`` for complete coverage).
* split-half: cells with fewer than 2 valid trials cannot be halved and are left out of the half-means
  of that submission (``split_half_missing``; equal to ``rank_inversion.split_half`` for N >= 2).

Within-task variance v is reported two ways: ``tb2_bootstrap.within_task_summary`` as is (a cell with
N = 1 counts as a covered task with zero unbiased variance, which biases v down once many cells have
N = 1) and v over cells with N >= 2 only (``v_n2``), the form that is used for the headline of this
sensitivity. f = v / (P (1 - P)), P = submission mean.

Usage: python robustness_tb2.py [--results results] [--n-boot 10000] [--n-splits 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import bootstrap_core as bc
import rank_inversion as R
import tb2_bootstrap
from tb2_load import select_analysis_set


def task_sampling_missing(c: bc.Counts, n_boot: int = bc.B, chunk: int = 1000) -> np.ndarray:
    """Task-sampling bootstrap that tolerates cells with N = 0 (task not covered by a submission).
    Same RNG stream as ``bootstrap_core.task_sampling``; for complete coverage the output is equal."""
    T = len(c.tasks)
    present = c.N > 0
    P = np.divide(c.K, c.N, out=np.zeros(c.K.shape), where=present)
    cover = present.astype(float)
    rng = np.random.default_rng(bc.seed_for(bc.SCHEME_TASK))
    out = np.empty((n_boot, len(c.subs)))
    for a in range(0, n_boot, chunk):
        b = min(chunk, n_boot - a)
        counts = rng.multinomial(T, np.full(T, 1.0 / T), size=b)
        num = counts @ P.T
        den = counts @ cover.T
        out[a:a + b] = num / den
    return out


def split_half_missing(c: bc.Counts, n_splits: int = 1000) -> pd.DataFrame:
    """Split-half reliability (see ``rank_inversion.split_half``) with cells of fewer than 2 trials
    left out of the half-means. Same RNG stream; equal to the original when every cell has N >= 2."""
    rng = np.random.default_rng(bc.seed_for(bc.SCHEME_SPLIT))
    h = c.N // 2
    use = h > 0
    cnt = use.sum(axis=1)
    hs = np.where(use, h, 1)
    m = c.N.sum() / h.sum()
    rows = []
    for k in range(n_splits):
        a = rng.hypergeometric(c.K, c.N - c.K, h)
        b = rng.hypergeometric(c.K - a, (c.N - c.K) - (h - a), h)
        ma = np.where(use, a / hs, 0.0).sum(axis=1) / cnt
        mb = np.where(use, b / hs, 0.0).sum(axis=1) / cnt
        rho = stats.spearmanr(ma, mb).statistic
        tau = stats.kendalltau(ma, mb).statistic
        sb = lambda r: m * r / (1 + (m - 1) * r)
        rows.append((k, rho, tau, sb(rho), sb(tau)))
    return pd.DataFrame(rows, columns=["split", "spearman", "kendall", "spearman_sb", "kendall_sb"])


def variance_n2(c: bc.Counts) -> pd.DataFrame:
    """Per submission: P (mean over covered tasks), v_n2 = mean over cells with N >= 2 of the unbiased
    p(1-p) N/(N-1), and f = v_n2 / (P(1-P)) (NaN at P = 0 or 1)."""
    present = c.N > 0
    p = np.divide(c.K, c.N, out=np.zeros(c.K.shape), where=present)
    n = c.N.astype(float)
    ok = n >= 2
    unb = np.divide(p * (1 - p) * n, n - 1, out=np.zeros(p.shape), where=ok)
    v = unb.sum(axis=1) / ok.sum(axis=1)
    P = p.sum(axis=1) / present.sum(axis=1)
    pq = P * (1 - P)
    f = np.divide(v, pq, out=np.full(P.shape, np.nan), where=pq > 0)
    return pd.DataFrame({"submission": c.subs, "mean_success": P, "v_n2": v, "f_n2": f})


def analyse_missing(long: pd.DataFrame, subs: list[str], tasks: list[str], n_boot: int = bc.B,
                    n_splits: int = 1000) -> dict:
    """Summary statistics of one scheme on a fixed submission list, tolerant of sparse cells."""
    c = bc.build_counts(long, subs, tasks)
    means = bc.macro_mean(c.K, c.N)
    order = bc.observed_order(c.subs, means)
    S_run = bc.run_to_run(c, n_boot)
    S_task = task_sampling_missing(c, n_boot)
    P = {"run": bc.flip_matrix(S_run, order), "task": bc.flip_matrix(S_task, order)}
    out: dict = {
        "n_submissions": len(subs), "n_tasks": len(tasks), "n_boot": n_boot, "n_splits": n_splits,
        "trials": int(c.N.sum()), "cells_N0": int((c.N == 0).sum()), "cells_N1": int((c.N == 1).sum()),
        "cells_below_5": int(((c.N > 0) & (c.N < 5)).sum()),
        "run_sd_median_pp": float(100 * np.median(S_run.std(axis=0, ddof=1))),
        "task_sd_median_pp": float(100 * np.median(S_task.std(axis=0, ddof=1))),
        "run_sd_analytic_median_pp": float(100 * np.median(bc.analytic_run_sd(c))),
    }
    iu = np.triu_indices(len(subs), 1)
    for key in ("run", "task"):
        adj = R.adjacent_shares(P[key])
        out[f"{key}_n_adjacent_pairs"] = adj["n_adjacent_pairs"]
        out[f"{key}_n_adjacent_flip_gt_5pct"] = adj["n_adjacent_flip_gt_5pct"]
        out[f"{key}_share_adjacent_flip_gt_5pct"] = adj["share_adjacent_flip_gt_5pct"]
        out[f"{key}_share_all_pairs_flip_gt_5pct"] = float((P[key][iu] > 0.05).mean())
        out[f"{key}_separable_positions_exact"], out[f"{key}_separable_positions_greedy"] = R.max_separable(P[key])
    # a pair that is not separable under BOTH schemes (flip probability above 5% under each)
    both = sum(1 for i in range(len(subs) - 1) if P["run"][i, i + 1] > 0.05 and P["task"][i, i + 1] > 0.05)
    out["both_n_adjacent_flip_gt_5pct"] = int(both)
    sh = split_half_missing(c, n_splits)
    out["split_half_spearman_mean"] = float(sh["spearman"].mean())
    out["split_half_spearman_sb_mean"] = float(sh["spearman_sb"].mean())
    out["split_half_kendall_mean"] = float(sh["kendall"].mean())
    wt = tb2_bootstrap.within_task_summary(c)
    v2 = variance_n2(c)
    out["v_existing_median"] = float(wt["mean_p1mp_unbiased"].median())
    out["f_existing_median"] = float(wt["f_run_noise_share"].dropna().median())
    out["v_n2_median"] = float(v2["v_n2"].median())
    out["f_n2_median"] = float(v2["f_n2"].dropna().median())
    out["n_f_defined"] = int(v2["f_n2"].notna().sum())
    out["order"] = [c.subs[i] for i in order]
    out["means"] = [float(means[i]) for i in order]
    return out


def run_sensitivity(long: pd.DataFrame, n_boot: int = bc.B, n_splits: int = 1000) -> dict:
    """Baseline (no-reward trials = failures, as in the existing results), missing-reward sensitivity
    (those trials dropped, same 72 submissions) and the strict re-application of the >= 5 trial rule."""
    subs, tasks, _ = select_analysis_set(long)
    dropped = long[long["no_reward"] == 0]
    res = {
        "n_trials_total": int(len(long)), "n_no_reward_trials": int(long["no_reward"].sum()),
        "n_no_reward_trials_in_analysis_set": int(long[long["submission"].isin(subs)]["no_reward"].sum()),
        "baseline": analyse_missing(long, subs, tasks, n_boot, n_splits),
        "missing": analyse_missing(dropped, subs, tasks, n_boot, n_splits),
    }
    strict_subs, _, _ = select_analysis_set(dropped)
    res["strict_rule_submissions"] = len(strict_subs)
    if len(strict_subs) >= 3:
        res["strict_rule"] = analyse_missing(dropped, strict_subs, tasks, n_boot, n_splits)
    # rank agreement between the two orders (same 72 submissions)
    b, m = res["baseline"], res["missing"]
    rb = {s: i for i, s in enumerate(b["order"])}
    rm = {s: i for i, s in enumerate(m["order"])}
    common = sorted(rb)
    res["spearman_rank_baseline_vs_missing"] = float(stats.spearmanr([rb[s] for s in common], [rm[s] for s in common]).statistic)
    res["max_abs_rank_shift"] = int(max(abs(rb[s] - rm[s]) for s in common))
    return res


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--n-boot", type=int, default=bc.B)
    ap.add_argument("--n-splits", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None, help="output json (default: <results>/robustness_2026-10-06_tb2_missing_reward.json)")
    a = ap.parse_args()
    long = pd.read_csv(a.results / "tb2_long.csv")
    res = run_sensitivity(long, a.n_boot, a.n_splits)
    out = a.out or a.results / "robustness_2026-10-06_tb2_missing_reward.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("baseline", "missing", "strict_rule")}, indent=2))


if __name__ == "__main__":
    main()
