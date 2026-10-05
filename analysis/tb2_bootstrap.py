"""Per-submission mean success, bootstrap intervals and within-task variance.

PRIMARY (run-to-run): tasks fixed, trials resampled within each task, B=10,000,
    with the N/(N-1) variance correction of ``bootstrap_core`` (resampling N trials
    with replacement understates the run variance by (N-1)/N).
    Columns ``run_sd`` and ``run_ci_*`` (SD and 95% percentile interval of the
    resampled submission mean) and ``run_sd_analytic`` (closed form, check).
SECONDARY (task-sampling): tasks resampled, per-task trial means fixed.
    Columns ``task_sd`` and ``task_ci_*``.
Seed: ``bootstrap_core.SEED`` = 20261008; see ``bootstrap_core.seed_for``.

Also writes the per-task variance summary used for a model-based
detectability bound: for each submission, the mean over tasks of p(1-p),
where p is the task success rate over trials (plug-in, and the unbiased
version p(1-p) n/(n-1)), and the share of tasks with mixed outcomes.

Usage: python tb2_bootstrap.py [--results results] [--n-boot 10000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap_core as bc
from tb2_load import select_analysis_set


def within_task_summary(c: bc.Counts) -> pd.DataFrame:
    present = c.N > 0
    p = np.divide(c.K, c.N, out=np.zeros(c.K.shape), where=present)
    pq = p * (1 - p)
    n = c.N.astype(float)
    unb = np.divide(pq * n, n - 1, out=np.full(pq.shape, np.nan), where=n > 1)
    cnt = present.sum(axis=1)
    mean = p.sum(axis=1) / cnt
    v_unb = np.nansum(np.where(present, unb, np.nan), axis=1) / cnt
    pq_mean = mean * (1 - mean)
    return pd.DataFrame({
        "submission": c.subs,
        "n_tasks": cnt,
        "mean_success": mean,
        "mean_p1mp_plugin": (pq * present).sum(axis=1) / cnt,
        "mean_p1mp_unbiased": v_unb,
        # share of the single-run outcome variance P(1-P) that is run-to-run noise: f = v / (P(1-P));
        # undefined (NaN) at P = 0 or 1
        "f_run_noise_share": np.divide(v_unb, pq_mean, out=np.full(mean.shape, np.nan), where=pq_mean > 0),
        "share_tasks_mixed": ((c.K > 0) & (c.K < c.N)).sum(axis=1) / cnt,
        "share_tasks_all_success": ((c.K == c.N) & present).sum(axis=1) / cnt,
        "share_tasks_all_fail": ((c.K == 0) & present).sum(axis=1) / cnt,
    })


def run(long: pd.DataFrame, n_boot: int = bc.B, exclude: tuple[str, ...] = ()):
    subs, tasks, _ = select_analysis_set(long)
    subs = [s for s in subs if s not in set(exclude)]
    c = bc.build_counts(long, subs, tasks)
    mean = bc.macro_mean(c.K, c.N)
    S_run = bc.run_to_run(c, n_boot)
    S_task = bc.task_sampling(c, n_boot)
    rlo, rhi = bc.percentile_ci(S_run)
    tlo, thi = bc.percentile_ci(S_task)
    meta = long.groupby("submission")[["agent", "model"]].first().reindex(subs)
    out = pd.DataFrame({
        "submission": subs, "agent": meta["agent"].to_numpy(), "model": meta["model"].to_numpy(),
        "n_trials": c.N.sum(axis=1), "n_tasks": len(tasks), "mean_success": mean,
        "run_sd": S_run.std(axis=0, ddof=1), "run_sd_analytic": bc.analytic_run_sd(c),
        "run_ci_low": rlo, "run_ci_high": rhi,
        "task_sd": S_task.std(axis=0, ddof=1), "task_ci_low": tlo, "task_ci_high": thi,
    }).sort_values("mean_success", ascending=False, kind="stable").reset_index(drop=True)
    return out, within_task_summary(c), c, S_run, S_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--n-boot", type=int, default=bc.B)
    a = ap.parse_args()
    long = pd.read_csv(a.results / "tb2_long.csv")
    out, wt, *_ = run(long, a.n_boot)
    out.to_csv(a.results / "tb2_bootstrap.csv", index=False)
    wt.to_csv(a.results / "tb2_within_task_variance.csv", index=False)
    print(out.head(10).to_string())


if __name__ == "__main__":
    main()
