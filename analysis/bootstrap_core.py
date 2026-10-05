"""Shared resampling code for the Terminal-Bench 2.0 re-analysis (RQ3).

Data model. For one analysis set of S submissions and T tasks, ``K[s, t]`` is
the number of successful trials and ``N[s, t]`` the number of trials of
submission ``s`` on task ``t``. A submission score is the macro mean
``mean_t K[s,t] / N[s,t]`` over the tasks the submission covers.

Two resampling schemes, both with B = 10,000 replicates and the fixed master
seed ``SEED`` (see ``seed_for``):

PRIMARY, run-to-run ("trial-only"): tasks are fixed. For each submission
independently, the N[s,t] trials of each task are resampled with replacement.
A resample of N trials from a task with k successes has Binomial(N, k/N)
successes, so the resample is drawn directly from that binomial.

Small-sample correction (decision log 2026-10-04, adjudication of simulated
review 1, M2). Resampling N trials with replacement from N trials gives a
task-level resample variance of phat(1-phat)/N, whose expectation is only
(N-1)/N of the true p(1-p)/N. The scheme therefore rescales each task's
resampled success rate about its observed rate,
    x* = phat + (X/N - phat) * sqrt(N / (N - 1)),
so that the resample variance is phat(1-phat)/(N-1), the unbiased estimate of
p(1-p)/N. The mean over tasks is then a rescaled bootstrap whose variance equals
(1/T^2) sum_t phat_t(1-phat_t)/(N_t-1) (``analytic_run_sd``). With the same N on
every task this is the bootstrap SD scaled by sqrt(N/(N-1)) (1.118 for N = 5).
Rescaling is done per task, so cells with N = 5, 6 or 10 trials are each
corrected by their own factor; the replicates remain draws from the same
streams (same seeds), so a rescaled replicate is a deterministic function of
the uncorrected one. A cell with N = 1 carries no information about its
variance and is left unscaled (the analysis set requires N >= 5). Rescaled task
rates can lie slightly outside [0, 1] for tasks with mixed outcomes; scores are
compared as they are. ``run_to_run(..., correct=False)`` reproduces the
uncorrected scheme.

SECONDARY, task-sampling: the per-task trial means are fixed. Tasks are
resampled with replacement, with the same task draw applied to every
submission in the replicate (the submissions share one task set, so the
comparison is paired).

There is no nested (tasks then trials) scheme: it counts the within-task
variance twice, once through the trial resample and again through the
duplicated tasks.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20261008
B = 10_000
SCHEME_RUN, SCHEME_TASK, SCHEME_SPLIT = 1, 2, 3  # second element of the seed key


def seed_for(scheme: int, name: str | None = None) -> np.random.SeedSequence:
    """Seed sequence for one purpose. Per-submission streams are keyed by a
    CRC32 of the submission name, so a submission's draws do not depend on
    which other submissions are in the analysis set."""
    key = [SEED, scheme]
    if name is not None:
        key.append(zlib.crc32(name.encode("utf-8")))
    return np.random.SeedSequence(key)


@dataclass
class Counts:
    subs: list[str]
    tasks: list[str]
    K: np.ndarray  # successes, shape (S, T)
    N: np.ndarray  # trials, shape (S, T); 0 where the task is absent


def build_counts(long: pd.DataFrame, subs: list[str] | None = None,
                 tasks: list[str] | None = None) -> Counts:
    """Aggregate a long table (submission, task_id, success) into K and N."""
    subs = sorted(long["submission"].unique()) if subs is None else list(subs)
    tasks = sorted(long["task_id"].unique()) if tasks is None else list(tasks)
    d = long[long["submission"].isin(subs) & long["task_id"].isin(tasks)]
    g = d.groupby(["submission", "task_id"])["success"].agg(["sum", "count"])
    K = g["sum"].unstack("task_id").reindex(index=subs, columns=tasks).fillna(0).to_numpy(dtype=np.int64)
    N = g["count"].unstack("task_id").reindex(index=subs, columns=tasks).fillna(0).to_numpy(dtype=np.int64)
    return Counts(subs, tasks, K, N)


def macro_mean(K: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Per-submission mean of task success rates over covered tasks."""
    present = N > 0
    p = np.divide(K, N, out=np.zeros(K.shape), where=present)
    return p.sum(axis=1) / present.sum(axis=1)


def run_to_run(c: Counts, n_boot: int = B, correct: bool = True) -> np.ndarray:
    """PRIMARY. Shape (n_boot, S): resampled submission scores, tasks fixed.

    ``correct=True`` (default) applies the N/(N-1) variance correction described in the
    module docstring (per-task rescaling of the deviation from the observed rate)."""
    out = np.empty((n_boot, len(c.subs)))
    for s, name in enumerate(c.subs):
        rng = np.random.default_rng(seed_for(SCHEME_RUN, name))
        present = c.N[s] > 0
        n = c.N[s][present]
        p = c.K[s][present] / n
        draws = rng.binomial(n, p, size=(n_boot, n.size))  # successes in resample
        x = draws / n
        if correct:
            scale = np.sqrt(np.divide(n, n - 1.0, out=np.ones(n.shape), where=n > 1))
            x = p + (x - p) * scale
        out[:, s] = x.mean(axis=1)
    return out


def analytic_run_sd(c: Counts) -> np.ndarray:
    """Analytic run-to-run SD of each submission mean, tasks fixed, independent runs:
    sqrt( sum_t phat_t (1 - phat_t) / (N_t - 1) ) / T over the T covered tasks (unbiased for
    the true SD under the Bernoulli model). Cells with N = 1 contribute 0."""
    present = c.N > 0
    n = c.N.astype(float)
    p = np.divide(c.K, c.N, out=np.zeros(c.K.shape), where=present)
    var = np.divide(p * (1 - p), n - 1.0, out=np.zeros(p.shape), where=n > 1)
    return np.sqrt(var.sum(axis=1)) / present.sum(axis=1)


def task_sampling(c: Counts, n_boot: int = B, chunk: int = 1000) -> np.ndarray:
    """SECONDARY. Shape (n_boot, S): tasks resampled, trial means fixed.

    Requires every submission to cover every task."""
    if (c.N == 0).any():
        raise ValueError("task_sampling needs complete task coverage")
    T = len(c.tasks)
    P = c.K / c.N  # (S, T)
    rng = np.random.default_rng(seed_for(SCHEME_TASK))
    out = np.empty((n_boot, len(c.subs)))
    for a in range(0, n_boot, chunk):
        b = min(chunk, n_boot - a)
        counts = rng.multinomial(T, np.full(T, 1.0 / T), size=b)  # task multiplicities
        out[a:a + b] = counts @ P.T / T
    return out


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    lo, hi = np.quantile(x, [a, 1 - a], axis=0)
    return lo, hi


def observed_order(subs: list[str], means: np.ndarray) -> np.ndarray:
    """Indices ordered best to worst by observed mean; ties broken by name."""
    return np.array(sorted(range(len(subs)), key=lambda i: (-round(float(means[i]), 12), subs[i])))


def flip_matrix(S: np.ndarray, order: np.ndarray) -> np.ndarray:
    """P[i, j] for ranks i < j in ``order``: probability that the observed
    lower-ranked submission scores above the higher-ranked one, ties counted
    half. Scores are rounded to 1e-9 before comparing. Entries with i >= j are
    NaN."""
    R = np.round(S[:, order], 9)
    n = R.shape[1]
    P = np.full((n, n), np.nan)
    for i in range(n - 1):
        hi, lo = R[:, i:i + 1], R[:, i + 1:]
        P[i, i + 1:] = (lo > hi).mean(axis=0) + 0.5 * (lo == hi).mean(axis=0)
    return P
