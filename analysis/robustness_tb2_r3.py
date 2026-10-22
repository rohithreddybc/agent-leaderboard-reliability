"""Terminal-Bench 2.0 robustness analyses, round 3 (2026-10-19).

New file. Existing scripts and outputs are not changed; this module reuses ``bootstrap_core``,
``rank_inversion``, ``tb2_bootstrap`` and ``tb2_load`` and writes only files whose names start with
``robustness_2026-10-19_tb2``.

Seed. Every random stream of this script is keyed to ``SEED`` = 20261019 (``bootstrap_core.SEED`` is
set to it while the new analyses run, so the per-submission and per-scheme streams of
``bootstrap_core.seed_for`` change with it; the overdispersion simulation uses
``np.random.default_rng([SEED, 4])``). The committed headline numbers (seed 20261008) are reproduced
first with the original seed (section 0) and the script stops if they differ. Because a seed change
moves Monte Carlo quantities slightly, the 72-submission baseline is also recomputed at the new seed
and the sensitivity analysis of section 1 is compared with THAT baseline.

Sections
  0  reproduction of the committed headline numbers
  1  task-set identity: trials without a task git commit, task ids, raw run metadata, and a
     sensitivity analysis without the submissions whose every trial has no commit
  2  overdispersion of runs: variance of run means against the independence prediction
  3  window correlations: split-half Spearman/Kendall for the top 20 and top 10 submissions, and
     rank-interval widths of the top 20
  4  two-sided flip threshold: adjacent pairs with flip probability > 2.5 % against > 5 %
  5  run share of the variance of a k-run mean, and the single-run and five-run SD of a mean
  6  generalizability study: crossed systems x tasks x runs random-effects ANOVA (moment estimators)

Usage: python robustness_tb2_r3.py [--results results] [--raw-root ../data/raw/tb2]
                                   [--n-boot 10000] [--n-splits 1000] [--n-sim 2000]
                                   [--skip-raw] [--scan-cache FILE.pkl]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import bootstrap_core as bc
import rank_inversion as R
import tb2_bootstrap
from tb2_load import normalize_task, select_analysis_set

SEED = 20261019
ORIGINAL_SEED = 20261008
STAMP = "2026-10-19"
PREFIX = f"robustness_{STAMP}_tb2_"
RUN_K = (1, 2, 5, 10)


@contextlib.contextmanager
def seeded(seed: int):
    """Set ``bootstrap_core.SEED`` (read at call time by ``seed_for``) for the duration of a block."""
    old = bc.SEED
    bc.SEED = seed
    try:
        yield
    finally:
        bc.SEED = old


# --------------------------------------------------------------------------------------------------
# Section 1: task-set identity
# --------------------------------------------------------------------------------------------------

def is_blank(x) -> pd.Series:
    """True where a task git commit is missing (NaN, None or an empty string)."""
    s = pd.Series(x)
    return s.isna() | (s.astype(str).str.strip() == "")


def blank_commit_by_submission(long: pd.DataFrame, analysed: list[str]) -> pd.DataFrame:
    """Per submission: trials, trials with no task git commit, whether all of them are blank, and
    whether the submission is in the analysis set."""
    d = long.assign(_blank=is_blank(long["task_commit"]).to_numpy())
    g = d.groupby("submission").agg(n_trials=("_blank", "size"), n_blank=("_blank", "sum"))
    g["share_blank"] = g["n_blank"] / g["n_trials"]
    g["all_blank"] = (g["n_blank"] == g["n_trials"]).astype(int)
    g["any_blank"] = (g["n_blank"] > 0).astype(int)
    g["in_analysis_set"] = g.index.isin(analysed).astype(int)
    runs = d[d["_blank"]].groupby("submission")["run"].agg(lambda x: "; ".join(sorted(set(map(str, x)))))
    g["runs_with_blank_commit"] = runs.reindex(g.index).fillna("")
    return g.reset_index()


def blank_commit_summary(table: pd.DataFrame, n_trials_total: int) -> dict:
    b = table[table["any_blank"] == 1]
    full = table[table["all_blank"] == 1]
    part = table[(table["any_blank"] == 1) & (table["all_blank"] == 0)]
    return {
        "n_trials_total": int(n_trials_total),
        "n_trials_blank": int(table["n_blank"].sum()),
        "share_trials_blank": float(table["n_blank"].sum() / n_trials_total),
        "n_submissions_total": int(len(table)),
        "n_submissions_with_any_blank": int(len(b)),
        "n_submissions_all_blank": int(len(full)),
        "n_submissions_all_blank_in_analysis_set": int(full["in_analysis_set"].sum()),
        "n_submissions_partially_blank": int(len(part)),
        "partially_blank": {r.submission: int(r.n_blank) for r in part.itertuples()},
        "all_blank_submissions": list(full["submission"]),
        "n_analysed": int(table["in_analysis_set"].sum()),
        "n_trials_blank_in_analysis_set": int(table.loc[table["in_analysis_set"] == 1, "n_blank"].sum()),
    }


def task_id_check(long: pd.DataFrame, blank_subs: list[str], tasks: list[str]) -> dict:
    """Task ids of the blank-commit submissions against the task universe and against the
    commit-bearing submissions."""
    ref = long[~long["submission"].isin(blank_subs)]
    ref_tasks = set(ref["task_id"])
    rows = []
    for s in blank_subs:
        d = long[long["submission"] == s]
        t = set(d["task_id"])
        per = d.groupby("task_id").size()
        rows.append({
            "submission": s, "n_tasks": len(t), "n_in_universe": len(t & set(tasks)),
            "n_not_in_universe": len(t - set(tasks)), "n_universe_missing": len(set(tasks) - t),
            "n_in_commit_bearing_ids": len(t & ref_tasks),
            "trials_per_task_min": int(per.min()), "trials_per_task_max": int(per.max()),
            "task_raw_prefixes": "; ".join(sorted({str(x).rsplit("/", 1)[0] if "/" in str(x) else "(none)"
                                                   for x in d["task_raw"]})),
        })
    return {"table": pd.DataFrame(rows), "n_ref_tasks": len(ref_tasks),
            "ref_tasks_equal_universe": ref_tasks == set(tasks)}


def _opt(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def scan_raw(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metadata fields of the raw tree (same layout as ``tb2_load.load_tree``).

    Returns (trials, runs). ``trials``: one row per trial ``result.json`` with the dataset ``source``,
    the task registry reference (``task_id.ref``), ``task_checksum`` and the git fields of both
    ``task_id`` and ``config.task``. ``runs``: one row per run directory and dataset of its
    ``config.json`` (job name, dataset name, version, ref, path)."""
    trials, runs = [], []
    for sub in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for run in sorted(p for p in sub.iterdir() if p.is_dir()):
            cf = run / "config.json"
            if cf.exists():
                c = json.loads(cf.read_text(encoding="utf-8"))
                for d in (c.get("datasets") or [{}]):
                    runs.append({"submission": sub.name, "run": run.name, "job_name": c.get("job_name"),
                                 "dataset_name": d.get("name"), "dataset_version": d.get("version"),
                                 "dataset_ref": d.get("ref"), "dataset_path": d.get("path"),
                                 "n_datasets": len(c.get("datasets") or []), "n_tasks_listed": len(c.get("tasks") or [])})
            else:
                runs.append({"submission": sub.name, "run": run.name, "job_name": None, "dataset_name": None,
                             "dataset_version": None, "dataset_ref": None, "dataset_path": None,
                             "n_datasets": 0, "n_tasks_listed": 0})
            for t in sorted(p for p in run.iterdir() if p.is_dir()):
                f = t / "result.json"
                if not f.exists():
                    continue
                r = json.loads(f.read_text(encoding="utf-8"))
                tid = r.get("task_id")
                ct = _opt(r, "config", "task") or {}
                trials.append({
                    "submission": sub.name, "run": run.name, "trial_name": r.get("trial_name"),
                    "task_id": normalize_task(r["task_name"]),
                    "source": r.get("source"), "task_checksum": r.get("task_checksum"),
                    "tid_is_dict": int(isinstance(tid, dict)),
                    "tid_ref": _opt(tid, "ref"), "tid_git_commit": _opt(tid, "git_commit_id"),
                    "tid_git_url": _opt(tid, "git_url"), "tid_path": _opt(tid, "path"),
                    "cfg_source": ct.get("source"), "cfg_ref": ct.get("ref"),
                    "cfg_git_commit": ct.get("git_commit_id"), "cfg_git_url": ct.get("git_url"),
                    "cfg_path": ct.get("path"),
                })
    return pd.DataFrame(trials), pd.DataFrame(runs)


def _join(x) -> str:
    return "; ".join(sorted({str(v) for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))}))


def metadata_by_submission(trials: pd.DataFrame, runs: pd.DataFrame, blank: dict) -> pd.DataFrame:
    """Task-set fields per submission: distinct dataset names/refs of the run configs and distinct
    ``source`` values of the trials."""
    t = trials.groupby("submission").agg(
        n_trials=("task_id", "size"), n_tasks=("task_id", "nunique"),
        trial_sources=("source", _join), n_distinct_task_checksums=("task_checksum", "nunique"),
        n_distinct_task_refs=("tid_ref", "nunique"), n_distinct_tid_git_commit=("tid_git_commit", "nunique"),
    )
    r = runs.groupby("submission").agg(
        n_run_dirs=("run", "nunique"), job_names=("job_name", _join), dataset_names=("dataset_name", _join),
        dataset_versions=("dataset_version", _join), dataset_refs=("dataset_ref", _join),
        dataset_paths=("dataset_path", _join),
    )
    out = t.join(r).reset_index()
    out["blank_commit_trials"] = out["submission"].map(blank).fillna(0).astype(int)
    return out


def metadata_task_check(trials: pd.DataFrame, blank_subs: list[str]) -> pd.DataFrame:
    """Per task: task checksums and registry refs seen in the blank-commit submissions against those
    seen in the other submissions."""
    isb = trials["submission"].isin(blank_subs)
    rows = []
    for task, d in trials.groupby("task_id"):
        b, o = d[isb.reindex(d.index)], d[~isb.reindex(d.index)]
        cb, co = set(b["task_checksum"].dropna()), set(o["task_checksum"].dropna())
        rb, ro = set(b["tid_ref"].dropna()), set(o["tid_ref"].dropna())
        rows.append({"task_id": task, "n_checksums_blank": len(cb), "n_checksums_other": len(co),
                     "checksums_blank_subset_of_other": int(bool(cb) and cb <= co),
                     "checksums_disjoint": int(bool(cb) and bool(co) and not (cb & co)),
                     "n_refs_blank": len(rb), "n_refs_other": len(ro),
                     "refs_blank_subset_of_other": int(bool(rb) and rb <= ro),
                     "other_has_checksum": int(bool(co))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Shared statistics for one set of submissions
# --------------------------------------------------------------------------------------------------

def split_half_windows(c: bc.Counts, order: np.ndarray, n_splits: int, windows: tuple[int, ...]) -> pd.DataFrame:
    """Split-half rank correlations as in ``rank_inversion.split_half`` (same random stream and
    split), computed on all submissions and on the top-k submissions by observed mean for each k in
    ``windows``. The windows are fixed by the full-data order, then ranked by the half-means."""
    if (c.N < 2).any():
        raise ValueError("split-half needs at least 2 trials in every cell")
    rng = np.random.default_rng(bc.seed_for(bc.SCHEME_SPLIT))
    h = c.N // 2
    idx = {"all": np.arange(len(c.subs))}
    for k in windows:
        if k < len(c.subs):
            idx[f"top{k}"] = np.asarray(order[:k])
    rows = []
    for s in range(n_splits):
        a = rng.hypergeometric(c.K, c.N - c.K, h)
        b = rng.hypergeometric(c.K - a, (c.N - c.K) - (h - a), h)
        ma, mb = (a / h).mean(axis=1), (b / h).mean(axis=1)
        row = {"split": s}
        for name, ix in idx.items():
            row[f"spearman_{name}"] = stats.spearmanr(ma[ix], mb[ix]).statistic
            row[f"kendall_{name}"] = stats.kendalltau(ma[ix], mb[ix]).statistic
        rows.append(row)
    return pd.DataFrame(rows)


def run_share(f: float, k: int) -> float:
    """Share of the variance of a k-run mean that is run noise, (f/k) / ((1 - f) + f/k): the single-run
    outcome variance splits into f (run noise) and 1 - f (between tasks); averaging k runs divides
    only the run part by k."""
    return (f / k) / ((1.0 - f) + f / k)


def core(long: pd.DataFrame, subs: list[str], tasks: list[str], n_boot: int, n_splits: int,
         windows: tuple[int, ...] = (10, 20)) -> dict:
    """All bootstrap statistics of one submission set. Uses the current ``bootstrap_core.SEED``."""
    c = bc.build_counts(long, subs, tasks)
    means = bc.macro_mean(c.K, c.N)
    order = bc.observed_order(c.subs, means)
    S_run = bc.run_to_run(c, n_boot)
    S_task = bc.task_sampling(c, n_boot)
    P_run, P_task = bc.flip_matrix(S_run, order), bc.flip_matrix(S_task, order)
    adj_run, adj_task = np.diag(P_run, 1), np.diag(P_task, 1)
    run_sd, task_sd = S_run.std(axis=0, ddof=1), S_task.std(axis=0, ddof=1)
    wt = tb2_bootstrap.within_task_summary(c)
    f = wt["f_run_noise_share"].dropna()
    v = wt["mean_p1mp_unbiased"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (run_sd / task_sd) ** 2
    ri_run = R.rank_intervals(S_run, order, c.subs)
    ri_task = R.rank_intervals(S_task, order, c.subs)
    sh = split_half_windows(c, order, n_splits, windows)
    out = {
        "n_submissions": len(subs), "n_tasks": len(tasks), "n_adjacent_pairs": len(subs) - 1,
        "run_sd_median_pp": float(100 * np.median(run_sd)), "task_sd_median_pp": float(100 * np.median(task_sd)),
        "run_sd_max_pp": float(100 * run_sd.max()), "task_sd_min_pp": float(100 * task_sd.min()),
        "task_sd_max_pp": float(100 * task_sd.max()),
        "var_ratio_run_over_task_median": float(np.nanmedian(ratio)),
        "v_median": float(v.median()), "f_median": float(f.median()), "n_f_defined": int(f.size),
        "f_q25": float(f.quantile(0.25)), "f_q75": float(f.quantile(0.75)),
        "run_share_k5_at_median_f": run_share(float(f.median()), 5),
        "adj_run_gt_5pct": int((adj_run > 0.05).sum()), "adj_run_gt_2_5pct": int((adj_run > 0.025).sum()),
        "adj_task_gt_5pct": int((adj_task > 0.05).sum()), "adj_task_gt_2_5pct": int((adj_task > 0.025).sum()),
        "rank_width_median_run": float(ri_run["interval_width"].median()),
        "rank_width_median_task": float(ri_task["interval_width"].median()),
        "split_half_n_splits": int(len(sh)),
    }
    for col in sh.columns[1:]:
        out[f"split_half_{col}_mean"] = float(sh[col].mean())
    for k in windows:
        if k <= len(subs):
            top = ri_run["observed_rank"] <= k
            topt = ri_task["observed_rank"] <= k
            out[f"rank_width_median_run_top{k}"] = float(ri_run.loc[top, "interval_width"].median())
            out[f"rank_width_median_task_top{k}"] = float(ri_task.loc[topt, "interval_width"].median())
            out[f"rank_width_max_run_top{k}"] = float(ri_run.loc[top, "interval_width"].max())
    arrays = {"c": c, "means": means, "order": order, "adj_run": adj_run, "adj_task": adj_task,
              "run_sd": run_sd, "task_sd": task_sd, "wt": wt, "ri_run": ri_run, "ri_task": ri_task, "sh": sh}
    return {"stats": out, "arrays": arrays}


def adjacent_table(res: dict) -> pd.DataFrame:
    a = res["arrays"]
    subs, order, means = a["c"].subs, a["order"], a["means"]
    return pd.DataFrame({
        "rank_hi": np.arange(1, len(order)), "submission_hi": [subs[i] for i in order[:-1]],
        "submission_lo": [subs[i] for i in order[1:]],
        "gap_pp": 100 * (means[order[:-1]] - means[order[1:]]),
        "p_flip_run": a["adj_run"], "p_flip_task": a["adj_task"],
    })


# --------------------------------------------------------------------------------------------------
# Section 0: reproduction
# --------------------------------------------------------------------------------------------------

def reproduce(long: pd.DataFrame, results: Path, n_boot: int, n_splits: int) -> dict:
    """Recompute the committed headline numbers with the ORIGINAL seed and compare them with the
    committed files. Raises ``AssertionError`` listing every mismatch."""
    committed = json.loads((results / "rank_inversion_summary.json").read_text(encoding="utf-8"))
    boot = pd.read_csv(results / "tb2_bootstrap.csv")
    wtc = pd.read_csv(results / "tb2_within_task_variance.csv")
    subs, tasks, _ = select_analysis_set(long)
    with seeded(ORIGINAL_SEED):
        res = core(long, subs, tasks, n_boot, n_splits)
    s = res["stats"]
    rows = [
        ("n_submissions", s["n_submissions"], committed["n_submissions"], 0),
        ("run_sd_median_pp", s["run_sd_median_pp"], 100 * boot["run_sd"].median(), 1e-9),
        ("task_sd_median_pp", s["task_sd_median_pp"], 100 * boot["task_sd"].median(), 1e-9),
        ("adj_run_gt_5pct", s["adj_run_gt_5pct"], committed["run"]["n_adjacent_flip_gt_5pct"], 0),
        ("adj_task_gt_5pct", s["adj_task_gt_5pct"], committed["task"]["n_adjacent_flip_gt_5pct"], 0),
        ("f_median", s["f_median"], wtc["f_run_noise_share"].dropna().median(), 1e-9),
        ("split_half_spearman_all_mean", s["split_half_spearman_all_mean"], committed["split_half"]["spearman_mean"], 1e-9),
        ("split_half_kendall_all_mean", s["split_half_kendall_all_mean"], committed["split_half"]["kendall_mean"], 1e-9),
        ("rank_width_median_run", s["rank_width_median_run"], committed["run"]["rank_interval_width_median"], 1e-9),
        ("rank_width_median_task", s["rank_width_median_task"], committed["task"]["rank_interval_width_median"], 1e-9),
    ]
    tab = pd.DataFrame(rows, columns=["quantity", "recomputed", "committed", "tolerance"])
    tab["match"] = (tab["recomputed"] - tab["committed"]).abs() <= tab["tolerance"]
    bad = tab[~tab["match"]]
    assert bad.empty, "reproduction failed:\n" + bad.to_string()
    return {"table": tab, "stats": s, "arrays": res["arrays"]}


# --------------------------------------------------------------------------------------------------
# Section 2: overdispersion of runs
# --------------------------------------------------------------------------------------------------

def complete_run_matrix(d: pd.DataFrame, tasks: list[str]) -> tuple[np.ndarray, list[str], dict]:
    """Run directories of one submission that hold exactly one trial of every task. Returns the
    (runs, tasks) 0/1 matrix, their names and a count dictionary."""
    X, names = [], []
    g = d.groupby("run")
    for run, x in g:
        if len(x) == len(tasks) and x["task_id"].nunique() == len(tasks) and set(x["task_id"]) == set(tasks):
            X.append(x.set_index("task_id")["success"].reindex(tasks).to_numpy(dtype=float))
            names.append(str(run))
    info = {"n_run_dirs": int(g.ngroups), "n_complete_runs": len(names), "trials_total": int(len(d)),
            "trials_used": len(names) * len(tasks)}
    return (np.array(X) if X else np.empty((0, len(tasks)))), names, info


def overdispersion_ratio(X: np.ndarray) -> tuple[float, float, float]:
    """(variance of the run means, independence prediction, ratio) for an (R, T) 0/1 matrix.

    Run mean m_r = mean_t X[r, t]. Observed variance V = sample variance of the m_r (ddof 1). Under
    independent runs with the tasks fixed, Var(m_r) = T^-2 sum_t p_t (1 - p_t); the unbiased estimate
    from the same R runs is pred = T^-2 sum_t phat_t (1 - phat_t) R / (R - 1). Ratio = V / pred
    (NaN when pred = 0)."""
    R_, T = X.shape
    m = X.mean(axis=1)
    V = float(m.var(ddof=1))
    p = X.mean(axis=0)
    pred = float((p * (1 - p)).sum() * R_ / (R_ - 1) / T ** 2)
    return V, pred, (V / pred if pred > 0 else float("nan"))


def overdispersion(long: pd.DataFrame, subs: list[str], tasks: list[str], n_sim: int = 2000,
                   seed: int = SEED) -> dict:
    """Overdispersion of run means for every submission with at least two complete runs, with a
    parametric null (independent Bernoulli(phat_t) runs, same R and T) simulated n_sim times."""
    rng = np.random.default_rng([seed, 4])
    rows, sim = [], []
    for s in subs:
        d = long[long["submission"] == s]
        X, names, info = complete_run_matrix(d, tasks)
        row = {"submission": s, **info, "n_runs_used": len(names), "var_run_means": np.nan,
               "var_independence": np.nan, "ratio": np.nan, "null_ratio_median": np.nan,
               "p_null_ratio_ge_observed": np.nan, "status": "", "reason": ""}
        if len(names) < 2:
            row["status"] = "cannot be checked"
            if info["n_run_dirs"] == 1:
                row["reason"] = "one run directory holds all trials; runs within it are not labelled"
            elif len(names) == 1:
                row["reason"] = "only one run directory has exactly one trial per task"
            else:
                row["reason"] = ("no run directory has exactly one trial per task "
                                 "(directories are task batches or hold several trials per task)")
            rows.append(row)
            continue
        V, pred, ratio = overdispersion_ratio(X)
        row.update(var_run_means=V, var_independence=pred, ratio=ratio)
        if not np.isfinite(ratio):
            row["status"] = "undefined"
            row["reason"] = "independence prediction is 0 (every task has identical outcomes across runs)"
            rows.append(row)
            continue
        p = X.mean(axis=0)
        Rn = X.shape[0]
        draws = (rng.random((n_sim, Rn, p.size)) < p).astype(float)
        m = draws.mean(axis=2)
        Vs = m.var(axis=1, ddof=1)
        ps = draws.mean(axis=1)
        pred_s = (ps * (1 - ps)).sum(axis=1) * Rn / (Rn - 1) / p.size ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(pred_s > 0, Vs / pred_s, np.nan)
        sim.append(rs)
        row["null_ratio_median"] = float(np.nanmedian(rs))
        row["p_null_ratio_ge_observed"] = float(np.nanmean(rs >= ratio))
        row["status"] = "checked" if len(names) == info["n_run_dirs"] else "checked (subset of directories)"
        if len(names) < info["n_run_dirs"]:
            row["reason"] = f"{len(names)} of {info['n_run_dirs']} run directories have exactly one trial per task"
        rows.append(row)
    tab = pd.DataFrame(rows)
    ok = tab[tab["status"].str.startswith("checked")]
    out = {"table": tab, "n_checked": int(len(ok)), "n_total": int(len(tab)),
           "n_cannot": int((tab["status"] == "cannot be checked").sum()),
           "n_undefined": int((tab["status"] == "undefined").sum())}
    if len(ok):
        r = ok["ratio"]
        out.update(ratio_median=float(r.median()), ratio_mean=float(r.mean()), ratio_min=float(r.min()),
                   ratio_max=float(r.max()), n_ratio_gt_1=int((r > 1).sum()),
                   n_runs_median=float(ok["n_runs_used"].median()))
        w = ok["n_runs_used"] - 1
        out["ratio_pooled_df_weighted"] = float((w * ok["var_run_means"]).sum() / (w * ok["var_independence"]).sum())
        S = np.column_stack(sim)
        med = np.nanmedian(S, axis=1)
        mean = np.nanmean(S, axis=1)
        out.update(null_median_of_medians=float(np.mean(med)), null_mean_of_means=float(np.mean(mean)),
                   p_null_median_ge_observed=float(np.mean(med >= out["ratio_median"])),
                   p_null_mean_ge_observed=float(np.mean(mean >= out["ratio_mean"])), n_sim=int(n_sim))
    return out


# --------------------------------------------------------------------------------------------------
# Section 5: single-run and five-run SD
# --------------------------------------------------------------------------------------------------

def sd_of_mean(v: np.ndarray, n_tasks: int, k: int) -> np.ndarray:
    """SD of the mean over n_tasks tasks of k-run task rates: sqrt(v / (k n_tasks)), v the mean
    within-task variance of a single run."""
    return np.sqrt(np.asarray(v, dtype=float) / (k * n_tasks))


# --------------------------------------------------------------------------------------------------
# Section 6: G-study
# --------------------------------------------------------------------------------------------------

def balanced_array(long: pd.DataFrame, subs: list[str], tasks: list[str], n_r: int = 5):
    """Systems x tasks x runs array for the submissions that have exactly n_r trials in every task
    cell; ``trial`` (1..n_r, ordered by run directory, start time and trial name) is the run index.
    Returns (X, kept, dropped) with ``dropped`` a table of submission and reason."""
    kept, dropped, mats = [], [], []
    for s in subs:
        d = long[long["submission"] == s]
        n = d.groupby("task_id").size().reindex(tasks).fillna(0).astype(int)
        if (n == n_r).all():
            d = d.sort_values(["task_id", "trial"], kind="stable")
            mats.append(d.set_index(["task_id", "trial"])["success"].unstack("trial").reindex(tasks).to_numpy(dtype=float))
            kept.append(s)
        else:
            vc = n.value_counts().sort_index()
            reason = "cells with " + ", ".join(f"{k} trials: {v}" for k, v in vc.items())
            dropped.append({"submission": s, "reason": reason, "n_trials": int(len(d))})
    X = np.stack(mats) if mats else np.empty((0, len(tasks), n_r))
    return X, kept, pd.DataFrame(dropped, columns=["submission", "reason", "n_trials"])


def anova_3way(X: np.ndarray) -> dict:
    """Mean squares and variance components of a fully crossed random-effects systems x tasks x runs
    design with one observation per cell (balanced), by the method of moments.

    With n_s, n_t, n_r levels and cell means over the omitted indices, the expected mean squares are
      MS_s  = s2_str + n_r s2_st + n_t s2_sr + n_t n_r s2_s
      MS_t  = s2_str + n_r s2_st + n_s s2_tr + n_s n_r s2_t
      MS_r  = s2_str + n_t s2_sr + n_s s2_tr + n_s n_t s2_r
      MS_st = s2_str + n_r s2_st,  MS_sr = s2_str + n_t s2_sr,  MS_tr = s2_str + n_s s2_tr,
      MS_str = s2_str
    where s2_str is the three-way interaction confounded with residual error. Components are solved
    from these equations; negative estimates are reported as is in ``raw`` and set to 0 in ``comp``."""
    S, T, Rn = X.shape
    g = X.mean()
    ms_, mt_, mr_ = X.mean(axis=(1, 2)), X.mean(axis=(0, 2)), X.mean(axis=(0, 1))
    mst, msr, mtr = X.mean(axis=2), X.mean(axis=1), X.mean(axis=0)
    ss = {
        "s": T * Rn * ((ms_ - g) ** 2).sum(),
        "t": S * Rn * ((mt_ - g) ** 2).sum(),
        "r": S * T * ((mr_ - g) ** 2).sum(),
        "st": Rn * ((mst - ms_[:, None] - mt_[None, :] + g) ** 2).sum(),
        "sr": T * ((msr - ms_[:, None] - mr_[None, :] + g) ** 2).sum(),
        "tr": S * ((mtr - mt_[:, None] - mr_[None, :] + g) ** 2).sum(),
    }
    resid = (X - mst[:, :, None] - msr[:, None, :] - mtr[None, :, :]
             + ms_[:, None, None] + mt_[None, :, None] + mr_[None, None, :] - g)
    ss["str"] = (resid ** 2).sum()
    df = {"s": S - 1, "t": T - 1, "r": Rn - 1, "st": (S - 1) * (T - 1), "sr": (S - 1) * (Rn - 1),
          "tr": (T - 1) * (Rn - 1), "str": (S - 1) * (T - 1) * (Rn - 1)}
    ms = {k: ss[k] / df[k] for k in ss}
    raw = {
        "str": ms["str"],
        "st": (ms["st"] - ms["str"]) / Rn,
        "sr": (ms["sr"] - ms["str"]) / T,
        "tr": (ms["tr"] - ms["str"]) / S,
        "s": (ms["s"] - ms["st"] - ms["sr"] + ms["str"]) / (T * Rn),
        "t": (ms["t"] - ms["st"] - ms["tr"] + ms["str"]) / (S * Rn),
        "r": (ms["r"] - ms["sr"] - ms["tr"] + ms["str"]) / (S * T),
    }
    comp = {k: max(0.0, v) for k, v in raw.items()}
    total = sum(comp.values())
    share = {k: (v / total if total > 0 else float("nan")) for k, v in comp.items()}
    return {"n_s": S, "n_t": T, "n_r": Rn, "ss": ss, "df": df, "ms": ms, "raw": raw, "comp": comp,
            "share": share, "total": total, "ss_total_check": float(((X - g) ** 2).sum())}


def anova_2way_replicated(X: np.ndarray) -> dict:
    """Systems x tasks design with n_r exchangeable replicates per cell (run labels ignored):
      MS_s = s2_e + n_r s2_st + n_t n_r s2_s,  MS_t = s2_e + n_r s2_st + n_s n_r s2_t,
      MS_st = s2_e + n_r s2_st,  MS_w = s2_e (pooled within-cell variance, n_r - 1 df per cell)."""
    S, T, Rn = X.shape
    g = X.mean()
    ms_, mt_, mst = X.mean(axis=(1, 2)), X.mean(axis=(0, 2)), X.mean(axis=2)
    MS_s = T * Rn * ((ms_ - g) ** 2).sum() / (S - 1)
    MS_t = S * Rn * ((mt_ - g) ** 2).sum() / (T - 1)
    MS_st = Rn * ((mst - ms_[:, None] - mt_[None, :] + g) ** 2).sum() / ((S - 1) * (T - 1))
    MS_w = ((X - mst[:, :, None]) ** 2).sum() / (S * T * (Rn - 1))
    raw = {"e": MS_w, "st": (MS_st - MS_w) / Rn, "s": (MS_s - MS_st) / (T * Rn), "t": (MS_t - MS_st) / (S * Rn)}
    comp = {k: max(0.0, v) for k, v in raw.items()}
    total = sum(comp.values())
    return {"raw": raw, "comp": comp, "total": total,
            "share": {k: (v / total if total > 0 else float("nan")) for k, v in comp.items()}}


def g_relative(comp: dict, n_t: int, n_r: int) -> float:
    """Relative generalizability coefficient for ranking systems (systems = object of measurement; tasks
    and runs random and crossed with systems), 3-way model:
        Ep2 = s2_s / (s2_s + s2_st / n_t + s2_sr / n_r + s2_str / (n_t n_r)).
    The system-independent components s2_t, s2_r, s2_tr do not enter a relative decision."""
    err = comp["st"] / n_t + comp["sr"] / n_r + comp["str"] / (n_t * n_r)
    return comp["s"] / (comp["s"] + err) if comp["s"] + err > 0 else float("nan")


def g_absolute(comp: dict, n_t: int, n_r: int) -> float:
    """Absolute coefficient (Phi), same model: s2_s / (s2_s + s2_t/n_t + s2_r/n_r + s2_st/n_t + s2_sr/n_r
    + s2_tr/(n_t n_r) + s2_str/(n_t n_r))."""
    err = (comp["t"] / n_t + comp["r"] / n_r + comp["st"] / n_t + comp["sr"] / n_r
           + comp["tr"] / (n_t * n_r) + comp["str"] / (n_t * n_r))
    return comp["s"] / (comp["s"] + err) if comp["s"] + err > 0 else float("nan")


def g_relative_2way(comp: dict, n_t: int, n_r: int) -> float:
    """Relative coefficient with exchangeable replicates: s2_s / (s2_s + s2_st / n_t + s2_e / (n_t n_r))."""
    err = comp["st"] / n_t + comp["e"] / (n_t * n_r)
    return comp["s"] / (comp["s"] + err) if comp["s"] + err > 0 else float("nan")


def gstudy(long: pd.DataFrame, subs: list[str], tasks: list[str], label: str, n_t_design: int = 89) -> dict:
    X, kept, dropped = balanced_array(long, subs, tasks)
    a3, a2 = anova_3way(X), anova_2way_replicated(X)
    rows = []
    names = {"s": "systems", "t": "tasks", "r": "runs (trial index)", "st": "systems x tasks",
             "sr": "systems x runs", "tr": "tasks x runs", "str": "systems x tasks x runs, residual"}
    for k in ("s", "t", "r", "st", "sr", "tr", "str"):
        rows.append({"set": label, "model": "3-way", "component": names[k], "mean_square": a3["ms"][k],
                     "df": a3["df"][k], "raw_estimate": a3["raw"][k], "variance_component": a3["comp"][k],
                     "share_of_total": a3["share"][k]})
    names2 = {"s": "systems", "t": "tasks", "st": "systems x tasks", "e": "within cell (runs and residual)"}
    for k in ("s", "t", "st", "e"):
        rows.append({"set": label, "model": "2-way replicated", "component": names2[k], "mean_square": np.nan,
                     "df": np.nan, "raw_estimate": a2["raw"][k], "variance_component": a2["comp"][k],
                     "share_of_total": a2["share"][k]})
    coef = {f"Ep2_3way_nt{n_t_design}_nr{n}": g_relative(a3["comp"], n_t_design, n) for n in (1, 5)}
    coef.update({f"Phi_3way_nt{n_t_design}_nr{n}": g_absolute(a3["comp"], n_t_design, n) for n in (1, 5)})
    coef.update({f"Ep2_2way_nt{n_t_design}_nr{n}": g_relative_2way(a2["comp"], n_t_design, n) for n in (1, 5)})
    return {"label": label, "n_s": a3["n_s"], "n_t": a3["n_t"], "n_r": a3["n_r"], "kept": kept,
            "dropped": dropped, "table": pd.DataFrame(rows), "coef": coef, "a3": a3, "a2": a2}


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------

def _f(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{x:.{d}f}"


def md_table(df: pd.DataFrame, floats: dict | None = None) -> list[str]:
    """Markdown table; each cell is formatted by the dtype of its own column (integers stay integers)."""
    floats = floats or {}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for i in range(len(df)):
        cells = []
        for c in df.columns:
            v = df[c].iloc[i]
            if isinstance(v, (float, np.floating)):
                cells.append(_f(float(v), floats.get(c, 4)))
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                cells.append(str(int(v)))
            else:
                cells.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def trial_index_overdispersion(long: pd.DataFrame, subs: list[str], tasks: list[str]) -> pd.DataFrame:
    """EXPLORATORY. For submissions with exactly 5 trials in every task cell, group trials into 'runs' by the
    within-cell trial index (run directory, start time, trial name) and compute the same V / pred ratio. For a
    submission whose five runs sit in one directory this grouping is by start-time rank within a task and need
    not coincide with real runs."""
    X, kept, _ = balanced_array(long, subs, tasks)
    nd = long.groupby("submission")["run"].nunique()
    rows = []
    for i, s in enumerate(kept):
        V, pred, ratio = overdispersion_ratio(X[i].T)
        rows.append({"submission": s, "n_run_dirs": int(nd[s]), "grouping": "run directory" if nd[s] > 1 else "trial index in one directory",
                     "var_run_means": V, "var_independence": pred, "ratio": ratio})
    return pd.DataFrame(rows)


def canonical_checksum_subs(trials: pd.DataFrame) -> tuple[list[str], int]:
    """Submissions whose per-task ``task_checksum`` vector equals the most common vector, and the
    number of distinct vectors."""
    cs = trials.groupby(["submission", "task_id"])["task_checksum"].first().unstack("task_id")
    keys = {s: tuple(cs.loc[s].tolist()) for s in cs.index}
    from collections import Counter
    cnt = Counter(keys.values())
    best = cnt.most_common(1)[0][0]
    return sorted(s for s, k in keys.items() if k == best), len(cnt)


SENS_ROWS = [
    ("n_submissions", "submissions analysed", 0), ("n_adjacent_pairs", "adjacent pairs", 0),
    ("run_sd_median_pp", "median SD of a submission mean over runs (pp)", 2),
    ("task_sd_median_pp", "median SD of a submission mean over tasks (pp)", 2),
    ("var_ratio_run_over_task_median", "median (run SD / task SD)^2", 4),
    ("f_median", "f, median", 4), ("run_share_k5_at_median_f", "run share of variance at k = 5, at median f", 4),
    ("adj_run_gt_5pct", "adjacent pairs with flip probability > 5%, run resampling", 0),
    ("adj_task_gt_5pct", "adjacent pairs with flip probability > 5%, task resampling", 0),
    ("split_half_spearman_all_mean", "split-half Spearman, mean", 4),
    ("split_half_kendall_all_mean", "split-half Kendall, mean", 4),
]


def sens_table(variants: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, label, d in SENS_ROWS:
        row = {"statistic": label}
        for name, st in variants.items():
            v = st[key]
            row[name] = int(v) if d == 0 else round(float(v), d)
        rows.append(row)
        if key in ("adj_run_gt_5pct", "adj_task_gt_5pct"):
            r2 = {"statistic": label.replace("adjacent pairs with", "share of adjacent pairs with")}
            for name, st in variants.items():
                r2[name] = round(st[key] / st["n_adjacent_pairs"], 4)
            rows.append(r2)
    return pd.DataFrame(rows, dtype=object)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--raw-root", type=Path, default=here.parent / "data/raw/tb2")
    ap.add_argument("--n-boot", type=int, default=bc.B)
    ap.add_argument("--n-splits", type=int, default=1000)
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--skip-raw", action="store_true", help="skip the scan of the raw run and trial metadata")
    ap.add_argument("--scan-cache", type=Path, default=None,
                    help="pickle file for the raw-metadata scan (read if present, else written); the scan takes minutes")
    a = ap.parse_args()
    res_dir = a.results
    long = pd.read_csv(res_dir / "tb2_long.csv")
    subs, tasks, _ = select_analysis_set(long)
    nb, ns = a.n_boot, a.n_splits
    summary: dict = {"seed": SEED, "n_boot": nb, "n_splits": ns, "n_sim": a.n_sim}
    L: list[str] = []
    w = L.append

    def save(name: str, df: pd.DataFrame) -> str:
        fn = f"{PREFIX}{name}.csv"
        df.to_csv(res_dir / fn, index=False)
        return fn

    w(f"# Terminal-Bench 2.0 robustness analyses, round 3 ({STAMP})")
    w("")
    w("Numbers only. Generated by `analysis/robustness_tb2_r3.py`; tests in `analysis/tests/test_robustness_tb2_r3.py`. "
      "New files only: no existing script or output was changed. Inputs: `analysis/results/tb2_long.csv` (the trial table of "
      "`tb2_load.py`) and, for the metadata part of section 1, the raw trial and run `result.json` / `config.json` files in "
      "`data/raw/tb2` (retrieval of 2026-10-04).")
    w("")
    w("## Definitions and conventions")
    w("")
    w(f"- Seed {SEED} for every random stream of this report (bootstrap_core.SEED is set to it for the new analyses; "
      f"B = {nb:,} bootstrap replicates, {ns:,} split-half splits, {a.n_sim:,} simulation replicates in section 2). "
      f"The committed numbers (seed {ORIGINAL_SEED}) are reproduced in section 0 and cited as 'committed' where used; "
      "every comparison of a new analysis with a baseline uses the baseline recomputed at the new seed.")
    w("- Analysis set: the 72 submissions with at least 5 trials on every one of the 89 tasks (`tb2_load.select_analysis_set`). "
      "Success: verifier reward >= 1.0; a trial with no reward counts as a failure (as in the committed analysis).")
    w("- Run-to-run resampling: tasks fixed, trials resampled within task with the N/(N-1) correction (`bootstrap_core.run_to_run`). "
      "Task resampling: tasks resampled, one draw for all submissions (`bootstrap_core.task_sampling`). "
      "Flip probability of an adjacent pair: share of replicates in which the lower-ranked submission scores above the higher-ranked one, ties half.")
    w("- f = v / (P (1 - P)), v = mean over tasks of the unbiased within-task variance p (1 - p) N / (N - 1), P = submission mean; "
      "f is undefined at P = 0 or 1. T = 89 tasks.")
    w("")

    # ------------------------------------------------------------------ section 0
    w("## 0. Reproduction of the committed headline numbers (seed %d)" % ORIGINAL_SEED)
    w("")
    rep = reproduce(long, res_dir, nb, ns)
    summary["reproduction"] = rep["table"].assign(match=rep["table"]["match"].astype(bool)).to_dict(orient="records")
    w("Recomputed with the original seed and compared with `rank_inversion_summary.json`, `tb2_bootstrap.csv` and "
      "`tb2_within_task_variance.csv`. All rows match, otherwise the script stops.")
    w("")
    t0 = rep["table"].copy()
    t0["match"] = t0["match"].map({True: "yes", False: "NO"})
    for c_ in ("recomputed", "committed"):
        t0[c_] = t0[c_].astype(float).round(6)
    w("\n".join(md_table(t0.drop(columns="tolerance"), {"recomputed": 6, "committed": 6})))
    w("")
    s0 = rep["stats"]
    w(f"- Submissions analysed (committed): {s0['n_submissions']}; adjacent pairs: {s0['n_adjacent_pairs']}.")
    w(f"- Median SD of a submission mean over runs (committed seed): {s0['run_sd_median_pp']:.2f} pp; over tasks: {s0['task_sd_median_pp']:.2f} pp.")
    w(f"- Adjacent pairs with flip probability > 5% (committed seed): run resampling {s0['adj_run_gt_5pct']} of {s0['n_adjacent_pairs']}; "
      f"task resampling {s0['adj_task_gt_5pct']} of {s0['n_adjacent_pairs']}.")
    w(f"- f, median (committed): {s0['f_median']:.3f}; split-half Spearman, mean: {s0['split_half_spearman_all_mean']:.3f}; Kendall: {s0['split_half_kendall_all_mean']:.3f}.")
    w("")

    # ------------------------------------------------------------------ new-seed baseline
    with seeded(SEED):
        base = core(long, subs, tasks, nb, ns)
    summary["baseline_72_new_seed"] = base["stats"]

    # ------------------------------------------------------------------ section 1
    w("## 1. Task-set identity")
    w("")
    tab = blank_commit_by_submission(long, subs)
    bsum = blank_commit_summary(tab, len(long))
    summary["blank_commit"] = bsum
    fn_b = save("blank_commit_by_submission", tab)
    w("### 1a. Trials with no task git commit (`task_id.git_commit_id` missing in the trial `result.json`)")
    w("")
    w(f"- Trials in the snapshot: {bsum['n_trials_total']:,}; trials with no task git commit: {bsum['n_trials_blank']:,} "
      f"({100 * bsum['share_trials_blank']:.1f}%). The reviewer's figures (4,010 of 32,803, 12.2%) are "
      f"{'confirmed' if (bsum['n_trials_blank'], bsum['n_trials_total']) == (4010, 32803) else 'NOT confirmed'}.")
    w(f"- Submissions in the snapshot: {bsum['n_submissions_total']}; with at least one blank-commit trial: {bsum['n_submissions_with_any_blank']}; "
      f"with every trial blank: {bsum['n_submissions_all_blank']} (all {bsum['n_submissions_all_blank_in_analysis_set']} of them in the 72-submission analysis set).")
    w(f"- Blank-commit trials inside the analysis set: {bsum['n_trials_blank_in_analysis_set']:,} of {int(tab.loc[tab['in_analysis_set'] == 1, 'n_trials'].sum()):,} trials of the analysis set "
      f"({100 * bsum['n_trials_blank_in_analysis_set'] / tab.loc[tab['in_analysis_set'] == 1, 'n_trials'].sum():.1f}%).")
    w("- Submissions with every trial blank: " + "; ".join(f"`{s}`" for s in bsum["all_blank_submissions"]) + ".")
    w("- Partially blank: " + ("; ".join(f"`{s}` ({n} trials)" for s, n in bsum["partially_blank"].items()) or "none") + ".")
    ledger = [r for r in tab["submission"] if r.startswith("0error-Ledger")]
    if ledger:
        runs_l = sorted(long.loc[long["submission"] == ledger[0], "run"].unique())
        w(f"- Run directory name(s) of `{ledger[0]}`: " + "; ".join(f"`{r}`" for r in runs_l) + ".")
    w("")
    blank_all = bsum["all_blank_submissions"]
    blank_any = list(tab.loc[tab["any_blank"] == 1, "submission"])
    chk = task_id_check(long, blank_any, tasks)
    fn_t = save("blank_commit_task_ids", chk["table"])
    w("### 1b. Task ids of the blank-commit submissions against the 89 tasks")
    w("")
    w(f"- The commit-bearing submissions carry {chk['n_ref_tasks']} distinct task ids; they equal the 89-task universe: {'yes' if chk['ref_tasks_equal_universe'] else 'NO'}.")
    tt = chk["table"]
    w(f"- Blank-commit submissions with all 89 task ids in the universe and none outside it: "
      f"{int(((tt['n_in_universe'] == 89) & (tt['n_not_in_universe'] == 0)).sum())} of {len(tt)}; "
      f"with exactly 5 trials on every task: {int(((tt['trials_per_task_min'] == 5) & (tt['trials_per_task_max'] == 5)).sum())} of {len(tt)}.")
    w("")
    w("\n".join(md_table(tt[["submission", "n_tasks", "n_in_universe", "n_not_in_universe", "n_universe_missing",
                              "trials_per_task_min", "trials_per_task_max", "task_raw_prefixes"]])))
    w("")

    canon_subs = None
    meta = None
    if a.skip_raw or not a.raw_root.exists():
        w("### 1c. Raw metadata")
        w("")
        w("Skipped: raw tree not scanned (`--skip-raw` or `%s` not found)." % a.raw_root)
        w("")
    else:
        if a.scan_cache is not None and a.scan_cache.exists():
            trials, runs = pd.read_pickle(a.scan_cache)
        else:
            trials, runs = scan_raw(a.raw_root)
            if a.scan_cache is not None:
                pd.to_pickle((trials, runs), a.scan_cache)
        blank_counts = dict(zip(tab["submission"], tab["n_blank"]))
        meta = metadata_by_submission(trials, runs, blank_counts)
        rawblank = int(is_blank(trials["tid_git_commit"]).sum())
        canon_subs, n_vec = canonical_checksum_subs(trials)
        mt = metadata_task_check(trials, blank_any)
        summary["metadata"] = {"n_trials_scanned": int(len(trials)), "n_run_dirs_scanned": int(runs.groupby(["submission", "run"]).ngroups),
                               "raw_blank_task_id_git_commit": rawblank, "n_distinct_checksum_vectors": int(n_vec),
                               "n_canonical_checksum_subs": len(canon_subs)}
        fn_m = save("metadata_by_submission", meta)
        fn_mt = save("metadata_task_check", mt)
        w("### 1c. What the raw metadata says about the task set")
        w("")
        w(f"Scanned {len(trials):,} trial `result.json` files and {runs.groupby(['submission', 'run']).ngroups} run directories. "
          f"Blank `task_id.git_commit_id` in the raw trials: {rawblank:,} (equals the count of 1a: {'yes' if rawblank == bsum['n_trials_blank'] else 'NO'}). "
          "Fields read: trial `source`, `task_id.ref`, `task_id.git_url`, `task_id.git_commit_id`, `task_checksum`, `config.task.*`; run `config.json` `datasets[*].name/version/ref/path`. "
          f"Per-submission rows: `{fn_m}`; per-task rows: `{fn_mt}`.")
        w("")
        gt = trials[trials["tid_git_commit"].notna()]
        w(f"- Commit-bearing trials: {len(gt):,}; distinct `task_id.git_commit_id`: {gt['tid_git_commit'].nunique()} ({'; '.join(sorted(gt['tid_git_commit'].unique()))}); "
          f"distinct `task_id.git_url`: {'; '.join(sorted(gt['tid_git_url'].dropna().unique()))}; trial `source` of these trials: {'; '.join(sorted(gt['source'].dropna().unique()))}.")
        w(f"- `task_checksum` is present on every trial; the {trials['submission'].nunique()} submissions fall into {n_vec} distinct per-task checksum vectors. "
          f"The largest group has {len(canon_subs)} submissions (the 'canonical' vector).")
        w("")
        cols = ["submission", "blank_commit_trials", "trial_sources", "dataset_names", "dataset_versions", "dataset_refs", "dataset_paths"]
        m10 = meta[meta["submission"].isin(blank_any)][cols].copy()
        m10["dataset_refs"] = m10["dataset_refs"].str.slice(0, 22)
        m10["dataset_paths"] = m10["dataset_paths"].str.slice(0, 60)
        canon_vec = trials[trials["submission"].isin(canon_subs)].groupby("task_id")["task_checksum"].agg(lambda x: x.mode().iloc[0])
        ndiff = {s_: int((trials[trials["submission"] == s_].groupby("task_id")["task_checksum"].first() != canon_vec).sum())
                 for s_ in m10["submission"]}
        m10["task_checksum_vs_canonical"] = [("identical on all 89 tasks" if ndiff[s_] == 0 else f"differs on {ndiff[s_]} of 89 tasks")
                                             for s_ in m10["submission"]]
        w("Blank-commit submissions (dataset fields of `config.json`; `None` means null; the checksum column compares `task_checksum` with the canonical vector):")
        w("")
        w("\n".join(md_table(m10)))
        w("")
        w("- Dataset versions: " + "; ".join(f"`{k}`: {v} run directories" for k, v in
                                        runs["dataset_version"].astype(str).value_counts().items()) + ".")
        w("- Dataset names: " + "; ".join(f"`{k}`: {v}" for k, v in runs["dataset_name"].astype(str).value_counts().items()) + " (run directories).")
        w("")
        led = meta[meta["submission"].str.startswith("0error-Ledger")]
        if len(led):
            r0 = led.iloc[0]
            w(f"- `{r0['submission']}`: run directory `{r0['job_names']}`; `config.json` dataset name `{r0['dataset_names']}`, dataset ref `{r0['dataset_refs']}`; "
              f"trial `source` `{r0['trial_sources']}`. Its metadata names the task set `terminal-bench-2-1`, not `terminal-bench-2`; "
              f"its `task_checksum` differs from the canonical vector on all 89 tasks.")
        w("- The raw files contain no task-set version string beyond these fields (dataset name, dataset version, registry ref, local path, trial `source`); "
          "they do not say whether a task's content differs between task sets. A different `task_checksum` shows that the hashed task directory (or the hashing route) differs; it does not show which files differ.")
        w("")

    # ------------------------------------------------------------------ sensitivity
    w("### 1d. Sensitivity: headline statistics without the blank-commit submissions")
    w("")
    ex_all = [s for s in blank_all]
    ex_any = [s for s in blank_any]
    results_variants = {"all 72": base}
    for name, ex in (("excl. 9 blank", ex_all), ("excl. 10 any blank", ex_any)):
        keep = [s for s in subs if s not in set(ex)]
        with seeded(SEED):
            results_variants[name] = core(long, keep, tasks, nb, ns)
    if canon_subs is not None:
        keep = [s for s in subs if s in set(canon_subs)]
        with seeded(SEED):
            results_variants["canonical checksum only"] = core(long, keep, tasks, nb, ns)
    vstats = {k: v["stats"] for k, v in results_variants.items()}
    summary["sensitivity"] = vstats
    st = sens_table(vstats)
    save("sensitivity_headline", st)
    for name, rv in results_variants.items():
        if name != "all 72":
            save("sensitivity_adjacent_pairs_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_"), adjacent_table(rv))
    w(f"Seed {SEED} for every column, so the columns are paired. 'excl. 9 blank' removes the {len(ex_all)} submissions whose every trial has no task git commit "
      f"(primary sensitivity); 'excl. 10 any blank' also removes `{', '.join(sorted(set(ex_any) - set(ex_all)))}` (5 blank-commit trials); "
      + ("'canonical checksum only' keeps the analysed submissions whose per-task `task_checksum` vector is the most common one (section 1c)."
         if canon_subs is not None else "the checksum-based column is absent (raw tree not scanned).") + " Adjacent pairs are recomputed for each smaller set.")
    w("")
    w("\n".join(md_table(st, {c_: 4 for c_ in st.columns})))
    w("")
    p9 = vstats["excl. 9 blank"]
    w("Labelled values for the primary sensitivity (excl. 9 blank, "
      f"{p9['n_submissions']} submissions):")
    w(f"- n submissions: {p9['n_submissions']}; adjacent pairs: {p9['n_adjacent_pairs']}.")
    w(f"- Median run SD: {p9['run_sd_median_pp']:.2f} pp; median task SD: {p9['task_sd_median_pp']:.2f} pp.")
    w(f"- Variance share at k = 5: median (run SD / task SD)^2 = {p9['var_ratio_run_over_task_median']:.4f}; (f/5)/((1-f)+f/5) at median f = {p9['run_share_k5_at_median_f']:.4f}.")
    w(f"- Unstable adjacent pairs (flip probability > 5%): run resampling {p9['adj_run_gt_5pct']} of {p9['n_adjacent_pairs']}; task resampling {p9['adj_task_gt_5pct']} of {p9['n_adjacent_pairs']}.")
    w(f"- f, median: {p9['f_median']:.4f} (IQR {p9['f_q25']:.4f} to {p9['f_q75']:.4f}); v, median: {p9['v_median']:.4f}.")
    w(f"- Split-half (mean over {ns:,} splits): Spearman {p9['split_half_spearman_all_mean']:.4f}, Kendall {p9['split_half_kendall_all_mean']:.4f}.")
    b72 = base["stats"]
    w(f"- Same quantities for all 72 at seed {SEED}: run SD {b72['run_sd_median_pp']:.2f} pp, task SD {b72['task_sd_median_pp']:.2f} pp, "
      f"unstable adjacent pairs {b72['adj_run_gt_5pct']}/{b72['n_adjacent_pairs']} (run) and {b72['adj_task_gt_5pct']}/{b72['n_adjacent_pairs']} (task), "
      f"f median {b72['f_median']:.4f}, Spearman {b72['split_half_spearman_all_mean']:.4f}, Kendall {b72['split_half_kendall_all_mean']:.4f}.")
    w("")

    # ------------------------------------------------------------------ section 2
    w("## 2. Overdispersion of runs")
    w("")
    od = overdispersion(long, subs, tasks, a.n_sim, SEED)
    odt = od.pop("table")
    summary["overdispersion"] = od
    fn_od = save("overdispersion", odt)
    w("A *run* is a run directory that holds exactly one trial of every one of the 89 tasks. For such a submission with R >= 2 complete runs: "
      "m_r = mean over tasks of the run's outcomes; V = sample variance of the R run means (ddof 1); independence prediction "
      "pred = T^-2 sum_t phat_t (1 - phat_t) R / (R - 1), the unbiased within-task estimate of Var(m_r) when runs are independent and tasks fixed (T = 89; phat_t from the same R runs); "
      "ratio = V / pred. Ratio near 1: independent runs; above 1: run means vary more than independence predicts (a shared run-level effect). "
      f"With R = 2 to 5 runs a single ratio is very noisy, so the null reference below simulates independent Bernoulli(phat_t) runs with the same R ({a.n_sim:,} replicates per submission) and computes the same ratio. "
      f"Per-submission rows: `{fn_od}`.")
    w("")
    w(f"- Submissions in the analysis set: {od['n_total']}; checked: {od['n_checked']}; cannot be checked: {od['n_cannot']}; undefined (independence prediction 0): {od['n_undefined']}.")
    if od["n_checked"]:
        w(f"- Ratio V / pred, median over the {od['n_checked']} checked submissions: {od['ratio_median']:.4f}; mean: {od['ratio_mean']:.4f}; min: {od['ratio_min']:.4f}; max: {od['ratio_max']:.4f}.")
        w(f"- Submissions with ratio > 1: {od['n_ratio_gt_1']} of {od['n_checked']}; median number of runs used: {od['n_runs_median']:.0f}.")
        w(f"- Degrees-of-freedom-weighted pooled ratio, sum (R-1) V / sum (R-1) pred: {od['ratio_pooled_df_weighted']:.4f}.")
        w(f"- Null reference (independent runs), median ratio over submissions, averaged over replicates: {od['null_median_of_medians']:.4f}; mean ratio: {od['null_mean_of_means']:.4f}.")
        w(f"- Share of null replicates with a median ratio at least the observed median: {od['p_null_median_ge_observed']:.4f}; with a mean ratio at least the observed mean: {od['p_null_mean_ge_observed']:.4f}.")
    w("")
    chk_df = odt[odt["status"].str.startswith("checked")][["submission", "n_runs_used", "var_run_means", "var_independence", "ratio", "null_ratio_median", "p_null_ratio_ge_observed", "reason"]]
    w("Checked submissions (variance values in units of share^2; `null_ratio_median` = median ratio under independence):")
    w("")
    w("\n".join(md_table(chk_df.sort_values("ratio"), {"var_run_means": 6, "var_independence": 6})))
    w("")
    cant = odt[~odt["status"].str.startswith("checked")]
    w(f"Cannot be checked or undefined ({len(cant)}): reason counts: "
      + "; ".join(f"{k}: {v}" for k, v in cant["reason"].value_counts().items()) + ".")
    w("")
    w("\n".join(md_table(cant[["submission", "n_run_dirs", "n_complete_runs", "trials_total", "status", "reason"]])))
    w("")
    te = trial_index_overdispersion(long, subs, tasks)
    save("overdispersion_trial_index_exploratory", te)
    te_ok = te[np.isfinite(te["ratio"])]
    w("Exploratory addition (not a run-level check): the same ratio with 'runs' defined by the within-cell trial index (1 to 5, ordered by run directory, start time, trial name) for every submission with exactly 5 trials in every task cell. "
      "For the submissions whose trials sit in one directory this groups trials by start-time rank within a task, which need not coincide with real runs. "
      f"Per-submission rows: `{PREFIX}overdispersion_trial_index_exploratory.csv`.")
    for lab, m in (("grouped by run directory", te_ok["grouping"] == "run directory"), ("grouped by trial index in one directory", te_ok["grouping"] != "run directory")):
        r = te_ok.loc[m, "ratio"]
        w(f"- Ratio V / pred, {lab}: {len(r)} submissions; median {r.median():.4f}; mean {r.mean():.4f}; min {r.min():.4f}; max {r.max():.4f}; ratio > 1 in {int((r > 1).sum())}.")
    summary["overdispersion_trial_index"] = {k: float(v) for k, v in te_ok.groupby("grouping")["ratio"].median().items()}
    w("")

    # ------------------------------------------------------------------ section 3
    w("## 3. Window correlations")
    w("")
    sh = base["arrays"]["sh"]
    rows3 = []
    for name, label in (("all", "all 72"), ("top20", "top 20"), ("top10", "top 10")):
        rows3.append({"window": label, "n_submissions": {"all": len(subs), "top20": 20, "top10": 10}[name],
                      "spearman_mean": sh[f"spearman_{name}"].mean(), "spearman_q025": sh[f"spearman_{name}"].quantile(0.025),
                      "spearman_q975": sh[f"spearman_{name}"].quantile(0.975), "kendall_mean": sh[f"kendall_{name}"].mean(),
                      "kendall_q025": sh[f"kendall_{name}"].quantile(0.025), "kendall_q975": sh[f"kendall_{name}"].quantile(0.975),
                      "n_splits_defined": int(sh[f"spearman_{name}"].notna().sum())})
    t3 = pd.DataFrame(rows3)
    save("window_split_half", t3)
    summary["windows"] = rows3
    w(f"Split-half rank correlation (the {ns:,} splits of `rank_inversion.split_half`, same random stream) restricted to the top 20 and top 10 submissions by full-data mean "
      "(the window is fixed from all trials, then the two half-rankings of those submissions are correlated). Seed %d." % SEED)
    w("")
    w("\n".join(md_table(t3)))
    w("")
    for r in rows3:
        w(f"- Split-half, {r['window']}: Spearman mean {r['spearman_mean']:.4f} (2.5% to 97.5% of splits {r['spearman_q025']:.4f} to {r['spearman_q975']:.4f}); "
          f"Kendall mean {r['kendall_mean']:.4f} ({r['kendall_q025']:.4f} to {r['kendall_q975']:.4f}).")
    w("")
    ri_run, ri_task = base["arrays"]["ri_run"], base["arrays"]["ri_task"]
    committed_ri = pd.read_csv(res_dir / "rank_intervals_run.csv")
    committed_rt = pd.read_csv(res_dir / "rank_intervals_task.csv")
    rows3b = []
    for scheme, ri, cri in (("run", ri_run, committed_ri), ("task", ri_task, committed_rt)):
        for k, lab in ((20, "top 20"), (10, "top 10"), (len(subs), "all 72")):
            sel = ri["observed_rank"] <= k
            csel = cri["observed_rank"] <= k
            rows3b.append({"scheme": scheme, "window": lab, "median_width": ri.loc[sel, "interval_width"].median(),
                           "mean_width": ri.loc[sel, "interval_width"].mean(), "max_width": ri.loc[sel, "interval_width"].max(),
                           "min_width": ri.loc[sel, "interval_width"].min(),
                           "committed_seed_median_width": cri.loc[csel, "interval_width"].median(),
                           "committed_seed_max_width": cri.loc[csel, "interval_width"].max()})
    t3b = pd.DataFrame(rows3b)
    save("window_rank_interval_widths", t3b)
    summary["rank_interval_widths"] = rows3b
    w("95% rank-interval width = (97.5th percentile rank) - (2.5th percentile rank) + 1 over bootstrap replicates (`rank_inversion.rank_intervals`); ranks are among all 72 submissions and the "
      f"window selects submissions by observed rank. `committed_seed_*` columns are read from the committed `rank_intervals_run.csv` and `rank_intervals_task.csv` (seed {ORIGINAL_SEED}).")
    w("")
    w("\n".join(md_table(t3b, {c_: 2 for c_ in t3b.columns})))
    w("")
    r20 = t3b[(t3b["scheme"] == "run") & (t3b["window"] == "top 20")].iloc[0]
    w(f"- Rank-interval width under run resampling, top 20: median {r20['median_width']:.2f} (seed {SEED}); {r20['committed_seed_median_width']:.2f} (committed seed {ORIGINAL_SEED}); max {r20['max_width']:.2f}.")
    r20t = t3b[(t3b["scheme"] == "task") & (t3b["window"] == "top 20")].iloc[0]
    w(f"- Rank-interval width under task resampling, top 20: median {r20t['median_width']:.2f} (seed {SEED}); {r20t['committed_seed_median_width']:.2f} (committed seed {ORIGINAL_SEED}).")
    w("")

    # ------------------------------------------------------------------ section 4
    w("## 4. Two-sided flip threshold")
    w("")
    ca = pd.read_csv(res_dir / "rank_inversion_adjacent.csv")
    rows4 = []
    for src, pr, pt in ((f"committed (seed {ORIGINAL_SEED})", ca["p_flip_run"].to_numpy(), ca["p_flip_task"].to_numpy()),
                        (f"recomputed (seed {SEED})", base["arrays"]["adj_run"], base["arrays"]["adj_task"])):
        for scheme, p in (("run", pr), ("task", pt)):
            n = len(p)
            rows4.append({"source": src, "resampling": scheme, "n_adjacent_pairs": n,
                          "n_flip_gt_5pct": int((p > 0.05).sum()), "share_flip_gt_5pct": float((p > 0.05).mean()),
                          "n_flip_gt_2_5pct": int((p > 0.025).sum()), "share_flip_gt_2_5pct": float((p > 0.025).mean()),
                          "n_between_2_5_and_5pct": int(((p > 0.025) & (p <= 0.05)).sum())})
    t4 = pd.DataFrame(rows4)
    save("flip_threshold_two_sided", t4)
    summary["flip_threshold"] = rows4
    w("A pair is separable at a two-sided 5% rule when its flip probability is at most 2.5% (one-sided 5% = z 1.645; two-sided 5% = z 1.96, the census rule). "
      "Adjacent pairs = ranks (i, i+1) of the 72 submissions (71 pairs). The committed rows read `rank_inversion_adjacent.csv`.")
    w("")
    w("\n".join(md_table(t4)))
    w("")
    for r in rows4:
        w(f"- {r['source']}, {r['resampling']} resampling: flip probability > 5%: {r['n_flip_gt_5pct']} of {r['n_adjacent_pairs']} ({100 * r['share_flip_gt_5pct']:.1f}%); "
          f"> 2.5%: {r['n_flip_gt_2_5pct']} of {r['n_adjacent_pairs']} ({100 * r['share_flip_gt_2_5pct']:.1f}%); between 2.5% and 5%: {r['n_between_2_5_and_5pct']}.")
    w("")

    # ------------------------------------------------------------------ section 5
    w("## 5. Run share of variance as a function of k")
    w("")
    wt = base["arrays"]["wt"]
    f_all = wt["f_run_noise_share"]
    v_all = wt["mean_p1mp_unbiased"].to_numpy()
    f_med = float(f_all.dropna().median())
    rows5 = []
    for k in RUN_K:
        per_sub = np.array([run_share(float(x), k) for x in f_all.dropna()])
        rows5.append({"k": k, "run_share_at_median_f": run_share(f_med, k), "run_share_median_over_submissions": float(np.median(per_sub)),
                      "sd_of_mean_median_pp": float(100 * np.median(sd_of_mean(v_all, len(tasks), k))),
                      "sd_of_mean_min_pp": float(100 * sd_of_mean(v_all, len(tasks), k).min()),
                      "sd_of_mean_max_pp": float(100 * sd_of_mean(v_all, len(tasks), k).max())})
    t5 = pd.DataFrame(rows5)
    save("run_share_by_k", t5)
    summary["run_share_by_k"] = rows5
    w(f"Run share of the variance of a k-run mean = (f/k) / ((1 - f) + f/k), with f the share of a single run's outcome variance that is run noise (so 1 - f is the between-task part, which averaging runs does not reduce). "
      f"Evaluated at the median f over the 72 submissions, f = {f_med:.4f} ({int(f_all.notna().sum())} submissions with f defined), and, as a check, as the median over submissions of each submission's own share. "
      f"SD of the mean over runs for one submission = sqrt(v / (k T)) with v the mean unbiased within-task variance and T = {len(tasks)}; reported as the median (min, max) over the 72 submissions, in percentage points. "
      f"v, median: {float(np.median(v_all)):.4f}.")
    w("")
    w("\n".join(md_table(t5, {c_: 4 for c_ in t5.columns})))
    w("")
    for r in rows5:
        w(f"- k = {r['k']}: run share at median f {r['run_share_at_median_f']:.4f}; median over submissions {r['run_share_median_over_submissions']:.4f}; "
          f"median SD of the mean over runs {r['sd_of_mean_median_pp']:.2f} pp (min {r['sd_of_mean_min_pp']:.2f}, max {r['sd_of_mean_max_pp']:.2f}).")
    w(f"- Single-run SD of the mean, sqrt(v/T), median over submissions: {rows5[0]['sd_of_mean_median_pp']:.2f} pp; five-run SD, sqrt(v/(5T)): {rows5[2]['sd_of_mean_median_pp']:.2f} pp. "
      f"For comparison, median run-to-run bootstrap SD (corrected, committed seed): {s0['run_sd_median_pp']:.2f} pp; median task-sampling SD: {s0['task_sd_median_pp']:.2f} pp.")
    w("")

    # ------------------------------------------------------------------ section 6
    w("## 6. G-study: systems x tasks x runs")
    w("")
    flagged = set(wt.loc[wt["share_tasks_mixed"] == 0, "submission"])
    sets6 = [("balanced subset of the 72", subs),
             ("balanced subset, without flagged and blank-commit submissions",
              [s for s in subs if s not in flagged and s not in set(blank_all)])]
    gres = []
    w("Design. Systems = submissions (random), tasks (random), runs (random). Observation X[s,t,r] in {0,1} = success of trial r of system s on task t. "
      "The run index r is the within-cell trial index of `tb2_load` (1 to 5, ordered by run directory, start time and trial name); it is an arbitrary label, "
      "so the run main effect and the systems x runs and tasks x runs interactions are expected to be near zero, and the three-way term is confounded with residual error. "
      "Only submissions with exactly 5 trials in every one of the 89 task cells enter; the others are dropped whole (no trial is discarded from a kept submission). "
      "Components are method-of-moments (ANOVA) estimators on the 0/1 scale; a negative estimate is reported as raw and set to 0 for shares and coefficients.")
    w("")
    w("Expected mean squares (n_s systems, n_t = 89 tasks, n_r = 5 runs; one observation per cell):")
    w("")
    w("```")
    w("MS_s   = s2_e + n_r s2_st + n_t s2_sr + n_t n_r s2_s")
    w("MS_t   = s2_e + n_r s2_st + n_s s2_tr + n_s n_r s2_t")
    w("MS_r   = s2_e + n_t s2_sr + n_s s2_tr + n_s n_t s2_r")
    w("MS_st  = s2_e + n_r s2_st ;  MS_sr = s2_e + n_t s2_sr ;  MS_tr = s2_e + n_s s2_tr ;  MS_str = s2_e")
    w("```")
    w("")
    w("with s2_e the systems x tasks x runs interaction confounded with residual error. Relative generalizability coefficient for ranking systems with n_t tasks and n_r runs per task (D-study):")
    w("")
    w("```")
    w("Ep2 = s2_s / (s2_s + s2_st / n_t + s2_sr / n_r + s2_e / (n_t n_r))")
    w("Phi = s2_s / (s2_s + (s2_t + s2_st) / n_t + (s2_r + s2_sr) / n_r + (s2_tr + s2_e) / (n_t n_r))   (absolute, for reference)")
    w("```")
    w("")
    w("Because the run labels are exchangeable, a second fit treats runs as replicates within a systems x tasks cell (s2_w = pooled within-cell variance): "
      "Ep2 = s2_s / (s2_s + s2_st / n_t + s2_w / (n_t n_r)). It is reported as 'Ep2 (replicate model)'.")
    w("")
    all_g_tables, dropped_all = [], []
    for label, ss in sets6:
        g = gstudy(long, ss, tasks, label, n_t_design=len(tasks))
        gres.append(g)
        all_g_tables.append(g["table"])
        w(f"### {label}")
        w("")
        w(f"- Systems kept: {g['n_s']} of {len(ss)} candidates; tasks: {g['n_t']}; runs per cell: {g['n_r']}; observations: {g['n_s'] * g['n_t'] * g['n_r']:,}.")
        d = g["dropped"]
        if len(d):
            dropped_all.append(d.assign(set=label))
            w(f"- Dropped ({len(d)} submissions): " + "; ".join(f"`{r.submission}` ({r.reason})" for r in d.itertuples()) + ".")
        else:
            w("- Dropped: none.")
        if g["n_s"] > 0:
            tb = g["table"].drop(columns=["set"])
            w("")
            w("\n".join(md_table(tb, {"mean_square": 6, "raw_estimate": 6, "variance_component": 6, "share_of_total": 4})))
            w("")
            c_ = g["coef"]
            w(f"- Ep2 with n_t = {len(tasks)}, n_r = 1 (3-way model): {c_[f'Ep2_3way_nt{len(tasks)}_nr1']:.4f}; n_r = 5: {c_[f'Ep2_3way_nt{len(tasks)}_nr5']:.4f}.")
            w(f"- Ep2 (replicate model) with n_t = {len(tasks)}, n_r = 1: {c_[f'Ep2_2way_nt{len(tasks)}_nr1']:.4f}; n_r = 5: {c_[f'Ep2_2way_nt{len(tasks)}_nr5']:.4f}.")
            w(f"- Phi (absolute) with n_t = {len(tasks)}, n_r = 1: {c_[f'Phi_3way_nt{len(tasks)}_nr1']:.4f}; n_r = 5: {c_[f'Phi_3way_nt{len(tasks)}_nr5']:.4f}.")
            sh_ = {k: g["a3"]["share"][k] for k in g["a3"]["share"]}
            w("- Shares of total variance (3-way): " + "; ".join(f"{k} {100 * v:.1f}%" for k, v in sh_.items()) + ".")
            w("")
        summary.setdefault("gstudy", {})[label] = {"n_s": g["n_s"], "coef": g["coef"], "comp": g["a3"]["comp"], "share": g["a3"]["share"],
                                                      "raw": g["a3"]["raw"], "comp_2way": g["a2"]["comp"], "dropped": list(d["submission"])}
    save("gstudy_components", pd.concat(all_g_tables, ignore_index=True))
    if dropped_all:
        save("gstudy_dropped", pd.concat(dropped_all, ignore_index=True))

    w("## Files")
    w("")
    w("`robustness_2026-10-19_tb2.md` (this report), `robustness_2026-10-19_tb2_summary.json`, and the CSV files beside it with the prefix `robustness_2026-10-19_tb2_`.")
    w("")
    (res_dir / f"robustness_{STAMP}_tb2.md").write_text("\n".join(L), encoding="utf-8")
    (res_dir / f"{PREFIX}summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                                   encoding="utf-8")
    print(f"wrote robustness_{STAMP}_tb2.md")


if __name__ == "__main__":
    main()
