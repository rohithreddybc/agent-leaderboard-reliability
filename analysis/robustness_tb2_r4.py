"""Terminal-Bench 2.0 robustness analysis, round 4 (2026-10-20): whole-run resampling.

New file. Existing scripts and outputs are not changed; this module reuses ``bootstrap_core``,
``robustness_tb2_r3`` and ``tb2_load`` and writes only files whose names start with
``robustness_2026-10-20_tb2``.

Question. The paper's run-to-run scheme resamples trials WITHIN each task (tasks fixed). A reviewer
can object that a leaderboard run is one whole pass over the 89 tasks, so the natural replicate unit is
the run, and that within-task resampling ignores any run-level effect shared across tasks. For the
submissions whose run directories each hold exactly one trial of every task (the 'identifiable' set of
``robustness_2026-10-19_tb2.md`` section 2), the run directory is a run and can be resampled as a unit.

Scheme (whole-run). X is the (R, T) 0/1 matrix of the R complete runs of a submission, m_r the run
means. Each replicate draws R runs with replacement and takes the mean of their run means. Small-sample
rescaling: the replicate mean is rescaled about the observed mean by sqrt(R / (R - 1)), which makes the
resample variance (1/R^2) sum_r (m_r - mbar)^2 R/(R-1) = s^2 / R, s^2 the ddof-1 variance of the run
means (the unbiased estimate of Var(mean of R runs)). This is the same N/(N-1) logic as in
``bootstrap_core.run_to_run``. The within-task scheme is ``bootstrap_core.run_to_run`` on all trials of
the same submissions (the scheme of the paper), and, as a matched check, on the complete runs only.

Identity. With the rescaling, the analytic whole-run SD is s / sqrt(R) and the analytic within-task SD
(``bootstrap_core.analytic_run_sd``) is sqrt(pred / R), pred the independence prediction of section 2
of the round-3 report, so (SD ratio)^2 = V / pred, the section-2 overdispersion ratio. The script checks
this identity and the identified set against the committed ``robustness_2026-10-19_tb2_overdispersion.csv``.

Seed 20261020. ``bootstrap_core.SEED`` is set to it for the within-task streams; the whole-run stream of
a submission is ``SeedSequence([SEED, 5, crc32(name)])``. Submissions are resampled independently of each
other under both schemes (as in the paper). Adjacent pairs are formed by sorting the identified
submissions by their mean over all trials (as the paper does); flip probability = share of replicates in
which the lower-ranked submission scores above the higher-ranked one, ties half.

Usage: python robustness_tb2_r4.py [--results results] [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap_core as bc
import robustness_tb2_r3 as T3
from tb2_load import select_analysis_set

SEED = 20261020
STAMP = "2026-10-20"
PREFIX = f"robustness_{STAMP}_tb2_"
STREAM_WHOLE_RUN = 5


# --------------------------------------------------------------------------------------------------
# identification of runs
# --------------------------------------------------------------------------------------------------

def identifiable_runs(long: pd.DataFrame, subs: list[str], tasks: list[str]) -> dict[str, np.ndarray]:
    """{submission: (R, T) 0/1 matrix of complete runs} for the submissions with R >= 2 complete runs
    (run directories with exactly one trial of every task) and a positive independence prediction (a
    submission whose tasks all have identical outcomes across runs has within-task SD 0 and an undefined
    ratio; it is left out, as in section 2 of the round-3 report)."""
    out = {}
    for s in subs:
        X, names, _ = T3.complete_run_matrix(long[long["submission"] == s], tasks)
        if X.shape[0] < 2:
            continue
        if analytic_within_task_sd(X) <= 0:
            continue
        out[s] = X
    return out


# --------------------------------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------------------------------

def whole_run_resample(X: np.ndarray, n_boot: int, rng: np.random.Generator, correct: bool = True) -> np.ndarray:
    """Replicate means of the submission when whole runs are resampled with replacement (shape (n_boot,)).
    ``correct`` applies the sqrt(R / (R - 1)) rescaling about the observed mean."""
    R = X.shape[0]
    m = X.mean(axis=1)
    idx = rng.integers(0, R, size=(n_boot, R))
    x = m[idx].mean(axis=1)
    if correct:
        x = m.mean() + (x - m.mean()) * np.sqrt(R / (R - 1.0))
    return x


def whole_run_stream(name: str, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, STREAM_WHOLE_RUN, zlib.crc32(name.encode("utf-8"))]))


def analytic_whole_run_sd(X: np.ndarray) -> float:
    """s / sqrt(R): the rescaled whole-run bootstrap SD in the limit of many replicates."""
    R = X.shape[0]
    return float(X.mean(axis=1).std(ddof=1) / np.sqrt(R))


def analytic_within_task_sd(X: np.ndarray) -> float:
    """Analytic within-task SD on the same R runs (``bootstrap_core.analytic_run_sd`` with N = R)."""
    R, T = X.shape
    p = X.mean(axis=0)
    return float(np.sqrt((p * (1 - p)).sum() / (R - 1)) / T)


def counts_from_runs(name: str, X: np.ndarray, tasks: list[str]) -> bc.Counts:
    R = X.shape[0]
    return bc.Counts([name], list(tasks), X.sum(axis=0).astype(np.int64)[None, :], np.full((1, X.shape[1]), R, dtype=np.int64))


def compare_schemes(long: pd.DataFrame, subs: list[str], tasks: list[str], n_boot: int, seed: int = SEED) -> dict:
    """Whole-run against within-task resampling for the identifiable submissions."""
    runs = identifiable_runs(long, subs, tasks)
    names = sorted(runs)
    if not names:
        raise ValueError("no identifiable submission")
    with T3.seeded(seed):
        c_all = bc.build_counts(long, names, tasks)
        S_task_all = bc.run_to_run(c_all, n_boot)
        S_task_match = np.column_stack([bc.run_to_run(counts_from_runs(n, runs[n], tasks), n_boot)[:, 0] for n in names])
    S_whole = np.column_stack([whole_run_resample(runs[n], n_boot, whole_run_stream(n, seed)) for n in names])
    mean_all = bc.macro_mean(c_all.K, c_all.N)
    rows = []
    for i, n in enumerate(names):
        X = runs[n]
        sd_w, sd_t, sd_m = S_whole[:, i].std(ddof=1), S_task_all[:, i].std(ddof=1), S_task_match[:, i].std(ddof=1)
        a_w, a_t = analytic_whole_run_sd(X), analytic_within_task_sd(X)
        rows.append({
            "submission": n, "n_runs_used": X.shape[0], "n_trials_all": int(c_all.N[i].sum()),
            "mean_all_trials": float(mean_all[i]), "mean_complete_runs": float(X.mean()),
            "sd_whole_run_pp": 100 * sd_w, "sd_within_task_pp": 100 * sd_t, "ratio": sd_w / sd_t,
            "sd_within_task_matched_pp": 100 * sd_m, "ratio_matched": sd_w / sd_m,
            "analytic_sd_whole_run_pp": 100 * a_w, "analytic_sd_within_task_pp": 100 * a_t,
            "analytic_ratio": a_w / a_t, "analytic_ratio_squared": (a_w / a_t) ** 2,
        })
    tab = pd.DataFrame(rows)
    order = bc.observed_order(names, mean_all)
    P = {k: bc.flip_matrix(S, order) for k, S in (("whole_run", S_whole), ("within_task", S_task_all), ("within_task_matched", S_task_match))}
    adj = pd.DataFrame({
        "rank_hi": np.arange(1, len(order)), "submission_hi": [names[i] for i in order[:-1]],
        "submission_lo": [names[i] for i in order[1:]],
        "gap_pp": 100 * (mean_all[order[:-1]] - mean_all[order[1:]]),
        "p_flip_whole_run": np.diag(P["whole_run"], 1), "p_flip_within_task": np.diag(P["within_task"], 1),
        "p_flip_within_task_matched": np.diag(P["within_task_matched"], 1),
    })
    return {"table": tab, "adjacent": adj, "names": names, "n_boot": n_boot}


def summarize(res: dict, threshold: float = 0.05) -> dict:
    t, a = res["table"], res["adjacent"]
    return {
        "n_submissions": int(len(t)), "n_adjacent_pairs": int(len(a)), "n_boot": int(res["n_boot"]), "flip_threshold": threshold,
        "sd_whole_run_median_pp": float(t["sd_whole_run_pp"].median()),
        "sd_within_task_median_pp": float(t["sd_within_task_pp"].median()),
        "ratio_median": float(t["ratio"].median()), "ratio_min": float(t["ratio"].min()), "ratio_max": float(t["ratio"].max()),
        "n_ratio_gt_1": int((t["ratio"] > 1).sum()),
        "ratio_matched_median": float(t["ratio_matched"].median()), "n_ratio_matched_gt_1": int((t["ratio_matched"] > 1).sum()),
        "analytic_ratio_median": float(t["analytic_ratio"].median()), "n_analytic_ratio_gt_1": int((t["analytic_ratio"] > 1).sum()),
        "flip_gt_whole_run": int((a["p_flip_whole_run"] > threshold).sum()),
        "flip_gt_within_task": int((a["p_flip_within_task"] > threshold).sum()),
        "flip_gt_within_task_matched": int((a["p_flip_within_task_matched"] > threshold).sum()),
        "pairs_flip_class_differs": int(((a["p_flip_whole_run"] > threshold) != (a["p_flip_within_task"] > threshold)).sum()),
    }


def check_against_committed(res: dict, overdisp_csv: Path) -> dict:
    """Compare the identified set and the identity (analytic ratio)^2 = V / pred with the committed
    section-2 table (``robustness_2026-10-19_tb2_overdispersion.csv``)."""
    od = pd.read_csv(overdisp_csv)
    ok = od[od["status"].astype(str).str.startswith("checked")].set_index("submission")
    t = res["table"].set_index("submission")
    same_set = set(ok.index) == set(t.index)
    common = sorted(set(ok.index) & set(t.index))
    diff = float((ok.loc[common, "ratio"] - t.loc[common, "analytic_ratio_squared"]).abs().max()) if common else float("nan")
    return {"same_set": bool(same_set), "n_committed": int(len(ok)), "n_here": int(len(t)),
            "max_abs_diff_ratio": diff, "only_committed": sorted(set(ok.index) - set(t.index)),
            "only_here": sorted(set(t.index) - set(ok.index))}


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------

def build_report(res: dict, summ: dict, chk: dict | None, n_boot: int) -> str:
    t, a = res["table"], res["adjacent"]
    L: list[str] = []
    w = L.append
    w(f"# Terminal-Bench 2.0 robustness analysis, round 4 ({STAMP}): whole-run resampling")
    w("")
    w("Numbers only. Generated by `analysis/robustness_tb2_r4.py`; tests in `analysis/tests/test_robustness_tb2_r4.py`. "
      "New files only: no existing script or output was changed. Input: `analysis/results/tb2_long.csv` "
      "(and, for the consistency check, the committed `robustness_2026-10-19_tb2_overdispersion.csv`).")
    w("")
    w("## Definitions and conventions")
    w("")
    w(f"- Seed {SEED}; B = {n_boot:,} replicates for every scheme. `bootstrap_core.SEED` is set to it for the within-task streams; "
      "the whole-run stream of a submission is `SeedSequence([20261020, 5, crc32(name)])`. Submissions are resampled independently of each other under both schemes.")
    w("- Identified set: submissions of the 72-submission analysis set (at least 5 trials on each of the 89 tasks) with at least 2 *complete* runs (run directories holding exactly one trial of every task) "
      "and a positive within-task variance; these are the 20 'checked' submissions of section 2 of `robustness_2026-10-19_tb2.md`.")
    w("- Whole-run resampling: R complete runs per submission, run means m_r; each replicate draws R runs with replacement and averages their means. "
      "**Small-sample rescaling:** the replicate mean is rescaled about the observed mean by sqrt(R/(R-1)), so that the resample variance is s^2/R, s^2 the ddof-1 variance of the run means "
      "(the same N/(N-1) logic as `bootstrap_core.run_to_run`).")
    w("- Within-task resampling (the paper's scheme): tasks fixed, trials resampled within task with the N/(N-1) rescaling (`bootstrap_core.run_to_run`) on ALL trials of the same submissions. "
      "'Matched' = the same scheme on the complete runs only (differs from the paper's scheme for one submission, whose fourth run directory is not complete).")
    w("- Ratio = SD (whole-run) / SD (within-task). Flip probability of an adjacent pair: share of replicates in which the lower-ranked submission scores above the higher-ranked one, ties half; "
      "pairs are adjacent in the order by mean over all trials among the identified submissions.")
    w("- Identity: with the rescaling, analytic whole-run SD = s/sqrt(R), analytic within-task SD = sqrt(pred/R), so (ratio)^2 = V/pred, the overdispersion ratio of round 3, section 2.")
    w("")
    w("## 1. SD of the submission mean")
    w("")
    w(f"- Submissions: {summ['n_submissions']}; complete runs used: median {int(t['n_runs_used'].median())} "
      f"(range {int(t['n_runs_used'].min())} to {int(t['n_runs_used'].max())}).")
    w(f"- Median SD of the submission mean, whole-run resampling: **{summ['sd_whole_run_median_pp']:.2f} pp**; within-task resampling (paper): **{summ['sd_within_task_median_pp']:.2f} pp**.")
    w(f"- Ratio whole-run / within-task: median **{summ['ratio_median']:.3f}** (min {summ['ratio_min']:.3f}, max {summ['ratio_max']:.3f}); "
      f"submissions with ratio > 1: **{summ['n_ratio_gt_1']} of {summ['n_submissions']}**.")
    w(f"- Matched within-task (complete runs only): median ratio {summ['ratio_matched_median']:.3f}; ratio > 1 for {summ['n_ratio_matched_gt_1']} of {summ['n_submissions']}.")
    w(f"- Analytic ratio, s/sqrt(R) over sqrt(pred/R): median {summ['analytic_ratio_median']:.3f}; > 1 for {summ['n_analytic_ratio_gt_1']} of {summ['n_submissions']}.")
    if chk is not None:
        w(f"- Consistency with the committed section 2: identified set identical: {'yes' if chk['same_set'] else 'NO'} "
          f"({chk['n_here']} here, {chk['n_committed']} committed); largest |committed ratio - (analytic ratio)^2| = {chk['max_abs_diff_ratio']:.2e}.")
    w("")
    cols = ["submission", "n_runs_used", "mean_all_trials", "sd_whole_run_pp", "sd_within_task_pp", "ratio", "ratio_matched", "analytic_ratio"]
    tt = t[cols].sort_values("ratio", ascending=False)
    w("| " + " | ".join(cols) + " |")
    w("|" + "---|" * len(cols))
    for r in tt.itertuples(index=False):
        w(f"| {r.submission} | {r.n_runs_used} | {r.mean_all_trials:.4f} | {r.sd_whole_run_pp:.3f} | {r.sd_within_task_pp:.3f} | {r.ratio:.3f} | {r.ratio_matched:.3f} | {r.analytic_ratio:.3f} |")
    w("")
    w("Per-submission rows: `robustness_2026-10-20_tb2_whole_run_vs_within_task.csv`.")
    w("")
    w("## 2. Adjacent-pair flip probabilities")
    w("")
    n = summ["n_adjacent_pairs"]
    w(f"- Adjacent pairs among the {summ['n_submissions']} identified submissions (sorted by mean): {n}.")
    w(f"- Pairs with flip probability > 5%: whole-run resampling **{summ['flip_gt_whole_run']} of {n}**; within-task resampling (paper) **{summ['flip_gt_within_task']} of {n}**; "
      f"matched within-task {summ['flip_gt_within_task_matched']} of {n}.")
    w(f"- Pairs whose > 5% classification differs between whole-run and within-task resampling: {summ['pairs_flip_class_differs']}.")
    w("- Note: these 20 submissions are not adjacent in the 72-submission ranking; the pairs are adjacent only within this set, so the counts are not comparable with the 63 of 71 of the paper.")
    w("")
    w("| rank | higher | lower | gap (pp) | flip, whole-run | flip, within-task | flip, within-task matched |")
    w("|---|---|---|---|---|---|---|")
    for r in a.itertuples(index=False):
        w(f"| {r.rank_hi} | {r.submission_hi} | {r.submission_lo} | {r.gap_pp:.2f} | {r.p_flip_whole_run:.4f} | {r.p_flip_within_task:.4f} | {r.p_flip_within_task_matched:.4f} |")
    w("")
    w("Per-pair rows: `robustness_2026-10-20_tb2_whole_run_adjacent_pairs.csv`.")
    w("")
    return "\n".join(L)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: --results)")
    ap.add_argument("--n-boot", type=int, default=bc.B)
    a = ap.parse_args()
    out = a.out or a.results
    out.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(a.results / "tb2_long.csv")
    subs, tasks, _ = select_analysis_set(long)
    res = compare_schemes(long, subs, tasks, a.n_boot)
    summ = summarize(res)
    od = a.results / "robustness_2026-10-19_tb2_overdispersion.csv"
    chk = check_against_committed(res, od) if od.exists() else None
    res["table"].to_csv(out / f"{PREFIX}whole_run_vs_within_task.csv", index=False)
    res["adjacent"].to_csv(out / f"{PREFIX}whole_run_adjacent_pairs.csv", index=False)
    (out / f"{PREFIX}summary.json").write_text(json.dumps({"seed": SEED, **summ, "consistency": chk}, indent=2), encoding="utf-8")
    (out / f"robustness_{STAMP}_tb2.md").write_text(build_report(res, summ, chk, a.n_boot), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    print("consistency:", chk)


if __name__ == "__main__":
    main()
