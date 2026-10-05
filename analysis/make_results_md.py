"""Assemble results/results.md from the CSV and JSON outputs of the analysis
scripts. Every number is read from those files; nothing is typed in here.

Usage: python make_results_md.py [--results results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap_core as bc

# Worked examples of the detectability bound (paper Sections 4 and 5): label, n tasks, k runs per system.
WORKED = (("illustration: n = 100, k = 1", 100, 1),
          ("Terminal-Bench 2.0 design: n = 89, k = 5", 89, 5),
          ("ReliabilityBench check: n = 20, k = 2", 20, 2))
Z_TWO, Z_ONE = 1.96, 1.645  # two-sided 95% (as in Eq. 1 of the paper) and one-sided 5% normal quantiles
F_PS = (0.2, 0.5, 0.8)  # success rates at which the f-parameterised bound is evaluated


def delta_min(v: float, n: int, k: int, z: float = Z_TWO) -> float:
    """Smallest observed difference of two k-run means that is significant at the level of z
    (v = within-task variance shared by both systems): z * sqrt(2 v / (n k)). In pp."""
    return 100 * z * float(np.sqrt(2 * v / (n * k)))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def build(res: Path) -> str:
    load = json.loads((res / "tb2_load_summary.json").read_text())
    counts = pd.read_csv(res / "tb2_load_counts.csv")
    boot = pd.read_csv(res / "tb2_bootstrap.csv")
    wt = pd.read_csv(res / "tb2_within_task_variance.csv")
    rk = json.loads((res / "rank_inversion_summary.json").read_text())
    tau = json.loads((res / "tau2_summary.json").read_text())
    tau_df = pd.read_csv(res / "tau2_consistency.csv")
    out = ["# RQ3 re-analysis results", "",
           "Numbers only. Every value is produced by a script in `analysis/`; see the CSV and JSON files in this folder.",
           f"Seed {bc.SEED}; B = {rk['n_boot']:,} bootstrap replicates; split-half splits = {rk['split_half']['n_splits']:,}.", ""]

    out += ["## Definitions", "",
            "- Success: verifier reward >= 1.0. A trial with no reward is counted as failure and flagged (`no_reward`).",
            "- Submission mean: mean over tasks of the task success rate (trials / task).",
            "- Leaderboard order: submissions sorted by observed mean, best first. The dataset holds no published score or rank.",
            "- PRIMARY (run-to-run): tasks fixed; trials resampled with replacement within each task, per submission independently. "
            "Small-sample correction: resampling N trials from N understates the variance by (N-1)/N, so each task's resampled rate is rescaled about its observed rate by sqrt(N/(N-1)) "
            "(`bootstrap_core.py` docstring); `run SD (analytic)` is the closed form sqrt(sum_t phat(1-phat)/(N_t-1))/T, which the corrected bootstrap SD must match.",
            "- SECONDARY (task-sampling): per-task trial means fixed; tasks resampled with replacement, same draw for every submission.",
            "- Flip probability of a pair: share of replicates in which the observed lower-ranked submission scores above the higher-ranked one, plus half the share of exact ties.",
            "- Separable pair: flip probability <= 0.05. Separable rank positions: size of the largest set of submissions in which every pair is separable.",
            "- Gap: observed difference in submission means, in percentage points (pp).",
            "- Analysis set: submissions with at least 5 trials on every task of the task union.", ""]

    out += ["## Terminal-Bench 2.0 data", "",
            f"Dataset revision and licence: `data/SOURCES.md`.", "",
            md_table(pd.DataFrame([
                ("submissions", load["n_submissions"]), ("tasks (union)", load["n_tasks_union"]),
                ("run directories", load["n_runs_total"]), ("trials", f"{load['n_trials_total']:,}"),
                ("task git commits", load["n_task_commits"]),
                ("trials with no verifier reward", load["n_no_reward_trials"]),
                ("(submission, task) cells", f"{load['cells_total']:,}"),
                ("cells absent", load["cells_absent"]), ("cells with 1-4 trials", load["cells_below_5"]),
                ("submissions with >= 5 trials on every task (analysis set)", load["analysis_set_submissions"]),
            ], columns=["quantity", "value"])), "",
            "Trials per (submission, task) cell:", "",
            md_table(pd.DataFrame(list(load["trials_per_cell_distribution"].items()), columns=["trials in cell", "cells"])), ""]
    excl = counts[counts["included"] == 0]
    out += [f"Excluded from the analysis set: {len(excl)} submissions.", ""]
    if len(excl):
        e = excl[["submission", "n_trials", "n_runs", "n_tasks", "trials_per_task_min", "trials_per_task_max", "reason"]]
        out += [md_table(e), ""]
    inc = counts[counts["included"] == 1]
    out += ["Trials per task within the analysis set (min / median / max across submissions of each submission's per-task minimum, median, maximum):", "",
            md_table(pd.DataFrame([
                ("per-task minimum", inc["trials_per_task_min"].min(), inc["trials_per_task_min"].median(), inc["trials_per_task_min"].max()),
                ("per-task median", inc["trials_per_task_median"].min(), inc["trials_per_task_median"].median(), inc["trials_per_task_median"].max()),
                ("per-task maximum", inc["trials_per_task_max"].min(), inc["trials_per_task_max"].median(), inc["trials_per_task_max"].max()),
            ], columns=["statistic", "min", "median", "max"])), ""]

    out += ["## Per-submission mean and intervals (analysis set)", "",
            "Values in percent. `run` = PRIMARY run-to-run; `task` = SECONDARY task-sampling. SD and 95% percentile interval of the resampled mean.", ""]
    b = boot.copy()
    t = pd.DataFrame({
        "rank": range(1, len(b) + 1), "submission": b["submission"], "trials": b["n_trials"],
        "mean": b["mean_success"].map(pct), "run SD": b["run_sd"].map(lambda x: pct(x, 2)),
        "run SD (analytic)": b["run_sd_analytic"].map(lambda x: pct(x, 2)),
        "run 95% CI": [f"{pct(l)} to {pct(h)}" for l, h in zip(b["run_ci_low"], b["run_ci_high"])],
        "task SD": b["task_sd"].map(lambda x: pct(x, 2)),
        "task 95% CI": [f"{pct(l)} to {pct(h)}" for l, h in zip(b["task_ci_low"], b["task_ci_high"])]})
    out += [md_table(t), "",
            "Run-to-run SD across the analysis set (pp): "
            f"min {pct(b['run_sd'].min(), 2)}, median {pct(b['run_sd'].median(), 2)}, max {pct(b['run_sd'].max(), 2)}. "
            f"Task-sampling SD (pp): min {pct(b['task_sd'].min(), 2)}, median {pct(b['task_sd'].median(), 2)}, max {pct(b['task_sd'].max(), 2)}.", ""]
    pos = b[b["run_sd_analytic"] > 0]
    ratio = pos["run_sd"] / pos["run_sd_analytic"]
    out += ["Check of the corrected bootstrap against the closed form (submissions with non-zero run variance, "
            f"{len(pos)} of {len(b)}): ratio bootstrap SD / analytic SD has median {ratio.median():.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}. "
            f"Median analytic run SD {pct(b['run_sd_analytic'].median(), 2)} pp.", ""]

    out += ["## Within-task variance (analysis set)", "",
            "p = task success rate over trials. `p(1-p)` is averaged over tasks; unbiased = p(1-p) n/(n-1). Per-submission rows: `tb2_within_task_variance.csv`.", "",
            md_table(pd.DataFrame([
                (c, f"{wt[c].min():.4f}", f"{wt[c].median():.4f}", f"{wt[c].max():.4f}")
                for c in ("mean_p1mp_plugin", "mean_p1mp_unbiased", "share_tasks_mixed", "share_tasks_all_success", "share_tasks_all_fail")
            ], columns=["quantity (across submissions)", "min", "median", "max"])), ""]

    flagged_names = set(counts.loc[(counts["included"] == 1) & ((counts["zero_within_task_variance"] == 1) | (counts["duplicate_trial_names"] > 0)), "submission"])
    f_all = wt["f_run_noise_share"].dropna()
    f_cl = wt.loc[~wt["submission"].isin(flagged_names), "f_run_noise_share"].dropna()
    q = lambda x: x.quantile([0.25, 0.75]).tolist()
    v_med = float(wt["mean_p1mp_unbiased"].median())
    f_med = float(f_all.median())
    out += ["## Run-noise share f and the detectability bound (worked numbers)", "",
            "f = v / (P(1-P)): the share of a single run's outcome variance that is run-to-run noise, with v the unbiased mean within-task variance of a submission and P its mean success. "
            "Undefined at P = 0 or 1. The detectability bound is the smallest observed difference between two systems that is significant at 95% (two-sided z = 1.96; "
            "50% power at a true difference equal to the bound), z * sqrt(2 v / (n k)); no power claim beyond that is made.", "",
            md_table(pd.DataFrame([
                ("v, median (min to max)", f"{v_med:.4f} ({wt['mean_p1mp_unbiased'].min():.4f} to {wt['mean_p1mp_unbiased'].max():.4f})"),
                (f"f, median (IQR; min to max), {len(f_all)} submissions", f"{f_med:.3f} ({q(f_all)[0]:.3f} to {q(f_all)[1]:.3f}; {f_all.min():.3f} to {f_all.max():.3f})"),
                (f"f, median (IQR; min to max), without the flagged submission, {len(f_cl)}", f"{f_cl.median():.3f} ({q(f_cl)[0]:.3f} to {q(f_cl)[1]:.3f}; {f_cl.min():.3f} to {f_cl.max():.3f})"),
                ("corr(v, P) across submissions", f"{wt['mean_p1mp_unbiased'].corr(wt['mean_success']):.3f}"),
            ], columns=["quantity (across the analysis set)", "value"])), "",
            "Bound in pp, with v = median v transported absolutely (z = 1.96 two-sided; z = 1.645 is the one-sided 5% rule used by the flip probability) "
            f"and with v = f x P(1-P) at f = {f_med:.3f} for P = {', '.join(f'{x:g}' for x in F_PS)} (both systems at P):", ""]
    rows_w = []
    for lab, n_, k_ in WORKED:
        rows_w.append((lab, f"{delta_min(v_med, n_, k_):.2f}", f"{delta_min(v_med, n_, k_, Z_ONE):.2f}",
                       *[f"{delta_min(f_med * P_ * (1 - P_), n_, k_):.2f}" for P_ in F_PS]))
    out += [md_table(pd.DataFrame(rows_w, columns=["case", "z = 1.96, v median", "z = 1.645, v median",
                                                   *[f"z = 1.96, f median, P = {P_:g}" for P_ in F_PS]])), "",
            f"For comparison, the bootstrap gap_max_unstable (run-to-run) of the Terminal-Bench 2.0 analysis set is {rk['run']['gap_max_unstable_pp']:.2f} pp "
            f"and gap_min_stable {rk['run']['gap_min_stable_pp']:.2f} pp (Rank inversion table).", ""]

    r, k = rk["run"], rk["task"]
    rows = [
        ("submissions / tasks / pairs", f"{rk['n_submissions']} / {rk['n_tasks']} / {rk['n_pairs']}", ""),
        ("adjacent pairs", r["n_adjacent_pairs"], k["n_adjacent_pairs"]),
        ("adjacent pairs with flip probability > 5%", f"{r['n_adjacent_flip_gt_5pct']} ({pct(r['share_adjacent_flip_gt_5pct'])}%)",
         f"{k['n_adjacent_flip_gt_5pct']} ({pct(k['share_adjacent_flip_gt_5pct'])}%)"),
        ("adjacent pairs with flip probability > 20%", f"{r['n_adjacent_flip_gt_20pct']} ({pct(r['share_adjacent_flip_gt_20pct'])}%)",
         f"{k['n_adjacent_flip_gt_20pct']} ({pct(k['share_adjacent_flip_gt_20pct'])}%)"),
        ("all pairs with flip probability > 5%", f"{pct(r['share_all_pairs_flip_gt_5pct'])}%", f"{pct(k['share_all_pairs_flip_gt_5pct'])}%"),
        ("separable rank positions (exact)", r["separable_positions_exact"], k["separable_positions_exact"]),
        ("separable rank positions (greedy from top)", r["separable_positions_greedy"], k["separable_positions_greedy"]),
        ("gap_max_unstable (pp)", f"{r['gap_max_unstable_pp']:.2f}", f"{k['gap_max_unstable_pp']:.2f}"),
        ("gap_min_stable (pp)", f"{r['gap_min_stable_pp']:.2f}", f"{k['gap_min_stable_pp']:.2f}"),
        ("gap_bin_stable (pp)", r["gap_bin_stable_pp"], k["gap_bin_stable_pp"]),
        ("Kendall tau, observed rank vs mean bootstrap rank", f"{r['observed_vs_mean_bootstrap_rank_kendall']:.4f}", f"{k['observed_vs_mean_bootstrap_rank_kendall']:.4f}"),
        ("observed rank outside own 95% rank interval", r["observed_rank_outside_95pct_interval"], k["observed_rank_outside_95pct_interval"]),
        ("rank interval width, median (positions)", f"{r['rank_interval_width_median']:.1f}", f"{k['rank_interval_width_median']:.1f}"),
        ("rank interval width, max (positions)", f"{r['rank_interval_width_max']:.1f}", f"{k['rank_interval_width_max']:.1f}"),
    ]
    out += ["## Rank inversion", "",
            md_table(pd.DataFrame(rows, columns=["quantity", "PRIMARY run-to-run", "SECONDARY task-sampling"])), "",
            "Files: `rank_inversion_pairs.csv`, `rank_inversion_adjacent.csv`, `rank_intervals_run.csv`, `rank_intervals_task.csv`, `rank_gap_bins.csv`.", ""]

    sens = res / "rank_inversion_summary_sensitivity_excl_MAYA.json"
    if sens.exists():
        x = json.loads(sens.read_text())
        rr, kk = x["run"], x["task"]
        flags = counts[(counts["included"] == 1) & ((counts["zero_within_task_variance"] == 1) | (counts["duplicate_trial_names"] > 0))]
        out += ["## Sensitivity: analysis set without flagged submissions", "",
                "Flag: every task has identical outcomes across all its trials (zero within-task variance), or trial names repeat within a task. "
                "Flagged members of the analysis set: " + ", ".join(
                    f"{r.submission} ({int(r.duplicate_trial_names)} repeated trial names, zero within-task variance = {int(r.zero_within_task_variance)})"
                    for r in flags.itertuples()) + ".", "",
                md_table(pd.DataFrame([
                    ("submissions", x["n_submissions"], x["n_submissions"]),
                    ("adjacent pairs with flip probability > 5%", f"{rr['n_adjacent_flip_gt_5pct']} ({pct(rr['share_adjacent_flip_gt_5pct'])}%)", f"{kk['n_adjacent_flip_gt_5pct']} ({pct(kk['share_adjacent_flip_gt_5pct'])}%)"),
                    ("adjacent pairs with flip probability > 20%", f"{rr['n_adjacent_flip_gt_20pct']} ({pct(rr['share_adjacent_flip_gt_20pct'])}%)", f"{kk['n_adjacent_flip_gt_20pct']} ({pct(kk['share_adjacent_flip_gt_20pct'])}%)"),
                    ("separable rank positions (exact)", rr["separable_positions_exact"], kk["separable_positions_exact"]),
                    ("gap_max_unstable (pp)", f"{rr['gap_max_unstable_pp']:.2f}", f"{kk['gap_max_unstable_pp']:.2f}"),
                    ("gap_min_stable (pp)", f"{rr['gap_min_stable_pp']:.2f}", f"{kk['gap_min_stable_pp']:.2f}"),
                    ("split-half Spearman-Brown Spearman, mean", f"{x['split_half']['spearman_sb_mean']:.4f}", ""),
                ], columns=["quantity", "PRIMARY run-to-run", "SECONDARY task-sampling"])), ""]

    sh = rk["split_half"]
    out += ["## Split-half reliability of the ranking", "",
            f"{sh['n_splits']:,} random splits of each task's trials into two disjoint halves; Spearman-Brown m = {sh['spearman_brown_m']:.3f}. Mean and 2.5%/97.5% quantiles over splits.", "",
            md_table(pd.DataFrame([
                (n, f"{sh[n + '_mean']:.4f}", f"{sh[n + '_q025']:.4f}", f"{sh[n + '_q975']:.4f}")
                for n in ("spearman", "kendall", "spearman_sb", "kendall_sb")
            ], columns=["statistic", "mean", "q2.5", "q97.5"])), "",
            "`*_sb`: Spearman-Brown corrected to the full trial count (applied to Kendall tau as a heuristic).", ""]

    out += ["## tau2-bench aggregates", "", tau["note"], "",
            f"Submissions with pass^1 to pass^4 present in at least one domain: {tau['n_submissions_complete']} "
            f"(of which with 4 trials stated in the submission file: {tau['submissions_with_stated_4_trials']}). "
            f"Submission-domain rows: {tau['n_rows_submission_domain']}; rows where pass^k increases with k: {tau['n_non_monotone_rows']}.", "",
            "Drop pass^1 - pass^4 (pp) by domain:", "",
            md_table(pd.DataFrame([
                (d, v["n_submissions"], f"{v['drop_4_pp_min']:.1f}", f"{v['drop_4_pp_median']:.1f}", f"{v['drop_4_pp_mean']:.1f}", f"{v['drop_4_pp_max']:.1f}")
                for d, v in tau["by_domain"].items()
            ], columns=["domain", "submissions", "min", "median", "mean", "max"])), "",
            "Per-submission drops (pp):", ""]
    d = tau_df[["submission", "domain", "trials_stated", "pass_1", "pass_2", "pass_3", "pass_4", "drop_2_pp", "drop_3_pp", "drop_4_pp"]].copy()
    for c in ("drop_2_pp", "drop_3_pp", "drop_4_pp"):
        d[c] = d[c].map(lambda x: f"{x:.2f}")
    d["trials_stated"] = d["trials_stated"].map(lambda x: "" if pd.isna(x) else int(x))
    out += [md_table(d), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    a = ap.parse_args()
    (a.results / "results.md").write_text(build(a.results), encoding="utf-8")


if __name__ == "__main__":
    main()
