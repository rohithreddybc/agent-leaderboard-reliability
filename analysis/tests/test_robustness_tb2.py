"""Tests for robustness_tb2.py (missing-reward sensitivity). Synthetic inputs only."""
import numpy as np
import pandas as pd
import pytest

import bootstrap_core as bc
import rank_inversion as R
import robustness_tb2 as T


def _long(n_subs=6, n_tasks=12, trials=6, seed=0, drop_frac=0.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, n_tasks)
    rows = []
    for s in range(n_subs):
        skill = 0.04 * s
        for t in range(n_tasks):
            for i in range(trials):
                miss = rng.random() < drop_frac
                succ = 0 if miss else int(rng.random() < min(0.95, base[t] + skill - 0.1))
                rows.append((f"s{s}", f"t{t}", i + 1, succ, int(miss)))
    df = pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success", "no_reward"])
    df["agent"], df["model"] = df["submission"], "m"
    return df


def test_task_sampling_missing_equals_original_on_complete_data():
    c = bc.build_counts(_long())
    assert np.array_equal(T.task_sampling_missing(c, 3000), bc.task_sampling(c, 3000))


def test_task_sampling_missing_skips_uncovered_tasks():
    c = bc.build_counts(_long(n_subs=3))
    c.K[0, :4], c.N[0, :4] = 0, 0                      # submission 0 covers 8 of the 12 tasks
    S = T.task_sampling_missing(c, 4000)
    assert np.isfinite(S).all() and ((S >= 0) & (S <= 1)).all()
    covered_mean = bc.macro_mean(c.K, c.N)[0]
    assert S[:, 0].mean() == pytest.approx(covered_mean, abs=0.01)   # unbiased for the mean over covered tasks
    with pytest.raises(ValueError):
        bc.task_sampling(c, 10)                          # the original refuses incomplete coverage


def test_split_half_missing_equals_original_when_every_cell_has_two_trials():
    c = bc.build_counts(_long(seed=2))
    pd.testing.assert_frame_equal(T.split_half_missing(c, 40), R.split_half(c, 40))


def test_split_half_missing_handles_sparse_cells():
    c = bc.build_counts(_long(seed=3))
    c.K[1, 0], c.N[1, 0] = 1, 1                          # a cell with one trial: cannot be halved
    c.K[2, 1], c.N[2, 1] = 0, 0                          # an uncovered cell
    with pytest.raises(ValueError):
        R.split_half(c, 5)
    sh = T.split_half_missing(c, 30)
    assert len(sh) == 30 and sh["spearman"].between(-1, 1).all()


def test_variance_n2_hand_example():
    long = pd.DataFrame({
        "submission": ["a"] * 9, "task_id": ["t1"] * 4 + ["t2"] * 4 + ["t3"],
        "trial": [1, 2, 3, 4, 1, 2, 3, 4, 1], "success": [1, 1, 0, 0, 1, 1, 1, 1, 1], "no_reward": 0})
    c = bc.build_counts(long)
    v = T.variance_n2(c).iloc[0]
    # cells with N >= 2: t1 (p = 0.5, N = 4): 0.25 * 4/3; t2: 0; t3 (N = 1) is left out
    assert v["v_n2"] == pytest.approx((0.25 * 4 / 3 + 0.0) / 2)
    assert v["mean_success"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert v["f_n2"] == pytest.approx(v["v_n2"] / (v["mean_success"] * (1 - v["mean_success"])))


def test_analyse_missing_reproduces_existing_analysis_on_complete_data():
    long = _long(seed=4)
    subs = sorted(long["submission"].unique())
    tasks = sorted(long["task_id"].unique())
    mine = T.analyse_missing(long, subs, tasks, n_boot=300, n_splits=25)
    summary, *_ = R.analyse(long, n_boot=300, n_splits=25)
    assert mine["run_n_adjacent_flip_gt_5pct"] == summary["run"]["n_adjacent_flip_gt_5pct"]
    assert mine["task_n_adjacent_flip_gt_5pct"] == summary["task"]["n_adjacent_flip_gt_5pct"]
    assert mine["run_separable_positions_exact"] == summary["run"]["separable_positions_exact"]
    assert mine["split_half_spearman_mean"] == pytest.approx(summary["split_half"]["spearman_mean"])
    assert mine["split_half_spearman_sb_mean"] == pytest.approx(summary["split_half"]["spearman_sb_mean"])


def test_run_sensitivity_drops_missing_trials_and_keeps_the_analysis_set():
    long = _long(seed=5, drop_frac=0.25)
    res = T.run_sensitivity(long, n_boot=200, n_splits=10)
    assert res["n_no_reward_trials"] == int(long["no_reward"].sum()) > 0
    assert res["baseline"]["trials"] == len(long)
    assert res["missing"]["trials"] == len(long) - res["n_no_reward_trials"]
    assert res["baseline"]["n_submissions"] == res["missing"]["n_submissions"] == 6
    assert res["strict_rule_submissions"] < 6                 # the 5-trial rule fails after dropping
    assert res["missing"]["cells_below_5"] > 0 and res["baseline"]["cells_below_5"] == 0
    assert -1 <= res["spearman_rank_baseline_vs_missing"] <= 1
