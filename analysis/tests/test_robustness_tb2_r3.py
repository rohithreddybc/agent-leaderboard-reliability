"""Tests for robustness_tb2_r3.py (round 3 robustness analyses). Tiny synthetic inputs only."""
import json

import numpy as np
import pandas as pd
import pytest

import bootstrap_core as bc
import rank_inversion as R
import robustness_tb2_r3 as T


def _long(n_subs=6, n_tasks=10, trials=5, seed=0, commit="abc"):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, n_tasks)
    rows = []
    for s in range(n_subs):
        skill = 0.05 * s
        for t in range(n_tasks):
            for i in range(trials):
                succ = int(rng.random() < min(0.95, max(0.05, base[t] + skill - 0.15)))
                rows.append((f"s{s}", f"t{t}", i + 1, succ, f"run{i}", commit, f"terminal-bench/t{t}"))
    df = pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success", "run", "task_commit", "task_raw"])
    df["agent"], df["model"], df["no_reward"] = df["submission"], "m", 0
    return df


# ---------------------------------------------------------------- section 1

def test_is_blank_handles_nan_none_and_empty():
    assert T.is_blank([None, np.nan, "", "  ", "abc"]).tolist() == [True, True, True, True, False]


def test_blank_commit_table_and_summary():
    df = _long(n_subs=4, n_tasks=3, trials=2)
    df["task_commit"] = df["task_commit"].astype(object)
    df.loc[df["submission"].isin(["s0", "s1"]), "task_commit"] = None           # two fully blank
    df.loc[(df["submission"] == "s2") & (df["task_id"] == "t0"), "task_commit"] = ""   # 2 blank trials
    tab = T.blank_commit_by_submission(df, analysed=["s0", "s1", "s2"])
    s = T.blank_commit_summary(tab, len(df))
    assert s["n_trials_total"] == 24 and s["n_trials_blank"] == 6 + 6 + 2
    assert s["n_submissions_all_blank"] == 2 and s["n_submissions_partially_blank"] == 1
    assert s["partially_blank"] == {"s2": 2}
    assert s["n_submissions_all_blank_in_analysis_set"] == 2
    assert s["share_trials_blank"] == pytest.approx(14 / 24)
    assert set(s["all_blank_submissions"]) == {"s0", "s1"}


def test_task_id_check():
    df = _long(n_subs=3, n_tasks=4, trials=2)
    df.loc[df["submission"] == "s2", "task_id"] = df.loc[df["submission"] == "s2", "task_id"].replace("t3", "tX")
    res = T.task_id_check(df, ["s2"], tasks=["t0", "t1", "t2", "t3"])
    row = res["table"].iloc[0]
    assert row["n_tasks"] == 4 and row["n_in_universe"] == 3 and row["n_not_in_universe"] == 1
    assert row["n_universe_missing"] == 1 and row["n_in_commit_bearing_ids"] == 3
    assert res["n_ref_tasks"] == 4 and res["ref_tasks_equal_universe"]


def _write_trial(d, task, trial, src, commit, ref, checksum):
    p = d / f"{task}__{trial}"
    p.mkdir(parents=True)
    (p / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{task}", "trial_name": f"{task}__{trial}", "source": src,
        "task_checksum": checksum, "task_id": {"org": "terminal-bench", "name": task, "ref": ref, "git_commit_id": commit},
        "config": {"task": {"source": src, "ref": ref, "git_commit_id": commit}},
        "verifier_result": {"rewards": {"reward": 1.0}}}), encoding="utf-8")


def test_scan_raw_and_metadata_checks(tmp_path):
    for sub, src, commit in (("A", "terminal-bench/terminal-bench-2", "c1"), ("B", "terminal-bench/terminal-bench-2-1", None)):
        run = tmp_path / sub / "job1"
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"job_name": "job1", "datasets": [
            {"name": src, "version": None, "ref": "sha256:" + sub}]}), encoding="utf-8")
        for task in ("t1", "t2"):
            chk = "same" if (task == "t1" or sub == "A") else "different"
            _write_trial(run, task, "x1", src, commit, "r-" + task, chk)
    trials, runs = T.scan_raw(tmp_path)
    assert len(trials) == 4 and len(runs) == 2
    meta = T.metadata_by_submission(trials, runs, {"B": 2})
    b = meta[meta["submission"] == "B"].iloc[0]
    assert b["dataset_names"] == "terminal-bench/terminal-bench-2-1" and b["blank_commit_trials"] == 2
    assert b["trial_sources"] == "terminal-bench/terminal-bench-2-1"
    chk = T.metadata_task_check(trials, ["B"]).set_index("task_id")
    assert chk.loc["t1", "checksums_blank_subset_of_other"] == 1
    assert chk.loc["t2", "checksums_blank_subset_of_other"] == 0 and chk.loc["t2", "checksums_disjoint"] == 1
    assert chk.loc["t1", "refs_blank_subset_of_other"] == 1


# ---------------------------------------------------------------- shared statistics

def test_seeded_sets_and_restores_seed():
    old = bc.SEED
    with T.seeded(123):
        assert bc.SEED == 123
        assert bc.seed_for(1).entropy == [123, 1]
    assert bc.SEED == old


def test_run_share_values():
    assert T.run_share(0.5, 1) == pytest.approx(0.5)
    assert T.run_share(0.369, 5) == pytest.approx((0.369 / 5) / (0.631 + 0.369 / 5))
    assert T.run_share(0.0, 5) == 0.0 and T.run_share(1.0, 5) == pytest.approx(1.0)
    assert T.run_share(0.4, 10) < T.run_share(0.4, 2)


def test_split_half_windows_all_equals_original():
    c = bc.build_counts(_long(n_subs=7, n_tasks=9, trials=6))
    means = bc.macro_mean(c.K, c.N)
    order = bc.observed_order(c.subs, means)
    new = T.split_half_windows(c, order, 25, windows=(3,))
    old = R.split_half(c, 25)
    assert np.allclose(new["spearman_all"], old["spearman"]) and np.allclose(new["kendall_all"], old["kendall"])
    assert {"spearman_top3", "kendall_top3"} <= set(new.columns)
    assert new["spearman_top3"].dropna().between(-1, 1).all()


def test_core_threshold_counts_and_window_columns():
    long = _long(n_subs=8, n_tasks=12, trials=5)
    subs = sorted(long["submission"].unique())
    tasks = sorted(long["task_id"].unique())
    res = T.core(long, subs, tasks, n_boot=400, n_splits=20, windows=(4,))
    s = res["stats"]
    assert s["n_adjacent_pairs"] == 7
    assert s["adj_run_gt_2_5pct"] >= s["adj_run_gt_5pct"] and s["adj_task_gt_2_5pct"] >= s["adj_task_gt_5pct"]
    assert 0 <= s["adj_run_gt_5pct"] <= 7
    assert "split_half_spearman_top4_mean" in s and "rank_width_median_run_top4" in s
    adj = T.adjacent_table(res)
    assert len(adj) == 7 and (adj["gap_pp"] >= -1e-9).all()
    assert int((adj["p_flip_run"] > 0.05).sum()) == s["adj_run_gt_5pct"]
    # same seed, same numbers
    assert T.core(long, subs, tasks, n_boot=400, n_splits=20, windows=(4,))["stats"] == s


def test_core_matches_rank_inversion_on_same_seed():
    long = _long(n_subs=8, n_tasks=12, trials=5)
    summary, _, adj, _, _ = R.analyse(long, n_boot=300, n_splits=10)
    s = T.core(long, sorted(long["submission"].unique()), sorted(long["task_id"].unique()), 300, 10)["stats"]
    assert s["adj_run_gt_5pct"] == summary["run"]["n_adjacent_flip_gt_5pct"]
    assert s["adj_task_gt_5pct"] == summary["task"]["n_adjacent_flip_gt_5pct"]
    assert s["split_half_spearman_all_mean"] == pytest.approx(summary["split_half"]["spearman_mean"])


# ---------------------------------------------------------------- section 2

def test_overdispersion_ratio_known_cases():
    Tn = 10
    X = np.vstack([np.ones(Tn), np.zeros(Tn)])        # runs fully opposed: V = 0.5, p = 0.5, pred = 0.5 / T
    V, pred, ratio = T.overdispersion_ratio(X)
    assert V == pytest.approx(0.5) and pred == pytest.approx(0.5 / Tn) and ratio == pytest.approx(Tn)
    V, pred, ratio = T.overdispersion_ratio(np.vstack([np.ones(Tn), np.ones(Tn)]))   # identical runs
    assert V == 0 and pred == 0 and np.isnan(ratio)
    X = np.array([[1.0, 0, 1, 0], [1.0, 0, 0, 1]])    # same mean, tasks 3 and 4 disagree
    V, pred, ratio = T.overdispersion_ratio(X)
    assert V == pytest.approx(0.0) and ratio == pytest.approx(0.0)


def test_complete_run_matrix_ignores_incomplete_directories():
    tasks = ["a", "b", "c"]
    rows = [("r1", t, 1) for t in tasks] + [("r2", t, 0) for t in tasks] + [("r3", "a", 1), ("r3", "a", 0)]
    d = pd.DataFrame(rows, columns=["run", "task_id", "success"])
    X, names, info = T.complete_run_matrix(d, tasks)
    assert names == ["r1", "r2"] and X.shape == (2, 3)
    assert info["n_run_dirs"] == 3 and info["n_complete_runs"] == 2 and info["trials_used"] == 6


def test_overdispersion_classifies_submissions_and_detects_shared_run_effect():
    rng = np.random.default_rng(3)
    tasks = [f"t{i}" for i in range(60)]
    p = rng.uniform(0.2, 0.8, 60)
    rows = []
    for r in range(5):                                   # "indep": five runs, independent
        for i, t in enumerate(tasks):
            rows.append(("indep", f"run{r}", t, int(rng.random() < p[i])))
    for r in range(5):                                   # "shared": a run-level shift of +-0.25 on every task
        shift = [0.25, -0.25, 0.25, -0.25, 0.0][r]
        for i, t in enumerate(tasks):
            rows.append(("shared", f"run{r}", t, int(rng.random() < np.clip(p[i] + shift, 0, 1))))
    for r in range(3):                                   # "single": one directory with 3 trials per task
        for i, t in enumerate(tasks):
            rows.append(("single", "job", t, int(rng.random() < p[i])))
    for r in range(2):                                   # "frozen": identical runs
        for i, t in enumerate(tasks):
            rows.append(("frozen", f"run{r}", t, i % 2))
    long = pd.DataFrame(rows, columns=["submission", "run", "task_id", "success"])
    res = T.overdispersion(long, ["indep", "shared", "single", "frozen"], tasks, n_sim=300, seed=1)
    tab = res["table"].set_index("submission")
    assert tab.loc["single", "status"] == "cannot be checked" and "one run directory" in tab.loc["single", "reason"]
    assert tab.loc["frozen", "status"] == "undefined"
    assert tab.loc["shared", "ratio"] > 2 * tab.loc["indep", "ratio"]
    assert tab.loc["shared", "p_null_ratio_ge_observed"] < 0.05
    assert res["n_checked"] == 2 and res["n_cannot"] == 1 and res["n_undefined"] == 1
    assert res["ratio_median"] == pytest.approx(tab.loc[["indep", "shared"], "ratio"].median())
    # deterministic for a fixed seed
    assert T.overdispersion(long, ["indep", "shared"], tasks, n_sim=300, seed=1)["null_median_of_medians"] == \
        T.overdispersion(long, ["indep", "shared"], tasks, n_sim=300, seed=1)["null_median_of_medians"]


# ---------------------------------------------------------------- section 5

def test_sd_of_mean():
    assert T.sd_of_mean(np.array([0.09]), 100, 1)[0] == pytest.approx(0.03)
    assert T.sd_of_mean(np.array([0.09]), 100, 5)[0] == pytest.approx(0.03 / np.sqrt(5))


# ---------------------------------------------------------------- section 6

def test_anova_additive_data_recovers_exactly():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=7), rng.normal(size=11)
    X = (a[:, None] + b[None, :])[:, :, None] * np.ones((1, 1, 5))
    r = T.anova_3way(X)
    assert r["comp"]["s"] == pytest.approx(a.var(ddof=1)) and r["comp"]["t"] == pytest.approx(b.var(ddof=1))
    for k in ("r", "st", "sr", "tr", "str"):
        assert r["comp"][k] == pytest.approx(0, abs=1e-12)
    assert r["share"]["s"] + r["share"]["t"] == pytest.approx(1.0)


def test_anova_sums_of_squares_add_up():
    X = np.random.default_rng(1).normal(size=(6, 9, 5))
    r = T.anova_3way(X)
    assert sum(r["ss"].values()) == pytest.approx(r["ss_total_check"])
    assert sum(r["df"].values()) == 6 * 9 * 5 - 1


def test_anova_recovers_known_components_and_two_way_agrees():
    rng = np.random.default_rng(2)
    S, Tn, Rn = 60, 80, 5
    sd = {"s": 1.0, "t": 0.8, "st": 0.6, "e": 0.5}
    X = (rng.normal(0, sd["s"], (S, 1, 1)) + rng.normal(0, sd["t"], (1, Tn, 1))
         + rng.normal(0, sd["st"], (S, Tn, 1)) + rng.normal(0, sd["e"], (S, Tn, Rn)))
    r3, r2 = T.anova_3way(X), T.anova_2way_replicated(X)
    assert r3["comp"]["s"] == pytest.approx(sd["s"] ** 2, rel=0.35)
    assert r3["comp"]["t"] == pytest.approx(sd["t"] ** 2, rel=0.25)
    assert r3["comp"]["st"] == pytest.approx(sd["st"] ** 2, rel=0.1)
    assert r3["comp"]["str"] == pytest.approx(sd["e"] ** 2, rel=0.05)
    assert r3["comp"]["r"] < 0.01 and r3["comp"]["sr"] < 0.01 and r3["comp"]["tr"] < 0.01
    assert r2["comp"]["s"] == pytest.approx(r3["comp"]["s"], rel=0.02)
    assert r2["comp"]["st"] == pytest.approx(r3["comp"]["st"], rel=0.05)
    assert r2["comp"]["e"] == pytest.approx(r3["comp"]["str"] + r3["comp"]["sr"] * 0, rel=0.05)


def test_g_coefficients_formulas():
    comp = {"s": 1.0, "t": 0.5, "r": 0.1, "st": 0.4, "sr": 0.2, "tr": 0.05, "str": 0.6}
    assert T.g_relative(comp, 10, 1) == pytest.approx(1.0 / (1.0 + 0.04 + 0.2 + 0.06))
    assert T.g_relative(comp, 10, 5) == pytest.approx(1.0 / (1.0 + 0.04 + 0.04 + 0.012))
    assert T.g_absolute(comp, 10, 1) == pytest.approx(1.0 / (1.0 + 0.05 + 0.1 + 0.04 + 0.2 + 0.005 + 0.06))
    assert T.g_relative(comp, 10, 5) > T.g_relative(comp, 10, 1)
    assert T.g_relative(comp, 1000, 1000) > 0.99
    two = {"s": 1.0, "t": 0.5, "st": 0.4, "e": 0.6}
    assert T.g_relative_2way(two, 10, 5) == pytest.approx(1.0 / (1.0 + 0.04 + 0.012))
    assert T.g_relative({"s": 0.0, "st": 0.0, "sr": 0.0, "str": 0.0}, 5, 5) != T.g_relative({"s": 0.0, "st": 0.0, "sr": 0.0, "str": 0.0}, 5, 5)  # NaN


def test_balanced_array_drops_unbalanced_and_lists_reason():
    long = _long(n_subs=4, n_tasks=6, trials=5)
    extra = long[(long["submission"] == "s1") & (long["task_id"] == "t0")].head(1).assign(trial=6)
    long = pd.concat([long, extra], ignore_index=True)                              # s1: one cell with 6 trials
    long = long[~((long["submission"] == "s2") & (long["task_id"] == "t1") & (long["trial"] > 1))]   # s2: one cell with 1 trial
    subs = ["s0", "s1", "s2", "s3"]
    X, kept, dropped = T.balanced_array(long, subs, sorted(long["task_id"].unique()))
    assert kept == ["s0", "s3"] and X.shape == (2, 6, 5)
    assert set(dropped["submission"]) == {"s1", "s2"}
    assert "6 trials: 1" in dropped.set_index("submission").loc["s1", "reason"]
    s0 = long[long["submission"] == "s0"].sort_values(["task_id", "trial"])["success"].to_numpy().reshape(6, 5)
    assert np.array_equal(X[0], s0)


def test_gstudy_end_to_end_small():
    long = _long(n_subs=8, n_tasks=15, trials=5)
    g = T.gstudy(long, sorted(long["submission"].unique()), sorted(long["task_id"].unique()), "test", n_t_design=89)
    assert g["n_s"] == 8 and g["n_t"] == 15 and g["n_r"] == 5 and g["dropped"].empty
    assert 0 <= g["coef"]["Ep2_3way_nt89_nr1"] <= g["coef"]["Ep2_3way_nt89_nr5"] <= 1
    assert g["coef"]["Phi_3way_nt89_nr5"] <= g["coef"]["Ep2_3way_nt89_nr5"] + 1e-12
    assert g["table"].loc[g["table"]["model"] == "3-way", "share_of_total"].sum() == pytest.approx(1.0)


def test_canonical_checksum_subs_groups_identical_vectors():
    rows = []
    for s, shift in (("a", 0), ("b", 0), ("c", 0), ("d", 1)):
        for t in ("t1", "t2"):
            rows.append((s, t, f"{t}-{shift}"))
    trials = pd.DataFrame(rows, columns=["submission", "task_id", "task_checksum"])
    subs, n_vec = T.canonical_checksum_subs(trials)
    assert subs == ["a", "b", "c"] and n_vec == 2


def test_sens_table_shares():
    st = {"n_submissions": 5, "n_adjacent_pairs": 4, "run_sd_median_pp": 1.234, "task_sd_median_pp": 3.9,
          "var_ratio_run_over_task_median": 0.1, "f_median": 0.37, "run_share_k5_at_median_f": 0.1,
          "adj_run_gt_5pct": 3, "adj_task_gt_5pct": 4, "split_half_spearman_all_mean": 0.98,
          "split_half_kendall_all_mean": 0.9}
    tab = T.sens_table({"x": st})
    row = tab[tab["statistic"].str.startswith("share of adjacent pairs") & tab["statistic"].str.contains("run")].iloc[0]
    assert row["x"] == 0.75
    assert tab.loc[tab["statistic"].str.startswith("median SD of a submission mean over runs"), "x"].iloc[0] == 1.23


def test_md_table_keeps_integers():
    df = pd.DataFrame({"k": [1, 2], "x": [0.5, 0.25]})
    lines = T.md_table(df)
    assert lines[2] == "| 1 | 0.5000 |" and lines[3] == "| 2 | 0.2500 |"


def test_trial_index_overdispersion_runs():
    long = _long(n_subs=3, n_tasks=8, trials=5)
    out = T.trial_index_overdispersion(long, sorted(long["submission"].unique()), sorted(long["task_id"].unique()))
    assert len(out) == 3 and (out["grouping"] == "run directory").all()
    long["run"] = "one"
    out1 = T.trial_index_overdispersion(long, sorted(long["submission"].unique()), sorted(long["task_id"].unique()))
    assert (out1["grouping"] == "trial index in one directory").all()
