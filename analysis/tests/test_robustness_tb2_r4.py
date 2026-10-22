"""Tests for robustness_tb2_r4.py (whole-run resampling). Tiny synthetic inputs only."""
import numpy as np
import pandas as pd
import pytest

import bootstrap_core as bc
import robustness_tb2_r3 as T3
import robustness_tb2_r4 as T4


def _long(n_subs=4, n_tasks=12, n_runs=5, seed=0, shared_sd=0.0):
    """Each submission: n_runs run directories with exactly one trial per task. ``shared_sd`` adds a
    run-level shift to the success probability (overdispersed runs)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, n_tasks)
    rows = []
    for s in range(n_subs):
        for r in range(n_runs):
            shift = rng.normal(0, shared_sd)
            for t in range(n_tasks):
                p = np.clip(base[t] + 0.06 * s + shift, 0.02, 0.98)
                rows.append((f"s{s}", f"t{t}", r + 1, int(rng.random() < p), f"run{r}"))
    df = pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success", "run"])
    return df


def _tasks(n):
    return [f"t{i}" for i in range(n)]


def test_identifiable_runs_requires_complete_runs_and_variance():
    df = _long(n_subs=3)
    # s1: a single run directory holding every trial -> not identifiable
    df.loc[df["submission"] == "s1", "run"] = "all"
    # s2: constant outcomes -> within-task variance 0
    df.loc[df["submission"] == "s2", "success"] = 1
    out = T4.identifiable_runs(df, ["s0", "s1", "s2"], _tasks(12))
    assert list(out) == ["s0"]
    assert out["s0"].shape == (5, 12)


def test_whole_run_resample_matches_analytic_sd_with_rescaling():
    rng = np.random.default_rng(1)
    X = (rng.random((5, 40)) < 0.5).astype(float)
    x = T4.whole_run_resample(X, 200_000, np.random.default_rng(2))
    assert x.std(ddof=1) == pytest.approx(T4.analytic_whole_run_sd(X), rel=0.02)
    assert x.mean() == pytest.approx(X.mean(), abs=2e-3)
    # without the rescaling the SD is smaller by sqrt((R-1)/R)
    x0 = T4.whole_run_resample(X, 200_000, np.random.default_rng(2), correct=False)
    assert x0.std(ddof=1) == pytest.approx(T4.analytic_whole_run_sd(X) * np.sqrt(4 / 5), rel=0.02)


def test_squared_analytic_ratio_equals_overdispersion_ratio():
    rng = np.random.default_rng(3)
    X = (rng.random((6, 30)) < rng.uniform(0.2, 0.8, 30)).astype(float)
    V, pred, ratio = T3.overdispersion_ratio(X)
    a = T4.analytic_whole_run_sd(X) / T4.analytic_within_task_sd(X)
    assert a ** 2 == pytest.approx(ratio, rel=1e-9)
    # and the within-task analytic SD equals bootstrap_core's formula on the same runs
    c = T4.counts_from_runs("x", X, _tasks(30))
    assert bc.analytic_run_sd(c)[0] == pytest.approx(T4.analytic_within_task_sd(X), rel=1e-12)


def test_overdispersed_runs_give_ratio_above_one_independent_runs_near_one():
    tasks = _tasks(60)
    ind = _long(n_subs=6, n_tasks=60, n_runs=8, seed=5, shared_sd=0.0)
    od = _long(n_subs=6, n_tasks=60, n_runs=8, seed=5, shared_sd=0.15)
    r_ind = T4.compare_schemes(ind, [f"s{i}" for i in range(6)], tasks, 3000)["table"]["analytic_ratio"].median()
    r_od = T4.compare_schemes(od, [f"s{i}" for i in range(6)], tasks, 3000)["table"]["analytic_ratio"].median()
    assert r_ind < 1.3
    assert r_od > 1.6 and r_od > r_ind


def test_compare_schemes_shapes_summary_and_determinism():
    df = _long(n_subs=5, n_tasks=15, n_runs=5, seed=7)
    subs = [f"s{i}" for i in range(5)]
    a = T4.compare_schemes(df, subs, _tasks(15), 1000)
    b = T4.compare_schemes(df, subs, _tasks(15), 1000)
    pd.testing.assert_frame_equal(a["table"], b["table"])
    pd.testing.assert_frame_equal(a["adjacent"], b["adjacent"])
    assert len(a["table"]) == 5 and len(a["adjacent"]) == 4
    s = T4.summarize(a)
    assert s["n_adjacent_pairs"] == 4 and 0 <= s["flip_gt_whole_run"] <= 4 and 0 <= s["n_ratio_gt_1"] <= 5
    # the adjacent table is ordered by the mean over all trials, best first
    assert (a["adjacent"]["gap_pp"] >= -1e-9).all()
    p = a["adjacent"][["p_flip_whole_run", "p_flip_within_task"]].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_flip_probability_extremes():
    # two submissions, large gap, small noise -> flip probability 0; identical runs-level means -> about 0.5
    rng = np.random.default_rng(9)
    tasks = _tasks(50)
    rows = []
    for s, p in (("a", 0.9), ("b", 0.2)):
        for r in range(5):
            for t in range(50):
                rows.append((s, f"t{t}", r + 1, int(rng.random() < p), f"run{r}"))
    df = pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success", "run"])
    res = T4.compare_schemes(df, ["a", "b"], tasks, 2000)
    assert res["adjacent"]["p_flip_whole_run"].iloc[0] == 0.0
    assert res["adjacent"]["p_flip_within_task"].iloc[0] == 0.0
    assert res["adjacent"]["submission_hi"].iloc[0] == "a"


def test_report_and_main_write_only_to_out(tmp_path, monkeypatch):
    df = _long(n_subs=3, n_tasks=89, n_runs=5, seed=11)
    # make a tiny results dir: tb2_long.csv with 89 tasks so select_analysis_set accepts it (5 trials per task)
    df["task_id"] = df["task_id"].str.replace("t", "task", regex=False)
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    df.to_csv(res_dir / "tb2_long.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--results", str(res_dir), "--out", str(out), "--n-boot", "500"])
    T4.main()
    names = sorted(p.name for p in out.iterdir())
    assert "robustness_2026-10-20_tb2.md" in names
    assert all(n.startswith("robustness_2026-10-20_tb2") for n in names)
    md = (out / "robustness_2026-10-20_tb2.md").read_text(encoding="utf-8")
    assert "sqrt(R/(R-1))" in md and "Pairs with flip probability > 5%" in md
