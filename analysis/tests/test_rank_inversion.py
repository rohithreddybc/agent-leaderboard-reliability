import json

import numpy as np
import pandas as pd
import pytest

import bootstrap_core as bc
import rank_inversion as R
from test_bootstrap_core import long_from


def test_adjacent_shares_and_separable():
    P = np.full((4, 4), np.nan)
    P[0, 1], P[1, 2], P[2, 3] = 0.30, 0.10, 0.01
    P[0, 2], P[0, 3], P[1, 3] = 0.001, 0.0, 0.0
    s = R.adjacent_shares(P)
    assert s["n_adjacent_pairs"] == 3
    assert s["share_adjacent_flip_gt_5pct"] == pytest.approx(2 / 3)
    assert s["share_adjacent_flip_gt_20pct"] == pytest.approx(1 / 3)
    exact, greedy = R.max_separable(P)
    assert exact == 3 and greedy == 3   # {0, 2, 3} pairwise separable


def test_max_separable_exact_beats_greedy():
    # nodes 0..3; 0 conflicts with 1 and 2, and {1,2,3} are mutually separable
    P = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(i + 1, 4):
            P[i, j] = 0.0
    P[0, 1] = P[0, 2] = 0.5
    exact, greedy = R.max_separable(P)
    assert exact == 3 and greedy == 2


def test_min_gap():
    pairs = pd.DataFrame({"gap_pp": [0.5, 1.5, 2.5, 4.5, 6.5], "p_flip_run": [0.4, 0.2, 0.06, 0.04, 0.0]})
    g, bins = R.min_gap(pairs, "p_flip_run")
    assert g["gap_max_unstable_pp"] == 2.5 and g["gap_min_stable_pp"] == 4.5 and g["gap_bin_stable_pp"] == 4.0
    assert set(bins["scheme"]) == {"p_flip_run"}


def test_split_half_noise_free_is_perfect():
    spec = {f"s{i}": {f"t{t}": [1 if t < 2 * i + 1 else 0] * 5 for t in range(12)} for i in range(5)}
    c = bc.build_counts(long_from(spec))
    sh = R.split_half(c, 20)
    assert np.allclose(sh.spearman, 1) and np.allclose(sh.kendall, 1)
    assert bc.seed_for(bc.SCHEME_SPLIT) is not None


def test_spearman_brown_value():
    sh = pd.DataFrame({"spearman": [0.5], "kendall": [0.3], "spearman_sb": [2.5 * 0.5 / (1 + 1.5 * 0.5)], "kendall_sb": [0.0]})
    out = R.summarize_split(sh, 2.5)
    assert out["spearman_sb_mean"] == pytest.approx(1.25 / 1.75) and out["spearman_brown_m"] == 2.5


def synthetic_long(S=6, T=30, k=5, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, T)
    spec = {}
    for s in range(S):
        p = np.clip(base + 0.03 * s - 0.08, 0.02, 0.98)
        spec[f"s{s}"] = {f"t{t:02d}": list((rng.random(k) < p[t]).astype(int)) for t in range(T)}
    df = long_from(spec)
    return df.assign(agent=df.submission, model="m")


def test_analyse_end_to_end_and_main_in_tmp(tmp_path, monkeypatch):
    long = synthetic_long()
    summary, pairs, adj, bins, sh = R.analyse(long, n_boot=300, n_splits=20)
    assert summary["n_submissions"] == 6 and summary["n_pairs"] == 15 and len(adj) == 5
    for key in ("run", "task"):
        s = summary[key]
        assert 0 <= s["share_adjacent_flip_gt_20pct"] <= s["share_adjacent_flip_gt_5pct"] <= 1
        assert 1 <= s["separable_positions_greedy"] <= s["separable_positions_exact"] <= 6
    assert pairs["gap_pp"].min() >= 0
    summary.pop("_rank_tables")
    long.to_csv(tmp_path / "tb2_long.csv", index=False)
    monkeypatch.setattr("sys.argv", ["rank_inversion.py", "--results", str(tmp_path), "--n-boot", "200", "--n-splits", "10"])
    R.main()
    assert (tmp_path / "rank_inversion_summary.json").exists() and (tmp_path / "split_half.csv").exists()
    assert json.loads((tmp_path / "rank_inversion_summary.json").read_text())["seed"] == 20261008


def test_bootstrap_script_outputs(tmp_path, monkeypatch):
    import tb2_bootstrap as TB
    long = synthetic_long()
    long.to_csv(tmp_path / "tb2_long.csv", index=False)
    monkeypatch.setattr("sys.argv", ["tb2_bootstrap.py", "--results", str(tmp_path), "--n-boot", "200"])
    TB.main()
    b = pd.read_csv(tmp_path / "tb2_bootstrap.csv")
    assert ((b.run_ci_low <= b.mean_success) & (b.mean_success <= b.run_ci_high)).all()
    w = pd.read_csv(tmp_path / "tb2_within_task_variance.csv")
    assert (w.mean_p1mp_unbiased >= w.mean_p1mp_plugin - 1e-12).all()
    assert np.allclose(w.share_tasks_mixed + w.share_tasks_all_success + w.share_tasks_all_fail, 1)


def test_bootstrap_script_run_sd_matches_analytic_and_f_column(tmp_path, monkeypatch):
    import tb2_bootstrap as TB
    long = synthetic_long(S=5, T=40, k=5, seed=11)
    long.to_csv(tmp_path / "tb2_long.csv", index=False)
    monkeypatch.setattr("sys.argv", ["tb2_bootstrap.py", "--results", str(tmp_path), "--n-boot", "4000"])
    TB.main()
    b = pd.read_csv(tmp_path / "tb2_bootstrap.csv")
    assert (b.run_sd / b.run_sd_analytic).between(0.93, 1.07).all()
    w = pd.read_csv(tmp_path / "tb2_within_task_variance.csv").set_index("submission")
    m = w.mean_success
    assert np.allclose(w.f_run_noise_share, w.mean_p1mp_unbiased / (m * (1 - m)))
    # the unbiased within-task variance cannot exceed the single-run outcome variance by much
    assert (w.f_run_noise_share < 1.05).all()


def test_f_is_nan_at_boundary_and_delta_min_worked_numbers():
    import make_results_md as M
    import tb2_bootstrap as TB
    import bootstrap_core as bc
    long = pd.DataFrame([("z", "t1", i, 0) for i in range(5)] + [("z", "t2", i, 0) for i in range(5)],
                        columns=["submission", "task_id", "trial", "success"])
    w = TB.within_task_summary(bc.build_counts(long))
    assert np.isnan(w.f_run_noise_share[0])
    # worked numbers of the paper: v = 0.0753, n = 100, k = 1 -> 7.6 pp; n = 89, k = 5 -> 3.6 pp
    assert M.delta_min(0.0753, 100, 1) == pytest.approx(7.606, abs=1e-3)
    assert M.delta_min(0.0753, 89, 5) == pytest.approx(3.606, abs=1e-3)
    assert M.delta_min(0.0753, 89, 5, M.Z_ONE) == pytest.approx(M.delta_min(0.0753, 89, 5) * 1.645 / 1.96)
