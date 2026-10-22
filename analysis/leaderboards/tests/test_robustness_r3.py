"""Tests for robustness_r3.py. Core functions on tiny synthetic inputs; one integration test reproduces the
committed headline (reads results/ only; conftest.py checks that nothing there changes). Files go to pytest temp dirs."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import census as C
import robustness_r3 as R3


def make_board(scores, n=100, rho=0.5, ks=None, lb="synthetic"):
    scores = np.array(sorted(scores, reverse=True), float)
    names = [f"e{i:02d}" for i in range(len(scores))]
    return R3.Board(lb, n, names, scores, np.ones(len(scores)) if ks is None else np.array(ks, float), rho, "own")


# ---- evaluator and tiers ---------------------------------------------------------------------------

def test_evaluate_matches_census_config():
    rng = np.random.default_rng(1)
    s = np.sort(rng.uniform(0.1, 0.9, 12))[::-1]
    s[3] = s[2]                                   # a tie
    names = [f"e{i:02d}" for i in range(12)]
    ks = np.ones(12)
    for rho in (0.0, 0.5):
        ev = R3.evaluate(s, ks, R3.fn_a(120, rho))
        st = C.census_config(list(s), names, list(ks), lambda p1, p2, k1, k2: C.bound_a(p1, p2, 120, rho))
        assert ev["n_tiers"] == st["n_tiers"] and ev["n_adjacent_below"] == st["n_adjacent_below"]
        assert ev["top1_vs_2_separable"] == st["top1_vs_2_separable"] and ev["top1_vs_5_separable"] == st["top1_vs_5_separable"]
        assert ev["share_all_below"] == pytest.approx(st["share_all_below"])


def test_greedy_tiers_batch_equals_scalar_rule():
    rng = np.random.default_rng(2)
    S = -np.sort(-rng.uniform(0.0, 1.0, (40, 15)), axis=1)
    S[:, 4] = S[:, 3]                              # ties
    fn = lambda p1, p2: C.bound_a(p1, p2, 80, 0.4)
    got = R3.greedy_tiers_batch(S, fn)
    want = [C.greedy_tiers(row, lambda i, j, row=row: float(fn(row[i], row[j]))) for row in S]
    assert list(got) == want


def test_tiers_known_example():
    # n = 100, rho = 0: SE_D at P = 0.5 is 0.0707, bound 0.1386. Gaps from the head: 0.1, 0.2 -> second entry joins, third opens tier 2.
    ev = R3.evaluate([0.6, 0.5, 0.4], [1, 1, 1], R3.fn_a(100, 0.0))
    assert ev["n_tiers"] == 2 and ev["top1_vs_2_separable"] is False


# ---- section 1 -------------------------------------------------------------------------------------

def test_confidence_values_and_edge_cases():
    assert float(R3.confidence(0.0, 0.05)) == pytest.approx(0.5)
    assert float(R3.confidence(1.96 * 0.05, 0.05)) == pytest.approx(stats.norm.cdf(1.96))
    assert float(R3.confidence(0.1, 0.0)) == 1.0 and float(R3.confidence(0.0, 0.0)) == 0.5
    assert R3.confidence([0.0, 0.1], [0.1, 0.1])[1] > R3.confidence([0.0, 0.1], [0.1, 0.1])[0]


def test_sec1_c_equals_phi_of_gap_over_se_and_bound_link():
    b = make_board([0.8, 0.7, 0.65, 0.6, 0.5, 0.3], n=200, rho=0.4)
    pairs, bd, summ = R3.sec1_order_confidence([b])
    r = pairs[(pairs["pair"] == "adjacent") & (pairs["rank_hi"] == 1)].iloc[0]
    sed = float(C.bound_a(0.8, 0.7, 200, 0.4)) / C.Z
    assert r["c"] == pytest.approx(stats.norm.cdf(0.1 / sed)) and r["se_d_pp"] == pytest.approx(100 * sed)
    # c >= 0.975 exactly when the pair is separable at 1.96
    ev = R3.evaluate(b.scores, b.ks, R3.fn_a(b.n, b.rho))
    adj = pairs[pairs["pair"] == "adjacent"]["c"].to_numpy()
    assert list(adj >= stats.norm.cdf(1.96) - 1e-12) == list(~ev["adj_below"])
    assert summ["n_adjacent_pairs"] == 5


# ---- section 2 -------------------------------------------------------------------------------------

def test_simulate_scores_moments():
    mu, n, rho = np.array([0.7, 0.5, 0.3]), 200, 0.6
    X = R3.simulate_scores(mu, n, rho, 60_000, np.random.default_rng(5))
    se = np.sqrt(mu * (1 - mu) / n)
    assert np.allclose(X.mean(axis=0), mu, atol=2e-3)
    assert np.allclose(X.std(axis=0), se, rtol=0.02)
    assert np.corrcoef(X.T)[0, 1] == pytest.approx(rho, abs=0.02) and np.corrcoef(X.T)[0, 2] == pytest.approx(rho, abs=0.02)
    assert X.min() >= 0 and X.max() <= 1
    with pytest.raises(ValueError):
        R3.simulate_scores(mu, n, 1.0, 5, np.random.default_rng(0))


def test_simulate_scores_degenerate_entries_are_constants():
    X = R3.simulate_scores(np.array([1.0, 0.5, 0.0]), 50, 0.5, 100, np.random.default_rng(0))
    assert (X[:, 0] == 1).all() and (X[:, 2] == 0).all()


def test_eb_shrink():
    obs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    mu, tau2 = R3.eb_shrink(obs, 100)
    se2 = obs * (1 - obs) / 100
    assert tau2 == pytest.approx(np.var(obs, ddof=1) - se2.mean())
    assert np.all(np.diff(mu) < 0) and mu[0] < obs[0] and mu[-1] > obs[-1] and mu.mean() == pytest.approx(obs.mean())
    B = tau2 / (tau2 + se2)
    assert np.allclose(mu, obs.mean() + B * (obs - obs.mean()))
    flat, t0 = R3.eb_shrink(np.array([0.5, 0.5, 0.5]), 10)
    assert t0 == 0.0 and np.allclose(flat, 0.5)
    tight, _ = R3.eb_shrink(np.array([0.9, 0.1]), 100_000)      # tiny SE: almost no shrinkage
    assert np.allclose(tight, [0.9, 0.1], atol=1e-3)


def test_bootstrap_board_is_reproducible_and_sane():
    b = make_board([0.8, 0.79, 0.6, 0.55, 0.3, 0.2], n=100, rho=0.5)
    a = R3.bootstrap_board(b, "plugin", 300, np.random.default_rng(3))
    c = R3.bootstrap_board(b, "plugin", 300, np.random.default_rng(3))
    assert {k: v for k, v in a.items() if k != 'tau2'} == {k: v for k, v in c.items() if k != 'tau2'}   # tau2 is NaN under the plug-in truth
    assert 0 <= a["p_top_correct"] <= 1 and a["p_top_correct_strict"] <= a["p_top_correct"] + 1e-12
    assert 0 <= a["exp_misordered_adjacent"] <= b.m - 1 and a["tiers_sim_q025"] <= a["tiers_sim_median"] <= a["tiers_sim_q975"]
    assert a["tiers_observed"] == R3.evaluate(b.scores, b.ks, R3.fn_a(100, 0.5))["n_tiers"]
    # a very large gap between #1 and the rest: the top is almost surely correct
    far = make_board([0.9, 0.4, 0.38, 0.3], n=400, rho=0.3)
    assert R3.bootstrap_board(far, "plugin", 400, np.random.default_rng(1))["p_top_correct"] > 0.99
    assert R3.bootstrap_board(b, "shrink", 100, np.random.default_rng(2))["tau2"] > 0


# ---- sections 3-5 ----------------------------------------------------------------------------------

def test_rho_sweep_monotone_in_rho_and_headline_row():
    boards = [make_board([0.8, 0.77, 0.7, 0.66, 0.6, 0.5, 0.2], n=150, rho=0.5, lb="a"),
              make_board([0.6, 0.58, 0.5, 0.45, 0.44, 0.3], n=90, rho=0.3, lb="b")]
    summ, per = R3.sec3_rho_sweep(boards, rhos=(0.0, 0.5, "headline", 0.9))
    assert list(summ["rho"]) == ["0", "0.5", "headline (board)", "0.9"]
    assert summ["n12_not_sep"].is_monotonic_decreasing                      # a wider-to-narrower bound separates more
    assert set(per["rho_setting"].astype(str)) == {"0.0", "0.5", "headline", "0.9"}


def test_lofo_families():
    med = {"SWE-bench Verified": 0.6, "SWE-bench Lite": 0.5, "SWE-bench test (full)": 0.54, "SWE-bench Multilingual": 0.56,
           "Terminal-Bench 1.0": 0.52, "Terminal-Bench 2.0": 0.42, "TheAgentCompany": 0.39}
    df = pd.DataFrame({"leaderboard": list(med), "rho_adjacent_median": list(med.values())})
    fam = R3.family_rhos(df)
    assert fam["SWE-bench"] == pytest.approx(np.median([0.6, 0.5, 0.54, 0.56]))
    f4, b4, pooled = R3.sec4_lofo(df)
    sw = f4.set_index("family")
    assert sw.loc["TheAgentCompany", "rho_predicted_by_others"] == pytest.approx(np.median([fam["SWE-bench"], 0.52, 0.42]))
    assert (sw["abs_error"] == (sw["rho_predicted_by_others"] - sw["rho_family"]).abs()).all()
    assert pooled == pytest.approx(np.median(list(fam.values()))) and len(b4) == 7


def test_rho_with_pooled_keeps_own():
    b1, b2 = make_board([0.5, 0.4], lb="x", rho=0.3), make_board([0.5, 0.4], lb="y", rho=0.6)
    b2.source = "pooled"
    assert R3.rho_with_pooled([b1, b2], 0.45) == {"x": 0.3, "y": 0.45}


# ---- section 6 and 7 -------------------------------------------------------------------------------

def test_kadj_bound():
    p1, p2, n, rho, f = 0.6, 0.5, 100, 0.5, 0.369
    base = float(C.bound_a(p1, p2, n, rho))
    assert float(R3.bound_a_kadj(p1, p2, 1.0, 1.0, n, rho, f)) == pytest.approx(base)           # k = 1: unchanged
    assert float(R3.bound_a_kadj(p1, p2, 1.0, 1.0, n, rho, f, literal=True)) == pytest.approx(base)
    # k = 5 on entry 1 only: variance factor 1 - f (1 - 1/k) on that entry's SE^2, covariance term unchanged
    s1, s2 = C.se(p1, n), C.se(p2, n)
    a1 = 1 - f * (1 - 1 / 5)
    want = C.Z * math.sqrt(s1 ** 2 * a1 + s2 ** 2 - 2 * rho * s1 * s2)
    assert float(R3.bound_a_kadj(p1, p2, 5.0, 1.0, n, rho, f)) == pytest.approx(want)
    lit = C.Z * math.sqrt(s1 ** 2 * a1 + s2 ** 2 - 2 * rho * s1 * math.sqrt(a1) * s2)
    assert float(R3.bound_a_kadj(p1, p2, 5.0, 1.0, n, rho, f, literal=True)) == pytest.approx(lit)
    assert want < base                                                                           # more runs: tighter
    assert float(R3.bound_a_kadj(p1, p2, 10.0, 1.0, n, rho, f)) < want


def test_sec6_no_change_when_all_single_run():
    b = make_board([0.8, 0.77, 0.7, 0.66, 0.6, 0.5], n=150, rho=0.5)
    cmp, summ = R3.sec6_kadj([b], 0.369)
    assert cmp["adj_verdicts_changed"].sum() == 0 and (cmp["tiers_headline"] == cmp["tiers_alt"]).all()


def test_adjusted_formulas():
    p, n = np.array([0.0, 0.5, 1.0]), 98
    pt, nt = R3.adjusted(p, n, "ac")
    assert nt == 100 and np.allclose(pt, [1 / 100, 50 / 100, 99 / 100])                          # (x + 1) / (n + 2)
    pw, nw = R3.adjusted(p, n, "wilson")
    z2 = C.Z ** 2
    assert nw == pytest.approx(98 + z2) and np.allclose(pw, (p * 98 + z2 / 2) / (98 + z2))
    # an entry at 0% has a positive adjusted SE, so the bound against it is no longer driven by one entry alone
    fn = R3.fn_a_adjusted(98, 0.0, "ac")
    assert float(fn(0.0, 0.0, 1, 1, 0, 1)) > 0 and float(C.bound_a(0.0, 0.0, 98, 0.0)) == 0.0
    assert R3.evaluate([0.0, 0.0], [1, 1], fn)["top1_vs_2_separable"] is False                  # a tie stays a tie


# ---- section 8 -------------------------------------------------------------------------------------

def test_cluster_se_singleton_clusters_equal_sample_se():
    rng = np.random.default_rng(0)
    y1, y2 = rng.integers(0, 2, 30).astype(float), rng.integers(0, 2, 30).astype(float)
    d = y1 - y2
    se = R3.cluster_se_diff(y1, y2, np.arange(30))
    assert se == pytest.approx(d.std(ddof=1) / math.sqrt(30))
    assert R3.paired_iid_se(y1, y2) == pytest.approx(d.std(ddof=0) / math.sqrt(30))
    assert math.isnan(R3.cluster_se_diff(y1, y2, np.zeros(30)))


def test_cluster_se_hand_value():
    y1, y2 = np.array([1, 1, 0, 0.0]), np.zeros(4)           # d = (1, 1, 0, 0), dbar = 0.5
    # clusters {t0, t1} and {t2, t3}: sums of (d - dbar) are +1 and -1; G = 2 -> sqrt(2/1 * 2) / 4 = 0.5
    assert R3.cluster_se_diff(y1, y2, ["a", "a", "b", "b"]) == pytest.approx(0.5)
    # perfectly clustered differences inflate the SE relative to the iid value
    assert R3.cluster_se_diff(y1, y2, ["a", "a", "b", "b"]) > R3.paired_iid_se(y1, y2)
    # mixing the clusters (each holds one 1 and one 0) removes the cluster effect
    assert R3.cluster_se_diff(y1, y2, ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_repo_prefix():
    assert R3.repo_prefix("django__django-12345") == "django__django"
    assert R3.repo_prefix("scikit-learn__scikit-learn-9999") == "scikit-learn__scikit-learn"
    assert R3.repo_prefix("sphinx-doc__sphinx-8") == "sphinx-doc__sphinx"


# ---- section 9 -------------------------------------------------------------------------------------

def test_assumed_rhos_clipping():
    assert R3.assumed_rhos(0.25) == [("true", 0.25), ("true-0.25", 0.0), ("true+0.25", 0.5)]
    assert R3.assumed_rhos(0.75)[2] == ("true+0.25", 0.95)


def test_calibration_cell_nominal_and_overconfident_rho():
    ok = R3.calibration_cell(100, 0.5, 0.5, 0.5, 6000, 7)
    assert abs(ok["false_separation_rate"] - 0.05) < 0.02 and abs(ok["power_at_bound"] - 0.5) < 0.04
    assert ok["mc_se_null"] == pytest.approx(math.sqrt(ok["false_separation_rate"] * (1 - ok["false_separation_rate"]) / 6000))
    over = R3.calibration_cell(100, 0.5, 0.25, 0.5, 6000, 7)               # assumed rho above the truth: too many false separations
    under = R3.calibration_cell(100, 0.5, 0.5, 0.25, 6000, 7)              # assumed rho below the truth: conservative
    assert over["false_separation_rate"] > 0.08 and under["false_separation_rate"] < 0.035
    assert R3.calibration_cell(100, 0.5, 0.5, 0.5, 500, 7) == R3.calibration_cell(100, 0.5, 0.5, 0.5, 500, 7)   # deterministic


def test_calibration_cell_infeasible_power_is_nan():
    c = R3.calibration_cell(30, 0.05, 0.75, 0.0, 200, 1)                   # delta/2 pushes the lower entry below 0
    assert math.isnan(c["power_at_bound"]) and not math.isnan(c["false_separation_rate"])


# ---- section 10 and 16 -----------------------------------------------------------------------------

def test_tau2_f_formula():
    df = pd.DataFrame({"pass_1": [60.0, 100.0, 50.0], "pass_2": [40.0, 100.0, np.nan], "domain": ["a", "a", "b"],
                       "trials_stated": [4, np.nan, np.nan]})
    f = R3.tau2_f(df)["f"]
    assert f.iloc[0] == pytest.approx((0.6 - 0.4) / (0.6 * 0.4))
    assert math.isnan(f.iloc[1]) and math.isnan(f.iloc[2])                  # pass^1 = 1; pass^2 missing


def test_feasible_coupling_bound():
    assert R3.feasible(0.5, 0.5) and not R3.feasible(0.51, 0.5)
    assert R3.feasible(0.6, 0.2, 0.5) is True and R3.feasible(0.7, 0.2, 0.5) is False      # sqrt(0.8 * 0.5) = 0.632
    assert R3.feasible(0.0, 1.0)


def test_sec10_on_temp_files(tmp_path):
    pd.DataFrame({"submission": list("abcd"), "domain": ["x", "x", "y", "y"], "trials_stated": [4, np.nan, 4, np.nan],
                  "pass_1": [60.0, 50.0, 70.0, 80.0], "pass_2": [40.0, 30.0, 55.0, 70.0]}).to_csv(tmp_path / "tau2_consistency.csv", index=False)
    pd.DataFrame({"f_run_noise_share": [0.2, 0.4, np.nan]}).to_csv(tmp_path / "tb2_within_task_variance.csv", index=False)
    t, feas, s = R3.sec10_tau2(tmp_path, [make_board([0.5, 0.4], rho=0.5)])
    assert s["n_rows_f"] == 4 and s["n_rows_trials_stated"] == 2 and s["tb2_n"] == 2 and s["tb2_median_f"] == pytest.approx(0.3)
    assert list(feas["feasible"]) == [True, True, True, False]               # rho 0.5 against f = 0.25, 0.369, 0.5, 0.75
    assert s["by_domain"][0]["domain"] == "x"


def test_sec16_feasible_counts():
    b = make_board([0.8, 0.79, 0.6, 0.55, 0.5, 0.2], n=100, rho=0.6)
    rows, summ = R3.sec16_rerun([b], {"low": 0.2, "high": 0.9, "max feasible": lambda bb: 1 - bb.rho})
    s = summ.set_index(["f_label", "form"])
    assert s.loc[("low", "k-adjusted"), "boards_feasible"] == 1 and s.loc[("high", "k-adjusted"), "boards_feasible"] == 0
    assert s.loc[("high", "k-adjusted"), "n12_not_sep_feasible"] == 0
    assert s.loc[("max feasible", "k-adjusted"), "boards_feasible"] == 1
    # with k = 1 both forms coincide
    assert s.loc[("low", "k-adjusted"), "n12_not_sep_all"] == s.loc[("low", "single-run"), "n12_not_sep_all"]
    # a larger run-noise share gives a wider bound, so fewer separable pairs
    assert s.loc[("high", "k-adjusted"), "n12_not_sep_all"] >= s.loc[("low", "k-adjusted"), "n12_not_sep_all"]


# ---- sections 11-15 --------------------------------------------------------------------------------

def test_leader_set_and_bonferroni():
    s = np.array([0.80, 0.79, 0.74, 0.60])
    u = R3.leader_set(s, 100, 0.5, C.Z)
    assert list(u) == [True, True, False]                                  # gaps 0.01 and 0.06 are inside 1.96 SE_D, 0.20 is not
    big = R3.leader_set(s, 100, 0.5, 3.5)
    assert big.sum() >= u.sum()                                          # a larger critical value admits more leaders
    assert R3.leader_set(np.array([0.5, 0.5, 0.3]), 100, 0.5, 1.96)[0]    # a tie is always in the set
    b = make_board([0.8, 0.79, 0.74, 0.6, 0.3], n=100, rho=0.5)
    r = R3.sec11_leaders([b]).iloc[0]
    assert r["set_size_uncorrected"] == r["others_uncorrected"] + 1 and r["z_bonferroni"] == pytest.approx(stats.norm.ppf(1 - 0.025 / 4))


def test_cohen_kappa():
    assert R3.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0])["kappa"] == 1.0
    k = R3.cohen_kappa([1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])
    po, pe = 7 / 8, (2 / 8) * (1 / 8) + (6 / 8) * (7 / 8)
    assert k["kappa"] == pytest.approx((po - pe) / (1 - pe))
    assert math.isnan(R3.cohen_kappa([0, 0, 0], [0, 0, 0])["kappa"])
    assert R3.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1])["kappa"] == pytest.approx(-1.0)


def test_sec13_on_temp_csv(tmp_path):
    df = pd.DataFrame({"leaderboard": ["A"] * 4 + ["B"] * 2, "pair": ["adjacent"] * 4 + ["adjacent", "#1 vs #5"],
                       "bound_separable": [True, True, False, False, False, True],
                       "test_separable": [True, False, False, False, False, True]})
    p = tmp_path / "pairs.csv"
    df.to_csv(p, index=False)
    per, out = R3.sec13_kappa(p)
    a = out["adjacent pairs"]
    assert (a["n"], a["n_both"], a["n_bound_only"], a["n_test_only"], a["n_neither"]) == (5, 1, 1, 0, 3)
    assert a["agree_among_either"] == pytest.approx(0.5) and out["adjacent + #1 vs #5"]["n"] == 6
    assert set(per["leaderboard"]) == {"A", "B"}


def test_two_way_components_exact_structure():
    s_eff, t_eff = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    d = R3.two_way_components(s_eff[:, None] + t_eff[None, :])
    assert d["var_residual"] == pytest.approx(0.0, abs=1e-12) and d["ep2"] == pytest.approx(1.0)
    assert d["var_systems"] == pytest.approx(np.var(s_eff, ddof=1)) and d["var_tasks"] == pytest.approx(np.var(t_eff, ddof=1))


def test_two_way_components_recovers_variances_and_ep2_formula():
    rng = np.random.default_rng(11)
    m, n = 60, 400
    Y = rng.normal(0, 1.0, (m, 1)) + rng.normal(0, 0.5, (1, n)) + rng.normal(0, 2.0, (m, n))
    d = R3.two_way_components(Y)
    assert d["var_systems"] == pytest.approx(1.0, abs=0.35) and d["var_tasks"] == pytest.approx(0.25, abs=0.05)
    assert d["var_residual"] == pytest.approx(4.0, rel=0.03)
    assert d["ep2"] == pytest.approx((d["ms_systems"] - d["ms_residual"]) / d["ms_systems"])
    assert d["ep2"] == pytest.approx(d["var_systems"] / (d["var_systems"] + d["var_residual"] / n))
    assert d["share_systems"] + d["share_tasks"] + d["share_residual"] == pytest.approx(1.0)
    # a pure-noise matrix has no system variance
    assert R3.two_way_components(rng.normal(0, 1, (30, 200)))["ep2"] < 0.35


def test_sec14_uses_census_order_and_reports_ep2():
    rng = np.random.default_rng(3)
    ability = np.array([0.9, 0.8, 0.6, 0.5, 0.3, 0.2])
    Y = (rng.uniform(size=(6, 150)) < ability[:, None]).astype(float)
    b = make_board(Y.mean(axis=1), n=150, rho=0.4, lb="syn")
    names = list(b.names)
    order = np.argsort(-Y.mean(axis=1), kind="stable")
    out = {"syn": {nm: Y[i] for nm, i in zip(names, order)}}
    base = R3.ref_rows([b])
    conf = pd.DataFrame({"leaderboard": ["syn"], "c_1v2": [0.7]})
    df = R3.sec14_decomposition([b], out, base, conf, {"syn": 2})
    assert df.loc[0, "systems"] == 6 and df.loc[0, "tasks"] == 150 and df.loc[0, "census_excluded_entries"] == 2
    assert 0.9 < df.loc[0, "ep2"] <= 1.0 and df.loc[0, "cell"] == "binary outcome"


def test_sec15_pair_rho_changes_only_top_checks():
    b = make_board([0.8, 0.77, 0.7, 0.66, 0.6, 0.5, 0.2], n=150, rho=0.4)
    R = np.full((b.m, b.m), 0.4)
    R[0, 1] = R[0, 4] = 0.95                                         # a very high top-pair rho narrows those two bounds
    df, summ = R3.sec15_pair_rho([b], {b.lb: R}, pd.DataFrame({"leaderboard": [b.lb], "rho_all_pairs_median": [0.3]}))
    r = df.iloc[0]
    assert r["bound12_pair_pp"] < r["bound12_headline_pp"] and r["bound15_pair_pp"] < r["bound15_headline_pp"]
    assert summ["n_1v2_above_board"] == 1 and summ["n_1v5_above_board"] == 1
    df2, _ = R3.sec15_pair_rho([b], {}, pd.DataFrame({"leaderboard": [b.lb], "rho_all_pairs_median": [np.nan]}))
    assert df2.iloc[0]["bound12_pair_pp"] == pytest.approx(df2.iloc[0]["bound12_headline_pp"])   # no per-task data: board rho kept


# ---- integration: the committed headline -----------------------------------------------------------

def test_headline_is_reproduced_from_committed_files():
    from pathlib import Path
    res = Path(C.__file__).resolve().parent / "results"
    boards, par = R3.load_boards(res)
    assert len(boards) == 22 and sum(b.source == "own" for b in boards) == 7
    rep = R3.reproduce(boards, res / "census.csv")
    assert rep["reproduced"] and rep["boards_identical"] == 22
    a = rep["aggregate"]
    assert (a["top1_vs_2_not_sep"], a["top1_vs_5_not_sep"], a["tiers_median"]) == (17, 9, 6.0)
    assert round(100 * a["adjacent_below_median"], 1) == 93.9


def test_reproduce_detects_a_changed_board():
    from pathlib import Path
    res = Path(C.__file__).resolve().parent / "results"
    boards, _ = R3.load_boards(res)
    boards[0].rho = 0.0                                              # a different rho must not pass the gate
    assert not R3.reproduce(boards, res / "census.csv")["reproduced"]
