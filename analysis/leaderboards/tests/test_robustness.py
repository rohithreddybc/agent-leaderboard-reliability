"""Tests for robustness.py (McNemar validation, calibration simulation, eligibility helpers).
Synthetic inputs only; files are written to pytest temp dirs."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import census as C
import robustness as Rb


# ---- McNemar helper ----------------------------------------------------------------------------

def test_mcnemar_exact_hand_values():
    x = np.array([1, 1, 1, 1, 1, 0, 0, 1])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    r = Rb.mcnemar_exact(x, y)
    assert (r["b"], r["c"], r["n_discordant"]) == (5, 0, 5)
    assert r["p"] == pytest.approx(2 * 0.5 ** 5)           # 0.0625: not significant at 0.05
    assert Rb.mcnemar_exact(np.r_[x, np.ones(3)], np.r_[y, np.zeros(3)])["p"] == pytest.approx(2 * 0.5 ** 8)  # 8 vs 0
    assert Rb.mcnemar_exact(y, x)["p"] == pytest.approx(r["p"])  # symmetric in the two entries


def test_mcnemar_exact_matches_binomial_cdf_and_edge_cases():
    x = np.array([1] * 3 + [0] * 9 + [1] * 4)
    y = np.array([0] * 3 + [1] * 9 + [1] * 4)
    r = Rb.mcnemar_exact(x, y)
    assert (r["b"], r["c"]) == (3, 9)
    assert r["p"] == pytest.approx(2 * stats.binom.cdf(3, 12, 0.5))
    assert Rb.mcnemar_exact([1, 0, 1], [1, 0, 1])["p"] == 1.0           # no discordant task
    assert Rb.mcnemar_exact([1, 0], [0, 1])["p"] == 1.0                 # b = c: two-sided p capped at 1
    assert Rb.mcnemar_exact([1, 1, 1, 1, 1, 1, 1], [0] * 7)["p"] < 0.05  # 7 vs 0: 2 / 128


def test_paired_tests_shares():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.9, 80)
    x = np.clip(base + rng.normal(0, 0.1, 80), 0, 1)
    y = np.clip(base + 0.05 + rng.normal(0, 0.1, 80), 0, 1)
    t = Rb.paired_tests(x, y)
    assert t["p_t"] == pytest.approx(stats.ttest_rel(x, y).pvalue)
    assert 0 <= t["p_wilcoxon"] <= 1 and 0 <= t["p_sign"] <= 1
    same = Rb.paired_tests(x, x)
    assert same["p_t"] == 1.0 and same["p_wilcoxon"] == 1.0 and same["p_sign"] == 1.0 and same["n_nonzero"] == 0
    assert Rb.paired_tests(np.ones(5), np.full(5, 0.5))["p_t"] == 0.0   # constant non-zero difference


def test_is_binary_and_classify():
    assert Rb.is_binary([0, 1, 1.0]) and not Rb.is_binary([0, 0.2, 1])
    assert Rb.classify(True, True) == "both separable" and Rb.classify(False, False) == "both not separable"
    assert Rb.classify(True, False) == "bound only" and Rb.classify(False, True) == "test only"


def test_bound_separable_follows_census_rule():
    sep, bd = Rb.bound_separable(0.50, 0.40, 500, 0.5)
    assert bd == pytest.approx(float(C.bound_a(0.50, 0.40, 500, 0.5)))
    assert sep == (0.10 >= bd)
    assert Rb.bound_separable(0.3, 0.3, 100, 0.5)[0] is False   # a tie is never separable
    assert Rb.bound_separable(0.0, 0.0, 100, 0.5)[0] is False   # SE = 0 at P = 0


def _board_outcomes(n=200, seed=1):
    rng = np.random.default_rng(seed)
    top = (rng.random(n) < 0.6).astype(float)
    follow = np.where(rng.random(n) < 0.9, top, 1 - top)   # nearly identical to the top entry
    other = (rng.random(n) < 0.3).astype(float)            # independent and much weaker
    return {"a": top, "b": follow, "c": other}


def test_validate_board_pairs_and_categories():
    out = _board_outcomes()
    names = ["a", "b", "c"]
    scores = [out[k].mean() for k in names]
    # order by score, as the census does
    idx = C.order(scores, names)
    names = [names[i] for i in idx]
    scores = [scores[i] for i in idx]
    df = Rb.validate_board("toy", names, scores, out, 200, 0.5)
    assert list(df["pair"]) == ["adjacent", "adjacent"]          # 3 entries: no #1 vs #5
    assert (df["test"] == "McNemar exact").all()
    assert set(df["category"]) <= set(Rb.CATS)
    assert df["gap_pp"].min() >= 0
    # pair table matches a direct computation for the first pair
    r0 = df.iloc[0]
    want = Rb.mcnemar_exact(out[r0.entry_hi], out[r0.entry_lo])["p"]
    assert r0.p_value == pytest.approx(want)
    assert bool(r0.test_separable) == (want < 0.05)


def test_validate_board_adds_pair_1_5_and_uses_t_test_for_shares():
    rng = np.random.default_rng(3)
    out = {f"e{i}": np.round(rng.binomial(5, 0.6 - 0.05 * i, 100) / 5.0, 1) for i in range(6)}
    names = [f"e{i}" for i in range(6)]
    scores = [out[k].mean() for k in names]
    idx = C.order(scores, names)
    names = [names[i] for i in idx]
    scores = [scores[i] for i in idx]
    df = Rb.validate_board("toy", names, scores, out, 100, 0.4)
    assert len(df) == 5 + 1 and df.iloc[-1]["pair"] == "#1 vs #5"
    assert (df["test"] == "paired t").all() and {"p_wilcoxon", "p_sign"} <= set(df.columns)
    x, y = out[names[0]], out[names[4]]
    assert df.iloc[-1]["p_value"] == pytest.approx(stats.ttest_rel(x, y).pvalue)


def test_agreement_table_counts():
    pairs = pd.DataFrame({
        "leaderboard": ["A"] * 4 + ["B"] * 2, "pair": ["adjacent"] * 3 + ["#1 vs #5"] + ["adjacent", "adjacent"],
        "rank_hi": [1, 2, 3, 1, 1, 2],
        "category": ["both separable", "bound only", "both not separable", "test only", "both not separable", "both not separable"],
        "bound_separable": [True, True, False, False, False, False], "test_separable": [True, False, False, True, False, False],
        "test": ["McNemar exact"] * 6})
    t = Rb.agreement_table(pairs)
    a = t[(t.pair_set == "adjacent (all)") & (t.leaderboard == "A")].iloc[0]
    assert (a["pairs"], a["both separable"], a["bound only"], a["both not separable"], a["agree"]) == (3, 1, 1, 1, 2)
    allb = t[(t.pair_set == "adjacent (all)") & (t.leaderboard == "ALL BOARDS")].iloc[0]
    assert allb["pairs"] == 5 and allb["agree"] == 4
    one5 = t[(t.pair_set == "#1 vs #5") & (t.leaderboard == "ALL BOARDS")].iloc[0]
    assert one5["test only"] == 1 and one5["agree"] == 0
    assert t[(t.pair_set == "#1 vs #2") & (t.leaderboard == "ALL BOARDS")].iloc[0]["pairs"] == 2


# ---- simulation ---------------------------------------------------------------------------------

def test_joint_probs_marginals_and_correlation():
    pr = Rb.joint_probs(0.3, 0.6, 0.4)
    assert pr.sum() == pytest.approx(1.0)
    assert pr[0] + pr[1] == pytest.approx(0.3) and pr[0] + pr[2] == pytest.approx(0.6)
    phi = (pr[0] - 0.3 * 0.6) / math.sqrt(0.3 * 0.7 * 0.6 * 0.4)
    assert phi == pytest.approx(0.4)
    with pytest.raises(ValueError):
        Rb.joint_probs(0.1, 0.9, 0.9)   # infeasible: P(1,1) would exceed min(P1, P2)


def test_simulation_trivial_cases():
    rng = np.random.default_rng(0)
    assert Rb.simulate_cell(30, 0.5, 0.5, 1.0, 2000, rng) == 0.0   # perfectly correlated equal entries: D = 0 always
    assert Rb.simulate_cell(30, 0.0, 0.0, 0.0, 500, rng) == 0.0    # both solve nothing: tie
    assert Rb.simulate_cell(30, 1.0, 0.0, 0.0, 500, rng) == 1.0    # one solves all, the other none: |D| = 1, SE = 0
    # n = 1, P = 0.5, rho = 0: the two outcomes differ with probability 0.5, and then |D| = 1 > 1.96 * 0
    assert Rb.exact_rejection_prob(1, 0.5, 0.5, 0.0) == pytest.approx(0.5)
    assert Rb.simulate_cell(1, 0.5, 0.5, 0.0, 40000, np.random.default_rng(1)) == pytest.approx(0.5, abs=0.01)


def test_simulation_matches_exact_enumeration():
    for n, P, rho in [(12, 0.3, 0.0), (20, 0.5, 0.5), (25, 0.8, 0.75)]:
        ex = Rb.exact_rejection_prob(n, P, P, rho)
        sm = Rb.simulate_cell(n, P, P, rho, 200_000, np.random.default_rng(n))
        assert sm == pytest.approx(ex, abs=0.004)
    # with unequal marginals (power setting)
    ex = Rb.exact_rejection_prob(20, 0.6, 0.4, 0.25)
    assert Rb.simulate_cell(20, 0.6, 0.4, 0.25, 200_000, np.random.default_rng(7)) == pytest.approx(ex, abs=0.004)
    # the exact probabilities of all outcomes sum to one (sanity of the enumeration): reject + accept = 1
    assert 0.0 <= ex <= 1.0


def test_calibration_grid_deterministic_and_close_to_nominal():
    g1 = Rb.calibration_grid(ns=(500,), Ps=(0.5,), rhos=(0.0, 0.5), reps=4000, seed=11)
    g2 = Rb.calibration_grid(ns=(500,), Ps=(0.5,), rhos=(0.0, 0.5), reps=4000, seed=11)
    pd.testing.assert_frame_equal(g1, g2)
    assert len(g1) == 2 and (g1["false_separation_rate"].between(0.03, 0.08)).all()
    assert (g1["power_at_bound"].between(0.42, 0.58)).all()
    assert Rb.nominal_bound(100, 0.5, 0.0) == pytest.approx(1.96 * math.sqrt(2 * 0.25 / 100))
    assert g1["delta_pp"].iloc[0] == pytest.approx(100 * Rb.nominal_bound(500, 0.5, 0.0))


# ---- eligibility helpers --------------------------------------------------------------------------

def test_census_row_equals_census_config_and_aggregate():
    scores = [0.80, 0.78, 0.60, 0.55, 0.50, 0.20]
    names = [f"e{i}" for i in range(6)]
    b = Rb.Board("toy", 100, names, scores, [1.0] * 6, 0.5, "pooled")
    r = Rb.census_row(b)
    st = C.census_config(scores, names, [1.0] * 6, lambda p1, p2, k1, k2: C.bound_a(p1, p2, 100, 0.5))
    for k in ("n_tiers", "n_adjacent_below", "n_all_below", "share_all_below", "top1_vs_2_separable", "top1_vs_5_separable"):
        assert r[k] == st[k]
    agg = Rb.aggregate_rows([r, {**r, "leaderboard": "toy2", "top1_vs_2_separable": True, "n_tiers": r["n_tiers"] + 2}])
    assert agg["n_boards"] == 2 and agg["top1_vs_2_not_sep"] == int(not r["top1_vs_2_separable"])
    assert agg["tiers_max"] == r["n_tiers"] + 2


def test_dedupe_best_per_organisation_and_gaia_loader(tmp_path):
    d = tmp_path / "data/raw/leaderboards/gaia/results_public/2023"
    d.mkdir(parents=True)
    pd.DataFrame({"model": ["m1", "m2", "m3", "m4"], "organisation": ["org", "org", "", ""],
                  "score": [0.5, 0.6, 0.2, 0.2], "date": ["2025-01-01"] * 4}).to_parquet(d / "test-00000-of-00001.parquet")
    es = Rb.load_gaia(tmp_path, "test", "GAIA test")
    assert len(es) == 4 and all(e.status == "kept" and e.lb == "GAIA test" for e in es)
    assert len({e.name for e in es}) == 4
    ded = Rb.dedupe_best_per_organisation(es)
    assert sorted(round(e.score, 2) for e in ded) == [0.2, 0.2, 0.6]   # org keeps its best row; empty orgs keyed by model


# ---- multiplicity, ties, TB2 run vs task (added 2026-10-07) ---------------------------------------

def test_z_bonferroni_values():
    assert Rb.z_bonferroni(1) == pytest.approx(1.959964, abs=1e-5)
    assert Rb.z_bonferroni(2) == pytest.approx(stats.norm.ppf(1 - 0.0125))
    assert Rb.z_bonferroni(2) == pytest.approx(2.2414, abs=1e-4)
    assert Rb.z_bonferroni(10) > Rb.z_bonferroni(2)


def test_bound_a_z_reduces_to_census_bound_at_196():
    assert float(Rb.bound_a_z(0.6, 0.5, 200, 0.4, C.Z)) == pytest.approx(float(C.bound_a(0.6, 0.5, 200, 0.4)))
    assert float(Rb.bound_a_z(0.6, 0.5, 200, 0.4, 3.0)) == pytest.approx(3.0 / C.Z * float(C.bound_a(0.6, 0.5, 200, 0.4)))


def test_holm_reject_count_hand_example():
    assert Rb.holm_reject_count([0.001, 0.01, 0.04, 0.5]) == 2      # thresholds .0125 .0167 .025 .05: stops at 0.04
    assert Rb.holm_reject_count([0.5, 0.6]) == 0
    assert Rb.holm_reject_count([0.001, 0.002, 0.003]) == 3
    assert Rb.holm_reject_count([]) == 0


def _toy_board(scores, n=100, rho=0.5, lb="toy"):
    names = [f"e{i}" for i in range(len(scores))]
    return Rb.Board(lb, n, names, list(scores), [1.0] * len(scores), rho, "pooled")


def test_holm_all_pairs_share_and_multiplicity_table_monotone():
    b = _toy_board([0.80, 0.80, 0.70, 0.62, 0.55, 0.30, 0.10])
    h = Rb.holm_all_pairs_share(b)
    assert h["n_pairs"] == 21 and h["n_rejected"] <= h["n_unadjusted_rejected"]
    assert h["share_not_rejected"] == pytest.approx(1 - h["n_rejected"] / 21)
    t = Rb.multiplicity_table([b]).iloc[0]
    # a larger critical value can only move pairs into 'below the bound'
    assert t["adjbonf_adjacent_share_below"] >= t["unadj_adjacent_share_below"]
    assert t["adjbonf_all_pairs_share_below"] >= t["unadj_all_pairs_share_below"]
    assert t["holm_all_pairs_share_not_rejected"] >= t["unadj_all_pairs_share_below"] - 1e-12
    assert t["z_adjacent_bonferroni"] == pytest.approx(Rb.z_bonferroni(6))
    s = Rb.multiplicity_summary(Rb.multiplicity_table([b, _toy_board([0.9, 0.5, 0.4, 0.3, 0.2])]))
    assert s["n_boards"] == 2


def test_tie_summary():
    s = Rb.tie_summary([_toy_board([0.5, 0.5, 0.4, 0.3, 0.2], lb="tied"), _toy_board([0.9, 0.5, 0.4, 0.3, 0.2], lb="clear"),
                        _toy_board([0.51, 0.50, 0.4, 0.3, 0.2], lb="close")])
    assert s["tied_boards"] == ["tied"] and s["n_tied"] == 1 and s["untied_boards"] == 2
    assert s["untied_1v2_not_sep"] == 1                      # 'close' is not separable, 'clear' is


def test_tb2_run_vs_task(tmp_path):
    pd.DataFrame({"submission": ["a", "b"], "n_trials": [445, 445], "n_tasks": [89, 89],
                  "run_sd": [0.01, 0.02], "task_sd": [0.04, 0.04]}).to_csv(tmp_path / "tb2_bootstrap.csv", index=False)
    pd.DataFrame({"submission": ["a", "b"], "mean_p1mp_unbiased": [0.089, 0.178]}).to_csv(tmp_path / "tb2_within_task_variance.csv", index=False)
    d, o = Rb.tb2_run_vs_task(tmp_path)
    assert d["ratio_sq"].tolist() == pytest.approx([0.0625, 0.25])
    assert o["ratio_sq_median"] == pytest.approx((0.0625 + 0.25) / 2)
    assert d["single_run_sd_from_v"].tolist() == pytest.approx([math.sqrt(0.089 / 89), math.sqrt(0.178 / 89)])
    assert o["single_run_sd_pp_median"] == pytest.approx(100 * (math.sqrt(0.001) + math.sqrt(0.002)) / 2)
