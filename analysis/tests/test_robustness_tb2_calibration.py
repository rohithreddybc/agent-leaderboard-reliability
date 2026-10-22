"""Tests for robustness_tb2_calibration.py. Synthetic inputs only."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import bootstrap_core as bc
import robustness_tb2_calibration as K


def test_poisson_binomial_pmf():
    p = np.array([0.2, 0.5, 0.9])
    pmf = K.poisson_binomial_pmf(p)
    assert pmf.sum() == pytest.approx(1.0) and len(pmf) == 4
    assert (pmf * np.arange(4)).sum() == pytest.approx(p.sum())
    assert pmf[0] == pytest.approx(0.8 * 0.5 * 0.1) and pmf[3] == pytest.approx(0.2 * 0.5 * 0.9)
    assert K.poisson_binomial_pmf(np.full(10, 0.3)) == pytest.approx(stats.binom.pmf(np.arange(11), 10, 0.3))
    assert K.count_pmf(np.array([0.3, 0.7]), 5).size == 11


def test_bootstrap_means_matches_bootstrap_core_distribution():
    rng = np.random.default_rng(0)
    r, T = 5, 89
    p = rng.uniform(0.05, 0.95, T)
    Kc = rng.binomial(r, p)
    rows = [(f"s", f"t{t}", i, int(i < Kc[t])) for t in range(T) for i in range(r)]
    c = bc.build_counts(pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success"]))
    ref = bc.run_to_run(c, 60_000)[:, 0]
    ph, dev = K.bootstrap_means(Kc[None, :], r, 60_000, np.random.default_rng(1))
    mine = ph[0] + np.sqrt(r / (r - 1)) * dev[0]
    assert mine.mean() == pytest.approx(ref.mean(), abs=2e-4)
    assert mine.std() == pytest.approx(ref.std(), rel=0.02)
    assert mine.std() == pytest.approx(bc.analytic_run_sd(c)[0], rel=0.02)   # corrected SD equals the closed form


def test_true_sd_and_shifted_copy():
    p = np.array([0.5, 0.5, 1.0, 0.0])
    assert K.true_sd(p, 5) == pytest.approx(np.sqrt((0.25 + 0.25) / 5) / 4)
    q = K.shifted_copy(np.array([0.4, 0.6, 0.8]), 0.05)
    assert q.mean() == pytest.approx(0.6 - 0.05) and ((q >= 0) & (q <= 1)).all()
    with pytest.raises(ValueError):
        K.shifted_copy(np.array([0.1, 0.1]), 0.2)


def test_true_flip_trivial_and_symmetric_cases():
    # A always solves, B never: the order never reverses
    t = K.true_flip_probability(np.ones(3), np.zeros(3), 5)
    assert (t["pi_plus"], t["pi_zero"], t["pi_minus"], t["true_flip"]) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    # identical entries on many tasks: ties are rare, the rerun reverses the observed order with probability near 1/2
    p = np.full(200, 0.5)
    t = K.true_flip_probability(p, p, 5)
    assert t["pi_plus"] == pytest.approx(t["pi_minus"]) and t["true_flip"] == pytest.approx(0.5)  # exactly 1/2 with half credit for ties
    # outcomes sum to one
    t = K.true_flip_probability(np.array([0.3, 0.6, 0.9]), np.array([0.2, 0.5, 0.8]), 3)
    assert t["pi_plus"] + t["pi_zero"] + t["pi_minus"] == pytest.approx(1.0)
    assert t["pi_plus"] > t["pi_minus"]


def test_simulate_submission_homogeneous_case_is_calibrated():
    res = K.simulate_submission(np.full(100, 0.5), 5, 600, 800, np.random.SeedSequence(3))
    assert res["true_sd"] == pytest.approx(np.sqrt(100 * 0.25 / 5) / 100)
    assert res["sd_ratio_corrected"] == pytest.approx(1.0, abs=0.03)
    assert res["sd_ratio_uncorrected"] == pytest.approx(np.sqrt(4 / 5), abs=0.03)
    assert 0.92 <= res["coverage_corrected"] <= 0.98
    assert res["coverage_uncorrected"] <= res["coverage_corrected"] + 0.02


def test_simulate_submission_degenerate_truth():
    res = K.simulate_submission(np.array([0.0, 1.0, 1.0]), 5, 50, 100, np.random.SeedSequence(0))
    assert res["true_sd"] == 0 and res["coverage_corrected"] == 1.0 and np.isnan(res["sd_ratio_corrected"])


def test_simulate_pair_far_apart_and_deterministic():
    pA, pB = np.full(60, 0.8), np.full(60, 0.3)
    a = K.simulate_pair(pA, pB, 5, 200, 400, np.random.SeedSequence(5))
    b = K.simulate_pair(pA, pB, 5, 200, 400, np.random.SeedSequence(5))
    assert a["mean_boot_flip_corrected"] == b["mean_boot_flip_corrected"]
    assert a["mean_boot_flip_corrected"] == 0.0 and a["true_flip"] < 1e-9 and a["share_observed_order_correct"] == 1.0
    assert a["gap_pp"] == pytest.approx(50.0)
    # equal entries: observed order is a coin flip, true reversal is 1/2, the mean bootstrap flip is clearly below 1/2
    e = K.simulate_pair(np.full(60, 0.5), np.full(60, 0.5), 5, 300, 400, np.random.SeedSequence(6))
    assert e["true_flip"] == pytest.approx(0.5) and 0.15 < e["mean_boot_flip_corrected"] < 0.40


def test_calibration_bins_and_representative():
    ds = pd.DataFrame({"r": 5, "true_gap_pp": 1.0, "f_corrected": [0.0, 0.005, 0.03, 0.08, 0.4, 0.7],
                       "f_uncorrected": 0.0, "true_reversal_given_order": [0.0, 0.1, 0.2, 0.3, 0.4, 0.9]})
    b = K.calibration_bins(ds)
    first = b[(b.f_lo == 0.0) & (b.f_hi == 0.01)].iloc[0]
    assert first.datasets == 2 and first.mean_true_reversal == pytest.approx(0.05)
    sep = b[b.note.notna()].iloc[0]                    # f <= 0.05: three datasets
    assert sep.datasets == 3 and sep.mean_true_reversal == pytest.approx(0.1) and sep.mean_boot_flip == pytest.approx((0.0 + 0.005 + 0.03) / 3)
    assert b[b.f_lo == 0.5].iloc[0].datasets == 1
    means = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2])
    zero = np.array([False, False, False, True, False, False])
    idx = K.representative(means, zero, n=3)
    assert idx[0] == 0 and idx[-1] == 1 and 3 not in idx          # lowest and highest score; the zero-variance one is skipped
