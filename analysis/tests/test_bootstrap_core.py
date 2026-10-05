import numpy as np
import pandas as pd
import pytest

import bootstrap_core as bc


def long_from(spec):
    """spec: {submission: {task: [0/1,...]}}"""
    rows = [(s, t, i + 1, v) for s, d in spec.items() for t, vs in d.items() for i, v in enumerate(vs)]
    return pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success"])


def test_counts_and_macro_mean():
    c = bc.build_counts(long_from({"a": {"t1": [1, 1, 0, 0], "t2": [1, 1, 1, 1]}, "b": {"t1": [0, 0], "t2": [1, 0]}}))
    assert c.K.tolist() == [[2, 4], [0, 1]] and c.N.tolist() == [[4, 4], [2, 2]]
    assert bc.macro_mean(c.K, c.N).tolist() == [0.75, 0.25]


def _spec(T=40, n=5, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, T)
    return {"a": {f"t{t}": list((rng.random(n) < p[t]).astype(int)) for t in range(T)}}


def test_run_to_run_matches_analytic_sd_and_is_deterministic():
    T, n = 40, 5
    c = bc.build_counts(long_from(_spec(T, n)))
    S1, S2 = bc.run_to_run(c, 4000), bc.run_to_run(c, 4000)
    assert np.array_equal(S1, S2)
    ph = c.K[0] / c.N[0]
    # corrected bootstrap SD equals the unbiased closed form sqrt(sum phat(1-phat)/(N-1)) / T ...
    unbiased = np.sqrt((ph * (1 - ph) / (c.N[0] - 1)).sum()) / T
    assert bc.analytic_run_sd(c)[0] == pytest.approx(unbiased)
    assert S1.std(ddof=1) == pytest.approx(unbiased, rel=0.04)
    assert S1.mean() == pytest.approx(bc.macro_mean(c.K, c.N)[0], abs=0.005)


def test_uncorrected_bootstrap_is_biased_low_by_sqrt_of_n_minus_1_over_n():
    T, n = 60, 5
    c = bc.build_counts(long_from(_spec(T, n, seed=4)))
    raw, fixed = bc.run_to_run(c, 20000, correct=False), bc.run_to_run(c, 20000)
    plug_in = np.sqrt((c.K[0] / c.N[0] * (1 - c.K[0] / c.N[0]) / c.N[0]).sum()) / T
    assert raw.std(ddof=1) == pytest.approx(plug_in, rel=0.02)
    assert fixed.std(ddof=1) / raw.std(ddof=1) == pytest.approx(np.sqrt(n / (n - 1)), rel=0.01)
    # same draws: the corrected replicate is a deterministic rescaling of the uncorrected one
    assert np.allclose(fixed.mean(), raw.mean(), atol=1e-3)


def test_correction_per_task_with_mixed_trial_counts_and_single_trial_cells():
    # cell sizes 5, 10 and 1: each task is scaled by its own factor; N = 1 is left unscaled
    spec = {"a": {"t1": [1, 0, 1, 0, 0], "t2": [1, 1, 1, 0, 0, 0, 1, 0, 0, 1], "t3": [1]}}
    c = bc.build_counts(long_from(spec))
    want = np.sqrt(0.4 * 0.6 / 4 + 0.5 * 0.5 / 9 + 0.0) / 3
    assert bc.analytic_run_sd(c)[0] == pytest.approx(want)
    S = bc.run_to_run(c, 40000)
    assert S.std(ddof=1) == pytest.approx(want, rel=0.02)


def test_correction_leaves_constant_cells_and_ordering_stream_unchanged():
    spec = {"a": {"t1": [1] * 5, "t2": [0] * 6}}
    c = bc.build_counts(long_from(spec))
    assert bc.analytic_run_sd(c)[0] == 0.0
    assert np.array_equal(bc.run_to_run(c, 50), bc.run_to_run(c, 50, correct=False))


def test_run_to_run_stream_independent_of_set_membership():
    spec = {"a": {"t1": [1, 0, 1, 0, 1], "t2": [1, 1, 0, 0, 0]}, "b": {"t1": [1, 1, 1, 1, 0], "t2": [0, 0, 0, 1, 0]}}
    full = bc.run_to_run(bc.build_counts(long_from(spec)), 500)
    only_b = bc.run_to_run(bc.build_counts(long_from(spec), subs=["b"]), 500)
    assert np.array_equal(full[:, 1], only_b[:, 0])


def test_task_sampling_sd_and_coverage_check():
    rng = np.random.default_rng(1)
    T = 60
    p = rng.uniform(0, 1, T)
    spec = {"a": {f"t{t}": list((rng.random(5) < p[t]).astype(int)) for t in range(T)}}
    c = bc.build_counts(long_from(spec))
    S = bc.task_sampling(c, 4000)
    pm = c.K[0] / c.N[0]
    assert S.std(ddof=1) == pytest.approx(pm.std(ddof=0) / np.sqrt(T), rel=0.06)
    bad = bc.build_counts(long_from({"a": {"t1": [1, 0], "t2": [1, 1]}, "b": {"t1": [1, 0]}}))
    with pytest.raises(ValueError):
        bc.task_sampling(bad, 10)


def test_no_trial_variance_when_outcomes_are_constant():
    spec = {"a": {"t1": [1] * 5, "t2": [0] * 5}}
    S = bc.run_to_run(bc.build_counts(long_from(spec)), 200)
    assert S.std() == 0 and S[0, 0] == 0.5


def test_flip_matrix_orientation_and_ties():
    S = np.array([[0.9, 0.5, 0.5], [0.9, 0.6, 0.5], [0.4, 0.6, 0.5]])  # columns: x, y, z
    order = np.array([0, 1, 2])
    P = bc.flip_matrix(S, order)
    assert P[0, 1] == pytest.approx(1 / 3)         # y above x only in row 3
    assert P[0, 2] == pytest.approx(1 / 3)         # z above x only in row 3
    assert P[1, 2] == pytest.approx(0.5 / 3)       # one exact tie, never above
    assert np.isnan(P[1, 0]) and np.isnan(P[2, 2])


def test_observed_order_breaks_ties_by_name():
    assert bc.observed_order(["b", "a", "c"], np.array([0.5, 0.5, 0.9])).tolist() == [2, 1, 0]
