"""Tests for estimate_rho.py and the empirical-rho / run-noise-share parts of census.py.
Synthetic inputs only; files are written to pytest temp dirs."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import census as C
import estimate_rho as E


# ---- rho definition ----------------------------------------------------------------------------

def test_rho_matrix_is_phi_for_binary_outcomes():
    rng = np.random.default_rng(0)
    P = (rng.random((5, 60)) < rng.uniform(0.2, 0.8, (5, 1))).astype(float)
    R = E.rho_matrix(P)
    assert np.allclose(R, np.corrcoef(P))
    assert np.allclose(np.diag(R), 1.0) and np.allclose(R, R.T)


def test_rho_hand_example():
    a = np.array([1, 1, 0, 0, 1, 0, 0, 0], float)
    b = np.array([1, 0, 0, 0, 1, 1, 0, 0], float)
    # P_a = P_b = 3/8, both solved on tasks 0 and 4: cov = 2/8 - 9/64 = 7/64
    want = (7 / 64) / (3 / 8 * 5 / 8)
    assert E.rho_matrix(np.vstack([a, b]))[0, 1] == pytest.approx(want)
    assert want == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_rho_single_run_equivalent_for_multi_run_rates():
    # Entries with k = 5 runs: rates are multiples of 1/5. The single-run-equivalent rho divides the
    # across-task covariance by sqrt(P(1-P)) (single-run variance), so it is below Pearson of the rates.
    rng = np.random.default_rng(1)
    base = rng.uniform(0.05, 0.95, 400)
    P = np.vstack([rng.binomial(5, base) / 5.0, rng.binomial(5, base) / 5.0])
    r_single, r_rates = E.rho_matrix(P)[0, 1], E.pearson_matrix(P)[0, 1]
    assert r_single < r_rates
    # single runs drawn from the same per-task rates have phi equal to r_single in expectation
    phis = [np.corrcoef((rng.random((2, 400)) < P).astype(float))[0, 1] for _ in range(300)]
    assert np.mean(phis) == pytest.approx(r_single, abs=0.02)


def test_rho_nan_when_entry_solves_all_or_none():
    P = np.array([[1.0] * 6, [1, 0, 1, 0, 1, 0], [0.0] * 6])
    R = E.rho_matrix(P)
    assert np.isnan(R[0, 1]) and np.isnan(R[1, 2]) and np.isnan(R[0, 0])
    assert R[1, 1] == pytest.approx(1.0)
    assert np.isnan(E.pearson_matrix(P)[0, 1])


def test_adjacent_and_summarise():
    M = np.arange(16, dtype=float).reshape(4, 4)
    assert E.adjacent(M).tolist() == [1.0, 6.0, 11.0]
    s = E.summarise(np.array([0.1, 0.2, 0.3, 0.4, np.nan]))
    assert s["n"] == 4 and s["median"] == pytest.approx(0.25)
    assert (s["q25"], s["q75"]) == pytest.approx((0.175, 0.325))
    assert (s["min"], s["max"]) == (0.1, 0.4)
    assert E.summarise(np.array([np.nan]))["n"] == 0


# ---- loaders on synthetic files -------------------------------------------------------------------

def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_swebench_outcomes_match_census_scores(tmp_path):
    ids = [f"i{i}" for i in range(10)]
    d = {"a": {"source": "r", "resolved": ids[:6], "covered_ids": ids, "meta": {"attempts": 1}},
         "b": {"source": "r", "resolved": ids[3:8], "covered_ids": ids, "meta": {"attempts": 1}},
         "nofile": {"source": None, "meta": {}}}
    _write(tmp_path / "data/raw/leaderboards/swe-bench/swebench_verified.json", d)
    out = E.swebench_outcomes(tmp_path, "verified", 10)
    assert set(out) == {"a", "b"}
    es = {e.name: e for e in C.load_swebench(tmp_path, "verified", "SWE-bench Verified", 10)}
    for k in out:
        assert out[k].mean() == pytest.approx(es[k].score)
    assert out["a"].tolist() == [1] * 6 + [0] * 4


def test_tb1_outcomes_count_repeated_ids(tmp_path):
    ref = [f"t{i}" for i in range(4)]
    _write(tmp_path / "data/raw/leaderboards/n_verification/tb1_registry_0.1.1_task_ids.json", {"task_ids": ref})
    # entry X: one file with 2 trials per task (ids repeat); entry Y: two files with one trial per task
    X = {"resolved_ids": ["t0", "t0", "t1"], "unresolved_ids": ["t1", "t2", "t2", "t3", "t3"]}
    Y1 = {"resolved_ids": ["t0", "t1"], "unresolved_ids": ["t2", "t3"]}
    Y2 = {"resolved_ids": ["t0"], "unresolved_ids": ["t1", "t2", "t3"]}
    _write(tmp_path / "data/raw/leaderboards/terminal-bench-1/tb1_run_results.json", {"files": {
        "results/terminal-bench-core@0.1.1/X/results.json": X,
        "results/terminal-bench-core@0.1.1/Y/a/results.json": Y1,
        "results/terminal-bench-core@0.1.1/Y/b/results.json": Y2}})
    out = E.tb1_outcomes(tmp_path)
    assert out["X"].tolist() == [1.0, 0.5, 0.0, 0.0]
    assert out["Y"].tolist() == [1.0, 0.5, 0.0, 0.0]


def test_tb2_and_tac_outcomes(tmp_path):
    rows = [("s", t, i, int((t == "t1") or (t == "t2" and i < 2))) for t in ("t1", "t2", "t3") for i in range(4)]
    (tmp_path / "analysis/results").mkdir(parents=True)
    pd.DataFrame(rows, columns=["submission", "task_id", "trial", "success"]).to_csv(
        tmp_path / "analysis/results/tb2_long.csv", index=False)
    assert E.tb2_outcomes(tmp_path)["s"].tolist() == [1.0, 0.5, 0.0]
    cp = lambda r: {"checkpoints": [{"result": r, "total": 1}, {"result": 1, "total": 1}]}
    full = {f"t{i}-image": cp(1 if i < 50 else 0) for i in range(175)}
    part = {f"t{i}-image": cp(1) for i in range(174)}
    _write(tmp_path / "data/raw/leaderboards/theagentcompany/tac_results.json",
           {"results": {"1.0.0": {"full": full, "part": part}}})
    out = E.tac_outcomes(tmp_path)
    assert set(out) == {"full"} and out["full"].sum() == 50


# ---- estimate() ---------------------------------------------------------------------------------

def _board(lb, m, n, seed):
    """m single-run entries on n tasks with a shared task-difficulty component, plus Entry objects."""
    rng = np.random.default_rng(seed)
    diff = rng.normal(size=n)
    out, ents = {}, []
    for i in range(m):
        y = (rng.normal(size=n) * 0.6 + diff + (m - i) * 0.08 > 0.4).astype(float)
        out[f"e{i:02d}"] = y
        ents.append(C.Entry(lb, f"e{i:02d}", float(y.mean()), 1.0))
    return out, ents


def test_estimate_pooled_own_and_order(tmp_path):
    outA, entA = _board("SWE-bench Lite", 12, 300, 1)
    outB, entB = _board("SWE-bench Verified", 12, 500, 2)
    entC = [C.Entry("HAL USACO", f"u{i:02d}", 0.9 - 0.05 * i, 1.0) for i in range(12)]   # aggregates only
    loaders = {"SWE-bench Lite": lambda root: outA, "SWE-bench Verified": lambda root: outB}
    est = E.estimate(entA + entB + entC, tmp_path, loaders)
    own = est.boards.set_index("leaderboard")
    med = [own.loc["SWE-bench Lite", "rho_adjacent_median"], own.loc["SWE-bench Verified", "rho_adjacent_median"]]
    assert est.pooled_median == pytest.approx(np.median(med))
    assert est.source == {"SWE-bench Lite": "own", "SWE-bench Verified": "own", "HAL USACO": "pooled"}
    assert est.rho_used["HAL USACO"] == est.pooled_median
    assert est.rho_used["SWE-bench Lite"] == med[0]
    # adjacent pairs follow the census order and the median equals a direct recomputation
    names, R = est.matrices["SWE-bench Lite"]
    srt = [e.name for e in sorted(entA, key=lambda e: (-e.score, e.name))]
    assert names == srt
    direct = [np.corrcoef(outA[a], outA[b])[0, 1] for a, b in zip(srt[:-1], srt[1:])]
    assert med[0] == pytest.approx(np.nanmedian(direct))
    assert len(est.pairs) == 22 and est.pairs.rho.between(-1, 1).all()


def test_estimate_raises_when_a_kept_entry_has_no_outcomes(tmp_path):
    outA, entA = _board("SWE-bench Lite", 12, 300, 1)
    outA.pop("e03")
    with pytest.raises(ValueError, match="no per-task outcomes"):
        E.estimate(entA, tmp_path, {"SWE-bench Lite": lambda root: outA})


def test_write_outputs(tmp_path):
    outA, entA = _board("SWE-bench Lite", 12, 300, 1)
    est = E.estimate(entA, tmp_path, {"SWE-bench Lite": lambda root: outA})
    E.write_outputs(est, tmp_path / "o")
    s = json.loads((tmp_path / "o/rho_summary.json").read_text())
    assert s["pooled_median_of_board_medians"] == pytest.approx(est.pooled_median) and s["n_boards_with_per_task_data"] == 1
    assert len(pd.read_csv(tmp_path / "o/rho_adjacent_pairs.csv")) == 11
    assert pd.read_csv(tmp_path / "o/rho_by_board.csv").rho_source.iloc[0].startswith("own")


# ---- census: f bound, pair-specific rho, run() families, summary ---------------------------------------

def test_bound_f_formula_and_relation_to_unpaired_bound():
    n, f = 89, 0.37
    p1, p2 = 0.8, 0.5
    want = 1.96 * math.sqrt(f / n * (0.8 * 0.2 + 0.5 * 0.5))
    assert C.bound_f(p1, p2, n, f) == pytest.approx(want)
    assert C.bound_f(p1, p2, n, f) == pytest.approx(math.sqrt(f) * C.bound_a(p1, p2, n, 0.0))
    assert C.bound_f(p1, p2, n, f, 5, 5) == pytest.approx(C.bound_f(p1, p2, n, f) / math.sqrt(5))
    assert C.bound_f(p1, p2, n, f, 1, 4) == pytest.approx(1.96 * math.sqrt(f / n * (0.16 + 0.25 / 4)))
    # v = f P(1-P) never exceeds P(1-P) for f <= 1, so no cap is needed; the bound is 0 when both P = 1
    assert C.bound_f(1.0, 1.0, n, 0.9) == 0.0


def test_census_config_pair_aware_bound_receives_positions():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    seen = []

    def fn(p1, p2, k1, k2, ia, ib):
        seen.append((np.atleast_1d(ia).tolist(), np.atleast_1d(ib).tolist()))
        return np.full(np.broadcast(p1, p2).shape, 0.05)

    st = C.census_config(scores, list("abcde"), [1] * 5, fn, pair=True)
    assert [0, 1, 2, 3] in [x[0] for x in seen] and [1, 2, 3, 4] in [x[1] for x in seen]
    assert st["n_adjacent_below"] == 0 and st["n_tiers"] == 5      # every adjacent gap 0.1 > 0.05


def _lite_entries(m=12):
    return [C.Entry("SWE-bench Lite", f"e{i:02d}", 0.9 - 0.05 * i, 1.0) for i in range(m)]


def _fake_estimate(ents, rho, pooled=0.5, source="own"):
    names = [e.name for e in ents]
    m = len(names)
    R = np.full((m, m), rho) + (1 - rho) * np.eye(m)
    boards = pd.DataFrame([{
        "leaderboard": "SWE-bench Lite", "n_tasks": 300, "n_entries": m, "k_runs_max": 1.0, "n_adjacent_pairs": m - 1,
        "n_adjacent_defined": m - 1, "rho_adjacent_median": rho, "rho_adjacent_q25": rho - 0.1, "rho_adjacent_q75": rho + 0.1,
        "rho_adjacent_min": rho - 0.2, "rho_adjacent_max": rho + 0.2, "rho_all_pairs_median": rho - 0.05,
        "rho_pearson_rates_adjacent_median": rho, "rho_used": rho, "rho_source": "own (per-task data)"}])
    return E.RhoEstimate(pd.DataFrame(), boards, pooled, {"SWE-bench Lite": (names, R)},
                         {"SWE-bench Lite": rho}, {"SWE-bench Lite": source})


def test_run_with_rho_and_f_adds_families():
    ents = _lite_entries()
    est = _fake_estimate(ents, 0.6)
    cen, pairs = C.run(ents, est, 0.37)
    lite = cen[(cen.leaderboard == "SWE-bench Lite") & (cen.set == "primary")]
    assert set(lite.bound) == {"a", "a_emp", "a_pooled", "a_pair", "b", "b_kadj", "b_f", "b_f_kadj"}
    assert len(lite) == 4 + 3 + 8 + 4 + 2 * 4          # rho sweep, emp/pooled/pair, v x cap, v k-adj., f x (single, k-adj.)
    emp = lite[lite.bound == "a_emp"].iloc[0]
    assert emp.param == pytest.approx(0.6) and "own" in emp.param_label
    # constant rho = 0.6 everywhere: the pair-specific result equals the board-level one
    pr = lite[lite.bound == "a_pair"].iloc[0]
    for col in ("n_adjacent_below", "n_all_below", "n_tiers"):
        assert pr[col] == emp[col]
    want_below = sum(0.05 < float(C.bound_a(0.9 - 0.05 * (i + 1), 0.9 - 0.05 * i, 300, 0.6)) for i in range(11))
    assert emp.n_adjacent_below == want_below
    fb = lite[(lite.bound == "b_f_kadj") & np.isclose(lite.param, 0.37)].iloc[0]
    assert fb.param_label == "f=TB2 median"
    assert {"rho_used", "bound_a_emp_pp", "bound_a_pair_pp", "bound_b_fTB2_kadj_pp"} <= set(pairs.columns)
    assert pairs.rho_pair.eq(0.6).all()


def test_run_without_rho_has_no_empirical_families():
    cen, pairs = C.run(_lite_entries())
    assert set(cen.bound) == {"a", "b", "b_kadj", "b_f", "b_f_kadj"}      # f sweep without the TB2 value
    assert not cen.bound.isin(["a_emp", "a_pooled", "a_pair"]).any()
    assert "rho_used" not in pairs.columns and "bound_b_fTB2_pp" not in pairs.columns


def test_tb2_median_f(tmp_path):
    pd.DataFrame({"submission": list("abcde"), "f_run_noise_share": [0.2, 0.3, np.nan, 0.5, 0.9]}).to_csv(
        tmp_path / "tb2_within_task_variance.csv", index=False)
    assert C.tb2_median_f(tmp_path) == pytest.approx(0.4)   # median of 0.2, 0.3, 0.5, 0.9


def test_summary_and_aggregates_with_rho(tmp_path):
    ents = _lite_entries()
    est = _fake_estimate(ents, 0.5)
    cen, _ = C.run(ents, est, 0.37)
    ag = C.aggregates(cen, ["SWE-bench Lite"], 0.37)
    labs = [a["config"] for a in ag]
    assert labs[0].startswith("(a) empirical rho") and any("f_TB2" in x for x in labs) and len(labs) == 9
    out = tmp_path / "s.md"
    C.write_summary(cen, ents, out, None, est, 0.37)
    txt = out.read_text(encoding="utf-8")
    assert "Task-level correlation rho between adjacent entries" in txt and "empirical rho [headline]" in txt
    assert "f = f_TB2 = 0.370" in txt and "pair-specific rho" in txt
    cen0, _ = C.run(ents)
    C.write_summary(cen0, ents, tmp_path / "s0.md")      # the path without rho and f still renders
    assert "rho = 0" in (tmp_path / "s0.md").read_text(encoding="utf-8")
