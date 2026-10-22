"""Tests for sepbound. Reference values come from the committed CSVs of the release copy (read only)."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sepbound
from sepbound import _norm

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
LOOKUP = REPO / "analysis" / "results" / "robustness_2026-10-20_secC_lookup.csv"
RESULTS = REPO / "analysis" / "leaderboards" / "results"

needs_lookup = pytest.mark.skipif(not LOOKUP.exists(), reason="release analysis CSVs not present")
needs_census = pytest.mark.skipif(not (RESULTS / "census.csv").exists(), reason="release census CSVs not present")


# ---------------------------------------------------------------- closed-form values

def test_item_bound_n100_p05_rho0():
    assert 100 * sepbound.item_sampling_bound(0.5, 0.5, 100, 0.0) == pytest.approx(13.859, abs=5e-4)


def test_item_bound_n100_p05_rho05():
    assert 100 * sepbound.item_sampling_bound(0.5, 0.5, 100, 0.5) == pytest.approx(9.8, abs=5e-3)


@needs_lookup
def test_lookup_csv_reproduced():
    rows = list(csv.DictReader(open(LOOKUP, newline="")))
    assert len(rows) == 48
    for r in rows:
        b = 100 * sepbound.item_sampling_bound(float(r["P"]), float(r["P"]), float(r["n"]), float(r["rho"]))
        assert b == pytest.approx(float(r["bound_pp"]), rel=1e-9)


def test_per_entry_n():
    a = sepbound.item_sampling_bound(0.5, 0.4, 100, 0.3, n2=400)
    se = math.sqrt(0.25 / 100), math.sqrt(0.24 / 400)
    assert a == pytest.approx(1.96 * math.sqrt(se[0] ** 2 + se[1] ** 2 - 0.6 * se[0] * se[1]))


def test_bound_symmetric_and_rho_monotone():
    assert sepbound.item_sampling_bound(0.6, 0.3, 200, 0.2) == pytest.approx(sepbound.item_sampling_bound(0.3, 0.6, 200, 0.2))
    assert sepbound.item_sampling_bound(0.6, 0.3, 200, 0.0) > sepbound.item_sampling_bound(0.6, 0.3, 200, 0.5)


def test_wang_rule_at_rho_half():
    # equal SEs, rho = 0.5: SE_D equals SE
    assert sepbound.item_sampling_bound(0.4, 0.4, 250, 0.5) == pytest.approx(1.96 * math.sqrt(0.24 / 250))


def test_separable_rule_and_ties():
    assert not sepbound.separable(0.5, 0.5, 100, 0.0)             # equal scores are always below the bound
    assert not sepbound.separable(0.0, 0.0, 100, 0.0)             # zero bound, zero gap
    assert sepbound.separable(0.0, 1.0, 100, 0.0)                 # zero SE_D, positive gap
    assert not sepbound.separable(0.50, 0.45, 100, 0.0)           # 5 pp < 13.8 pp
    assert sepbound.separable(0.65, 0.45, 100, 0.0)               # 20 pp > 13.8 pp
    assert sepbound.separable(0.45, 0.65, 100, 0.0)               # order of arguments is irrelevant


def test_separable_vectorised():
    out = sepbound.separable([0.5, 0.65], [0.45, 0.45], 100)
    assert out.tolist() == [False, True]


def test_order_confidence():
    assert sepbound.order_confidence(0.5, 0.5, 100) == pytest.approx(0.5)
    assert sepbound.order_confidence(0.6, 0.4, 100, 0.0) > 0.99
    # gap equal to the bound gives Phi(1.96); here the scores themselves fix SE_D
    gap = sepbound.item_sampling_bound(0.5, 0.4, 100, 0.2)
    c = sepbound.order_confidence(0.5, 0.4, 100, 0.2)
    assert c == pytest.approx(float(_norm.cdf(0.1 / (gap / 1.96))))
    assert sepbound.order_confidence(1.0, 0.0, 10) == 1.0           # SE_D = 0, positive gap


def test_rerun_bound():
    # equal scores, k = 1: sqrt(f) times the unpaired item-sampling bound (rho = 0)
    r = sepbound.rerun_bound(0.5, 0.5, 100, 0.369, k=1)
    assert r == pytest.approx(math.sqrt(0.369) * sepbound.item_sampling_bound(0.5, 0.5, 100, 0.0))
    # Eq. detectability: f = v / (P(1-P)); v = 0.1 -> 1.96 sqrt(2 v / (n k))
    assert sepbound.rerun_bound(0.5, 0.5, 100, 0.4, k=4) == pytest.approx(1.96 * math.sqrt(2 * 0.1 / (100 * 4)))
    # entries at 0 or 1 contribute nothing; both at 0 or 1 gives zero
    assert sepbound.rerun_bound(0.0, 1.0, 100, 0.369) == 0.0
    assert sepbound.rerun_bound(0.5, 0.5, 100, 0.369, k=2, k2=4) < sepbound.rerun_bound(0.5, 0.5, 100, 0.369, k=1)


# ---------------------------------------------------------------- leader quantities

def test_leader_posterior_basic():
    assert sepbound.leader_posterior([0.9, 0.3, 0.2], 500, 0.0, draws=4000) > 0.999
    p = sepbound.leader_posterior([0.5, 0.5], 100, 0.0, draws=40000, seed=1)  # exact tie: the tied set is the whole board
    assert p == 1.0
    # two entries, flat prior: equals the order confidence (symmetry of the normal)
    c = sepbound.order_confidence(0.52, 0.50, 100, 0.5)
    assert sepbound.leader_posterior([0.52, 0.50], 100, 0.5, draws=200000, seed=3) == pytest.approx(c, abs=0.01)


def test_leader_posterior_reproducible_and_rho():
    a = sepbound.leader_posterior([0.5, 0.48, 0.47], 200, 0.5, 5000, seed=7)
    assert a == sepbound.leader_posterior([0.5, 0.48, 0.47], 200, 0.5, 5000, seed=7)
    hi = sepbound.leader_posterior([0.5, 0.48, 0.47], 200, 0.9, 20000, seed=7)
    lo = sepbound.leader_posterior([0.5, 0.48, 0.47], 200, 0.0, 20000, seed=7)
    assert hi > lo                                                # shared task noise cancels in the comparison


def test_leader_posterior_negative_rho_runs():
    assert 0.0 < sepbound.leader_posterior([0.5, 0.48, 0.47], 200, -0.3, 5000, seed=0) < 1.0


def test_leader_set():
    s = [0.50, 0.49, 0.30, 0.50]
    ls = sepbound.leader_set(s, 100, 0.0)
    assert ls[0] == 0 and set(ls) == {0, 1, 3}                    # includes the leader, the tie and the close entry
    assert set(sepbound.leader_set(s, 100, 0.0, bonferroni=False)) <= set(ls)
    assert sepbound.leader_set([0.5], 100) == [0]


def test_leader_set_bonferroni_is_superset_and_z():
    s = np.linspace(0.6, 0.3, 12)
    unc = set(sepbound.leader_set(s, 100, 0.5, bonferroni=False))
    bon = set(sepbound.leader_set(s, 100, 0.5, bonferroni=True))
    assert unc <= bon
    assert sepbound.z_bonferroni(1) == pytest.approx(1.959964, abs=1e-6)
    assert sepbound.z_bonferroni(4) == pytest.approx(2.497705, abs=1e-5)


def test_tiers_algorithm1():
    # worked example: n = 100, rho = 0.5 -> bounds 9.4 to 9.8 pp
    t = sepbound.tiers([0.50, 0.45, 0.40, 0.35, 0.30], 100, 0.5)
    assert [list(g) for g in t] == [[0, 1], [2, 3], [4]]
    assert len(sepbound.tiers([0.5, 0.5, 0.5], 100, 0.0)) == 1    # ties never open a tier
    assert sepbound.tiers([], 100) == []
    # input order does not matter
    t2 = sepbound.tiers([0.30, 0.50, 0.35, 0.45, 0.40], 100, 0.5)
    assert [[x for x in g] for g in t2] == [[1, 3], [4, 2], [0]]


def test_check_leaderboard_report():
    names = list("ABCDE")
    rep = sepbound.check_leaderboard(names, [0.5, 0.45, 0.4, 0.35, 0.3], 100, 0.5)
    assert rep["order"] == names and rep["n_tiers"] == 3
    assert rep["top1_vs_2"]["separable"] is False and rep["top1_vs_5"]["separable"] is True
    assert rep["top1_vs_2"]["bound"] == pytest.approx(sepbound.item_sampling_bound(0.5, 0.45, 100, 0.5))
    assert rep["leader_set"] == ["A", "B", "C"]
    assert "Tiers: 3" in rep["text"] and "below the bound" in rep["text"]
    assert sepbound.check_leaderboard(["a", "b"], [0.5, 0.4], 100)["top1_vs_5"] is None


def test_input_validation():
    with pytest.raises(ValueError):
        sepbound.item_sampling_bound(50, 45, 100)                 # percent, not proportion
    with pytest.raises(ValueError):
        sepbound.item_sampling_bound(0.5, 0.4, 0)
    with pytest.raises(ValueError):
        sepbound.item_sampling_bound(0.5, 0.4, 100, rho=1.5)
    assert sepbound.to_proportions([72.4, 71.8]).tolist() == pytest.approx([0.724, 0.718])
    assert sepbound.to_proportions([0.724, 0.718]).tolist() == [0.724, 0.718]


# ---------------------------------------------------------------- pure-Python normal fallback

@pytest.mark.parametrize("p", [1e-9, 1e-4, 0.00625, 0.025, 0.3, 0.5, 0.975, 0.99375, 1 - 1e-6])
def test_ppf_fallback_matches_reference(p):
    try:
        from scipy import stats
    except Exception:
        pytest.skip("scipy not installed")
    assert _norm.ppf_py(p) == pytest.approx(float(stats.norm.ppf(p)), rel=1e-9, abs=1e-12)


def test_cdf_fallback():
    assert _norm.cdf_py(1.96) == pytest.approx(0.9750021048517795, abs=1e-12)
    assert float(_norm.cdf_py(np.inf)) == 1.0


# ---------------------------------------------------------------- census reproductions

def _census():
    rows = list(csv.DictReader(open(RESULTS / "census.csv", newline="")))
    ents = list(csv.DictReader(open(RESULTS / "census_entries.csv", newline="")))
    par = json.load(open(RESULTS / "census_parameters.json"))
    return rows, ents, par


def _board(ents, lb):
    kept = [(float(e["score"]), e["entry"]) for e in ents if e["leaderboard"] == lb and e["status"] == "kept"]
    kept.sort(key=lambda t: (-t[0], t[1]))
    return [e for _, e in kept], np.array([s for s, _ in kept])


def _row(rows, lb, bound, param=None):
    hit = [r for r in rows if r["leaderboard"] == lb and r["set"] == "primary" and r["bound"] == bound
           and (param is None or abs(float(r["param"]) - param) < 1e-12)]
    assert len(hit) == 1
    return hit[0]


# (leaderboard, n) as in the census
BOARDS = [("SWE-bench Lite", 300), ("Terminal-Bench 2.0", 89), ("TheAgentCompany", 175), ("SWE-bench Verified", 500)]


@needs_census
@pytest.mark.parametrize("lb,n", BOARDS)
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75])
def test_census_fixed_rho(lb, n, rho):
    rows, ents, _ = _census()
    names, s = _board(ents, lb)
    ref = _row(rows, lb, "a", rho)
    assert len(s) == int(ref["n_entries"])
    t = sepbound.tiers(s, n, rho, names=names)
    assert len(t) == int(ref["n_tiers"])
    b12 = 100 * sepbound.item_sampling_bound(s[0], s[1], n, rho)
    assert b12 == pytest.approx(float(ref["bound_1_2_pp"]), rel=1e-9)
    assert sepbound.separable(s[0], s[1], n, rho) == (ref["top1_vs_2_separable"] == "True")
    assert sepbound.separable(s[0], s[4], n, rho) == (ref["top1_vs_5_separable"] == "True")
    assert 100 * (s[0] - s[1]) == pytest.approx(float(ref["gap_1_2_pp"]), abs=1e-9)


@needs_census
@pytest.mark.parametrize("lb,n", BOARDS)
def test_census_headline_rho_tiers_and_adjacent(lb, n):
    rows, ents, par = _census()
    names, s = _board(ents, lb)
    rho = par["rho_used"][lb]
    ref = _row(rows, lb, "a_emp")
    assert float(ref["param"]) == pytest.approx(rho)
    assert len(sepbound.tiers(s, n, rho, names=names)) == int(ref["n_tiers"])
    below = [not sepbound.separable(s[i], s[i + 1], n, rho) for i in range(len(s) - 1)]
    assert sum(below) == int(ref["n_adjacent_below"])
    assert sepbound.separable(s[0], s[1], n, rho) == (ref["top1_vs_2_separable"] == "True")


@needs_census
def test_census_known_verdicts():
    # SWE-bench Lite: #1 and #2 are not separable at any census rho; TheAgentCompany: separable once rho >= 0.25
    rows, ents, par = _census()
    _, lite = _board(ents, "SWE-bench Lite")
    _, tac = _board(ents, "TheAgentCompany")
    rho_tac = par["rho_used"]["TheAgentCompany"]
    assert not sepbound.separable(lite[0], lite[1], 300, par["rho_used"]["SWE-bench Lite"])
    assert not sepbound.separable(tac[0], tac[1], 175, 0.0)
    assert sepbound.separable(tac[0], tac[1], 175, rho_tac)


@needs_census
def test_census_leader_quantities_consistent():
    # leader set is the leader plus the entries below the bound; Bonferroni set contains the uncorrected set
    _, ents, par = _census()
    names, s = _board(ents, "TheAgentCompany")
    rho = par["rho_used"]["TheAgentCompany"]
    unc = sepbound.leader_set(s, 175, rho, bonferroni=False)
    bon = sepbound.leader_set(s, 175, rho)
    assert set(unc) <= set(bon) and unc[0] == 0
    manual = [0] + [j for j in range(1, len(s)) if not sepbound.separable(s[0], s[j], 175, rho)]
    assert unc == manual
    p = sepbound.leader_posterior(s, 175, rho, draws=20000, seed=0)
    assert 0.5 < p < 1.0


# ---------------------------------------------------------------- CLI

def test_cli_runs_and_autodetects_percent():
    out = subprocess.run([sys.executable, "-m", "sepbound", "--n", "100", "--rho", "0.5", "50", "45", "40", "35", "30"],
                         cwd=PKG, capture_output=True, text=True, check=True).stdout
    assert "#1 vs #2: gap 5.00 pp, bound 9.78 pp -> below the bound" in out
    assert "#1 vs #5: gap 20.00 pp, bound 9.42 pp -> separable" in out
    assert "Tiers: 3" in out
    out2 = subprocess.run([sys.executable, "-m", "sepbound", "--n", "100", "--rho", "0.5", "0.5", "0.45", "0.4", "0.35", "0.3"],
                          cwd=PKG, capture_output=True, text=True, check=True).stdout
    assert out2 == out


def test_cli_error_exit():
    r = subprocess.run([sys.executable, "-m", "sepbound", "--n", "0", "50", "40"], cwd=PKG, capture_output=True, text=True)
    assert r.returncode == 2 and "error" in r.stderr


# ---------------------------------------------------------------- README table

def test_readme_rule_of_thumb_table_matches_package():
    text = (PKG / "README.md").read_text(encoding="utf-8")
    for n in (50, 100, 200, 500, 1000):
        b0 = 100 * sepbound.item_sampling_bound(0.5, 0.5, n, 0.0)
        b5 = 100 * sepbound.item_sampling_bound(0.5, 0.5, n, 0.5)
        assert f"| {n} | {b0:.1f} | {b5:.1f} |" in text
