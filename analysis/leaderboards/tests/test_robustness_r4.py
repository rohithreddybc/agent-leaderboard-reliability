"""Tests for robustness_r4.py. Core functions on tiny synthetic inputs; one integration test reads the committed
census outputs (read only; conftest.py checks that nothing under results/ changes). Files go to pytest temp dirs."""
import math

import numpy as np
import pandas as pd
import pytest

import census as C
import robustness_r3 as R3
import robustness_r4 as R4

NEEDS_RAW = pytest.mark.skipif(not (R4.HERE.parents[1] / "data" / "raw").exists(),
                               reason="needs data/raw, which is not redistributed (licences)")


COLS = ["leaderboard", "entry", "status", "reason", "score", "k_runs", "x_attempts", "x_covered_ids_n", "x_run_files",
        "x_in_rq3_analysis_set", "x_no_reward_trials", "x_score_alt_final_field", "x_src_file", "x_trials_stated",
        "x_trials_used", "x_benchmark_tuned", "x_std_err_reported", "x_se_consistent", "x_k_assumed"]


def ent_df(rows):
    """rows: dicts with leaderboard, entry, score, k_runs and any x_ field; status defaults to kept."""
    out = []
    for r in rows:
        d = {c: "" for c in COLS}
        d.update({"status": "kept"})
        d.update({k: str(v) for k, v in r.items()})
        out.append(d)
    return pd.DataFrame(out, columns=COLS)


def make_board(scores, ks=None, n=100, rho=0.5, lb="B1"):
    scores = np.array(scores, float)
    names = [f"{lb}-e{i:02d}" for i in range(len(scores))]
    return R3.Board(lb, n, names, scores, np.ones(len(scores)) if ks is None else np.array(ks, float), rho, "own")


# ---- classification -------------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [("2+", True), ("2", True), ("3", True), ("best-of-3", True), ("pass@5", True),
                                          ("1", False), ("None", False), ("", False), ("nan", False)])
def test_attempts_tag(tag, expected):
    assert R4._attempts_gt1(tag) is expected


def test_classify_every_class():
    ent = ent_df([
        {"leaderboard": "S", "entry": "a", "score": 0.5, "k_runs": 1, "x_attempts": "1"},
        {"leaderboard": "S", "entry": "b", "score": 0.5, "k_runs": 1, "x_attempts": "None"},
        {"leaderboard": "S", "entry": "c", "score": 0.5, "k_runs": 1, "x_attempts": "2+"},
        {"leaderboard": "S", "entry": "d", "score": 0.5, "k_runs": 1, "x_attempts": "2"},
        {"leaderboard": "T", "entry": "e", "score": 0.5, "k_runs": 5},
        {"leaderboard": "T", "entry": "f", "score": 0.5, "k_runs": 1},
        {"leaderboard": "M", "entry": "g", "score": 0.5, "k_runs": 1, "x_k_assumed": "1.0"},
        {"leaderboard": "A", "entry": "h", "score": 0.5, "k_runs": 1, "x_trials_stated": "1"},
        {"leaderboard": "A", "entry": "i", "score": 0.5, "k_runs": 1},               # blank trial count on a board that carries one
        {"leaderboard": "Q", "entry": "j", "score": 0.5, "k_runs": 4, "x_trials_used": "4"},
        {"leaderboard": "Q", "entry": "k", "score": 0.5, "k_runs": 4, "x_trials_stated": "", "x_trials_used": "4"},
        {"leaderboard": "S", "entry": "x", "score": 0.5, "k_runs": 1, "status": "excluded"},
    ])
    c = R4.classify_entries(ent).set_index("entry")["protocol"].to_dict()
    assert c == {"a": "single-run", "b": "single-run", "c": "selection-based", "d": "selection-based", "e": "averaged",
                 "f": "single-run", "g": "unknown", "h": "single-run", "i": "unknown", "j": "averaged", "k": "averaged"}
    # strict variant: a board whose single run is only assumed becomes unknown
    c2 = R4.classify_entries(ent, assumed_boards=("T",)).set_index("entry")["protocol"].to_dict()
    assert c2["f"] == "unknown" and c2["e"] == "averaged"      # a stated k > 1 stays averaged
    assert c2["a"] == "single-run"


def test_selection_beats_unknown_and_averaged():
    ent = ent_df([{"leaderboard": "S", "entry": "a", "score": 0.5, "k_runs": 5, "x_attempts": "2+", "x_k_assumed": "1"}])
    assert R4.classify_entries(ent)["protocol"].iloc[0] == "selection-based"


def test_board_counts_and_subset_drop_small_boards():
    ent = ent_df([{"leaderboard": "B1", "entry": f"B1-e{i:02d}", "score": 0.9 - 0.1 * i, "k_runs": 1} for i in range(4)]
                 + [{"leaderboard": "B1", "entry": "B1-e04", "score": 0.4, "k_runs": 1, "x_attempts": "2+"}]
                 + [{"leaderboard": "B2", "entry": "B2-e00", "score": 0.9, "k_runs": 1},
                    {"leaderboard": "B2", "entry": "B2-e01", "score": 0.8, "k_runs": 1, "x_attempts": "2+"},
                    {"leaderboard": "B2", "entry": "B2-e02", "score": 0.7, "k_runs": 1, "x_k_assumed": "1"}])
    b1 = make_board([0.9, 0.8, 0.7, 0.6, 0.4], lb="B1")
    b2 = make_board([0.9, 0.8, 0.7], lb="B2")
    cls = R4.classify_entries(ent)
    cnt = R4.board_protocol_counts(cls, ["B1", "B2"]).set_index("leaderboard")
    assert cnt.loc["B1", ["single-run", "selection-based", "dropped", "retained"]].tolist() == [4, 1, 1, 4]
    assert cnt.loc["B2", ["single-run", "selection-based", "unknown", "dropped", "retained"]].tolist() == [1, 1, 1, 2, 1]
    assert bool(cnt.loc["B1", "board_kept"]) and not bool(cnt.loc["B2", "board_kept"])
    sub = R4.subset_boards([b1, b2], cls)
    assert [b.lb for b in sub] == ["B1"]
    assert sub[0].names == ["B1-e00", "B1-e01", "B1-e02", "B1-e03"]
    np.testing.assert_allclose(sub[0].scores, [0.9, 0.8, 0.7, 0.6])
    assert sub[0].n == b1.n and sub[0].rho == b1.rho


def test_duplicate_entry_names_are_refused():
    ent = ent_df([{"leaderboard": "B1", "entry": "dup", "score": 0.9, "k_runs": 1}, {"leaderboard": "B1", "entry": "dup", "score": 0.8, "k_runs": 1}])
    b = make_board([0.9, 0.8], lb="B1")
    b.names[:] = ["dup", "dup"]
    with pytest.raises(ValueError):
        R4.subset_boards([b], R4.classify_entries(ent))


# ---- k-adjusted evaluation --------------------------------------------------------------------------

def test_kadj_equals_headline_when_all_single_run_and_shrinks_with_k():
    s = [0.72, 0.70, 0.66, 0.60, 0.59, 0.40]
    b1 = make_board(s)
    base = R3.ref_rows([b1])[0]
    ka = R4.kadj_rows([b1], 0.369)[0]
    assert (ka["n_tiers"], ka["n_adjacent_below"]) == (base["n_tiers"], base["n_adjacent_below"])
    assert ka["bound12"] == pytest.approx(base["bound12"])
    b5 = make_board(s, ks=[5] * 6)
    k5 = R4.kadj_rows([b5], 0.369)[0]
    assert k5["bound12"] < base["bound12"]
    # hand computation of the k-adjusted bound for the top pair
    a = math.sqrt(1 - 0.369 * (1 - 1 / 5))
    s1, s2 = C.se(0.72, 100), C.se(0.70, 100)
    expect = 1.96 * math.sqrt((s1 * a) ** 2 + (s2 * a) ** 2 - 2 * 0.5 * s1 * s2)
    assert k5["bound12"] == pytest.approx(expect)


def test_rho_zero_bound_is_wider_than_positive_rho():
    b = make_board([0.72, 0.70, 0.66, 0.60, 0.59, 0.40], rho=0.6)
    r = R4.kadj_rows([b], 0.369)[0]
    r0 = R4.kadj_rows([b], 0.369, rho_zero=True)[0]
    assert r0["bound12"] > r["bound12"] and r0["n_tiers"] <= r["n_tiers"]


def test_headline_line_denominators():
    big = make_board([0.8, 0.7, 0.6, 0.5, 0.4, 0.3], lb="big")
    small = make_board([0.8, 0.7, 0.6], lb="small")
    line = R4.headline_line(R4.kadj_rows([big, small], 0.369))
    assert line["boards"] == 2 and line["boards_with_5"] == 1
    assert line["entries"] == 9 and line["adj_pairs"] == 7


# ---- C lookup ------------------------------------------------------------------------------------

def test_item_bound_values():
    assert float(R4.item_bound_pp(100, 0.5, 0.0)) == pytest.approx(100 * 1.96 * math.sqrt(2 * 0.25 / 100))
    assert float(R4.item_bound_pp(2294, 0.3, 0.5)) == pytest.approx(100 * 1.96 * math.sqrt(2 * 0.21 * 0.5 / 2294))
    # rho = 1 removes all item-sampling noise; the bound is the C.bound_a value at equal scores
    assert float(R4.item_bound_pp(100, 0.4, 1.0)) == 0.0
    assert float(R4.item_bound_pp(120, 0.4, 0.3)) == pytest.approx(100 * float(C.bound_a(0.4, 0.4, 120, 0.3)))


def test_lookup_table_shape_symmetry_and_mdd():
    lk = R4.lookup_table()
    assert len(lk) == 2 * 3 * 8 and set(lk["n"]) == set(R4.LOOKUP_N) and set(lk["rho"]) == {0.0, 0.5}
    np.testing.assert_allclose(lk["mdd_80pct_power_pp"], 1.43 * lk["bound_pp"])
    g = lk.set_index(["rho", "P", "n"])["bound_pp"]
    assert g[(0.0, 0.3, 50)] == pytest.approx(g[(0.0, 0.7, 50)])
    assert g[(0.5, 0.5, 89)] == pytest.approx(g[(0.0, 0.5, 89)] * math.sqrt(0.5))
    assert g[(0.0, 0.5, 50)] > g[(0.0, 0.5, 2294)]
    assert round(g[(0.0, 0.5, 100)], 2) == 13.86


# ---- D leader set ----------------------------------------------------------------------------------

def test_leader_set_excludes_entry_one_and_size_adds_one():
    b = make_board([0.70, 0.69, 0.68, 0.50, 0.30], n=100, rho=0.0)
    others = R3.leader_set(b.scores, b.n, b.rho, 1.96)
    assert len(others) == 4                                 # entries 2..m only
    D = R4.sectionD([b], None)
    t = D["table"].iloc[0]
    assert t["set_size_uncorrected"] == t["others_uncorrected"] + 1 == int(others.sum()) + 1
    assert D["excludes_entry1"] and D["uncorrected"]["size_equals_others_plus_1"] and D["bonferroni"]["size_equals_others_plus_1"]
    assert D["uncorrected"]["size_min"] >= 1


# ---- integration on the committed census (read only) -------------------------------------------------

@NEEDS_RAW
def test_committed_census_subset_and_report(tmp_path, monkeypatch):
    res = R4.HERE / "results"
    boards, par = R3.load_boards(res)
    assert R3.reproduce(boards, res / "census.csv")["reproduced"]
    ent = pd.read_csv(res / "census_entries.csv", dtype=str, keep_default_na=False)
    B = R4.sectionB(boards, ent, float(par["f_tb2_median"]))
    assert sum(B["protocol_totals"].values()) == 775 and B["protocol_totals"]["selection-based"] == 94
    assert B["protocol_totals"]["unknown"] == 28 and B["dropped_boards"] == ["MLE-bench"]
    s = B["summary"]
    assert (s["rows_all_kadj"]["sep12_not"], s["rows_all_kadj"]["sep15_not"], s["rows_all_kadj"]["tiers_median"]) == (16, 8, 7.0)   # round 3, section 6
    assert (s["rows_sub_kadj"]["sep12_not"], s["rows_sub_kadj"]["boards"]) == (15, 21)
    D = R4.sectionD(boards, R4.ANALYSIS / "results" / "robustness_2026-10-19_sec11_leader_set.csv")
    assert D["matches_committed"] is True and D["excludes_entry1"]
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--out", str(out)])
    R4.main()
    names = sorted(p.name for p in out.iterdir())
    assert all(n.startswith("robustness_2026-10-20") for n in names)
    assert "robustness_2026-10-20.md" in names and "robustness_2026-10-20_secC_lookup.csv" in names
    md = (out / "robustness_2026-10-20.md").read_text(encoding="utf-8")
    assert "protocol-compatible subset" in md and "excludes entry 1" in md

# ---- E bound versus paired test ---------------------------------------------------------------------

def _pairs_df(rows):
    return pd.DataFrame(rows, columns=["score_hi", "score_lo", "n_tasks", "rho_used"])


def test_bound_verdicts_follow_the_census_rule():
    # n = 100, P = 0.5 and 0.4: SE1 = 0.05, SE2 = sqrt(0.24 / 100); rho = 0 bound = 1.96 sqrt(0.0025 + 0.0024) = 0.1372
    df = _pairs_df([(0.50, 0.40, 100, 0.5),     # gap 0.10: below the bound at rho = 0 (0.137) and at rho = 0.5 (0.0955 < 0.10: separable)
                    (0.50, 0.50, 100, 0.0),     # tie: never separable
                    (0.80, 0.50, 100, 0.9)])    # gap 0.30: separable at any rho
    assert R4.bound_verdicts(df).tolist() == [True, False, True]
    assert R4.bound_verdicts(df, 0.0).tolist() == [False, False, True]
    assert R4.bound_verdicts(df, 0.5).tolist() == [True, False, True]
    # a wider bound (smaller rho) can only remove separable verdicts
    assert (R4.bound_verdicts(df, 0.0) <= R4.bound_verdicts(df, 0.5)).all()


def test_agreement_stats_hand_example():
    bound = [True, True, True, False, False, False, False, False]
    test = [True, True, False, False, False, False, False, False]
    k = R4.agreement_stats(bound, test)
    assert (k["n_both"], k["n_bound_only"], k["n_test_only"], k["n_neither"], k["n_either"]) == (2, 1, 0, 5, 3)
    assert k["agree"] == pytest.approx(7 / 8) and k["agree_among_either"] == pytest.approx(2 / 3)
    pe = (3 / 8) * (2 / 8) + (5 / 8) * (6 / 8)
    assert k["kappa"] == pytest.approx((7 / 8 - pe) / (1 - pe))
    assert R4.agreement_stats([True, True], [True, True])["kappa"] != R4.agreement_stats([True, True], [True, True])["kappa"]   # nan: one category
    assert R4.agreement_stats([False, False], [False, False])["agree_among_either"] != R4.agreement_stats([False, False], [False, False])["agree_among_either"]


def test_sectionE_on_committed_pairs_reproduces_round3_and_orders_rho():
    par = pd.read_json(R4.HERE / "results" / "census_parameters.json", typ="series")
    E = R4.sectionE(R4.ANALYSIS / "results" / "robustness_2026-10-06_mcnemar_pairs.csv", float(par["pooled_median_rho"]))
    assert E["n_pairs"] == 401 and E["reproduces_file"]
    est = E["estimated"]                                   # round 3, section 13
    assert (est["n_both"], est["n_neither"], est["n_bound_only"], est["n_test_only"]) == (21, 373, 7, 0)
    assert est["kappa"] == pytest.approx(0.848, abs=5e-4) and est["agree_among_either"] == pytest.approx(0.75)
    # a smaller rho widens the bound: no more bound-separable pairs than at the pooled rho
    assert E["rho0"]["n_bound_sep"] <= E["pooled"]["n_bound_sep"]
    for key in ("estimated", "rho0", "pooled"):
        v = E[key]
        assert v["n"] == 401 and v["n_both"] + v["n_bound_only"] + v["n_test_only"] + v["n_neither"] == 401


# ---- F full primary analysis ----------------------------------------------------------------------------

def test_primary_variants_drop_only_appworld():
    bs = [make_board([0.9, 0.8], lb="AppWorld test_normal"), make_board([0.9, 0.7], lb="AppWorld test_challenge"),
          make_board([0.9, 0.6], lb="WebArena (Google Sheet)")]
    v = R4.primary_variants(bs)
    assert [b.lb for b in v["P1"]] == [b.lb for b in bs]
    assert [b.lb for b in v["P2"]] == ["WebArena (Google Sheet)"]


def test_top_counts_ties_and_boards_without_five_entries():
    tie = make_board([0.60, 0.60, 0.50, 0.40, 0.30, 0.20], lb="tie")              # #1 = #2 (tied, not separable)
    clear = make_board([0.90, 0.50, 0.40, 0.30, 0.20, 0.10], lb="clear")          # both top checks separable
    near = make_board([0.60, 0.59, 0.58, 0.57, 0.56, 0.20], lb="near")            # #1 vs #2 and #1 vs #5 not separable, no tie
    small = make_board([0.8, 0.8, 0.1], lb="small")                              # tie, fewer than 5 entries
    rows = R4.rows_se([tie, clear, near, small], "single", 0.369)
    c = R4.top_counts(rows)
    assert (c["boards"], c["boards_with_5"], c["sep12_not"], c["tied12"], c["sep15_not"], c["tied15"]) == (4, 3, 3, 2, 1, 0)


def test_bound_fn_se_choices_agree_for_single_run_and_differ_for_averaged():
    s = [0.72, 0.70, 0.66, 0.60, 0.59, 0.40]
    b1, b5 = make_board(s), make_board(s, ks=[5] * 6)
    r1 = R4.rows_se([b1], "single", 0.369)[0]
    assert R4.rows_se([b1], "kadj", 0.369)[0]["bound12"] == pytest.approx(r1["bound12"])
    assert R4.rows_se([b5], "single", 0.369)[0]["bound12"] == pytest.approx(r1["bound12"])    # (i) ignores k
    assert R4.rows_se([b5], "kadj", 0.369)[0]["bound12"] < r1["bound12"]
    assert R4.rows_se([b5], "single", 0.369, 0.0)[0]["bound12"] > r1["bound12"]               # an explicit rho overrides the headline
    with pytest.raises(ValueError):
        R4.bound_fn_se(b1, "other", 0.369)


def test_nonpositive_variance_needs_rho_above_the_scale_product():
    b = make_board([0.60, 0.60, 0.50], ks=[5, 5, 5])
    assert R4.nonpositive_variance_pairs([b], 0.5, 0.369) == {"pairs": 2, "nonpositive": 0}       # a^2 = 0.705 > 0.5
    assert R4.nonpositive_variance_pairs([b], 0.9, 0.369)["nonpositive"] >= 1                     # 2 (a^2 - rho) s^2 < 0 for equal scores
    one = make_board([0.60, 0.60, 0.50])
    assert R4.nonpositive_variance_pairs([one], 0.9, 0.369)["nonpositive"] == 0                   # k = 1: a^2 = 1 > rho
    assert float(R4.kadj_variance(0.6, 0.6, 1, 1, 100, 0.9, 0.369)) > 0
    assert float(R4.kadj_variance(0.6, 0.6, 5, 5, 100, 0.9, 0.369)) < 0


def test_order_confidence_values_and_tie():
    from scipy.stats import norm
    b = make_board([0.60, 0.60, 0.50, 0.40], rho=0.5)
    oc = R4.order_confidence_F([b], "single", 0.369)
    assert oc["c12"][0] == 0.5                                                  # a tie carries no order information
    c23 = norm.cdf(0.10 / (float(C.bound_a(0.60, 0.50, 100, 0.5)) / 1.96))
    c34 = norm.cdf(0.10 / (float(C.bound_a(0.50, 0.40, 100, 0.5)) / 1.96))
    assert oc["board_shares"][0] == pytest.approx(np.mean([True, c23 < 0.8, c34 < 0.8]))
    assert oc["pairs"] == 3 and oc["min_c12"] == 0.5 and oc["n_c12_lt"] == 1
    assert oc["pairs_lt"] == int(1 + (c23 < 0.8) + (c34 < 0.8))


def test_order_confidence_kadj_is_higher_for_averaged_entries():
    b = make_board([0.60, 0.57, 0.50, 0.45], ks=[5, 5, 5, 5])
    assert R4.order_confidence_F([b], "kadj", 0.369)["c12"][0] > R4.order_confidence_F([b], "single", 0.369)["c12"][0]


def test_simulator_matches_round3_when_scale_is_one_and_flags_non_psd():
    mu = np.array([0.7, 0.6, 0.5, 0.4])
    x, psd = R4.simulate_posterior_draws(mu, 100, 0.5, np.ones(4), 50, np.random.default_rng(1))
    np.testing.assert_allclose(x, R3.simulate_scores(mu, 100, 0.5, 50, np.random.default_rng(1)))
    assert psd is True
    # a^2 < rho: not positive semi-definite, eigenvalue fallback; the empirical covariance is close to the clipped target
    ak = np.full(4, math.sqrt(0.7))                                              # a^2 = 0.7 < rho = 0.8
    x, psd = R4.simulate_posterior_draws(mu, 1000, 0.8, ak, 40000, np.random.default_rng(2))
    assert psd is False
    se = np.sqrt(mu * (1 - mu) / 1000)
    w, V = np.linalg.eigh(0.8 * np.outer(se, se) + np.diag((ak * se) ** 2 - 0.8 * se ** 2))
    target = (V * np.clip(w, 0, None)) @ V.T
    np.testing.assert_allclose(np.cov(x.T), target, atol=0.15 * se.max() ** 2)
    with pytest.raises(ValueError):
        R4.simulate_posterior_draws(mu, 100, 1.0, np.ones(4), 5, np.random.default_rng(0))


def test_leader_probability_tied_set_counts_as_leader():
    tied = make_board([0.80, 0.80, 0.10], n=1000, rho=0.5)
    p = R4.p_leader_highest(tied, "single", 0.369, 4000, np.random.default_rng(3))
    assert p["n_tied_top"] == 2 and p["psd"] and p["p"] > 0.999                   # either tied entry counts; the third never leads
    draws = R4.simulate_posterior_draws(tied.scores, 1000, 0.5, np.ones(3), 4000, np.random.default_rng(3))[0]
    strict = float((draws.argmax(axis=1) == 0).mean())
    assert 0.4 < strict < 0.6                                                      # the single entry at position 1 would get about 0.5
    clear = make_board([0.90, 0.30, 0.20], n=1000)
    assert R4.p_leader_highest(clear, "single", 0.369, 500, np.random.default_rng(4))["p"] == 1.0
    close = make_board([0.60, 0.59, 0.58], n=100)
    pc = R4.p_leader_highest(close, "single", 0.369, 4000, np.random.default_rng(5))["p"]
    assert 0.3 < pc < 0.7
    assert R4.p_leader_highest(close, "single", 0.369, 4000, np.random.default_rng(5))["p"] == pc      # same seed, same answer


def test_leader_set_F_matches_round3_and_kadj_shrinks_the_set():
    b = make_board([0.70, 0.69, 0.68, 0.50, 0.30], n=100, rho=0.0)
    np.testing.assert_array_equal(R4.leader_set_F(b, "single", 0.369, 1.96), R3.leader_set(b.scores, b.n, b.rho, 1.96))
    np.testing.assert_array_equal(R4.leader_set_F(b, "kadj", 0.369, 1.96), R4.leader_set_F(b, "single", 0.369, 1.96))   # all k = 1
    b5 = make_board([0.70, 0.69, 0.68, 0.50, 0.30], n=100, rho=0.0, ks=[5] * 5)
    assert R4.leader_set_F(b5, "kadj", 0.369, 1.96).sum() <= R4.leader_set_F(b5, "single", 0.369, 1.96).sum()
    t = make_board([0.70, 0.70, 0.50], n=10000, rho=0.5)                            # a tied entry is always in the set
    assert R4.leader_set_F(t, "single", 0.369, 1.96)[0]
    assert R4.leader_set_F(t, "single", 0.369, 3.0).sum() >= R4.leader_set_F(t, "single", 0.369, 1.96).sum()


def test_sectionF_core_structure_and_seed_stability():
    b1 = make_board([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], lb="A")
    b2 = make_board([0.6, 0.6, 0.5, 0.4, 0.3, 0.2], lb="AppWorld test_normal")
    allb = [b1, b2]
    out = R4.sectionF_core(R4.primary_variants(allb), allb, 0.369, reps=300)
    assert set(out["boards"]["variant"]) == {"P1", "P2"} and len(out["boards"]) == (2 + 1) * 2
    assert len(out["sweep"]) == 2 * 2 * (1 + len(R4.F_RHOS))
    p = out["boards"].pivot_table(index=["leaderboard", "se"], columns="variant", values="p_leader_highest")
    assert p.loc[("A", "single"), "P1"] == p.loc[("A", "single"), "P2"]          # a board has the same stream in both variants
    out2 = R4.sectionF_core(R4.primary_variants(allb), allb, 0.369, reps=300)
    pd.testing.assert_frame_equal(out["boards"], out2["boards"])
    sw = out["sweep"]
    head = sw[(sw["variant"] == "P1") & (sw["se"] == "single") & (sw["rho_setting"] == "headline")].iloc[0]
    assert (head["boards"], head["tied12"], head["sep12_not"]) == (2, 1, 1)


def test_subset_test_pairs_use_subset_adjacency_and_agreement_counts():
    rng = np.random.default_rng(7)
    n = 100
    vecs = [(rng.random(n) < 0.5).astype(float) for _ in range(4)]
    names = ["e0", "e1", "e2", "e3"]
    outc = {"B1": dict(zip(names, vecs))}
    scores = [float(v.mean()) for v in vecs]
    order = sorted(range(4), key=lambda i: (-scores[i], names[i]))
    full = R3.Board("B1", n, [names[i] for i in order], np.array([scores[i] for i in order]), np.ones(4), 0.5, "own")
    sub = R3.Board("B1", n, [full.names[0], full.names[2], full.names[3]], full.scores[[0, 2, 3]], np.ones(3), 0.5, "own")
    pr = R4.subset_test_pairs([sub], outc, 0.369)
    assert len(pr) == 2 and set(pr["pair"]) == {"adjacent"}
    assert pr["entry_hi"].tolist() == [sub.names[0], sub.names[1]] and pr["entry_lo"].tolist() == [sub.names[1], sub.names[2]]
    assert (pr["rank_hi"].tolist(), pr["rank_lo"].tolist()) == ([1, 2], [2, 3])        # adjacent in the subset, not in the full order
    assert (pr["bound_separable"].to_numpy() == pr["bound_kadj_separable"].to_numpy()).all()    # all k = 1
    ag = R4.agreement_F(pr)
    assert ag["pairs"] == 2 and ag["boards"] == 1
    assert ag["single"]["n_both"] + ag["single"]["n_bound_only"] + ag["single"]["n_test_only"] + ag["single"]["n_neither"] == 2
    other = R3.Board("B2", n, ["x", "y"], np.array([0.6, 0.5]), np.ones(2), 0.5, "pooled")      # no outcomes: skipped
    assert len(R4.subset_test_pairs([sub, other], outc, 0.369)) == 2


def test_subset_test_pairs_kadj_bound_is_narrower_for_averaged_entries():
    rng = np.random.default_rng(8)
    n = 50
    vecs = [np.round(rng.random(n) * 5) / 5 for _ in range(3)]          # success shares of 5 runs
    names = ["e0", "e1", "e2"]
    scores = [float(v.mean()) for v in vecs]
    order = sorted(range(3), key=lambda i: (-scores[i], names[i]))
    sub = R3.Board("B1", n, [names[i] for i in order], np.array([scores[i] for i in order]), np.full(3, 5.0), 0.3, "own")
    pr = R4.subset_test_pairs([sub], {"B1": dict(zip(names, vecs))}, 0.369)
    assert (pr["bound_kadj_pp"] < pr["bound_pp"]).all() and set(pr["test"]) == {"paired t"}


@NEEDS_RAW
def test_sectionF_on_committed_census_matches_sectionB_and_round3():
    res = R4.HERE / "results"
    boards, par = R3.load_boards(res)
    ent = pd.read_csv(res / "census_entries.csv", dtype=str, keep_default_na=False)
    f = float(par["f_tb2_median"])
    B = R4.sectionB(boards, ent, f)
    outc, _, _ = R3.load_outcomes(R4.HERE.parents[1], boards)
    gate = R4.gate_validate_full(boards, outc, R4.ANALYSIS / "results" / "robustness_2026-10-06_mcnemar_pairs.csv")
    assert gate == {"pairs": 401, "reproduced": True}
    F = R4.sectionF(B["subset"], boards, f, outc, reps=300, gate=gate)
    assert len(F["variants"]["P1"]) == 21 and len(F["variants"]["P2"]) == 19
    assert not any(lb.startswith("AppWorld") for lb in F["variants"]["P2"])
    sw = F["sweep"].set_index(["variant", "se", "rho_setting"]).sort_index()
    s = B["summary"]
    assert sw.loc[("P1", "kadj", "headline"), "sep12_not"] == s["rows_sub_kadj"]["sep12_not"] == 15
    assert sw.loc[("P1", "kadj", "headline"), "sep15_not"] == s["rows_sub_kadj"]["sep15_not"] == 6
    assert sw.loc[("P1", "single", "headline"), "sep12_not"] == s["rows_sub_headline_nok"]["sep12_not"] == 16
    assert sw.loc[("P1", "kadj", "0"), "sep12_not"] == s["rows_sub_kadj_rho0"]["sep12_not"] == 17
    assert F["summary"][("P1", "kadj")]["head"]["adj_below"] == s["rows_sub_kadj"]["adj_below"]
    assert F["summary"][("P1", "single")]["head"]["tiers_median"] == s["rows_sub_headline_nok"]["tiers_median"]
    Bs = R4.sectionB(boards, ent, f, assumed_boards=("AppWorld test_normal", "AppWorld test_challenge"))      # same boards as P2
    assert F["summary"][("P2", "kadj")]["head"]["adj_below"] == Bs["summary"]["rows_sub_kadj"]["adj_below"]
    one = sw.xs(("P1", "single"))["sep12_not"]                                                                  # a larger rho narrows the bound
    assert one["0"] >= one["0.25"] >= one["0.5"] >= one["0.75"] >= one["0.9"]
    assert R4.nonpositive_variance_pairs(B["subset"], 0.5, f)["nonpositive"] == 0 and max(x.rho for x in B["subset"]) < 1 - f
    assert F["agreement"]["pairs"] == len(F["test_pairs"]) and F["agreement"]["boards"] == 7
    assert (F["boards"]["n_tied_top"] >= 1).all() and F["boards"]["p_leader_highest"].between(0, 1).all()
    assert F["summary"][("P1", "single")]["boards_not_psd"] == 0 and F["summary"][("P1", "kadj")]["boards_not_psd"] == 0


def test_nonpositive_variance_ignores_pairs_with_a_zero_se():
    b = make_board([0.5, 0.0, 0.0], ks=[1, 1, 1])
    assert R4.nonpositive_variance_pairs([b], 0.5, 0.369) == {"pairs": 2, "nonpositive": 0}      # (0, 0) has variance 0 at any rho
