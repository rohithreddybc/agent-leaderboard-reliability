"""Tests for census.py. Synthetic inputs only; files are written to pytest temp dirs."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import census as C


# ---- bounds --------------------------------------------------------------------------

def test_se_binomial():
    assert C.se(0.5, 100) == pytest.approx(0.05)
    assert C.se(0.0, 100) == 0.0 and C.se(1.0, 100) == 0.0


def test_bound_a_unpaired_equal_scores():
    # SE1 = SE2 = 0.05 -> 1.96 * sqrt(2 * 0.0025)
    assert C.bound_a(0.5, 0.5, 100, 0.0) == pytest.approx(1.96 * math.sqrt(0.005))


def test_bound_a_rho_one_equal_se_is_zero_and_monotone_in_rho():
    assert C.bound_a(0.4, 0.4, 200, 1.0) == pytest.approx(0.0)
    b = [float(C.bound_a(0.6, 0.5, 300, r)) for r in C.RHOS]
    assert b == sorted(b, reverse=True)


def test_bound_a_unequal_scores_formula():
    s1, s2 = math.sqrt(0.9 * 0.1 / 500), math.sqrt(0.5 * 0.5 / 500)
    for rho in (0.0, 0.5):
        want = 1.96 * math.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
        assert C.bound_a(0.9, 0.5, 500, rho) == pytest.approx(want)


def test_bound_b_formula_and_cap():
    n, v = 89, 0.0753
    assert C.bound_b(0.5, 0.5, n, v) == pytest.approx(1.96 * math.sqrt(2 * v / n))
    # near the ceiling P(1-P) = 0.0099 < v, so the capped bound is smaller than the uncapped one
    capped = C.bound_b(0.99, 0.5, n, v, cap=True)
    assert capped == pytest.approx(1.96 * math.sqrt(2 * 0.0099 / n))
    assert capped < C.bound_b(0.99, 0.5, n, v, cap=False)
    assert C.bound_b(1.0, 1.0, n, v, cap=True) == 0.0


def test_bound_b_k_adjustment():
    n, v = 80, 0.0753
    assert C.bound_b(0.5, 0.5, n, v, True, 5, 5) == pytest.approx(C.bound_b(0.5, 0.5, n, v) / math.sqrt(5))


def test_wilson_known_values():
    lo, hi = C.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = C.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert all(math.isnan(x) for x in C.wilson(0, 0))


# ---- ordering, tiers, census_config ----------------------------------------------------

def test_order_ties_by_name():
    assert C.order([0.5, 0.7, 0.5], ["b", "x", "a"]) == [1, 2, 0]


def test_greedy_tiers_hand_example():
    s = [0.90, 0.88, 0.86, 0.70, 0.69, 0.40]
    fixed = lambda i, j: 0.05  # separable iff gap >= 0.05
    # head 0.90: 0.88, 0.86 join (gaps 0.02, 0.04); 0.70 opens tier 2: 0.69 joins; 0.40 opens tier 3
    assert C.greedy_tiers(s, fixed) == 3


def test_greedy_tiers_uses_head_not_neighbour():
    s = [0.90, 0.86, 0.82, 0.78]  # neighbour gaps 0.04 < 0.05, but head-to-0.82 gap is 0.08 >= 0.05
    assert C.greedy_tiers(s, lambda i, j: 0.05) == 2


def _const(b):
    return lambda p1, p2, k1, k2: np.full(np.broadcast(p1, p2).shape, b)


def test_census_config_counts():
    scores = [0.50, 0.49, 0.30, 0.29, 0.05]
    st = C.census_config(scores, list("abcde"), [1] * 5, _const(0.05))
    # adjacent gaps: .01 .19 .01 .24 -> two below 0.05
    assert (st["n_adjacent_pairs"], st["n_adjacent_below"]) == (4, 2)
    assert st["share_adjacent_below"] == pytest.approx(0.5)
    assert (st["wilson_lo"], st["wilson_hi"]) == pytest.approx(C.wilson(2, 4))
    # all 10 pairs: below 0.05 are (a,b) and (c,d) only
    assert (st["n_all_pairs"], st["n_all_below"]) == (10, 2)
    assert st["top1_vs_2_separable"] is False          # gap 0.01 < 0.05
    assert st["top1_vs_5_separable"] is True           # gap 0.45
    assert st["n_tiers"] == 3                          # {a,b} {c,d} {e}
    assert st["gap_1_2_pp"] == pytest.approx(1.0)


def test_census_config_few_entries():
    assert C.census_config([0.5], ["a"], [1], _const(0.05)) == {"n_entries": 1}
    st = C.census_config([0.5, 0.2, 0.1, 0.05], list("abcd"), [1] * 4, _const(0.05))
    assert st["top1_vs_5_separable"] is None and math.isnan(st["gap_1_5_pp"])


# ---- granularity -----------------------------------------------------------------------

def test_recover_count():
    assert C.recover_count(24.74, 97, 4) == 96                  # 96/388 = 24.742, shown to 2 decimals
    assert C.recover_count(24.80, 97, 4) is None
    assert C.recover_count(39.69072164948454, 97, 4) == 154     # full precision
    assert C.recover_count(61.0, 50, 2, "61.00%") == 61
    assert C.recover_count(61.0, 50, 1, "61.00%") is None       # 30.5 tasks
    assert C.recover_count(72.0, 50, 1, "72.00%") == 36


# ---- loaders on synthetic files -----------------------------------------------------------

def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_load_swebench_rules(tmp_path):
    ids = ["i1", "i2", "i3", "i4"]
    mk = lambda resolved, pct=None: {
        "source": "results.json", "resolved": resolved, "covered_ids": ids,
        "meta": {"attempts": "1", "resolved_pct_reported": pct}}
    d = {
        "good": mk(["i1", "i2"]),
        "good_pct": mk(["i1"], pct=25.0),
        "bad_pct": mk(["i1"], pct=30.0),
        "dup": mk(["i1", "i1"]),
        "outside": mk(["i1", "zzz"]),
        "nofile": {"source": None, "meta": {}},
    }
    _write(tmp_path / "data/raw/leaderboards/swe-bench/swebench_verified.json", d)
    es = {e.name: e for e in C.load_swebench(tmp_path, "verified", "SWE-bench Verified", 4)}
    assert es["good"].status == "kept" and es["good"].score == 0.5
    assert es["good_pct"].status == "kept" and es["good_pct"].score == 0.25
    for k in ("bad_pct", "dup", "outside", "nofile"):
        assert es[k].status == "excluded" and es[k].score is None and es[k].reason


def test_load_hal_mini(tmp_path):
    p = tmp_path / "data/raw/leaderboards/hal"
    p.mkdir(parents=True)
    pd.DataFrame({
        "Rank": ["1", "2", "3"], "Scaffold": ["s"] * 3, "Primary": ["a", "b", "c"],
        "Accuracy": ["72.00%", "61.00% (-7.00/+7.00)", "61.00%"], "Runs": ["1", "2", "1"],
    }).to_csv(p / "swebench_verified_mini_entries.csv", index=False)
    es = C.load_hal_mini(tmp_path)
    assert [e.status for e in es] == ["kept", "kept", "excluded"]
    assert es[0].score == 0.72 and es[1].score == 0.61 and es[1].k == 2.0


def test_load_tau2_banking_filters(tmp_path):
    T = tmp_path / "data/raw/tau2"

    def sub(name, p1, ver="1.0.1"):
        _write(T / name / "submission.json", {
            "results": {"banking_knowledge": {"pass_1": p1, "pass_4": 1.0}},
            "methodology": {"tau2_bench_version": ver, "notes": "4 trials"}})

    sub("ok", 100 * 154 / 388)
    sub("old", 20.0, ver="0.2.1-dev")
    sub("voice", 20.0)
    sub("badgrain", 24.80)
    _write(T / "manifest.json", {"submissions": ["ok", "old", "badgrain"], "voice_submissions": ["voice"]})
    es = {e.name: e for e in C.load_tau2_banking(tmp_path)}
    assert es["ok"].status == "kept" and es["ok"].score == pytest.approx(154 / 388) and es["ok"].k == 4
    assert all(es[k].status == "excluded" for k in ("old", "voice", "badgrain"))


def test_load_tac_incomplete_excluded(tmp_path):
    cp = lambda r: {"checkpoints": [{"result": r, "total": 1}], "final": {"result": r, "total": 1}}
    full = {f"t{i}-image": cp(1 if i < 50 else 0) for i in range(175)}
    part = {f"t{i}-image": cp(1) for i in range(174)}
    _write(tmp_path / "data/raw/leaderboards/theagentcompany/tac_results.json",
           {"results": {"1.0.0": {"full": full, "part": part}}})
    es = {e.name: e for e in C.load_tac(tmp_path)}
    assert es["full"].score == pytest.approx(50 / 175)
    assert es["part"].status == "excluded"


# ---- driver ----------------------------------------------------------------------------

def _synthetic_entries(m=12):
    lb = "SWE-bench Lite"
    return [C.Entry(lb, f"e{i:02d}", 0.9 - 0.05 * i, 1.0, extra={"attempts": "2+" if i == 0 else "1"})
            for i in range(m)]


def test_run_rows_and_flags():
    cen, pairs = C.run(_synthetic_entries())
    lite = cen[(cen.leaderboard == "SWE-bench Lite") & (cen.set == "primary")]
    # 4 rho + 4 v x (capped, uncapped) + 4 k-adjusted + 3 f values x (single-run, k-adjusted); no empirical rho or TB2 f given
    assert len(lite) == 4 + 8 + 4 + 6 and lite.meets_min_10_entries.all()
    a0 = lite[(lite.bound == "a") & (lite.param == 0.0)].iloc[0]
    want_below = sum(0.05 < float(C.bound_a(0.9 - 0.05 * (i + 1), 0.9 - 0.05 * i, 300, 0.0)) for i in range(11))
    assert a0.n_adjacent_below == want_below
    sens = cen[(cen.leaderboard == "SWE-bench Lite") & cen.set.str.startswith("sensitivity")]
    assert (sens.n_entries == 11).all()             # the attempts = 2+ entry is dropped
    assert cen[cen.leaderboard == "SWE-bench Verified"].empty   # no entries: fails selection test T4, not analysed
    assert len(pairs) == 11 + 10                     # primary + sensitivity adjacent pairs


def test_run_excluded_entries_are_not_counted():
    es = _synthetic_entries()
    es.append(C.Entry("SWE-bench Lite", "bad", None, 1.0, status="excluded", reason="x"))
    cen, _ = C.run(es)
    assert cen[(cen.leaderboard == "SWE-bench Lite") & (cen.set == "primary")].n_entries.eq(12).all()


# ---- ties, leaderboard selection, extension loaders ----------------------------------------

def test_equal_scores_are_never_separable():
    # two entries at 100%: SE = 0, bound = 0, gap = 0 -> still "below" (a tie cannot be separable)
    scores = [1.0, 1.0, 0.8, 0.8, 0.5]
    zero = lambda p1, p2, k1, k2: np.zeros(np.broadcast(p1, p2).shape)
    st = C.census_config(scores, list("abcde"), [1] * 5, zero)
    assert st["n_adjacent_below"] == 2 and st["top1_vs_2_separable"] is False
    assert st["n_all_below"] == 2
    assert C.greedy_tiers([1.0, 1.0, 0.8], lambda i, j: 0.0) == 2   # tied entry stays in tier 1


def test_swebench_percentage_checked_at_reported_precision(tmp_path):
    ids = [f"i{i}" for i in range(300)]
    d = {
        "one_dec": {"source": "r", "resolved": ids[:194], "covered_ids": ids, "meta": {"attempts": 1, "resolved_pct_reported": 64.7}},
        "wrong": {"source": "r", "resolved": ids[:194], "covered_ids": ids, "meta": {"attempts": 1, "resolved_pct_reported": 65.7}},
        "full": {"source": "r", "resolved": ids[:150], "covered_ids": ids, "meta": {"attempts": 1, "resolved_pct_reported": 50.0}},
    }
    _write(tmp_path / "data/raw/leaderboards/swe-bench/swebench_multilingual.json", d)
    es = {e.name: e for e in C.load_swebench(tmp_path, "multilingual", "SWE-bench Multilingual", 300)}
    assert es["one_dec"].status == "kept" and es["full"].status == "kept"
    assert es["wrong"].status == "excluded"


def _tau2_sub(T, name, dom, p1, ver, notes="4 trials", p4=50.0):
    _write(T / name / "submission.json", {
        "results": {dom: {"pass_1": p1, "pass_4": p4}},
        "methodology": {"tau2_bench_version": ver, "notes": notes}})


def test_load_tau2_domain_rules(tmp_path):
    T = tmp_path / "data/raw/tau2"
    _tau2_sub(T, "new", "airline", 100 * 120 / 200, "1.0.1")
    _tau2_sub(T, "old", "airline", 60.0, "0.1.3")
    _tau2_sub(T, "orig", "airline", 60.0, "original tau-bench")
    _tau2_sub(T, "unlisted", "airline", 60.0, "1.0.1")
    _tau2_sub(T, "voice", "airline", 60.0, "1.0.1")
    _tau2_sub(T, "tel_old", "telecom", 100 * 228 / 456, "0.1.3")
    _tau2_sub(T, "tel_new", "telecom", 100 * 228 / 456, "1.0.1")
    _write(T / "manifest.json", {"submissions": ["new", "orig", "tel_new"], "legacy_submissions": ["old", "tel_old"],
                                 "voice_submissions": ["voice"]})
    air = {e.name: e for e in C.load_tau2(tmp_path, "tau2-bench airline")}
    assert air["new"].status == "kept" and air["new"].score == pytest.approx(0.6) and air["new"].k == 4
    for k in ("old", "orig", "unlisted", "voice"):
        assert air[k].status == "excluded"
    assert "1.0.0" in air["old"].reason and "manifest" in air["unlisted"].reason
    tel = {e.name: e for e in C.load_tau2(tmp_path, "tau2-bench telecom")}
    assert tel["tel_old"].status == "kept" and tel["tel_new"].status == "kept"   # telecom tasks unchanged since 0.1.3


def test_load_appworld(tmp_path):
    mk = lambda i, a, b: {"id": i, "method": {"name": "m"}, "llm": {"name": "l"}, "date": "2026-01-01",
                          "test_normal": {"all": {"task_goal_completion": a}},
                          "test_challenge": {"all": {"task_goal_completion": b}}}
    _write(tmp_path / "data/raw/leaderboards/appworld/_leaderboard.json",
           [mk("x1", 8.9, 38.9), mk("x2", 9.0, 40.0)])
    n = {e.name.split("[")[1]: e for e in C.load_appworld(tmp_path, "test_normal", "AppWorld test_normal")}
    assert n["x1]"].score == pytest.approx(15 / 168) and n["x2]"].status == "excluded"   # 9.0% of 168 = 15.12
    c = {e.name.split("[")[1]: e for e in C.load_appworld(tmp_path, "test_challenge", "AppWorld test_challenge")}
    assert c["x1]"].status == "excluded" and c["x2]"].score == pytest.approx(167 / 417)   # 40.05% printed 40.0


def test_load_browsergym_rules(tmp_path):
    base = tmp_path / "data/raw/leaderboards/browsergym/results"

    def row(agent, score, **kw):
        r = {"agent_name": agent, "study_id": "s", "benchmark": "WorkArena-L2", "score": score, "std_err": 3.3,
             "followed_evaluation_protocol": "Yes", "comments": "NA", "original_or_reproduced": "Original"}
        r.update(kw)
        return r

    _write(base / "a/workarena-l2.json", [
        row("ok", 47.7),                                                    # 112/235 = 47.66
        row("noprot", 47.7, followed_evaluation_protocol="No", comments="Increased max_steps"),
        row("341", 47.7, comments="341 tasks (full benchmark)"),
        row("grain", 47.9),                                                 # not a multiple of 1/235
        row("otherbench", 47.7, benchmark="WebArena"),
    ])
    es = {e.name.split(" [")[0]: e for e in C.load_browsergym(tmp_path, "WorkArena-L2", "WorkArena L2")}
    assert set(es) == {"ok", "noprot", "341", "grain"}
    assert es["ok"].status == "kept" and es["ok"].score == pytest.approx(112 / 235)
    assert all(es[k].status == "excluded" for k in ("noprot", "341", "grain"))
    assert "different task set" in es["341"].reason
    assert es["ok"].extra["se_consistent"] == 1     # std_err 3.3 vs binomial 3.26: recorded for a sensitivity set, not an exclusion test


def test_load_webarena_sheet_blocks(tmp_path):
    p = tmp_path / "data/raw/leaderboards/webarena"
    p.mkdir(parents=True)
    (p / "leaderboard_gid0.csv").write_text(
        "a,Open?,Model Size (billion),Model,Success Rate (%),Result Source,Work,Traj,Note\n"
        "02/2026,x,-,A,74.3,S,W,L,\n"
        "01/2025,x,-,B,7.12,S,W,L,\n"
        ",,,,,,,,\n"
        "WebArena Subset,,,,,,,,\n"
        "03/2024,-,Sub,43.7,Sub,Sub,,x,Reddit subset\n"
        "Human Performance,,,,,,,,\n"
        ",,Human,78.24,WebArena,,,,Selected tasks\n", encoding="utf-8")
    es = {e.name: e for e in C.load_webarena_sheet(tmp_path)}
    assert es["A / S (02/2026)"].status == "kept" and es["A / S (02/2026)"].score == pytest.approx(603 / 812)
    assert es["B / S (01/2025)"].status == "excluded"
    assert es["Sub / 03/2024"].status == "excluded" and "different task set" in es["Sub / 03/2024"].reason
    assert es["Human / human"].status == "excluded"


def test_load_androidworld_trials(tmp_path):
    p = tmp_path / "data/raw/leaderboards/androidworld"
    p.mkdir(parents=True)
    (p / "leaderboard_gid0.csv").write_text(
        "warning line,,,,,,,,,,,,\n"
        'Rank,Release Date,Result Source,Model Type,Open?,Model Size,Model,"Screen\nRepresentation","Success Rate\n(pass@1)","Number\nof trials","Success Rate\n(pass@k)","Trajectory\nsubmissions",Note\n'
        "1,08/2026,X,AI agent,x,-,m1,s,100,1,,,\n"
        "2,08/2026,Y,AI agent,x,-,m2,s,99.1,,,,\n"
        "3,08/2026,Z,AI agent,x,-,m3,s,80.0,3,,,\n"
        "4,08/2026,W,AI agent,x,-,m4,s,78.0,1,,,\n", encoding="utf-8")
    es = {e.name.split(" / ")[0]: e for e in C.load_androidworld(tmp_path)}
    assert es["m1"].score == 1.0 and es["m2"].status == "kept" and es["m2"].score == pytest.approx(115 / 116)
    assert es["m3"].status == "excluded" and es["m4"].status == "excluded"    # 80.0 not a multiple of 1/348; 78.0 of 1/116


def test_load_mlebench_table(tmp_path):
    p = tmp_path / "data/raw/leaderboards/mlebench"
    p.mkdir(parents=True)
    hdr = "| Agent | LLM(s) used | Low | Medium | High | All (%) | Running Time (hours) | Date | Source Code Available | Grading Reports Available |\n|---|---|---|---|---|---|---|---|---|---|\n"
    (p / "README.md").write_text(
        "## Leaderboard\n" + hdr +
        "| [A](http://a) | m1 | 1 | 2 | 3 | 63.11 ± 0.44 | 24 | 2026-03-06 | X | ✓ |\n"
        "| [B](http://b)<br>(Org) | m2[^4] | 1 | 2 | 3 | 61.33 ± 0.77[^3] | 24 | 2026-01-05 | X | ✓ |\n"
        "### Additional Leaderboard Submissions\n" + hdr.replace("Date |", "Date | Notes |") +
        "| [C](http://c) | m3 | 1 | 2 | 3 | 77.78 ± 0.44 | 24 | 2026-02-03 | note | X | ✓ |\n\n[^2]: x\n", encoding="utf-8")
    es = {e.name: e for e in C.load_mlebench(tmp_path)}
    assert es["A / m1 (2026-03-06)"].status == "kept" and es["A / m1 (2026-03-06)"].score == pytest.approx(0.6311)
    assert es["B / m2 (2026-01-05)"].status == "excluded" and "padded" in es["B / m2 (2026-01-05)"].reason
    assert es["C / m3 (2026-02-03)"].status == "excluded"


def test_load_hal_generic(tmp_path):
    p = tmp_path / "data/raw/leaderboards/hal"
    p.mkdir(parents=True)
    pd.DataFrame({
        "Rank": ["1", "2", "3"], "Scaffold": ["s"] * 3, "Primary": ["a", "b", "c"],
        "Accuracy": ["9.23%", "6.92% (-0.77/+0.77)", "9.30%"], "Runs": ["1", "2", "1"],
    }).to_csv(p / "scicode_entries.csv", index=False)
    es = C.load_hal(tmp_path, "HAL SciCode")
    assert [e.status for e in es] == ["kept", "kept", "excluded"]                  # 9.30% is not a multiple of 1/65
    assert es[0].score == pytest.approx(6 / 65) and es[1].score == pytest.approx(9 / 130) and es[1].k == 2.0


def test_uniquify_and_status():
    es = [C.Entry("HAL USACO", "x", 0.5), C.Entry("HAL USACO", "x", 0.4), C.Entry("HAL USACO", "y", 0.3)]
    assert [e.name for e in C.uniquify(es)] == ["x", "x #2", "y"]
    many = [C.Entry("HAL USACO", f"e{i}", 0.1 * i % 1) for i in range(10)]
    few = [C.Entry("HAL GAIA", f"g{i}", 0.5) for i in range(9)] + [C.Entry("HAL GAIA", "drop", None, status="excluded", reason="x")]
    st = {lb: (s, why) for lb, s, why in C.leaderboard_status(many + few)}
    assert st["HAL USACO"][0] == "analysed" and st["HAL GAIA"][0] == "excluded" and "T4" in st["HAL GAIA"][1]
    assert st["GAIA test"][0] == "excluded" and "not verifiable" in st["GAIA test"][1]
    assert len(C.CANDIDATES) == 26 and len(set(C.CANDIDATES)) == 26


def test_release_tables_have_only_derived_columns(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    pd.DataFrame({"path": ["leaderboards/hal/usaco.html"], "bytes": [1], "sha256": ["ab" * 32]}).to_csv(
        root / "data/leaderboards_manifest.csv", index=False)
    es = [C.Entry("HAL USACO", f"e{i}", 0.1 + 0.01 * i, 1.0, extra={"src_file": "hal/usaco.html"}) for i in range(10)]
    es.append(C.Entry("HAL USACO", "bad", None, status="excluded", reason="r"))
    kept, ex = C.release_tables(es, root)
    assert list(kept.columns) == ["leaderboard", "entry", "score", "n_tasks", "k_runs", "source_url", "snapshot_date", "source_file_sha256"]
    assert (kept.source_file_sha256 == "ab" * 32).all() and (kept.n_tasks == 307).all() and len(kept) == 10
    assert list(ex.columns) == ["leaderboard", "entry", "reason"] and len(ex) == 1


def test_aggregates_counts():
    es = [C.Entry("HAL USACO", f"e{i:02d}", 0.9 - 0.05 * i, 1.0) for i in range(12)]
    cen, _ = C.run(es)
    a = C.aggregates(cen, ["HAL USACO"])
    assert a[0]["config"] == "(a) rho = 0 [conservative]" and a[0]["n_lb"] == 1 and a[0]["tiers_median"] == a[0]["tiers_min"]
