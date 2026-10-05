# SPDX-License-Identifier: MIT
"""RQ2 leaderboard census: how many adjacent leaderboard gaps are inside the 95% noise bound?

Input: the public single-run leaderboard snapshots in ``data/raw/leaderboards`` (plus
``data/raw/tau2`` and ``analysis/results/tb2_long.csv`` for Terminal-Bench 2.0), and the
verified task count n of each leaderboard (``N_VERIFIED``; sources in
``research/leaderboard-n-verification.md``).

For each leaderboard the kept entries are sorted by score (best first). For every adjacent
pair (and every pair) the gap is compared with two bounds. Definitions are fixed in
``decisions/decision-log.md`` ("Census bound correction"):

(a) HEADLINE, no transport: the PAIRED item-sampling bound. SE = sqrt(P(1-P)/n) from the entry's own
    score and the leaderboard's task count n. For a difference,
        bound = 1.96 * sqrt(SE1^2 + SE2^2 - 2*rho*SE1*SE2),
    where rho is the task-level correlation between the two entries' per-task outcomes that shared
    tasks induce. rho is estimated from per-task results (``estimate_rho.py``) for the 7 boards that
    have them (one value per board: the median over adjacent pairs); the other boards use the pooled
    median of those per-board medians. Variants kept for comparison: the pooled median for every
    board, each pair's own rho on the 7 per-task boards (``a_pair``), rho = 0 (the scores-only
    unpaired bound, the conservative case) and the sweep rho in {0.25, 0.5, 0.75}. This is the
    uncertainty for generalising to the task population; it already contains run-to-run variation.

(b) Fixed-task-set rerun bound. Run-to-run variance only, with the entry's within-task variance
    v = f * P(1-P) transported as a FRACTION f of its outcome variance (decision log 2026-10-04,
    adjudication of simulated review 1, M3). Per pair, with k runs per entry,
        bound = 1.96 * sqrt((f/n) * (P1(1-P1)/k1 + P2(1-P2)/k2)),
    so v <= P(1-P) holds for every f <= 1 and no cap is needed. f is swept over
    {0.25, 0.5, f_TB2, 0.75} where f_TB2 is the median over the 72 Terminal-Bench 2.0 analysis-set
    submissions of v/(P(1-P)) (``analysis/results/tb2_within_task_variance.csv``). ``b_f_kadj`` is
    the k-adjusted form above, ``b_f`` the single-run form (k1 = k2 = 1; for entries that average
    k > 1 runs it overstates rerun noise). The earlier parameterisation on absolute v in
    {0.025, 0.0753, 0.1618, 0.2022} with a cap min(v, P1(1-P1), P2(1-P2)) is kept in
    ``census.csv`` as families ``b`` and ``b_kadj`` and is superseded.

A pair is "below" the bound when gap < bound (strict), and a pair with equal scores is always below (at P = 0 or 1 the binomial SE is 0, so a tie would otherwise count as separable). Entries are ordered by score, then by
name (so ties are deterministic). A pair is "separable" when gap >= bound.

Tiers (greedy): the top entry opens tier 1; each next entry joins the current tier while it
is NOT separable from the tier's first (best) entry; the first entry that is separable from
the tier head opens the next tier.

Scope: every leaderboard that passes tests T1-T3 of the selection rule (26) is analysed unless a
written criterion excludes it (n not verifiable from documentation, or fewer than 10 kept entries,
test T4); see ``leaderboard_status``. Derived per-entry statistics for release are written to
``<out>/release`` (no raw leaderboard record).

Usage: python census.py [--root <project root>] [--out <output dir>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

Z = 1.96
RHOS = (0.0, 0.25, 0.5, 0.75)
VS = (0.025, 0.0753, 0.1618, 0.2022)  # legacy absolute-v sweep (families b, b_kadj)
FS = (0.25, 0.5, 0.75)                # run-noise share f = v / (P(1-P)); the TB2 median f is added at run time
MIN_ENTRIES = 10  # selection rule, test T4

# Task counts n and the documentation they were read from: research/leaderboard-n-verification.md
# These are the 26 leaderboards that pass tests T1-T3 of the selection rule in the inventory
# (research/leaderboard-inventory.md). A leaderboard is analysed when its n is verified from
# documentation (listed here) and at least MIN_ENTRIES entries remain after the entry-level checks.
N_VERIFIED = {
    "SWE-bench Verified": 500,
    "SWE-bench Lite": 300,
    "SWE-bench test (full)": 2294,
    "Terminal-Bench 1.0": 80,
    "Terminal-Bench 2.0": 89,
    "TheAgentCompany": 175,
    "tau2-bench banking_knowledge": 97,
    "HAL SWE-bench Verified Mini": 50,
    # --- the 18 added when the census was extended from 8 to every passing leaderboard
    "SWE-bench Multilingual": 300,
    "tau2-bench airline": 50,
    "tau2-bench telecom": 114,
    "AppWorld test_normal": 168,
    "AppWorld test_challenge": 417,
    "WebArena (BrowserGym)": 812,
    "WebArena (Google Sheet)": 812,
    "WorkArena L1": 330,
    "WorkArena L2": 235,
    "AndroidWorld": 116,
    "MLE-bench": 75,
    "HAL GAIA": 165,
    "HAL SciCode": 65,
    "HAL ScienceAgentBench": 102,
    "HAL USACO": 307,
}
# Passing leaderboards whose n cannot be verified from documentation: excluded by that written criterion.
N_UNVERIFIED = {
    "tau2-bench retail": "n not verifiable: the benchmark paper's Table 1 gives 115 retail tasks, the repository's base split holds 114 task ids, and no documentation sentence states 114",
    "GAIA test": "n not verifiable: the paper states 300 test questions (and 166 development questions), the dataset card states no count, and every one of the 3,817 scores is a multiple of 1/301, not of 1/300",
    "GAIA validation": "n not verifiable: the paper states 166 development questions, the dataset card states no count, and every one of the 96 scores is a multiple of 1/165 (the HAL page states 165 for its own evaluation)",
}
CANDIDATES = tuple(N_VERIFIED) + tuple(N_UNVERIFIED)  # 26
PRIMARY = tuple(N_VERIFIED)  # leaderboards eligible for analysis (before the T4 count)


# --------------------------------------------------------------------------- statistics

def se(p, n: float):
    p = np.asarray(p, dtype=float)
    return np.sqrt(p * (1.0 - p) / n)


def bound_a(p1, p2, n: float, rho: float = 0.0):
    """95% bound on a score difference from binomial SEs with task-level correlation rho."""
    s1, s2 = se(p1, n), se(p2, n)
    var = s1 ** 2 + s2 ** 2 - 2.0 * rho * s1 * s2
    return Z * np.sqrt(np.maximum(var, 0.0))


def bound_b(p1, p2, n: float, v: float, cap: bool = True, k1=1.0, k2=1.0):
    """95% bound on a difference from run-to-run variance only (fixed task set).
    Variance of an entry mean = v/(n k). With k1 = k2 = 1 this is 1.96*sqrt(2 v / n)."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    if cap:
        veff = np.minimum(v, np.minimum(p1 * (1 - p1), p2 * (1 - p2)))
    else:
        veff = np.full(np.broadcast(p1, p2).shape, float(v))
    return Z * np.sqrt(veff / n * (1.0 / np.asarray(k1, float) + 1.0 / np.asarray(k2, float)))


def bound_f(p1, p2, n: float, f: float, k1=1.0, k2=1.0):
    """95% bound on a difference from run-to-run variance only, with v_j = f * P_j(1-P_j).
    Variance of an entry mean = v_j/(n k_j); with k1 = k2 = 1 this is sqrt(f) times the unpaired
    item-sampling bound."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    var = (f / n) * (p1 * (1 - p1) / np.asarray(k1, float) + p2 * (1 - p2) / np.asarray(k2, float))
    return Z * np.sqrt(var)


def wilson(k: int, m: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/m."""
    if m == 0:
        return (float("nan"), float("nan"))
    ph = k / m
    den = 1 + z * z / m
    c = (ph + z * z / (2 * m)) / den
    h = z * math.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def order(scores, names):
    """Indices that sort entries best-first; ties broken by name."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], names[i]))


def greedy_tiers(sorted_scores, bound_fn) -> int:
    """Number of greedy tiers. bound_fn(i, j) = bound for sorted entries i < j."""
    if len(sorted_scores) == 0:
        return 0
    tiers, head = 1, 0
    for j in range(1, len(sorted_scores)):
        gap = sorted_scores[head] - sorted_scores[j]
        if not (gap < bound_fn(head, j) or gap == 0):  # separable from the tier head
            tiers += 1
            head = j
    return tiers


def census_config(scores, names, ks, bound_fn, pair: bool = False):
    """Statistics for one leaderboard and one bound configuration.

    bound_fn(p1, p2, k1, k2) -> bound array (vectorised over pairs); k = runs per entry.
    With ``pair=True`` bound_fn(p1, p2, k1, k2, ia, ib) also receives the positions (0 = best) of the
    two entries in the sorted order, for bounds that depend on the pair (a pair-specific rho)."""
    idx = order(scores, names)
    s = np.array([scores[i] for i in idx])
    kk = np.array([ks[i] for i in idx], dtype=float)
    m = len(s)
    out = {"n_entries": m}
    if m < 2:
        return out

    def bm(p1, p2, k1, k2, ia, ib):
        return bound_fn(p1, p2, k1, k2, ia, ib) if pair else bound_fn(p1, p2, k1, k2)

    gaps = s[:-1] - s[1:]
    b_adj = bm(s[:-1], s[1:], kk[:-1], kk[1:], np.arange(m - 1), np.arange(1, m))
    below = (gaps < b_adj) | (gaps == 0)
    ii, jj = np.triu_indices(m, 1)
    gap_all = s[ii] - s[jj]
    b_all = bm(s[ii], s[jj], kk[ii], kk[jj], ii, jj)
    below_all = (gap_all < b_all) | (gap_all == 0)
    lo, hi = wilson(int(below.sum()), len(below))
    out.update(
        n_adjacent_pairs=len(below), n_adjacent_below=int(below.sum()),
        share_adjacent_below=float(below.mean()), wilson_lo=lo, wilson_hi=hi,
        n_all_pairs=len(below_all), n_all_below=int(below_all.sum()),
        share_all_below=float(below_all.mean()),
        median_adjacent_gap_pp=float(np.median(gaps) * 100), median_adjacent_bound_pp=float(np.median(b_adj) * 100),
    )
    out["n_tiers"] = greedy_tiers(s, lambda i, j: float(bm(s[i], s[j], kk[i], kk[j], i, j)))
    out["gap_1_2_pp"] = float(gaps[0] * 100)
    out["bound_1_2_pp"] = float(b_adj[0] * 100)
    out["top1_vs_2_separable"] = bool(not below[0])
    if m >= 5:
        g15 = s[0] - s[4]
        b15 = float(bm(s[0], s[4], kk[0], kk[4], 0, 4))
        out.update(gap_1_5_pp=float(g15 * 100), bound_1_5_pp=b15 * 100, top1_vs_5_separable=bool(not (g15 < b15 or g15 == 0)))
    else:
        out.update(gap_1_5_pp=float("nan"), bound_1_5_pp=float("nan"), top1_vs_5_separable=None)
    return out


# --------------------------------------------------------------------------- entries

@dataclass
class Entry:
    lb: str
    name: str
    score: float | None            # share of tasks solved, in [0, 1]
    k: float = 1.0                 # runs averaged into the score
    status: str = "kept"           # "kept" or "excluded"
    reason: str = ""
    extra: dict = field(default_factory=dict)


def decimals_of(x) -> int:
    s = repr(float(x)) if not isinstance(x, str) else x
    s = s.replace("%", "").strip()
    return len(s.split(".")[1]) if "." in s else 0


def recover_count(pct, n: int, k: int, shown: str | None = None) -> int | None:
    """Integer c such that 100*c/(n*k) equals the reported percentage at its displayed
    precision; None if no integer fits. `shown` is the printed form when the value was
    rounded for display (e.g. '72.00%')."""
    d = decimals_of(shown if shown is not None else pct)
    tol = 1e-6 if d >= 8 else 0.5 * 10 ** (-d) + 1e-9
    c = round(float(pct) * n * k / 100.0)
    return c if abs(100.0 * c / (n * k) - float(pct)) <= tol else None


def _stated_trials(d: dict) -> int | None:
    txt = json.dumps(d.get("trajectory_files") or {}) + " " + str((d.get("methodology") or {}).get("notes") or "")
    found = set(re.findall(r"(\d+)\s*trials", txt))
    return int(found.pop()) if len(found) == 1 else None


def load_swebench(root: Path, split: str, lb: str, n: int) -> list[Entry]:
    d = json.loads((root / "data/raw/leaderboards/swe-bench" / f"swebench_{split}.json").read_text(encoding="utf-8"))
    full = [frozenset(r["covered_ids"]) for r in d.values() if r["source"] and len(r["covered_ids"]) == n]
    canon = set(full[0]) if full else set()
    assert all(f == canon for f in full), f"{lb}: full-coverage entries disagree on the task set"
    assert len(canon) == n, f"{lb}: reference task set has {len(canon)} ids, expected {n}"
    out = []
    for key, r in sorted(d.items()):
        meta = r.get("meta") or {}
        att = str(meta.get("attempts"))
        e = Entry(lb, key, None, 1.0, extra={"attempts": att, "covered_ids_n": len(r.get("covered_ids") or [])})
        res = r.get("resolved")
        if r["source"] is None:
            e.status, e.reason = "excluded", "no per-instance result file in the repository"
        elif res is None:
            e.status, e.reason = "excluded", "no list of resolved instance ids"
        elif len(res) != len(set(res)):
            e.status, e.reason = "excluded", "resolved list contains duplicate instance ids"
        elif not set(res) <= canon:
            e.status, e.reason = "excluded", f"{len(set(res) - canon)} resolved ids are outside the {n}-task set"
        else:
            c = len(res)
            pct = meta.get("resolved_pct_reported")
            # compare at the reported precision (two decimals for most splits, one for Multilingual)
            tol = max(0.0051, 0.5 * 10 ** (-decimals_of(pct)) + 1e-9) if pct is not None else 0.0
            if pct is not None and abs(round(100 * c / n, 2) - float(pct)) > tol:
                e.status = "excluded"
                e.reason = (f"reported resolved {pct}% disagrees with the id list ({c}/{n} = {100 * c / n:.2f}%): "
                            "possible different task set or incomplete list")
            else:
                e.score = c / n
        out.append(e)
    return out


def load_tb1(root: Path) -> list[Entry]:
    d = json.loads((root / "data/raw/leaderboards/terminal-bench-1/tb1_run_results.json").read_text(encoding="utf-8"))["files"]
    ref = json.loads((root / "data/raw/leaderboards/n_verification/tb1_registry_0.1.1_task_ids.json").read_text(encoding="utf-8"))
    refset = set(ref["task_ids"])
    by: dict[str, list[dict]] = {}
    for p, v in d.items():
        by.setdefault(p.split("/")[2], []).append(v)
    out = []
    for key, runs in sorted(by.items()):
        e = Entry("Terminal-Bench 1.0", key, None, 1.0)
        res = sum(v["n_resolved"] for v in runs)
        tot = sum(v["n_resolved"] + v["n_unresolved"] for v in runs)
        sets_ok = all((set(v["resolved_ids"]) | set(v["unresolved_ids"])) == refset for v in runs)
        counts_ok = all(v["n_resolved"] == len(v["resolved_ids"]) and v["n_unresolved"] == len(v["unresolved_ids"]) for v in runs)
        if not sets_ok:
            e.status, e.reason = "excluded", "task ids differ from the 80 ids of terminal-bench-core 0.1.1"
        elif not counts_ok:
            e.status, e.reason = "excluded", "n_resolved/n_unresolved disagree with the id lists"
        elif tot % len(refset):
            e.status, e.reason = "excluded", f"{tot} trials is not a multiple of 80"
        else:
            e.score, e.k = res / tot, tot / len(refset)
            e.extra = {"run_files": len(runs)}
        out.append(e)
    return out


def load_tb2(root: Path) -> list[Entry]:
    df = pd.read_csv(root / "analysis/results/tb2_long.csv", usecols=["submission", "task_id", "success"])
    counts = pd.read_csv(root / "analysis/results/tb2_load_counts.csv").set_index("submission")
    union = set(df["task_id"])
    out = []
    for sub, g in sorted(df.groupby("submission")):
        e = Entry("Terminal-Bench 2.0", sub, None, 1.0)
        if set(g["task_id"]) != union or len(union) != 89:
            e.status, e.reason = "excluded", "task set differs from the 89-task union"
        else:
            e.score, e.k = float(g["success"].mean()), len(g) / 89
            e.extra = {"in_rq3_analysis_set": int(counts.loc[sub, "included"]),
                       "no_reward_trials": int(counts.loc[sub, "no_reward_trials"])}
        out.append(e)
    return out


def load_tac(root: Path) -> list[Entry]:
    t = json.loads((root / "data/raw/leaderboards/theagentcompany/tac_results.json").read_text(encoding="utf-8"))["results"]["1.0.0"]
    out = []
    for key, tasks in sorted(t.items()):
        e = Entry("TheAgentCompany", key, None, 1.0)
        ids = {x.replace("-image", "") for x in tasks}
        if len(ids) != 175:
            e.status, e.reason = "excluded", f"{len(ids)} of 175 task result files present"
        else:
            ok = sum(all(c["result"] == c["total"] for c in v["checkpoints"]) for v in tasks.values())
            okf = sum(v["final"]["result"] == v["final"]["total"] for v in tasks.values())
            e.score = ok / 175
            e.extra = {"score_alt_final_field": okf / 175}
        out.append(e)
    return out


# tau2-bench versions whose scores are on the same task definitions, per domain (CHANGELOG at
# commit 5bfa7e37): banking_knowledge grading and tasks changed in 1.0.1; the airline (27 tasks) and
# retail (26 tasks) task definitions were fixed in 1.0.0; no CHANGELOG item changes the telecom tasks.
TAU2_DOMAINS = {
    "tau2-bench banking_knowledge": ("banking_knowledge", 97, {"1.0.1"},
        "banking_knowledge scores before 1.0.1 are not comparable (CHANGELOG 1.0.1)"),
    "tau2-bench airline": ("airline", 50, {"1.0.0", "1.0.1"},
        "airline task definitions were fixed in 1.0.0 (27 tasks, CHANGELOG 1.0.0), so earlier scores are not on the same tasks"),
    "tau2-bench retail": ("retail", 114, {"1.0.0", "1.0.1"},
        "retail task definitions were fixed in 1.0.0 (26 tasks, CHANGELOG 1.0.0), so earlier scores are not on the same tasks"),
    "tau2-bench telecom": ("telecom", 114, {"0.1.3", "v0.1.3", "1.0.0", "1.0.1"},
        "version not among those whose telecom tasks match (no CHANGELOG item changes them from 0.1.3 on)"),
}


def load_tau2(root: Path, lb: str) -> list[Entry]:
    dom, n, versions, why = TAU2_DOMAINS[lb]
    T2 = root / "data/raw/tau2"
    man = json.loads((T2 / "manifest.json").read_text(encoding="utf-8"))
    grp = {nm: g for g, lst in man.items() for nm in lst}
    out = []
    for f in sorted(T2.glob("*/submission.json")):
        name = f.parent.name
        if name.startswith("A_EXAMPLE"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        r = (d.get("results") or {}).get(dom)
        if not (r and r.get("pass_1") is not None):
            continue
        e = Entry(lb, name, None, 1.0, extra={"src_file": f"web/leaderboard/public/submissions/{name}/submission.json"})
        ver = str((d.get("methodology") or {}).get("tau2_bench_version"))
        if name not in grp:
            e.status, e.reason = "excluded", "not listed in the submissions manifest (not an active leaderboard entry)"
        elif grp.get(name) == "voice_submissions":
            e.status, e.reason = "excluded", "voice submission (different modality and protocol)"
        elif ver not in versions:
            e.status, e.reason = "excluded", f"tau2-bench version {ver}: {why}"
        else:
            ks = [int(k.split("_")[1]) for k in r if k.startswith("pass_") and r[k] is not None]
            k = _stated_trials(d) or max(ks)
            c = recover_count(r["pass_1"], n, k)
            if c is None:
                e.status = "excluded"
                e.reason = f"pass_1 = {r['pass_1']} is not a multiple of 1/{n * k} (n = {n}, k = {k} trials): possible different task set"
            else:
                e.score, e.k = c / (n * k), float(k)
                e.extra.update({"trials_stated": _stated_trials(d), "trials_used": k})
        out.append(e)
    return out


HAL_FILES = {  # leaderboard -> (file stem under data/raw/leaderboards/hal, model column)
    "HAL SWE-bench Verified Mini": ("swebench_verified_mini", "Primary"),
    "HAL GAIA": ("gaia", "Primary"),
    "HAL SciCode": ("scicode", "Primary"),
    "HAL ScienceAgentBench": ("scienceagentbench", "Models"),
    "HAL USACO": ("usaco", "Primary"),
}


def load_hal(root: Path, lb: str) -> list[Entry]:
    stem, mcol = HAL_FILES[lb]
    df = pd.read_csv(root / f"data/raw/leaderboards/hal/{stem}_entries.csv", dtype=str)
    n = N_VERIFIED[lb]
    out = []
    for _, r in df.iterrows():
        name = f"rank{r['Rank']} {r['Scaffold']} / {r[mcol]}"
        e = Entry(lb, name, None, 1.0, extra={"src_file": f"hal/{stem}.html"})
        shown = r["Accuracy"].split("%")[0].strip() + "%"
        k = int(r["Runs"])
        c = recover_count(float(shown.rstrip("%")), n, k, shown)
        if c is None:
            e.status = "excluded"
            e.reason = f"accuracy {shown} is not a multiple of 1/{n * k} (n = {n}, {k} run(s))"
        else:
            e.score, e.k = c / (n * k), float(k)
        out.append(e)
    return out


def load_hal_mini(root: Path) -> list[Entry]:
    return load_hal(root, "HAL SWE-bench Verified Mini")


def load_appworld(root: Path, split: str, lb: str) -> list[Entry]:
    n = N_VERIFIED[lb]
    d = json.loads((root / "data/raw/leaderboards/appworld/_leaderboard.json").read_text(encoding="utf-8"))
    out = []
    for r in d:
        name = f"{r['method']['name']} / {r['llm']['name']} ({r['date']}) [{r['id']}]"
        e = Entry(lb, name, None, 1.0, extra={"src_file": "appworld/_leaderboard.json"})
        pct = r[split]["all"]["task_goal_completion"]
        c = recover_count(pct, n, 1)
        if c is None:
            e.status = "excluded"
            e.reason = f"task goal completion {pct} is not a multiple of 1/{n} at the printed precision: possible different task set or rounding"
        else:
            e.score = c / n
        out.append(e)
    return out


def load_browsergym(root: Path, bench: str, lb: str) -> list[Entry]:
    """BrowserGym leaderboard rows for one benchmark. An entry is excluded when its own fields say
    it did not follow the evaluation protocol, when its comment states a task count other than n,
    or when its score is not a multiple of 1/n at the printed precision. The reported standard error
    is not an exclusion test (it is recorded as ``se_consistent`` for a sensitivity set)."""
    n = N_VERIFIED[lb]
    out = []
    base = root / "data/raw/leaderboards/browsergym/results"
    for f in sorted(base.glob("*/*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("benchmark") != bench:
                continue
            pct = float(r["score"])
            p = pct / 100.0
            name = f"{r['agent_name']} [{r.get('study_id', '')}] {r.get('original_or_reproduced', '')}".strip()
            se_rep = float(r.get("std_err") or 0.0)
            se_bin = 100.0 * math.sqrt(p * (1 - p) / n)
            e = Entry(lb, name, None, 1.0, extra={
                "src_file": f"browsergym/results/{f.parent.name}/{f.name}",
                "benchmark_tuned": r.get("benchmark_tuned"), "std_err_reported": se_rep,
                "se_consistent": int(abs(se_rep - se_bin) <= 0.25)})
            stated = {int(x) for x in re.findall(r"(\d+)\s+tasks", str(r.get("comments") or ""))}
            c = recover_count(pct, n, 1)
            if r.get("followed_evaluation_protocol") != "Yes":
                e.status, e.reason = "excluded", f"the entry's own field says the evaluation protocol was not followed ({r.get('comments')})"
            elif stated and stated != {n}:
                e.status, e.reason = "excluded", f"the entry's comment states {sorted(stated)} tasks, not {n}: different task set"
            elif c is None:
                e.status, e.reason = "excluded", f"score {r['score']} is not a multiple of 1/{n} at the printed precision"
            else:
                e.score = c / n
            out.append(e)
    return out


def _sheet_rows(df: pd.DataFrame, col: str):
    for _, r in df.iterrows():
        raw = r[col]
        yield r, (None if pd.isna(raw) else str(raw).strip())


def load_webarena_sheet(root: Path) -> list[Entry]:
    """Main block of the WebArena Google Sheet (rows above the first blank row). The sheet's later
    blocks, 'WebArena Subset' (Reddit subset) and 'Human Performance' (selected tasks), are on
    other task sets and are excluded."""
    n = N_VERIFIED["WebArena (Google Sheet)"]
    with open(root / "data/raw/leaderboards/webarena/leaderboard_gid0.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    ci = {h: i for i, h in enumerate(hdr)}
    out = []
    section = "main"
    for r in rows[1:]:
        if not any(x.strip() for x in r):
            if section == "main":
                section = "after_main"
            continue
        first = r[0].strip()
        if first.startswith("WebArena Subset"):
            section = "subset"
            continue
        if first.startswith("Human Performance"):
            section = "human"
            continue
        if section == "main":
            raw, model, src, date = r[ci["Success Rate (%)"]].strip(), r[ci["Model"]], r[ci["Result Source"]], r[ci["a"]]
            if not raw:
                continue
            e = Entry("WebArena (Google Sheet)", f"{model} / {src} ({date})", None, 1.0, extra={"src_file": "webarena/leaderboard_gid0.csv"})
            try:
                c = recover_count(float(raw), n, 1, raw)
            except ValueError:
                c = None
            if c is None:
                e.status = "excluded"
                e.reason = f"score {raw} is not a multiple of 1/{n} at the printed precision: possible subset of tasks or rounding"
            else:
                e.score = c / n
            out.append(e)
        elif section in ("subset", "human") and len(r) >= 9 and r[3].strip():
            note = r[8].strip()
            nm = r[2].strip() or "Human"
            out.append(Entry("WebArena (Google Sheet)", f"{nm} / {r[0].strip() or section}", None, 1.0, "excluded",
                             f"listed under the sheet's '{'WebArena Subset' if section == 'subset' else 'Human Performance'}' block ({note}): a different task set",
                             {"src_file": "webarena/leaderboard_gid0.csv"}))
    return out


def load_androidworld(root: Path) -> list[Entry]:
    n = N_VERIFIED["AndroidWorld"]
    df = pd.read_csv(root / "data/raw/leaderboards/androidworld/leaderboard_gid0.csv", skiprows=1, dtype=str)
    df.columns = [re.sub(r"\s+", " ", c) for c in df.columns]
    out = []
    for r, raw in _sheet_rows(df, "Success Rate (pass@1)"):
        if raw is None:
            continue
        tr = r["Number of trials"]
        k = int(float(tr)) if isinstance(tr, str) and tr.strip() else 1
        name = f"{r['Model']} / {r['Result Source']} ({r['Release Date']})"
        e = Entry("AndroidWorld", name, None, float(k),
                  extra={"src_file": "androidworld/leaderboard_gid0.csv", "trials_stated": tr if isinstance(tr, str) else ""})
        try:
            c = recover_count(float(raw), n, k, raw)
        except ValueError:
            c = None
        if c is None:
            e.status = "excluded"
            e.reason = f"pass@1 {raw} is not a multiple of 1/{n * k} (n = {n}, {k} trial(s)) at the printed precision: possible different task set or rounding"
        else:
            e.score = c / (n * k)
        out.append(e)
    return out


def load_mlebench(root: Path) -> list[Entry]:
    """MLE-bench README table (main leaderboard). The score is the 'All (%)' column, the mean over
    seeds of the share of the 75 competitions with any medal. The number of seeds per entry is not
    stated, so k is recorded as 1 (a labelled assumption that only affects bound (b)). Rows whose
    footnote says incomplete seeds were padded with failing scores, and the 'additional submissions'
    that the README calls not directly comparable, are excluded."""
    md = (root / "data/raw/leaderboards/mlebench/README.md").read_text(encoding="utf-8")
    main, rest = md.split("### Additional Leaderboard Submissions")
    extra = rest.split("[^2]")[0]

    def rows(block: str):
        return [ln for ln in block.splitlines()
                if ln.startswith("| ") and not ln.startswith("| Agent") and not ln.startswith("|---")]

    def clean(x: str) -> str:
        x = re.sub(r"\]\([^)]*\)", "", x)
        x = re.sub(r"<br>.*", "", x)
        return re.sub(r"\[\^\d\]", "", x).replace("[", "").strip()

    out = []
    for ln in rows(main):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        name = f"{clean(c[0])} / {clean(c[1])} ({c[7]})"
        e = Entry("MLE-bench", name, None, 1.0, extra={"src_file": "mlebench/README.md", "k_assumed": 1})
        m = re.match(r"([\d.]+)", c[5])
        if "[^3]" in ln:
            e.status, e.reason = "excluded", "README footnote 3: incomplete seeds were padded with failing scores (different scoring)"
        elif not m or not 0 <= float(m.group(1)) <= 100:
            e.status, e.reason = "excluded", f"no readable All (%) score: {c[5]!r}"
        else:
            e.score = float(m.group(1)) / 100.0
        out.append(e)
    for ln in rows(extra):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        out.append(Entry("MLE-bench", f"{clean(c[0])} / {clean(c[1])} ({c[7]})", None, 1.0, "excluded",
                         "listed under Additional Leaderboard Submissions: not directly comparable (test-set feedback)",
                         {"src_file": "mlebench/README.md"}))
    return out


def load_all(root: Path) -> list[Entry]:
    es: list[Entry] = []
    es += load_swebench(root, "verified", "SWE-bench Verified", N_VERIFIED["SWE-bench Verified"])
    es += load_swebench(root, "lite", "SWE-bench Lite", N_VERIFIED["SWE-bench Lite"])
    es += load_swebench(root, "test", "SWE-bench test (full)", N_VERIFIED["SWE-bench test (full)"])
    es += load_swebench(root, "multilingual", "SWE-bench Multilingual", N_VERIFIED["SWE-bench Multilingual"])
    es += load_tb1(root)
    es += load_tb2(root)
    es += load_tac(root)
    for lb in TAU2_DOMAINS:
        if lb in N_VERIFIED:  # tau2 retail is excluded at leaderboard level (n not verifiable) and not loaded
            es += load_tau2(root, lb)
    es += load_appworld(root, "test_normal", "AppWorld test_normal")
    es += load_appworld(root, "test_challenge", "AppWorld test_challenge")
    es += load_browsergym(root, "WebArena", "WebArena (BrowserGym)")
    es += load_webarena_sheet(root)
    es += load_browsergym(root, "WorkArena-L1", "WorkArena L1")
    es += load_browsergym(root, "WorkArena-L2", "WorkArena L2")
    es += load_androidworld(root)
    es += load_mlebench(root)
    for lb in HAL_FILES:
        es += load_hal(root, lb)
    return uniquify(es)


def uniquify(es: list[Entry]) -> list[Entry]:
    """Entry names must be unique within a leaderboard (bounds are keyed by name): a repeated
    name gets a row suffix in file order."""
    seen: dict[tuple[str, str], int] = {}
    for e in es:
        key = (e.lb, e.name)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            e.name = f"{e.name} #{seen[key]}"
    return es


def load_tau2_banking(root: Path) -> list[Entry]:
    return load_tau2(root, "tau2-bench banking_knowledge")


# --------------------------------------------------------------------------- census driver

def kept_counts(entries: list[Entry]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entries:
        out[e.lb] = out.get(e.lb, 0) + (e.status == "kept")
    return out


def census_leaderboards(entries: list[Entry]) -> list[str]:
    """Leaderboards that are analysed: n verified from documentation (N_VERIFIED) and at least
    MIN_ENTRIES kept entries after the entry-level checks (selection test T4)."""
    kc = kept_counts(entries)
    return [lb for lb in PRIMARY if kc.get(lb, 0) >= MIN_ENTRIES]


def leaderboard_status(entries: list[Entry], side_kept: dict[str, int] | None = None) -> list[tuple[str, str, str]]:
    """(leaderboard, 'analysed' | 'excluded', reason) for the 26 candidates, in a fixed order."""
    kc = kept_counts(entries)
    side_kept = side_kept or {}
    rows = []
    for lb in CANDIDATES:
        if lb in N_UNVERIFIED:
            why = N_UNVERIFIED[lb]
            if lb in side_kept:
                why += f"; also only {side_kept[lb]} entries remain after the entry-level checks"
            rows.append((lb, "excluded", why))
        elif kc.get(lb, 0) >= MIN_ENTRIES:
            rows.append((lb, "analysed", f"{kc[lb]} kept entries"))
        else:
            rows.append((lb, "excluded", f"fewer than {MIN_ENTRIES} entries after the entry-level checks (test T4): {kc.get(lb, 0)} kept"))
    return rows


def variants(entries: list[Entry]) -> dict[tuple[str, str], list[Entry]]:
    """(leaderboard, set label) -> kept entries. 'primary' is the rule applied as written;
    'sensitivity' sets drop entries a reader could argue are not like-for-like."""
    out: dict[tuple[str, str], list[Entry]] = {}
    for lb in census_leaderboards(entries):
        kept = [e for e in entries if e.lb == lb and e.status == "kept"]
        out[(lb, "primary")] = kept
        if lb.startswith("SWE-bench") and any(e.extra.get("attempts") in ("2+", "2") for e in kept):
            out[(lb, "sensitivity: attempts = 1 or unset")] = [e for e in kept if e.extra.get("attempts") not in ("2+", "2")]
        if lb == "TheAgentCompany":
            out[(lb, "sensitivity: success = final.result == final.total")] = [
                Entry(e.lb, e.name, e.extra["score_alt_final_field"], e.k, extra=e.extra) for e in kept]
        if lb == "Terminal-Bench 2.0":
            out[(lb, "sensitivity: RQ3 analysis set (72)")] = [e for e in kept if e.extra.get("in_rq3_analysis_set") == 1]
        if lb in ("WebArena (BrowserGym)", "WorkArena L1", "WorkArena L2"):
            out[(lb, "sensitivity: reported std_err within 0.25 pp of binomial")] = [e for e in kept if e.extra.get("se_consistent") == 1]
    return out


def tb2_median_f(results_dir: Path) -> float:
    """Median over the Terminal-Bench 2.0 analysis-set submissions of f = v / (P(1-P)), the run-to-run
    share of a submission's single-run outcome variance (``tb2_within_task_variance.csv``,
    column ``f_run_noise_share``; submissions at P = 0 or 1 are undefined and dropped)."""
    w = pd.read_csv(results_dir / "tb2_within_task_variance.csv")
    return float(w["f_run_noise_share"].dropna().median())


def run(entries: list[Entry], rho=None, f_tb2: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Census statistics for every analysed leaderboard and set.

    rho: an ``estimate_rho.RhoEstimate`` (adds the empirical-rho families a_emp, a_pooled, a_pair);
    f_tb2: the Terminal-Bench 2.0 median run-noise share (adds it to the f sweep of b_f, b_f_kadj)."""
    rows, pair_rows = [], []
    fs = [(f, f"f={f:g}") for f in FS] + ([(f_tb2, "f=TB2 median")] if f_tb2 is not None else [])
    for (lb, label), es in variants(entries).items():
        n = N_VERIFIED[lb]
        names = [e.name for e in es]
        scores = [e.score for e in es]
        k = {e.name: e.k for e in es}
        meets = len(es) >= MIN_ENTRIES
        kl = [k[nm] for nm in names]
        # (family, param name, param, capped, label, bound fn, pair-aware)
        cfg = [("a", "rho", r, "", f"rho={r:g}", (lambda p1, p2, k1, k2, r=r: bound_a(p1, p2, n, r)), False) for r in RHOS]
        r_own = None
        if rho is not None:
            r_own = rho.rho_used[lb]
            cfg.append(("a_emp", "rho", r_own, "", "empirical rho (" + ("own median" if rho.source[lb] == "own" else "pooled median") + ")",
                        (lambda p1, p2, k1, k2: bound_a(p1, p2, n, r_own)), False))
            cfg.append(("a_pooled", "rho", rho.pooled_median, "", "pooled median rho",
                        (lambda p1, p2, k1, k2: bound_a(p1, p2, n, rho.pooled_median)), False))
            if lb in rho.matrices:
                mnames, R = rho.matrices[lb]
                pos = {nm: i for i, nm in enumerate(mnames)}
                ix = [pos[names[i]] for i in order(scores, names)]
                Rs = np.nan_to_num(R[np.ix_(ix, ix)], nan=r_own)  # rho is moot where SE = 0
                cfg.append(("a_pair", "rho", float("nan"), "", "pair-specific rho",
                            (lambda p1, p2, k1, k2, ia, ib, Rs=Rs: bound_a(p1, p2, n, Rs[ia, ib])), True))
        for v in VS:
            for cap in (True, False):
                cfg.append(("b", "v", v, cap, f"v={v:g}", (lambda p1, p2, k1, k2, v=v, cap=cap: bound_b(p1, p2, n, v, cap)), False))
        for v in VS:
            cfg.append(("b_kadj", "v", v, True, f"v={v:g}", (lambda p1, p2, k1, k2, v=v: bound_b(p1, p2, n, v, True, k1, k2)), False))
        for f, fl in fs:
            cfg.append(("b_f", "f", f, "", fl, (lambda p1, p2, k1, k2, f=f: bound_f(p1, p2, n, f)), False))
            cfg.append(("b_f_kadj", "f", f, "", fl, (lambda p1, p2, k1, k2, f=f: bound_f(p1, p2, n, f, k1, k2)), False))
        for fam, pname, par, cap, plabel, fn, pair in cfg:
            st = census_config(scores, names, kl, fn, pair=pair)
            rows.append({"leaderboard": lb, "set": label, "n_tasks": n, "meets_min_10_entries": meets,
                         "bound": fam, "param_name": pname, "param": par, "param_label": plabel, "capped": cap, **st})
        # adjacent-pair table
        idx = order(scores, names)
        for r_, (i, j) in enumerate(zip(idx[:-1], idx[1:]), start=1):
            p1, p2 = scores[i], scores[j]
            row = {"leaderboard": lb, "set": label, "rank_hi": r_, "entry_hi": names[i], "entry_lo": names[j],
                   "score_hi": p1, "score_lo": p2, "gap_pp": (p1 - p2) * 100, "k_hi": k[names[i]], "k_lo": k[names[j]]}
            for rr in RHOS:
                row[f"bound_a_rho{rr}_pp"] = float(bound_a(p1, p2, n, rr)) * 100
            if rho is not None:
                row["rho_used"] = r_own
                row["bound_a_emp_pp"] = float(bound_a(p1, p2, n, r_own)) * 100
                row["bound_a_pooled_pp"] = float(bound_a(p1, p2, n, rho.pooled_median)) * 100
                if lb in rho.matrices:
                    mnames, R = rho.matrices[lb]
                    pos = {nm: q for q, nm in enumerate(mnames)}
                    rp = R[pos[names[i]], pos[names[j]]]
                    row["rho_pair"] = float(rp)
                    row["bound_a_pair_pp"] = float(bound_a(p1, p2, n, r_own if np.isnan(rp) else rp)) * 100
            row["bound_b_v0.0753_capped_pp"] = float(bound_b(p1, p2, n, 0.0753, True)) * 100
            row["bound_b_v0.0753_capped_kadj_pp"] = float(bound_b(p1, p2, n, 0.0753, True, k[names[i]], k[names[j]])) * 100
            if f_tb2 is not None:
                row["bound_b_fTB2_pp"] = float(bound_f(p1, p2, n, f_tb2)) * 100
                row["bound_b_fTB2_kadj_pp"] = float(bound_f(p1, p2, n, f_tb2, k[names[i]], k[names[j]])) * 100
            pair_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def entries_table(entries: list[Entry]) -> pd.DataFrame:
    return pd.DataFrame([{
        "leaderboard": e.lb, "entry": e.name, "status": e.status, "reason": e.reason,
        "score": e.score, "k_runs": e.k, **{f"x_{k}": v for k, v in e.extra.items()}} for e in entries])


GH_EXP = "https://github.com/swe-bench/experiments"
TAU2_URL = "https://github.com/sierra-research/tau2-bench (web/leaderboard/public/submissions)"
BG_URL = "https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard"
SNAPSHOTS = {
    # leaderboard: (source URL, snapshot description, licence status, existing Wayback capture, snapshot date)
    "SWE-bench Verified": (GH_EXP, "commit 40f164d5, 2026-09-03", "no licence file", "none found; none submitted", "2026-09-03"),
    "SWE-bench Lite": (GH_EXP, "commit 40f164d5, 2026-09-03", "no licence file", "none found; none submitted", "2026-09-03"),
    "SWE-bench test (full)": (GH_EXP, "commit 40f164d5, 2026-09-03", "no licence file", "none found; none submitted", "2026-09-03"),
    "SWE-bench Multilingual": (GH_EXP, "commit 40f164d5, 2026-09-03", "no licence file", "none found; none submitted", "2026-09-03"),
    "Terminal-Bench 1.0": ("https://github.com/laude-institute/terminal-bench-leaderboard", "commit 3c08c2e1, 2025-11-14", "no licence file", "20251208110037 (predates local snapshot)", "2025-11-14"),
    "Terminal-Bench 2.0": ("https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard", "revision 572b2614 (modified 2026-05-15), retrieved 2026-10-04", "Apache-2.0", "none found; none submitted", "2026-05-15"),
    "TheAgentCompany": ("https://github.com/TheAgentCompany/experiments", "commit f9b1e941, 2026-08-01", "no licence file", "none found; none submitted", "2026-08-01"),
    "tau2-bench banking_knowledge": (TAU2_URL, "commit 5bfa7e37, 2026-09-28", "MIT", "20260221060820 (predates local snapshot)", "2026-09-28"),
    "tau2-bench airline": (TAU2_URL, "commit 5bfa7e37, 2026-09-28", "MIT", "20260221060820 (predates local snapshot)", "2026-09-28"),
    "tau2-bench telecom": (TAU2_URL, "commit 5bfa7e37, 2026-09-28", "MIT", "20260221060820 (predates local snapshot)", "2026-09-28"),
    "AppWorld test_normal": ("https://github.com/stonybrooknlp/appworld-leaderboard", "commit c1f56015, 2026-09-02", "no licence file", "none found; none submitted", "2026-09-02"),
    "AppWorld test_challenge": ("https://github.com/stonybrooknlp/appworld-leaderboard", "commit c1f56015, 2026-09-02", "no licence file", "none found; none submitted", "2026-09-02"),
    "WebArena (BrowserGym)": (BG_URL, "revision 294ebe1d, 2026-04-14", "MIT (Space card)", "none found; none submitted", "2026-04-14"),
    "WebArena (Google Sheet)": ("https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ", "CSV export retrieved 2026-10-04", "none stated", "20260930023707 (predates local snapshot)", "2026-10-04"),
    "WorkArena L1": (BG_URL, "revision 294ebe1d, 2026-04-14", "MIT (Space card)", "none found; none submitted", "2026-04-14"),
    "WorkArena L2": (BG_URL, "revision 294ebe1d, 2026-04-14", "MIT (Space card)", "none found; none submitted", "2026-04-14"),
    "AndroidWorld": ("https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo", "CSV export retrieved 2026-10-04", "none stated (community-submitted, unverified)", "none found; none submitted", "2026-10-04"),
    "MLE-bench": ("https://github.com/openai/mle-bench", "commit 507f92e1, 2026-04-24", "MIT", "none found; none submitted", "2026-04-24"),
    "HAL SWE-bench Verified Mini": ("https://hal.cs.princeton.edu/swebench_verified_mini", "HTML retrieved 2026-10-04", "none stated", "none found; none submitted", "2026-10-04"),
    "HAL GAIA": ("https://hal.cs.princeton.edu/gaia", "HTML retrieved 2026-10-04", "none stated", "20260922043251 (predates local snapshot)", "2026-10-04"),
    "HAL SciCode": ("https://hal.cs.princeton.edu/scicode", "HTML retrieved 2026-10-04", "none stated", "none found; none submitted", "2026-10-04"),
    "HAL ScienceAgentBench": ("https://hal.cs.princeton.edu/scienceagentbench", "HTML retrieved 2026-10-04", "none stated", "none found; none submitted", "2026-10-04"),
    "HAL USACO": ("https://hal.cs.princeton.edu/usaco", "HTML retrieved 2026-10-04", "none stated", "none found; none submitted", "2026-10-04"),
}
# default raw source file per leaderboard (paths as in the manifests); per-entry files override via extra["src_file"]
LB_SOURCE_FILE = {
    "SWE-bench Verified": "leaderboards/swe-bench/swebench_verified.json",
    "SWE-bench Lite": "leaderboards/swe-bench/swebench_lite.json",
    "SWE-bench test (full)": "leaderboards/swe-bench/swebench_test.json",
    "SWE-bench Multilingual": "leaderboards/swe-bench/swebench_multilingual.json",
    "Terminal-Bench 1.0": "leaderboards/terminal-bench-1/tb1_run_results.json",
    "TheAgentCompany": "leaderboards/theagentcompany/tac_results.json",
}


def _row(cen, lb, label, fam, par=None, capped=None):
    """The one census row for (leaderboard, set, family[, parameter]). ``par`` may be omitted for
    families that hold one row per board (a_emp, a_pooled, a_pair)."""
    q = cen[(cen.leaderboard == lb) & (cen.set == label) & (cen.bound == fam)]
    if par is not None:
        q = q[np.isclose(q.param.astype(float), par)]
    if fam == "b":
        q = q[q.capped == (True if capped is None else capped)]
    assert len(q) == 1, (lb, fam, par, len(q))
    return q.iloc[0]


def _frac(r):
    return f"{int(r.n_adjacent_below)}/{int(r.n_adjacent_pairs)} ({100 * r.share_adjacent_below:.1f}%)"


def _pct(x):
    return f"{100 * x:.1f}%"


def _sep(ok):
    if ok is None or (isinstance(ok, float) and math.isnan(ok)):
        return "n/a"
    return "separable" if ok else "not separable"


def agg_configs(f_tb2: float | None) -> list[tuple[str, float | None, str]]:
    """(family, parameter, label) of the configurations summarised across leaderboards."""
    out = [("a_emp", None, "(a) empirical rho [headline]"), ("a_pooled", None, "(a) pooled median rho"),
           ("a", 0.0, "(a) rho = 0 [conservative]"), ("a", 0.5, "(a) rho = 0.5")]
    if f_tb2 is not None:
        fl = sorted([(f, f"{f:g}") for f in FS] + [(f_tb2, f"f_TB2 = {f_tb2:.3f}")])
        out += [("b_f_kadj", f, f"(b) f = {lab}, k-adj.") for f, lab in fl]
        out += [("b_f", f_tb2, f"(b) f = f_TB2, single-run form")]
    return out


def aggregates(cen: pd.DataFrame, lbs: list[str], f_tb2: float | None = None) -> list[dict]:
    """Across-leaderboard summary per bound configuration (primary sets)."""
    has_emp = (cen.bound == "a_emp").any()
    out = []
    for fam, par, lab in agg_configs(f_tb2):
        if fam in ("a_emp", "a_pooled") and not has_emp:
            continue
        rs = [_row(cen, lb, "primary", fam, par) for lb in lbs]
        tiers = np.array([int(r.n_tiers) for r in rs])
        out.append({
            "config": lab, "fam": fam, "par": par, "n_lb": len(rs),
            "top1_vs_2_not_sep": int(sum(not r.top1_vs_2_separable for r in rs)),
            "top1_vs_5_not_sep": int(sum(not r.top1_vs_5_separable for r in rs)),
            "tiers_median": float(np.median(tiers)), "tiers_min": int(tiers.min()), "tiers_max": int(tiers.max()),
            "all_pairs_below_median": float(np.median([r.share_all_below for r in rs])),
            "all_pairs_below_min": float(min(r.share_all_below for r in rs)),
            "all_pairs_below_max": float(max(r.share_all_below for r in rs)),
            "adjacent_below_median": float(np.median([r.share_adjacent_below for r in rs])),
            "adjacent_below_min": float(min(r.share_adjacent_below for r in rs)),
            "adjacent_ge_90": int(sum(r.share_adjacent_below >= 0.9 for r in rs)),
            "entries_median": float(np.median([r.n_entries for r in rs])),
            "tiers_per_entry_median": float(np.median([r.n_tiers / r.n_entries for r in rs])),
        })
    return out


def write_summary(cen: pd.DataFrame, entries: list[Entry], path: Path, side_kept: dict[str, int] | None = None,
                  rho=None, f_tb2: float | None = None) -> None:
    L: list[str] = []
    w = L.append
    et = entries_table(entries)
    lbs = census_leaderboards(entries)
    status = leaderboard_status(entries, side_kept)
    nl = len(lbs)
    w("# RQ2 leaderboard census: summary")
    w("")
    w("Numbers only. Generated by `analysis/leaderboards/census.py`; every figure below is in `census.csv`, "
      "`census_pairs.csv`, `census_entries.csv` or `rho_by_board.csv`. Snapshot of 2026-10-04 (retrieval dates per leaderboard below). "
      "Task counts n: `research/leaderboard-n-verification.md`.")
    w("")
    w("Revision of 2026-10-04 (adjudication of simulated review 1, decision log): the headline bound is the paired item-sampling bound with an estimated rho, "
      "the rerun-bound sweep is parameterised on the run-noise share f, and rho = 0 is kept as the conservative case. Old and new values: `analysis/CHANGES-2026-10-04.md`.")
    w("")
    w("## Definitions")
    w("")
    w("- Score P = share of the n tasks solved (mean over k runs where an entry reports k runs). Entries sorted by P, best first; ties by entry name.")
    w("- SE = sqrt(P(1-P)/n), n = task count of the leaderboard (not n x k).")
    w("- rho: the correlation, across the n tasks, between two entries' per-task outcomes. For single-run entries it is the Pearson (phi) correlation of the two 0/1 outcome vectors; for entries averaging k > 1 runs it is the expected phi between one run of each (across-task covariance of the per-task success rates over sqrt(P1(1-P1) P2(1-P2))), the form the bound needs (`estimate_rho.py`). Per board: the median over adjacent pairs (census order) and the interquartile range. 7 of the 22 boards have per-task data locally; the other 15 use the pooled median, the median of the 7 per-board medians.")
    w("- Bound (a), headline, the paired item-sampling bound: 1.96 x sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2) with rho = the board's own median adjacent-pair rho (7 boards) or the pooled median (15 boards). rho = 0 (unpaired, scores only) is the conservative case and the sweep rho in {0.25, 0.5, 0.75} is kept; `a_pooled` applies the pooled median to every board and `a_pair` (7 per-task boards) uses each pair's own rho. No transport assumption. Answers: does the ordering generalise beyond these tasks.")
    w("- Bound (b), fixed-task-set rerun, run-to-run variance only: 1.96 x sqrt((f/n)(P1(1-P1)/k1 + P2(1-P2)/k2)), where the within-task variance of an entry is v = f P(1-P) (f = share of the single-run outcome variance that is run noise, transported from Terminal-Bench 2.0; v <= P(1-P) for any f <= 1, so no cap). f in {0.25, 0.5, f_TB2, 0.75}, f_TB2 = the median over the 72 TB2 analysis-set submissions of v/(P(1-P)) with v the unbiased mean within-task variance"
      + (f" (f_TB2 = {f_tb2:.4f})" if f_tb2 is not None else "")
      + ". `b_f_kadj` is the form above (k = runs per entry); `b_f` sets k1 = k2 = 1, which overstates rerun noise for entries averaging k > 1 runs. MLE-bench does not state k, so its k is 1 and both forms coincide. The earlier absolute-v sweep (v in {0.025, 0.0753, 0.1618, 0.2022}, capped) is superseded and kept in `census.csv` as families `b`, `b_kadj`.")
    w("- A pair is below the bound when gap < bound (strict), or when the two scores are equal (at P = 0 or 1 the binomial SE is 0, so a tie would otherwise count as separable). Separable = not below. The share of adjacent gaps below the bound has a Wilson 95% CI over the adjacent pairs; the share over all pairs has no CI (pairs are not independent).")
    w("- Tiers (greedy): the best entry opens tier 1; each later entry joins the tier while it is not separable from the tier's best entry; the first entry separable from it opens the next tier.")
    w(f"- Selection: the 26 leaderboards that pass tests T1-T3 of `decisions/architecture-v1.md` (inventory: `research/leaderboard-inventory.md`) are analysed unless a written criterion excludes them: n not verifiable from documentation, or fewer than {MIN_ENTRIES} kept entries after the entry-level checks (T4). {nl} of 26 are analysed.")
    w("")
    w("## Leaderboards analysed and excluded (26 passing T1-T3)")
    w("")
    w("| leaderboard | status | reason / kept entries |")
    w("|---|---|---|")
    for lb, st, why in status:
        w(f"| {lb} | {st} | {why} |")
    w("")
    w("## Leaderboards kept, snapshots, licence status")
    w("")
    w("| leaderboard | n | kept / listed | k per entry | source | snapshot | licence | existing Wayback capture |")
    w("|---|---|---|---|---|---|---|---|")
    for lb in lbs:
        g = et[et.leaderboard == lb]
        kept = g[g.status == "kept"]
        kr = f"{kept.k_runs.min():g}" if kept.k_runs.min() == kept.k_runs.max() else f"{kept.k_runs.min():g} to {kept.k_runs.max():g}"
        src, snap, lic, wb, _ = SNAPSHOTS[lb]
        w(f"| {lb} | {N_VERIFIED[lb]} | {len(kept)} / {len(g)} | {kr} | {src} | {snap} | {lic} | {wb} |")
    w("")
    nolic = [lb for lb in lbs if lb not in [x for x in lbs if ("MIT" in SNAPSHOTS[x][2] or "Apache" in SNAPSHOTS[x][2])]]
    lic = [lb for lb in lbs if ("MIT" in SNAPSHOTS[lb][2] or "Apache" in SNAPSHOTS[lb][2])]
    w(f"{len(nolic)} of the {nl} analysed leaderboards rest on sources with no stated licence or terms; the other {len(lic)} ({', '.join(lic)}) carry an open licence. `census.py` releases derived numbers only (`results/release/`); no raw leaderboard file is redistributed. Whether the no-licence sources are acceptable for release is the author's decision.")
    w("")
    has_rho = rho is not None
    if has_rho:
        own = rho.boards[rho.boards.rho_source.str.startswith("own")]
        w("## Task-level correlation rho between adjacent entries")
        w("")
        w("Per board with per-task data: the correlation of the adjacent entries' per-task outcomes (definition above), median over adjacent pairs with the 25th and 75th percentiles, the range over pairs, the median over all pairs for reference, and (k > 1 only) the Pearson correlation of the per-task success rates themselves, which is higher because rates carry less run noise than single runs. Pairs: `rho_adjacent_pairs.csv`.")
        w("")
        w("| leaderboard | n | entries | max k | adjacent pairs defined | median (IQR) | range | all-pairs median | Pearson of rates, median |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in own.itertuples():
            pr = f"{r.rho_pearson_rates_adjacent_median:.3f}" if r.k_runs_max > 1 else "same"
            w(f"| {r.leaderboard} | {int(r.n_tasks)} | {int(r.n_entries)} | {r.k_runs_max:g} | {int(r.n_adjacent_defined)} of {int(r.n_adjacent_pairs)} | "
              f"{r.rho_adjacent_median:.3f} ({r.rho_adjacent_q25:.3f} to {r.rho_adjacent_q75:.3f}) | {r.rho_adjacent_min:.3f} to {r.rho_adjacent_max:.3f} | "
              f"{r.rho_all_pairs_median:.3f} | {pr} |")
        med = own.rho_adjacent_median.to_numpy()
        w("")
        w(f"Pooled median (median of the {len(own)} per-board medians): {rho.pooled_median:.4f}; per-board medians range {med.min():.3f} to {med.max():.3f}, "
          f"quartiles {np.percentile(med, 25):.3f} and {np.percentile(med, 75):.3f}. The {nl - len(own)} boards without per-task data use {rho.pooled_median:.4f}. "
          "rho is not constant along a board: the median over all pairs is lower than over adjacent pairs, so the board-level adjacent median is somewhat high for distant pairs; the pair-specific sensitivity below measures the effect.")
        w("")
    w("## Headline: share of adjacent gaps below bound (a)")
    w("")
    w("Cells: below / adjacent pairs (share). rho used = the board's own median adjacent rho (own) or the pooled median (pooled). Wilson 95% CI of the share at the rho used.")
    w("")
    if has_rho:
        w("| leaderboard | rho used | empirical rho [headline] | Wilson 95% CI | rho = 0 [conservative] | rho = 0.25 | rho = 0.5 | rho = 0.75 | pooled median rho |")
        w("|---|---|---|---|---|---|---|---|---|")
        for lb in lbs:
            e_ = _row(cen, lb, "primary", "a_emp")
            rs = [_row(cen, lb, "primary", "a", r) for r in RHOS]
            pl = _row(cen, lb, "primary", "a_pooled")
            w(f"| {lb} | {e_.param:.3f} ({rho.source[lb]}) | {_frac(e_)} | {_pct(e_.wilson_lo)} to {_pct(e_.wilson_hi)} | "
              + " | ".join(_frac(r) for r in rs) + f" | {_frac(pl)} |")
    else:
        w("| leaderboard | rho = 0 | rho = 0.25 | rho = 0.5 | rho = 0.75 | Wilson 95% CI, rho = 0 |")
        w("|---|---|---|---|---|---|")
        for lb in lbs:
            rs = [_row(cen, lb, "primary", "a", r) for r in RHOS]
            w(f"| {lb} | " + " | ".join(_frac(r) for r in rs) + f" | {_pct(rs[0].wilson_lo)} to {_pct(rs[0].wilson_hi)} |")
    w("")
    if f_tb2 is not None:
        w("## Fixed-task-set rerun bound (b), sweep over the run-noise share f")
        w("")
        w("Cells: below / adjacent pairs (share), k-adjusted form. The last column is the single-run form at f_TB2 (k1 = k2 = 1).")
        w("")
        fl = sorted([(f, f"f = {f:g}") for f in FS] + [(f_tb2, f"f = f_TB2 = {f_tb2:.3f}")])
        w("| leaderboard | " + " | ".join(lab for _, lab in fl) + " | Wilson 95% CI, f_TB2 | single-run form, f_TB2 |")
        w("|---|" + "---|" * (len(fl) + 2))
        for lb in lbs:
            rs = [_row(cen, lb, "primary", "b_f_kadj", f) for f, _ in fl]
            at = [r for (f, _), r in zip(fl, rs) if f == f_tb2][0]
            sr = _row(cen, lb, "primary", "b_f", f_tb2)
            w(f"| {lb} | " + " | ".join(_frac(r) for r in rs) + f" | {_pct(at.wilson_lo)} to {_pct(at.wilson_hi)} | {_frac(sr)} |")
        w("")
    w("## All pairs, tiers, top entry")
    w("")
    w("All pairs: share of all C(m,2) pairs below the bound. Tiers: greedy count. Top entry: is #1 separable from #2 / from #5 (gap / bound in percentage points, in brackets).")
    w("")
    w("| leaderboard | bound | all pairs below | tiers | #1 vs #2 | #1 vs #5 |")
    w("|---|---|---|---|---|---|")
    table_cfgs = [c for c in agg_configs(f_tb2) if (c[0] == "a_emp") or (c[0] == "a" and c[1] == 0.0)
                  or (c[0] == "b_f_kadj" and f_tb2 is not None and np.isclose(c[1], f_tb2))]
    for lb in lbs:
        for fam, par, lab in table_cfgs:
            if fam == "a_emp" and not has_rho:
                continue
            r = _row(cen, lb, "primary", fam, par)
            w(f"| {lb} | {lab} | {_pct(r.share_all_below)} | {int(r.n_tiers)} | "
              f"{_sep(r.top1_vs_2_separable)} ({r.gap_1_2_pp:.2f} / {r.bound_1_2_pp:.2f}) | "
              f"{_sep(r.top1_vs_5_separable)} ({r.gap_1_5_pp:.2f} / {r.bound_1_5_pp:.2f}) |")
    w("")
    w("Tiers over the full rho and f sweeps: `census.csv`, column `n_tiers`.")
    w("")
    w(f"## Across the {nl} leaderboards")
    w("")
    w("Counts and medians over the analysed leaderboards (primary sets). Medians are over leaderboards, not entries.")
    w("")
    w("| bound | #1 not separable from #2 | #1 not separable from #5 | tiers: median (min to max) | median tiers per entry | all pairs below: median (min to max) | adjacent below: median (min) | adjacent share >= 90% |")
    w("|---|---|---|---|---|---|---|---|")
    for a in aggregates(cen, lbs, f_tb2):
        w(f"| {a['config']} | {a['top1_vs_2_not_sep']} of {a['n_lb']} | {a['top1_vs_5_not_sep']} of {a['n_lb']} | "
          f"{a['tiers_median']:g} ({a['tiers_min']} to {a['tiers_max']}) | {a['tiers_per_entry_median']:.3f} | "
          f"{_pct(a['all_pairs_below_median'])} ({_pct(a['all_pairs_below_min'])} to {_pct(a['all_pairs_below_max'])}) | "
          f"{_pct(a['adjacent_below_median'])} ({_pct(a['adjacent_below_min'])}) | {a['adjacent_ge_90']} of {a['n_lb']} |")
    w("")
    if has_rho:
        w("## Sensitivity: each pair's own rho (boards with per-task data)")
        w("")
        w("The headline applies one rho per board (the adjacent-pair median). Here each pair uses its own estimated rho (a_pair).")
        w("")
        w("| leaderboard | bound | adjacent below | all pairs below | tiers | #1 vs #2 | #1 vs #5 |")
        w("|---|---|---|---|---|---|---|")
        for lb in lbs:
            if lb not in rho.matrices:
                continue
            for fam, lab in (("a_emp", "board median rho"), ("a_pair", "pair-specific rho")):
                r = _row(cen, lb, "primary", fam)
                w(f"| {lb} | {lab} | {_frac(r)} | {_pct(r.share_all_below)} | {int(r.n_tiers)} | {_sep(r.top1_vs_2_separable)} | {_sep(r.top1_vs_5_separable)} |")
        w("")
    w("## Sensitivity sets")
    w("")
    hdr = ["leaderboard", "set", "kept"] + (["(a) empirical rho"] if has_rho else []) + ["(a) rho = 0", "(a) rho = 0.5"] + \
          (["(b) f = f_TB2, k-adj."] if f_tb2 is not None else [])
    w("| " + " | ".join(hdr) + " |")
    w("|" + "---|" * len(hdr))
    sens = sorted({(r.leaderboard, r.set) for r in cen.itertuples() if r.set != "primary"},
                  key=lambda t: (lbs.index(t[0]), t[1]))
    for lb, label in sens:
        a0, a5 = _row(cen, lb, label, "a", 0.0), _row(cen, lb, label, "a", 0.5)
        flag = "" if a0.meets_min_10_entries else " (fewer than 10 entries)"
        cells = [lb, label, f"{int(a0.n_entries)}{flag}"]
        if has_rho:
            cells.append(_frac(_row(cen, lb, label, "a_emp")))
        cells += [_frac(a0), _frac(a5)]
        if f_tb2 is not None:
            cells.append(_frac(_row(cen, lb, label, "b_f_kadj", f_tb2)))
        w("| " + " | ".join(cells) + " |")
    w("")
    w("## Exclusions by entry")
    w("")
    ex = et[et.status == "excluded"]
    w(f"{len(ex)} entries excluded from the {nl} analysed leaderboards and the airline split ({len(et)} entries read; tau2-bench retail, GAIA test and GAIA validation were not read because their n is not verifiable). "
      "Exclusions for leaderboards that fail the selection rule are in `research/leaderboard-entry-exclusions.csv` and `research/leaderboard-inventory.md`.")
    w("")
    w("| leaderboard | entry | reason |")
    w("|---|---|---|")
    for r in ex.itertuples():
        w(f"| {r.leaderboard} | {r.entry} | {r.reason} |")
    w("")
    w("## Entry-level checks")
    w("")
    w("- SWE-bench splits: every resolved id lies in the split's task set (the id set shared by all entries that list all n instances); where the metadata reports a resolved percentage it equals 100 x count / n at the reported precision (two decimals for Verified, Lite and test; one for Multilingual), else the entry is excluded (above). Entries with attempts = 2+ are kept in the primary set and dropped in the sensitivity set.")
    w("- Terminal-Bench 1.0: for every run file, resolved + unresolved ids equal the 80 task ids of terminal-bench-core 0.1.1 in the official registry; trials per entry are a multiple of 80 (k = 5 for all 26 entries).")
    w("- Terminal-Bench 2.0: every kept submission has trials on all 89 tasks of the union; k = trials / 89 ranges from 1 to 10.")
    w("- TheAgentCompany: success = every checkpoint passed (the paper's full completion score); a sensitivity set uses final.result == final.total, which differs from the checkpoint rule on 55 of 3,056 task results in the files.")
    w("- tau2-bench: voice submissions, entries absent from the submissions manifest and entries on versions whose task definitions differ are excluded (banking_knowledge: only 1.0.1; airline and retail: only 1.0.0 and 1.0.1, because 27 airline and 26 retail tasks were fixed in 1.0.0; telecom: 0.1.3 onward). pass_1 is recovered to an integer count out of n x k (k = trials stated in the notes or, failing that, the largest pass_k reported), allowing for the printed rounding.")
    w("- HAL pages: accuracy x n x Runs is an integer at the printed precision for every kept entry; the point value of a multi-run entry is read as the mean of its runs. The pages do not define accuracy; that every kept score is a multiple of 1/n is the evidence for a per-task binary metric.")
    w("- AppWorld: task goal completion (the leaderboard's primary column, `all` level) is a multiple of 1/n at one decimal; one run per entry is assumed (the file states none).")
    w("- BrowserGym leaderboard (WebArena, WorkArena L1, L2): entries are excluded when their own field says the evaluation protocol was not followed, when their comment states another task count, or when the score is not a multiple of 1/n at the printed precision. The reported std_err is not an exclusion test; a sensitivity set drops entries whose std_err is more than 0.25 pp from sqrt(P(1-P)/n). Entries flagged benchmark_tuned are kept.")
    w("- WebArena and AndroidWorld Google Sheets: only the main block of the WebArena sheet is read (its 'WebArena Subset' and 'Human Performance' blocks are other task sets); scores must be multiples of 1/(n x trials) at the printed precision; AndroidWorld trials default to 1 where the cell is blank.")
    w("- MLE-bench: score = the README's `All (%)` column (mean over seeds of the medal share over the 75 competitions); rows with the footnote 'padded incomplete seeds with failing scores' and the two 'additional submissions' (test-set feedback) are excluded; k is not stated and is set to 1.")
    w("- Terminal-Bench 1.0: entries 20250911_chaterm_claude-4-sonnet and 20251010_Chaterm_claude-4-5-sonnet contain one identical run file (2025-09-11__02-03-58, same resolved ids); both are kept.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- release (derived statistics only)

def _manifest_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in ("leaderboards_manifest.csv", "tau2_manifest.csv", "tb2_manifest.csv"):
        p = root / "data" / f
        if p.exists():
            m = pd.read_csv(p, dtype=str)
            out.update(dict(zip(m["path"], m["sha256"])))
    return out


def _entry_source_path(e: Entry, hashes: dict[str, str]) -> str | None:
    if e.lb == "Terminal-Bench 2.0":
        for ext in ("metadata.yaml", "metadata.yml"):
            p = f"submissions/terminal-bench/2.0/{e.name}/{ext}"
            if p in hashes:
                return p
        return None
    src = e.extra.get("src_file")
    if src:
        return src if src.startswith("web/") else f"leaderboards/{src}"
    return LB_SOURCE_FILE.get(e.lb)


def release_tables(entries: list[Entry], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derived per-entry statistics of the analysed leaderboards (kept entries), and the entry-level
    exclusions. No raw leaderboard record is copied: only entry name, score, n, k, source URL,
    snapshot date and the SHA-256 of the source file held locally."""
    hashes = _manifest_hashes(root)
    lbs = census_leaderboards(entries)
    rows, ex = [], []
    for lb in lbs:
        url, _, _, _, date = SNAPSHOTS[lb]
        for e in sorted((x for x in entries if x.lb == lb), key=lambda x: x.name):
            if e.status == "kept":
                p = _entry_source_path(e, hashes)
                rows.append({"leaderboard": lb, "entry": e.name, "score": round(e.score, 10), "n_tasks": N_VERIFIED[lb],
                             "k_runs": e.k, "source_url": url, "snapshot_date": date,
                             "source_file_sha256": hashes.get(p, "") if p else ""})
            else:
                ex.append({"leaderboard": lb, "entry": e.name, "reason": e.reason})
    return pd.DataFrame(rows), pd.DataFrame(ex)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


RELEASE_README = """# Derived leaderboard statistics (release)

Derived per-entry statistics of the RQ2 census (`analysis/leaderboards/census.py`). Only these
columns are released: leaderboard, entry name, score (share of the n tasks solved), n_tasks, k_runs,
source_url, snapshot_date and source_file_sha256 (SHA-256 of the file held locally from that
source, as listed in `data/leaderboards_manifest.csv`, `data/tau2_manifest.csv` and
`data/tb2_manifest.csv`). No raw leaderboard file, trajectory, contact detail or free-text field of
a source is redistributed. Sources without a stated licence or terms are listed in
`census_summary.md`; the released values are derived numbers (counts divided by n), and whether
they may be published is the author's decision. `census_exclusions_release.csv` lists entries
excluded by an entry-level check, with the reason.

`rho_by_board_release.csv` and `rho_adjacent_pairs_release.csv` (added 2026-10-04) hold the
estimated task-level correlation rho between adjacent entries, computed from per-task outcomes
that are not released (`analysis/leaderboards/estimate_rho.py`): per board the median, quartiles and
range over adjacent pairs and the rho used in the headline bound, and per adjacent pair the entry
names, scores and rho. They are derived numbers of the same status as the scores.
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parents[1]))
    ap.add_argument("--out", default=str(here / "results"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import estimate_rho  # imported here: estimate_rho imports this module

    entries = load_all(root)
    # leaderboards excluded for an unverifiable n are not analysed; tau2 retail is read only to state its entry count
    side = {"tau2-bench retail": sum(e.status == "kept" for e in load_tau2(root, "tau2-bench retail"))}
    est = estimate_rho.estimate(entries, root)
    f_tb2 = tb2_median_f(root / "analysis/results")
    cen, pairs = run(entries, est, f_tb2)
    cen.to_csv(out / "census.csv", index=False)
    pairs.to_csv(out / "census_pairs.csv", index=False)
    entries_table(entries).to_csv(out / "census_entries.csv", index=False)
    estimate_rho.write_outputs(est, out)
    (out / "census_parameters.json").write_text(json.dumps({
        "pooled_median_rho": est.pooled_median, "f_tb2_median": f_tb2, "rho_used": est.rho_used,
        "rho_source": est.source, "f_sweep": sorted(list(FS) + [f_tb2]), "rho_sweep": list(RHOS)}, indent=2), encoding="utf-8")
    write_summary(cen, entries, out / "census_summary.md", side, est, f_tb2)
    rel = out / "release"
    rel.mkdir(parents=True, exist_ok=True)
    kept, excl = release_tables(entries, root)
    kept.to_csv(rel / "census_entries_release.csv", index=False)
    excl.to_csv(rel / "census_exclusions_release.csv", index=False)
    rb = est.boards.copy()
    rb.to_csv(rel / "rho_by_board_release.csv", index=False)
    est.pairs[["leaderboard", "rank_hi", "entry_hi", "entry_lo", "score_hi", "score_lo", "k_hi", "k_lo", "rho", "rho_pearson_rates"]].to_csv(
        rel / "rho_adjacent_pairs_release.csv", index=False)
    (rel / "README.md").write_text(RELEASE_README, encoding="utf-8")
    print(f"entries read {len(entries)}, kept {sum(e.status == 'kept' for e in entries)}; "
          f"{len(census_leaderboards(entries))} leaderboards analysed; wrote {out}")


if __name__ == "__main__":
    main()
