# SPDX-License-Identifier: MIT
"""Task-level correlation rho between adjacent leaderboard entries, estimated from per-task results.

Why. The headline census bound is the PAIRED item-sampling bound
    1.96 * sqrt(SE1^2 + SE2^2 - 2 rho SE1 SE2),     SE = sqrt(P(1-P)/n),
where rho is the correlation, across tasks, between the two entries' task outcomes
(decision log 2026-10-04, adjudication of simulated review 1, M1). This script estimates rho
from every analysed leaderboard whose per-task outcomes are held locally.

Definition. For entries i and j with per-task success rates p_i(t), p_j(t) over the n tasks t
(a task outcome is 0/1 for an entry with one run; for an entry with k > 1 runs, p(t) is the share
of its runs that solved task t),
    P_i = mean_t p_i(t),
    rho_ij = [ mean_t p_i(t) p_j(t) - P_i P_j ] / sqrt( P_i (1-P_i) P_j (1-P_j) ).
For two single-run entries this is exactly the Pearson (phi) correlation of the two per-task binary
outcome vectors. For entries with k > 1 runs it is the expected phi between one randomly chosen run
of each entry: the numerator is the across-task covariance of the success rates (runs of different
entries are independent given the task) and the denominator is the single-run outcome variance
P(1-P), which is the variance the bound uses in SE. Pearson's correlation of the per-task rates
themselves (``rho_pearson_rates``, a recorded secondary column) is larger for k > 1 because the
rates carry less run noise than a single run; it is not the quantity the bound needs.
rho is undefined (NaN) when an entry solves none or all of the tasks.

Pairs. "Adjacent" means adjacent in the census order (score descending, ties by entry name) over
the kept entries of the primary set. Per board the script reports the median over adjacent pairs and
the interquartile range (25th and 75th percentiles, linear interpolation), plus the range and, for
reference, the median over all pairs.

Boards with per-task data (7 of the 22 analysed): SWE-bench Verified, Lite, test (full) and
Multilingual (resolved instance ids), Terminal-Bench 1.0 (resolved task ids per run), Terminal-Bench
2.0 (trial outcomes, ``analysis/results/tb2_long.csv``), TheAgentCompany (checkpoint results per
task). The other 15 publish aggregates only; they use the pooled median, the median of the per-board
medians of the 7 (``pooled_median``).

Usage: python estimate_rho.py [--root <project root>] [--out <output dir>]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import census as C

# ----------------------------------------------------------------------------- statistics


def rho_matrix(P: np.ndarray) -> np.ndarray:
    """Single-run-equivalent correlation between every pair of entries.

    P: (m entries, n tasks) per-task success rates in [0, 1]. Returns the (m, m) matrix of
    rho_ij defined in the module docstring; NaN where an entry has P(1-P) = 0."""
    P = np.asarray(P, dtype=float)
    n = P.shape[1]
    mean = P.mean(axis=1)
    cov = P @ P.T / n - np.outer(mean, mean)
    sd = np.sqrt(mean * (1.0 - mean))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.outer(sd, sd)
    r[(sd == 0)[:, None] | (sd == 0)[None, :]] = np.nan
    return r


def pearson_matrix(P: np.ndarray) -> np.ndarray:
    """Pearson correlation of the per-task rate vectors themselves (population moments)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[1]
    mean = P.mean(axis=1)
    cov = P @ P.T / n - np.outer(mean, mean)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.outer(sd, sd)
    r[(sd < 1e-12)[:, None] | (sd < 1e-12)[None, :]] = np.nan
    return r


def adjacent(M: np.ndarray) -> np.ndarray:
    """Entries (i, i+1) of a square matrix: the adjacent pairs of the sorted order."""
    return np.array([M[i, i + 1] for i in range(M.shape[0] - 1)])


def summarise(x: np.ndarray) -> dict:
    """Median, IQR and range of the defined (non-NaN) values."""
    x = np.asarray(x, dtype=float)
    ok = x[~np.isnan(x)]
    if ok.size == 0:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}
    q25, q75 = np.percentile(ok, [25, 75])
    return {"n": int(ok.size), "median": float(np.median(ok)), "q25": float(q25), "q75": float(q75),
            "min": float(ok.min()), "max": float(ok.max())}


# ----------------------------------------------------------------------------- per-task outcome loaders
# Each returns {entry name: per-task success-rate vector} on a common, sorted task order.


def swebench_outcomes(root: Path, split: str, n: int) -> dict[str, np.ndarray]:
    d = json.loads((root / "data/raw/leaderboards/swe-bench" / f"swebench_{split}.json").read_text(encoding="utf-8"))
    full = [frozenset(r["covered_ids"]) for r in d.values() if r["source"] and len(r["covered_ids"]) == n]
    canon = sorted(full[0])
    pos = {t: i for i, t in enumerate(canon)}
    out = {}
    for key, r in d.items():
        res = r.get("resolved")
        if r["source"] is None or res is None:
            continue
        v = np.zeros(len(canon))
        for t in set(res):
            if t in pos:
                v[pos[t]] = 1.0
        out[key] = v
    return out


def tb1_outcomes(root: Path) -> dict[str, np.ndarray]:
    """Per-task share of trials resolved. A run file holds one trial per task, or several (an id
    then repeats in the resolved and unresolved lists), so resolved and total trials are counted
    per task id and summed over the entry's run files."""
    d = json.loads((root / "data/raw/leaderboards/terminal-bench-1/tb1_run_results.json").read_text(encoding="utf-8"))["files"]
    ref = sorted(json.loads((root / "data/raw/leaderboards/n_verification/tb1_registry_0.1.1_task_ids.json").read_text(encoding="utf-8"))["task_ids"])
    pos = {t: i for i, t in enumerate(ref)}
    res: dict[str, np.ndarray] = {}
    tot: dict[str, np.ndarray] = {}
    for p, v in d.items():
        key = p.split("/")[2]
        r = res.setdefault(key, np.zeros(len(ref)))
        t = tot.setdefault(key, np.zeros(len(ref)))
        for x in v["resolved_ids"]:
            r[pos[x]] += 1
            t[pos[x]] += 1
        for x in v["unresolved_ids"]:
            t[pos[x]] += 1
    return {k: res[k] / tot[k] for k in res}


def tb2_outcomes(root: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(root / "analysis/results/tb2_long.csv", usecols=["submission", "task_id", "success"])
    rate = df.groupby(["submission", "task_id"])["success"].mean().unstack("task_id")
    rate = rate.reindex(columns=sorted(rate.columns))
    return {s: rate.loc[s].to_numpy(dtype=float) for s in rate.index}


def tac_outcomes(root: Path) -> dict[str, np.ndarray]:
    t = json.loads((root / "data/raw/leaderboards/theagentcompany/tac_results.json").read_text(encoding="utf-8"))["results"]["1.0.0"]
    out = {}
    for key, tasks in t.items():
        by = {x.replace("-image", ""): all(c["result"] == c["total"] for c in v["checkpoints"]) for x, v in tasks.items()}
        if len(by) != 175:
            continue
        out[key] = np.array([float(by[k]) for k in sorted(by)])
    return out


LOADERS = {
    "SWE-bench Verified": lambda root: swebench_outcomes(root, "verified", C.N_VERIFIED["SWE-bench Verified"]),
    "SWE-bench Lite": lambda root: swebench_outcomes(root, "lite", C.N_VERIFIED["SWE-bench Lite"]),
    "SWE-bench test (full)": lambda root: swebench_outcomes(root, "test", C.N_VERIFIED["SWE-bench test (full)"]),
    "SWE-bench Multilingual": lambda root: swebench_outcomes(root, "multilingual", C.N_VERIFIED["SWE-bench Multilingual"]),
    "Terminal-Bench 1.0": tb1_outcomes,
    "Terminal-Bench 2.0": tb2_outcomes,
    "TheAgentCompany": tac_outcomes,
}


# ----------------------------------------------------------------------------- estimation


@dataclass
class RhoEstimate:
    pairs: pd.DataFrame            # adjacent pairs of every per-task board
    boards: pd.DataFrame           # one row per analysed leaderboard
    pooled_median: float           # median of the per-board adjacent medians (per-task boards)
    matrices: dict                 # lb -> (entry names in matrix order, (m, m) rho matrix)
    rho_used: dict                 # lb -> rho for the headline bound
    source: dict                   # lb -> "own" | "pooled"


def estimate(entries: list[C.Entry], root: Path, loaders: dict | None = None) -> RhoEstimate:
    loaders = LOADERS if loaders is None else loaders
    lbs = C.census_leaderboards(entries)
    pair_rows, board_rows, matrices = [], {}, {}
    own: dict[str, float] = {}
    for lb in lbs:
        if lb not in loaders:
            continue
        outc = loaders[lb](root)
        kept = [e for e in entries if e.lb == lb and e.status == "kept"]
        missing = [e.name for e in kept if e.name not in outc]
        if missing:
            raise ValueError(f"{lb}: no per-task outcomes for {len(missing)} kept entries, e.g. {missing[0]}")
        idx = C.order([e.score for e in kept], [e.name for e in kept])
        names = [kept[i].name for i in idx]
        ks = [kept[i].k for i in idx]
        P = np.vstack([outc[nm] for nm in names])
        R, Rp = rho_matrix(P), pearson_matrix(P)
        matrices[lb] = (names, R)
        ra, rpa = adjacent(R), adjacent(Rp)
        for r_, (a, b) in enumerate(zip(names[:-1], names[1:]), start=1):
            pair_rows.append({"leaderboard": lb, "rank_hi": r_, "entry_hi": a, "entry_lo": b,
                              "score_hi": kept[idx[r_ - 1]].score, "score_lo": kept[idx[r_]].score,
                              "k_hi": ks[r_ - 1], "k_lo": ks[r_],
                              "rho": ra[r_ - 1], "rho_pearson_rates": rpa[r_ - 1]})
        s_adj, s_all = summarise(ra), summarise(R[np.triu_indices(len(names), 1)])
        s_pear = summarise(rpa)
        own[lb] = s_adj["median"]
        board_rows[lb] = {
            "leaderboard": lb, "n_tasks": P.shape[1], "n_entries": len(names), "k_runs_max": max(ks),
            "n_adjacent_pairs": len(names) - 1, "n_adjacent_defined": s_adj["n"],
            "rho_adjacent_median": s_adj["median"], "rho_adjacent_q25": s_adj["q25"], "rho_adjacent_q75": s_adj["q75"],
            "rho_adjacent_min": s_adj["min"], "rho_adjacent_max": s_adj["max"],
            "rho_all_pairs_median": s_all["median"], "rho_all_pairs_q25": s_all["q25"], "rho_all_pairs_q75": s_all["q75"],
            "rho_pearson_rates_adjacent_median": s_pear["median"],
        }
    pooled = float(np.median([v for v in own.values() if not np.isnan(v)])) if own else float("nan")
    rows, used, src = [], {}, {}
    for lb in lbs:
        if lb in board_rows and not np.isnan(board_rows[lb]["rho_adjacent_median"]):
            used[lb], src[lb] = board_rows[lb]["rho_adjacent_median"], "own"
            rows.append({**board_rows[lb], "rho_used": used[lb], "rho_source": "own (per-task data)"})
        else:
            used[lb], src[lb] = pooled, "pooled"
            rows.append({"leaderboard": lb, "n_tasks": C.N_VERIFIED[lb],
                         "n_entries": int(sum(e.lb == lb and e.status == "kept" for e in entries)),
                         "rho_used": pooled, "rho_source": "pooled median (aggregates only)"})
    return RhoEstimate(pd.DataFrame(pair_rows), pd.DataFrame(rows), pooled, matrices, used, src)


def write_outputs(est: RhoEstimate, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    est.pairs.to_csv(out / "rho_adjacent_pairs.csv", index=False)
    est.boards.to_csv(out / "rho_by_board.csv", index=False)
    per_task = est.boards[est.boards["rho_source"].str.startswith("own")]
    med = per_task["rho_adjacent_median"].to_numpy()
    q25, q75 = np.percentile(med, [25, 75])
    (out / "rho_summary.json").write_text(json.dumps({
        "pooled_median_of_board_medians": est.pooled_median,
        "n_boards_with_per_task_data": int(len(per_task)),
        "board_medians_min": float(med.min()), "board_medians_max": float(med.max()),
        "board_medians_q25": float(q25), "board_medians_q75": float(q75),
        "rho_used": est.rho_used, "rho_source": est.source,
        "definition": "single-run-equivalent phi between adjacent entries (census order); see estimate_rho.py docstring",
    }, indent=2), encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parents[1]))
    ap.add_argument("--out", default=str(here / "results"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    est = estimate(C.load_all(root), root)
    write_outputs(est, out)
    cols = ["leaderboard", "n_entries", "rho_adjacent_median", "rho_adjacent_q25", "rho_adjacent_q75", "rho_used"]
    print(est.boards[cols].to_string(index=False))
    print(f"pooled median of per-board medians: {est.pooled_median:.4f}")


if __name__ == "__main__":
    main()
