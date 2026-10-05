"""Rank-inversion analysis of the Terminal-Bench 2.0 leaderboard submissions.

Definitions
-----------
Leaderboard order: submissions sorted by observed mean success (macro mean over
tasks), best first, ties broken by name. The dataset holds no published score
or rank; this is the order its raw trials imply.

Flip probability of a pair (higher-ranked i, lower-ranked j): the share of
bootstrap replicates in which j scores above i, plus half the share of exact
ties. It is computed under
  PRIMARY   run-to-run: tasks fixed, trials resampled within task, per
            submission independently;
  SECONDARY task-sampling: tasks resampled, same draw for all submissions.
Both use B = 10,000 and the seed scheme in ``bootstrap_core``. The run-to-run scheme
carries the N/(N-1) variance correction described there (decision log, 2026-10-04); the
values before the correction are in ``analysis/CHANGES-2026-10-04.md``.

Separable pair: flip probability <= 0.05. Distinct rank positions that are
statistically separable: the size of the largest set of submissions in which
every pair is separable (exact maximum clique).

Minimum stable gap, three definitions (percentage points, pp):
  gap_max_unstable  largest observed gap among pairs with flip probability > 0.05;
                    every pair with a larger gap is separable.
  gap_min_stable    smallest observed gap among pairs with flip probability <= 0.05.
  gap_bin_stable    lower edge of the first 1-pp gap bin from which the mean flip
                    probability stays <= 0.05 in every later non-empty bin.

Split-half reliability: for each of 1,000 random splits, each task's trials
are permuted and two disjoint halves of floor(n/2) trials are drawn per
submission (one trial is left out when n is odd). Submissions are ranked by
half-mean in each half; Spearman and Kendall (tau-b) between the two rankings
are recorded. Spearman-Brown correction to the full trial count:
r_full = m r / (1 + (m - 1) r) with m = sum(n) / sum(floor(n/2)) over cells.
The same transform is applied to Kendall tau as a heuristic.

Usage: python rank_inversion.py [--results results] [--n-boot 10000] [--n-splits 1000]
                                [--exclude SUB ... --suffix _sensitivity]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

import bootstrap_core as bc
import tb2_bootstrap

THRESHOLDS = (0.05, 0.20)


def pair_table(subs, means, order, P_run, P_task) -> pd.DataFrame:
    rows = []
    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            rows.append((subs[a], subs[b], i + 1, j + 1, 100 * (means[a] - means[b]), P_run[i, j], P_task[i, j]))
    return pd.DataFrame(rows, columns=["submission_hi", "submission_lo", "rank_hi", "rank_lo", "gap_pp",
                                       "p_flip_run", "p_flip_task"])


def adjacent_shares(P: np.ndarray) -> dict:
    adj = np.array([P[i, i + 1] for i in range(P.shape[0] - 1)])
    out = {"n_adjacent_pairs": int(adj.size)}
    for t in THRESHOLDS:
        out[f"share_adjacent_flip_gt_{int(t * 100)}pct"] = float((adj > t).mean())
        out[f"n_adjacent_flip_gt_{int(t * 100)}pct"] = int((adj > t).sum())
    return out


def max_separable(P: np.ndarray, thr: float = 0.05) -> tuple[int, int]:
    """(exact max pairwise-separable set, greedy chain from the top)."""
    n = P.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((i, j) for i in range(n) for j in range(i + 1, n) if P[i, j] <= thr)
    exact = len(nx.max_weight_clique(G, weight=None)[0])
    chain = [0]
    for j in range(1, n):
        if all(P[i, j] <= thr for i in chain):
            chain.append(j)
    return exact, len(chain)


def min_gap(pairs: pd.DataFrame, col: str, thr: float = 0.05) -> tuple[dict, pd.DataFrame]:
    unstable = pairs[pairs[col] > thr]
    stable = pairs[pairs[col] <= thr]
    d = pairs.assign(bin=np.floor(pairs["gap_pp"]).astype(int))
    bins = d.groupby("bin").agg(n_pairs=(col, "size"), mean_flip=(col, "mean"), max_flip=(col, "max")).reset_index()
    ok = (bins["mean_flip"] <= thr)[::-1].cummin()[::-1]  # all later bins stable
    edge = float(bins.loc[ok, "bin"].min()) if ok.any() else float("nan")
    res = {
        "gap_max_unstable_pp": float(unstable["gap_pp"].max()) if len(unstable) else 0.0,
        "gap_min_stable_pp": float(stable["gap_pp"].min()) if len(stable) else float("nan"),
        "gap_bin_stable_pp": edge,
    }
    return res, bins.assign(scheme=col)


def rank_intervals(S: np.ndarray, order: np.ndarray, subs: list[str]) -> pd.DataFrame:
    ranks = stats.rankdata(-np.round(S, 9), axis=1, method="average")
    lo, med, hi = np.quantile(ranks, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame({
        "submission": subs, "observed_rank": np.argsort(order) + 1,
        "rank_low": lo, "rank_median": med, "rank_high": hi, "rank_mean": ranks.mean(axis=0),
        "interval_width": hi - lo + 1,
    })


def split_half(c: bc.Counts, n_splits: int = 1000) -> pd.DataFrame:
    if (c.N < 2).any():
        raise ValueError("split-half needs at least 2 trials in every cell")
    rng = np.random.default_rng(bc.seed_for(bc.SCHEME_SPLIT))
    h = c.N // 2
    m = c.N.sum() / h.sum()
    rows = []
    for k in range(n_splits):
        a = rng.hypergeometric(c.K, c.N - c.K, h)
        b = rng.hypergeometric(c.K - a, (c.N - c.K) - (h - a), h)
        ma, mb = (a / h).mean(axis=1), (b / h).mean(axis=1)
        rho = stats.spearmanr(ma, mb).statistic
        tau = stats.kendalltau(ma, mb).statistic
        sb = lambda r: m * r / (1 + (m - 1) * r)
        rows.append((k, rho, tau, sb(rho), sb(tau)))
    return pd.DataFrame(rows, columns=["split", "spearman", "kendall", "spearman_sb", "kendall_sb"])


def summarize_split(sh: pd.DataFrame, m: float) -> dict:
    out = {"n_splits": int(len(sh)), "spearman_brown_m": float(m)}
    for col in ("spearman", "kendall", "spearman_sb", "kendall_sb"):
        out[f"{col}_mean"] = float(sh[col].mean())
        out[f"{col}_q025"], out[f"{col}_q975"] = (float(x) for x in sh[col].quantile([0.025, 0.975]))
    return out


def analyse(long: pd.DataFrame, n_boot: int = bc.B, n_splits: int = 1000, exclude: tuple[str, ...] = ()):
    boot, wt, c, S_run, S_task = tb2_bootstrap.run(long, n_boot, exclude)
    means = bc.macro_mean(c.K, c.N)
    order = bc.observed_order(c.subs, means)
    P = {"run": bc.flip_matrix(S_run, order), "task": bc.flip_matrix(S_task, order)}
    pairs = pair_table(c.subs, means, order, P["run"], P["task"])
    summary = {"n_submissions": len(c.subs), "n_tasks": len(c.tasks), "n_pairs": len(pairs),
               "n_boot": n_boot, "seed": bc.SEED}
    bins = []
    for key, col in (("run", "p_flip_run"), ("task", "p_flip_task")):
        s = {f"{k}": v for k, v in adjacent_shares(P[key]).items()}
        ex, gr = max_separable(P[key])
        s["separable_positions_exact"], s["separable_positions_greedy"] = ex, gr
        s["share_all_pairs_flip_gt_5pct"] = float((pairs[col] > 0.05).mean())
        g, b = min_gap(pairs, col)
        s.update(g)
        bins.append(b)
        ri = rank_intervals(S_run if key == "run" else S_task, order, c.subs)
        s["observed_vs_mean_bootstrap_rank_kendall"] = float(stats.kendalltau(ri["observed_rank"], ri["rank_mean"]).statistic)
        s["observed_rank_outside_95pct_interval"] = int(((ri["observed_rank"] < ri["rank_low"]) |
                                                         (ri["observed_rank"] > ri["rank_high"])).sum())
        s["rank_interval_width_median"] = float(ri["interval_width"].median())
        s["rank_interval_width_max"] = float(ri["interval_width"].max())
        summary[key] = s
        summary.setdefault("_rank_tables", {})[key] = ri
    adj = pd.DataFrame({
        "rank_hi": np.arange(1, len(order)), "submission_hi": [c.subs[i] for i in order[:-1]],
        "submission_lo": [c.subs[i] for i in order[1:]],
        "gap_pp": 100 * (means[order[:-1]] - means[order[1:]]),
        "p_flip_run": [P["run"][i, i + 1] for i in range(len(order) - 1)],
        "p_flip_task": [P["task"][i, i + 1] for i in range(len(order) - 1)],
    })
    sh = split_half(c, n_splits)
    summary["split_half"] = summarize_split(sh, c.N.sum() / (c.N // 2).sum())
    return summary, pairs, adj, pd.concat(bins), sh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--n-boot", type=int, default=bc.B)
    ap.add_argument("--n-splits", type=int, default=1000)
    ap.add_argument("--exclude", nargs="*", default=[], help="submissions to drop (sensitivity run)")
    ap.add_argument("--suffix", default="", help="output file suffix, e.g. _sensitivity")
    a = ap.parse_args()
    long = pd.read_csv(a.results / "tb2_long.csv")
    summary, pairs, adj, bins, sh = analyse(long, a.n_boot, a.n_splits, tuple(a.exclude))
    tables = summary.pop("_rank_tables")
    summary["excluded"] = list(a.exclude)
    x = a.suffix
    pairs.to_csv(a.results / f"rank_inversion_pairs{x}.csv", index=False)
    adj.to_csv(a.results / f"rank_inversion_adjacent{x}.csv", index=False)
    bins.to_csv(a.results / f"rank_gap_bins{x}.csv", index=False)
    sh.to_csv(a.results / f"split_half{x}.csv", index=False)
    for k, t in tables.items():
        t.to_csv(a.results / f"rank_intervals_{k}{x}.csv", index=False)
    (a.results / f"rank_inversion_summary{x}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
