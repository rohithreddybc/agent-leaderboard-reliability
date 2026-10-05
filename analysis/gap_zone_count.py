"""Count separable pairs in the mixed zone between gap_min_stable and gap_max_unstable.

Reads analysis/results/rank_inversion_pairs.csv (and the MAYA-excluded sensitivity file when
--sensitivity is given). A pair is separable when its flip probability is <= 0.05.
For each scheme (run-to-run, task sampling) the mixed zone is the closed interval
[gap_min_stable, gap_max_unstable] taken from rank_inversion_summary.json. Prints the number of
pairs in the zone, how many are separable, and checks that every pair above the zone is separable
and every pair below it is unstable.

Usage: python gap_zone_count.py [--sensitivity]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true", help="use the excl_MAYA files")
    a = ap.parse_args()
    suf = "_sensitivity_excl_MAYA" if a.sensitivity else ""
    pairs = pd.read_csv(RES / f"rank_inversion_pairs{suf}.csv")
    summ = json.loads((RES / f"rank_inversion_summary{suf}.json").read_text())
    for scheme, col in (("run", "p_flip_run"), ("task", "p_flip_task")):
        lo, hi = summ[scheme]["gap_min_stable_pp"], summ[scheme]["gap_max_unstable_pp"]
        sep = pairs[col] <= 0.05
        zone = (pairs.gap_pp >= lo - 1e-9) & (pairs.gap_pp <= hi + 1e-9)
        above, below = pairs.gap_pp > hi + 1e-9, pairs.gap_pp < lo - 1e-9
        print(f"[{scheme}] zone {lo:.2f}-{hi:.2f} pp: {int(zone.sum())} pairs, "
              f"{int((zone & sep).sum())} separable, {int((zone & ~sep).sum())} unstable")
        print(f"[{scheme}] above zone: {int(above.sum())} pairs, all separable: {bool(sep[above].all())}")
        print(f"[{scheme}] below zone: {int(below.sum())} pairs, any separable: {bool(sep[below].any())}")


if __name__ == "__main__":
    main()
