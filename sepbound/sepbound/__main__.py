"""Command line: python -m sepbound --n 500 --rho 0.5 72.4 71.8 70.1 ..."""
from __future__ import annotations

import argparse
import sys

from .core import check_leaderboard, to_proportions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sepbound",
                                 description="Check from published scores whether leaderboard entries are separable.")
    ap.add_argument("scores", nargs="+", type=float,
                    help="scores in percent or as proportions (read as percent if any value exceeds 1)")
    ap.add_argument("--n", required=True, help="number of tasks; one value, or one comma-separated value per entry")
    ap.add_argument("--rho", type=float, default=0.0, help="task-level correlation between entries (default 0)")
    ap.add_argument("--names", default=None, help="comma-separated entry names (default: 1, 2, ...)")
    ap.add_argument("--draws", type=int, default=20000, help="Monte Carlo draws for the leader probability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=15, help="rows of the entry table to print")
    a = ap.parse_args(argv)
    try:
        if len(a.scores) < 2:
            raise ValueError("need at least two scores")
        s = to_proportions(a.scores)
        nv = [float(x) for x in a.n.split(",")]
        n = nv[0] if len(nv) == 1 else nv
        names = a.names.split(",") if a.names else [str(i) for i in range(1, len(s) + 1)]
        if len(names) != len(s):
            raise ValueError("--names must give one name per score")
        rep = check_leaderboard(names, s, n, a.rho, draws=a.draws, seed=a.seed, max_rows=a.rows)
    except ValueError as e:
        print(f"sepbound: error: {e}", file=sys.stderr)
        return 2
    print(rep["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
