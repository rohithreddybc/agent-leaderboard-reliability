"""Print every number that paper/sections/08-illustration.tex uses, from results/ only.

Run from anywhere: python harness/summarize_for_paper.py
Reads results/runs.jsonl (primary), results/superseded_runs.jsonl and results/summary.csv (cross-check).
Writes nothing. Failed runs (error is not null) count as unsuccessful, as in summarize.py.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from summarize import load, pass_hat_k  # noqa: E402  (same pass^k definition as summary.csv)

RES = HERE / "results"


def v_unbiased(counts):
    """Mean over tasks (n >= 2) of p(1-p) * n/(n-1), the Section 7 within-task variance."""
    vals = [(c / n) * (1 - c / n) * n / (n - 1) for c, n in counts if n >= 2]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    runs = load(RES / "runs.jsonl")
    old = load(RES / "superseded_runs.jsonl")
    quota = load(RES / "quota_events.jsonl")
    print(f"runs in runs.jsonl: {len(runs)}; superseded: {len(old)}; quota events: {len(quota)}")
    print("runs by condition:", dict(Counter(r["condition"] for r in runs)))
    print("run dates:", dict(Counter((r["condition"], r["model"], r["date"]) for r in runs)))
    print("tasks:", len({r["task_id"] for r in runs}), "| max_steps:", set(r["max_steps"] for r in runs))
    print("superseded by model and task:", dict(Counter((r["model"], r["task_id"]) for r in old)))
    print("total tool calls:", sum(r["tool_calls"] for r in runs), "| invalid:", sum(r["invalid_tool_calls"] for r in runs))
    print("tool calls by condition and model (calls; tool_errors; invalid):")
    cm = defaultdict(lambda: [0, 0, 0, 0])
    for r in runs:
        c = cm[(r["condition"], r["model"])]
        c[0] += r["tool_calls"]; c[1] += r["tool_errors"]; c[2] += r["invalid_tool_calls"]; c[3] += 1
    for k, c in sorted(cm.items()):
        print(f"  {k[0]:8s} {k[1]:22s} runs {c[3]:3d}; calls {c[0]:4d}; tool_errors {c[1]:3d}; invalid {c[2]}")
    d = [c for k, c in cm.items() if k[0] == "default"]
    print(f"  default condition, all models: runs {sum(c[3] for c in d)}; calls {sum(c[0] for c in d)}; tool_errors {sum(c[1] for c in d)}")
    print(f"  all conditions: runs {len(runs)}; calls {sum(c[0] for c in cm.values())}; tool_errors {sum(c[1] for c in cm.values())}")
    print()

    groups = defaultdict(list)
    for r in runs:
        groups[(r["condition"], r["model"])].append(r)
    summ = {(s["condition"], s["model"]): s for s in csv.DictReader((RES / "summary.csv").open(encoding="utf-8"))}

    for (cond, model), rs in sorted(groups.items()):
        by = defaultdict(list)
        for r in rs:
            by[r["task_id"]].append(r)
        counts = [(sum(x["success"] for x in v), len(v)) for v in by.values()]
        mixed = [t for t, v in by.items() if len(v) >= 2 and 0 < sum(x["success"] for x in v) < len(v)]
        multi = sum(1 for _, n in counts if n >= 2)
        failed = [r for r in rs if r["error"]]
        # sensitivity: drop failed runs, then recount mixed tasks
        by_ok = {t: [x for x in v if not x["error"]] for t, v in by.items()}
        mixed_ok = [t for t, v in by_ok.items() if len(v) >= 2 and 0 < sum(x["success"] for x in v) < len(v)]
        calls = sum(r["tool_calls"] for r in rs)
        inv = sum(r["invalid_tool_calls"] for r in rs)
        k5 = [n for _, n in counts if n >= 5]
        print(f"[{cond}] {model}")
        print(f"  runs {len(rs)}; tasks {len(by)}; runs/task min {min(n for _, n in counts)} max {max(n for _, n in counts)}; tasks with 5 runs {len(k5)}")
        print(f"  successes {sum(r['success'] for r in rs)}; mean success {sum(r['success'] for r in rs) / len(rs):.4f}")
        print("  pass^k (task mean over tasks with n>=k; tasks counted): " + "; ".join(
            f"k={k}: {pass_hat_k(counts, k):.4f} ({sum(1 for _, n in counts if n >= k)})" if pass_hat_k(counts, k) is not None else f"k={k}: n/a"
            for k in range(1, 6)))
        print(f"  mixed tasks {len(mixed)} of {multi} with >=2 runs = {len(mixed) / multi:.4f}; tasks: {sorted(mixed)}")
        print(f"  failed runs {len(failed)}: {dict(Counter((r['task_id'], r['error']) for r in failed))}")
        print(f"  mixed tasks after dropping failed runs: {len(mixed_ok)} = {len(mixed_ok) / multi:.4f}; tasks: {sorted(mixed_ok)}")
        unsucc = sum(1 for r in rs if not r["success"])
        print(f"  unsuccessful runs {unsucc}, of which failed (API error) {len(failed)}")
        print(f"  tool calls {calls}; invalid {inv}; rate {inv / calls:.4f}")
        terr = sum(r["tool_errors"] for r in rs)
        print(f"  tool errors (valid calls the world rejected) {terr} of {calls} calls = {terr / calls:.4f}")
        v = v_unbiased(counts)
        print(f"  within-task variance v (unbiased, Section 7 definition): {v:.4f}")
        s = summ.get((cond, model))
        if s:
            ok = (abs(float(s["mean_success"]) - sum(r["success"] for r in rs) / len(rs)) < 1e-4
                  and int(s["n_runs"]) == len(rs) and int(s["failed_run_count"]) == len(failed)
                  and int(s["tasks_with_mixed_outcomes"]) == len(mixed))
            print(f"  matches summary.csv: {ok}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
