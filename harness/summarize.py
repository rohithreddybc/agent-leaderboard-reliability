"""Compute results/summary.csv and results/per_task.csv from results/runs.jsonl only.

Every number is derived from the JSONL. Runs that ended in an API error are kept: they
count as unsuccessful in mean_success and pass^k, and are also counted in failed_run_count.

pass^k for one task with c successes in n runs is C(c,k)/C(n,k) (the unbiased estimate of
the probability that k runs drawn without replacement all succeed); the model value is
the mean over tasks that have at least k runs.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
K_MAX = 5


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pass_hat_k(counts: list[tuple[int, int]], k: int) -> float | None:
    """counts: (successes, runs) per task. Mean of C(c,k)/C(n,k) over tasks with n>=k."""
    vals = [comb(c, k) / comb(n, k) for c, n in counts if n >= k]
    return sum(vals) / len(vals) if vals else None


def summarize(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["condition"], r["model"])].append(r)
    summary, per_task_rows = [], []
    for (cond, model), rs in sorted(groups.items()):
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_task[r["task_id"]].append(r)
        counts = {t: (sum(1 for r in v if r["success"]), len(v)) for t, v in sorted(by_task.items())}
        for t, (c, n) in counts.items():
            per_task_rows.append({"condition": cond, "model": model, "task_id": t, "successes": c, "runs": n})
        n_tasks = len(counts)
        multi = [(c, n) for c, n in counts.values() if n >= 2]
        mixed = sum(1 for c, n in multi if 0 < c < n)
        total_calls = sum(r["tool_calls"] for r in rs)
        invalid = sum(r["invalid_tool_calls"] for r in rs)
        row = {
            "condition": cond,
            "model": model,
            "temperature_setting": rs[0]["temperature_setting"],
            "first_run_date": min(r["date"] for r in rs),
            "last_run_date": max(r["date"] for r in rs),
            "n_tasks": n_tasks,
            "n_runs": len(rs),
            "n_tasks_with_5_runs": sum(1 for _, n in counts.values() if n == K_MAX),
            "mean_success": round(sum(r["success"] for r in rs) / len(rs), 4),
            "per_task_success_counts": ";".join(f"{t}:{c}/{n}" for t, (c, n) in counts.items()),
        }
        for k in range(1, K_MAX + 1):
            v = pass_hat_k(list(counts.values()), k)
            row[f"pass_hat_{k}"] = "" if v is None else round(v, 4)
        row.update({
            "tasks_with_mixed_outcomes": mixed,
            "share_tasks_mixed": round(mixed / len(multi), 4) if multi else "",
            "tool_calls_total": total_calls,
            "invalid_tool_calls": invalid,
            "invalid_tool_call_rate": round(invalid / total_calls, 4) if total_calls else "",
            "failed_run_count": sum(1 for r in rs if r["error"]),
            "mean_steps": round(sum(r["steps"] for r in rs) / len(rs), 3),
            "mean_total_tokens": round(sum(r["total_tokens"] for r in rs) / len(rs), 1),
            "total_tokens": sum(r["total_tokens"] for r in rs),
            "mean_latency_s": round(sum(r["latency_s"] for r in rs) / len(rs), 3),
        })
        summary.append(row)
    return summary, per_task_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    runs_path = Path(argv[0]) if argv else RESULTS / "runs.jsonl"
    out_dir = Path(argv[1]) if argv and len(argv) > 1 else runs_path.parent
    summary, per_task = summarize(load(runs_path))
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "per_task.csv", per_task)
    print(f"wrote {out_dir / 'summary.csv'} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
