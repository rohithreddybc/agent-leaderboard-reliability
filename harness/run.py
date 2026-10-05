"""Resumable runner. One worker thread per model (quotas are per model).

Order inside a worker is repeat-major (all tasks for repeat 1, then repeat 2, ...), so a
daily-quota stop leaves the task x repeat grid evenly covered. A run is appended to the
JSONL only when it completes (success, task failure, or an unrecovered API error). A run
interrupted by a daily-quota stop is not logged; it is re-run on resume.

    python run.py --condition default            # provider-default temperature
    python run.py --condition temp0              # temperature 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from agent import MAX_STEPS, run_one
from client import GroqClient, QuotaExhausted
from tasks import TASKS

MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CONDITIONS = {"default": None, "temp0": 0.0}  # None = omit the parameter (provider default)
RESULTS = Path(__file__).parent / "results"
_lock = threading.Lock()


def load_done(path: Path) -> set[tuple]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["condition"], r["model"], r["task_id"], r["repeat"]))
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def pending(model: str, condition: str, task_ids: list[str], repeats: int, done: set) -> list[tuple]:
    return [
        (t.id, r)
        for r in range(1, repeats + 1)
        for t in TASKS
        if t.id in task_ids and (condition, model, t.id, r) not in done
    ]


def worker(model, condition, queue, out: Path, quota_log: Path, stats: dict, client_factory, max_steps):
    client = client_factory()
    temp = CONDITIONS[condition]
    by_id = {t.id: t for t in TASKS}
    for task_id, rep in queue:
        try:
            rec = run_one(client, model, by_id[task_id], rep, condition, temp, max_steps)
        except QuotaExhausted as e:
            append_jsonl(quota_log, {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model, "condition": condition, "abandoned_task": task_id,
                "abandoned_repeat": rep, "message": str(e)[:300]})
            stats[model] = ("quota", stats.get(model, (None, 0))[1])
            print(f"[{model}] daily quota reached; checkpointed. Re-run to resume.", flush=True)
            return
        append_jsonl(out, rec)
        n = stats.get(model, (None, 0))[1] + 1
        stats[model] = ("running", n)
        print(f"[{model}] {condition} {task_id} r{rep} success={rec['success']} steps={rec['steps']} "
              f"tok={rec['total_tokens']} err={rec['error']}", flush=True)
    stats[model] = ("done", stats.get(model, (None, 0))[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS), default="default")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tasks", nargs="+", default=[t.id for t in TASKS])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", type=Path, default=RESULTS / "runs.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="max runs per model this invocation")
    args = ap.parse_args(argv)

    done = load_done(args.out)
    quota_log = args.out.parent / "quota_events.jsonl"
    stats: dict = {}
    threads = []
    for m in args.models:
        q = pending(m, args.condition, args.tasks, args.repeats, done)
        if args.limit is not None:
            q = q[: args.limit]
        print(f"[{m}] {len(q)} runs pending", flush=True)
        th = threading.Thread(
            target=worker,
            args=(m, args.condition, q, args.out, quota_log, stats, lambda: GroqClient(), args.max_steps),
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    quota = [m for m, (s, _) in stats.items() if s == "quota"]
    return 3 if quota else 0


if __name__ == "__main__":
    sys.exit(main())
