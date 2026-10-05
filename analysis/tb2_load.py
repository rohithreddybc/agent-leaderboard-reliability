"""Parse Terminal-Bench 2.0 leaderboard submissions into one long table.

Input layout (as in the Hugging Face dataset, trimmed by ``fetch_data.py``)::

    <root>/<submission>/metadata.yaml
    <root>/<submission>/<run>/<trial>/result.json      # one file per trial
    <root>/<submission>/<run>/{config,result}.json     # job-level, not used

Output columns: submission, agent, model, task_id, trial, success, plus
run, trial_name, reward, no_reward, started_at, task_commit, task_raw
(the task name as written in result.json; ``task_id`` is its normalised form).

``success`` is 1 when the verifier reward is >= 1.0 and 0 otherwise. A trial
with no verifier reward (infrastructure error, timeout before verification)
is kept as ``success = 0`` with ``no_reward = 1``; pass ``--drop-no-reward``
to remove such trials instead. ``trial`` is a 1-based index within
(submission, task), ordered by run directory, start time and trial name.

Usage:
    python tb2_load.py [--root ../data/raw/tb2] [--out results] [--drop-no-reward]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

EXPECTED_TRIALS = 5  # leaderboard rule: at least five trials per task


def normalize_task(name: str) -> str:
    """Canonical task id: drop a ``terminal-bench/`` style prefix and replace
    dots by hyphens (``install-windows-3.11`` and ``install-windows-3-11`` are
    the same task, spelled differently across submissions)."""
    return name.rsplit("/", 1)[-1].replace(".", "-")


def parse_trial(r: dict) -> dict:
    """Extract the fields used from one trial ``result.json`` (already parsed)."""
    rewards = ((r.get("verifier_result") or {}).get("rewards")) or {}
    reward = rewards.get("reward")
    cfg_agent = ((r.get("config") or {}).get("agent")) or {}
    tid = r.get("task_id") or {}
    return {
        "task_id": normalize_task(r["task_name"]),
        "task_raw": r["task_name"],
        "trial_name": r.get("trial_name"),
        "reward": None if reward is None else float(reward),
        "no_reward": int(reward is None),
        "started_at": r.get("started_at"),
        "task_commit": tid.get("git_commit_id") if isinstance(tid, dict) else None,
        "config_model": cfg_agent.get("model_name"),
        "config_agent": cfg_agent.get("import_path") or cfg_agent.get("name"),
    }


def read_metadata(sub_dir: Path) -> tuple[str, str]:
    """(agent, model) display names from metadata.yaml/.yml; falls back to the
    folder name split on the first double underscore."""
    for name in ("metadata.yaml", "metadata.yml"):
        p = sub_dir / name
        if p.exists():
            try:
                m = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                agent = m.get("agent_display_name")
                models = [x.get("model_display_name") or x.get("model_name") for x in (m.get("models") or [])]
                models = [x for x in models if x]
                if agent and models:
                    return str(agent), "+".join(models)
            except yaml.YAMLError:
                pass
    head, _, tail = sub_dir.name.partition("__")
    return head, tail or sub_dir.name


def load_tree(root: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        agent, model = read_metadata(sub)
        for run in sorted(p for p in sub.iterdir() if p.is_dir()):
            for trial in sorted(p for p in run.iterdir() if p.is_dir()):
                f = trial / "result.json"
                if not f.exists():
                    continue
                row = parse_trial(json.loads(f.read_text(encoding="utf-8")))
                row.update(submission=sub.name, agent=agent, model=model, run=run.name)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["success"] = (df["reward"].fillna(0.0) >= 1.0).astype(int)
    df = df.sort_values(["submission", "task_id", "run", "started_at", "trial_name"], kind="stable")
    df["trial"] = df.groupby(["submission", "task_id"]).cumcount() + 1
    cols = ["submission", "agent", "model", "task_id", "trial", "success", "run", "trial_name",
            "reward", "no_reward", "started_at", "task_commit", "task_raw"]
    return df[cols + [c for c in df.columns if c not in cols]].reset_index(drop=True)


def select_analysis_set(long: pd.DataFrame, min_trials: int = EXPECTED_TRIALS):
    """Submissions with at least ``min_trials`` trials on every task of the
    task universe (all tasks seen in the table). Returns (subs, tasks,
    table with one row per submission: included, reason)."""
    tasks = sorted(long["task_id"].unique())
    n = long.groupby(["submission", "task_id"]).size().unstack("task_id").reindex(columns=tasks).fillna(0)
    rows = []
    for s, r in n.iterrows():
        missing = int((r == 0).sum())
        short = int(((r > 0) & (r < min_trials)).sum())
        reason = []
        if missing:
            reason.append(f"{missing} tasks absent")
        if short:
            reason.append(f"{short} tasks with <{min_trials} trials")
        rows.append({"submission": s, "included": int(not reason), "reason": "; ".join(reason)})
    tab = pd.DataFrame(rows)
    return list(tab.loc[tab.included == 1, "submission"]), tasks, tab


def summarize(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tasks = sorted(long["task_id"].unique())
    n = long.groupby(["submission", "task_id"]).size().unstack("task_id").reindex(columns=tasks).fillna(0).astype(int)
    meta = long.groupby("submission")[["agent", "model"]].first()
    per = pd.DataFrame({
        "agent": meta["agent"], "model": meta["model"],
        "n_trials": long.groupby("submission").size(),
        "n_runs": long.groupby("submission")["run"].nunique(),
        "n_tasks": (n > 0).sum(axis=1),
        "trials_per_task_min": n.where(n > 0).min(axis=1).astype(int),
        "trials_per_task_median": n.where(n > 0).median(axis=1),
        "trials_per_task_max": n.max(axis=1),
        "tasks_below_5_trials": ((n > 0) & (n < EXPECTED_TRIALS)).sum(axis=1),
        "tasks_absent": (n == 0).sum(axis=1),
        # trials missing relative to five per task on every task of the universe
        "missing_trials_vs_5": (EXPECTED_TRIALS - n).clip(lower=0).sum(axis=1),
        "no_reward_trials": long.groupby("submission")["no_reward"].sum(),
        # data-quality flags: repeated trial names inside a task, and no outcome
        # variation between trials of any task (every cell all-success or all-fail)
        "duplicate_trial_names": long.groupby("submission").apply(
            lambda x: int(x.duplicated(["task_id", "trial_name"]).sum()), include_groups=False),
        "zero_within_task_variance": long.groupby(["submission", "task_id"])["success"].nunique().groupby("submission").max().eq(1).astype(int),
        "mean_success": long.groupby("submission")["success"].mean(),
    }).reset_index()
    cell = n.to_numpy().ravel()
    dist = pd.Series(cell).value_counts().sort_index()
    summary = {
        "n_submissions": int(long["submission"].nunique()),
        "n_tasks_union": len(tasks),
        "n_trials_total": int(len(long)),
        "n_runs_total": int(long.groupby("submission")["run"].nunique().sum()),
        "n_no_reward_trials": int(long["no_reward"].sum()),
        "n_task_commits": int(long["task_commit"].nunique()),
        "cells_total": int(cell.size),
        "cells_absent": int((cell == 0).sum()),
        "cells_below_5": int(((cell > 0) & (cell < EXPECTED_TRIALS)).sum()),
        "trials_per_cell_distribution": {str(int(k)): int(v) for k, v in dist.items()},
        "submissions_with_every_task_at_least_5": int((n.min(axis=1) >= EXPECTED_TRIALS).sum()),
        "duplicate_run_trialname_rows": int(long.duplicated(["submission", "run", "trial_name"]).sum()),
    }
    return per, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "data/raw/tb2")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--drop-no-reward", action="store_true")
    a = ap.parse_args()
    long = load_tree(a.root)
    if a.drop_no_reward:
        long = long[long["no_reward"] == 0].copy()
        long["trial"] = long.groupby(["submission", "task_id"]).cumcount() + 1
    a.out.mkdir(parents=True, exist_ok=True)
    per, summary = summarize(long)
    subs, tasks, tab = select_analysis_set(long)
    summary["analysis_set_submissions"] = len(subs)
    summary["analysis_set_rule"] = f">= {EXPECTED_TRIALS} trials on every task of the {len(tasks)}-task union"
    long.to_csv(a.out / "tb2_long.csv", index=False)
    per.merge(tab, on="submission").to_csv(a.out / "tb2_load_counts.csv", index=False)
    (a.out / "tb2_load_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
