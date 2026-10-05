import csv
import json
from math import comb

import run as runmod
import summarize
from client import GroqClient
from fakes import FakeResp, Scripted, completion


def rec(model, task, rep, ok, cond="default", err=None, calls=1, invalid=0):
    return {"condition": cond, "model": model, "date": "2026-10-04", "task_id": task, "repeat": rep,
            "temperature_setting": "provider_default", "success": ok, "steps": 2, "tool_calls": calls,
            "invalid_tool_calls": invalid, "total_tokens": 100, "latency_s": 1.0, "error": err}


def test_pass_hat_k_matches_hand_computation():
    counts = [(5, 5), (3, 5), (0, 5)]
    assert summarize.pass_hat_k(counts, 1) == (1 + 0.6 + 0) / 3
    assert abs(summarize.pass_hat_k(counts, 2) - (1 + comb(3, 2) / comb(5, 2) + 0) / 3) < 1e-12
    assert summarize.pass_hat_k(counts, 5) == (1 + 0 + 0) / 3
    assert summarize.pass_hat_k([(2, 3)], 4) is None


def test_summary_rows(tmp_path):
    runs = []
    for r in range(1, 6):
        runs.append(rec("m1", "t01", r, True))
        runs.append(rec("m1", "t02", r, r <= 3, invalid=1 if r == 1 else 0, calls=2))
        runs.append(rec("m1", "t03", r, False, err="timeout" if r == 5 else None))
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(x) for x in runs) + "\n")
    summarize.main([str(tmp_path / "runs.jsonl"), str(tmp_path)])
    with (tmp_path / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    s = rows[0]
    assert s["n_runs"] == "15" and s["n_tasks"] == "3"
    assert float(s["mean_success"]) == round(8 / 15, 4)
    assert s["per_task_success_counts"] == "t01:5/5;t02:3/5;t03:0/5"
    assert s["tasks_with_mixed_outcomes"] == "1" and float(s["share_tasks_mixed"]) == round(1 / 3, 4)
    assert s["failed_run_count"] == "1"
    assert float(s["invalid_tool_call_rate"]) == round(1 / (5 + 10 + 5), 4)
    assert float(s["pass_hat_1"]) == round((1 + 0.6 + 0) / 3, 4)


def test_pending_skips_done():
    done = {("default", "m", "t01", 1), ("default", "m", "t01", 2)}
    q = runmod.pending("m", "default", ["t01", "t02"], 2, done)
    assert q == [("t02", 1), ("t02", 2)]


def test_worker_quota_checkpoint_and_resume(tmp_path):
    out = tmp_path / "runs.jsonl"
    quota = tmp_path / "quota_events.jsonl"
    t = Scripted([completion("12"), FakeResp(429, {"error": {"message": "tokens per day (TPD) reached"}})])
    stats = {}
    q = [("t01", 1), ("t02", 1), ("t03", 1)]
    runmod.worker("m", "default", q, out, quota, stats,
                  lambda: GroqClient(transport=t, sleep=lambda s: None, api_key="k"), 10)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["task_id"] for r in lines] == ["t01"]  # abandoned run is not logged
    ev = [json.loads(x) for x in quota.read_text().splitlines()]
    assert ev[0]["abandoned_task"] == "t02" and stats["m"][0] == "quota"
    left = runmod.pending("m", "default", ["t01", "t02", "t03"], 1, runmod.load_done(out))
    assert left == [("t02", 1), ("t03", 1)]
