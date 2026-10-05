import json

import pandas as pd
import pytest

import tb2_load as L


def write_trial(root, sub, run, trial, task, reward, started="2026-01-01T00:00:00"):
    d = root / sub / run / trial
    d.mkdir(parents=True)
    r = {"task_name": task, "trial_name": trial, "started_at": started,
         "task_id": {"git_commit_id": "abc"},
         "config": {"agent": {"model_name": "m", "import_path": "x"}},
         "verifier_result": None if reward is None else {"rewards": {"reward": reward}}}
    (d / "result.json").write_text(json.dumps(r))


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "tb2"
    (root / "A__M").mkdir(parents=True)
    (root / "A__M" / "metadata.yaml").write_text(
        'agent_display_name: "Agent A"\nmodels:\n  - model_name: "m"\n    model_display_name: "Model M"\n')
    for k in range(5):
        write_trial(root, "A__M", "run1", f"t1__{k}", "t1", 1.0 if k < 3 else 0.0, f"2026-01-0{k+1}")
        write_trial(root, "A__M", "run1", f"t2__{k}", "t2", 1.0)
    write_trial(root, "B__N", "run1", "t1__x", "t1", None)  # no reward, task t2 absent, 1 trial
    return root


def test_parse_and_index(tree):
    df = L.load_tree(tree)
    a = df[df.submission == "A__M"]
    assert len(df) == 11 and a.agent.iloc[0] == "Agent A" and a.model.iloc[0] == "Model M"
    t1 = a[a.task_id == "t1"].sort_values("trial")
    assert list(t1.trial) == [1, 2, 3, 4, 5] and list(t1.success) == [1, 1, 1, 0, 0]
    b = df[df.submission == "B__N"].iloc[0]
    assert b.no_reward == 1 and b.success == 0 and b.agent == "B" and b.model == "N"


def test_analysis_set_and_summary(tree):
    df = L.load_tree(tree)
    subs, tasks, tab = L.select_analysis_set(df)
    assert subs == ["A__M"] and tasks == ["t1", "t2"]
    assert "absent" in tab.set_index("submission").loc["B__N", "reason"]
    per, s = L.summarize(df)
    assert s["n_submissions"] == 2 and s["n_tasks_union"] == 2 and s["n_trials_total"] == 11
    assert s["cells_absent"] == 1 and s["cells_below_5"] == 1
    row = per.set_index("submission").loc["B__N"]
    assert row.missing_trials_vs_5 == 4 + 5 and row.no_reward_trials == 1


def test_main_writes_only_to_out(tree, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["tb2_load.py", "--root", str(tree), "--out", str(out)])
    L.main()
    assert (out / "tb2_long.csv").exists() and (out / "tb2_load_summary.json").exists()
    assert len(pd.read_csv(out / "tb2_long.csv")) == 11


def test_normalize_task():
    assert L.normalize_task("terminal-bench/install-windows-3.11") == "install-windows-3-11"
    assert L.normalize_task("install-windows-3-11") == "install-windows-3-11"
    assert L.normalize_task("fix-git") == "fix-git"
