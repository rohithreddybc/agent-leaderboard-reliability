"""Consistency drop from tau2-bench leaderboard aggregates.

Each ``submission.json`` reports, per domain, pass^1..pass^4 as percentages.
For every submission and domain with all four values present this script
reports the consistency drop pass^1 - pass^k (k = 2, 3, 4) in percentage
points and the ratio pass^k / pass^1.

LIMIT: these are domain-level aggregates. They carry no per-task or per-trial
outcomes, so trial-level or task-level resampling (the Terminal-Bench
bootstrap in ``bootstrap_core``) cannot be applied, and no confidence interval
for any value here can be computed from them. The numbers are descriptive.

Submissions whose directory name starts with ``A_EXAMPLE`` (template examples)
are excluded. The category (text, voice, legacy) comes from ``manifest.json``.

Usage: python tau2_bounds.py [--root ../data/raw/tau2] [--results results]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

KEYS = ("pass_1", "pass_2", "pass_3", "pass_4")
NOTE = ("Aggregates only: no per-task or per-trial outcomes, so no trial-level or "
        "task-level resampling and no confidence intervals are possible.")


def categories(root: Path) -> dict[str, str]:
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cat = {"submissions": "text", "voice_submissions": "voice", "legacy_submissions": "legacy"}
    return {name: cat[k] for k, names in m.items() if k in cat for name in names}


def stated_trials(d: dict) -> int | None:
    """Trial count stated in trajectory file names or methodology notes, if any."""
    txt = json.dumps(d.get("trajectory_files") or {}) + " " + str((d.get("methodology") or {}).get("notes") or "")
    found = set(re.findall(r"(\d+)\s*trials", txt))
    return int(found.pop()) if len(found) == 1 else None


def load(root: Path) -> pd.DataFrame:
    cat = categories(root)
    rows = []
    for f in sorted(root.glob("*/submission.json")):
        name = f.parent.name
        if name.startswith("A_EXAMPLE"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for dom, r in (d.get("results") or {}).items():
            r = r or {}
            if any(r.get(k) is None for k in KEYS):
                continue
            p = [float(r[k]) for k in KEYS]
            rows.append({
                "submission": name, "model_name": d.get("model_name"), "category": cat.get(name, "unlisted"),
                "domain": dom, "trials_stated": stated_trials(d),
                **{k: v for k, v in zip(KEYS, p)},
                "drop_2_pp": p[0] - p[1], "drop_3_pp": p[0] - p[2], "drop_4_pp": p[0] - p[3],
                "ratio_4_over_1": p[3] / p[0] if p[0] > 0 else None,
                "monotone_non_increasing": int(all(p[i] >= p[i + 1] for i in range(3))),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    s = {"note": NOTE, "n_rows_submission_domain": int(len(df)),
         "n_submissions_complete": int(df["submission"].nunique()),
         "n_non_monotone_rows": int((df["monotone_non_increasing"] == 0).sum()),
         "by_category": {k: int(v) for k, v in df.groupby("category")["submission"].nunique().items()},
         "submissions_with_stated_4_trials": int(df.loc[df["trials_stated"] == 4, "submission"].nunique()),
         "by_domain": {}}
    for dom, g in df.groupby("domain"):
        s["by_domain"][dom] = {
            "n_submissions": int(len(g)),
            "drop_4_pp_mean": float(g["drop_4_pp"].mean()), "drop_4_pp_min": float(g["drop_4_pp"].min()),
            "drop_4_pp_max": float(g["drop_4_pp"].max()), "drop_4_pp_median": float(g["drop_4_pp"].median()),
        }
    return s


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=here.parent / "data/raw/tau2")
    ap.add_argument("--results", type=Path, default=here / "results")
    a = ap.parse_args()
    df = load(a.root)
    a.results.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.results / "tau2_consistency.csv", index=False)
    (a.results / "tau2_summary.json").write_text(json.dumps(summarize(df), indent=2), encoding="utf-8")
    print(json.dumps(summarize(df), indent=2))


if __name__ == "__main__":
    main()
