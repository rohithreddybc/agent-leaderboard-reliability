import json

import tau2_bounds as T


def make(root, name, results, notes=""):
    d = root / name
    d.mkdir(parents=True)
    (d / "submission.json").write_text(json.dumps({"model_name": name, "results": results,
                                                    "methodology": {"notes": notes}}))


def test_drops_and_filters(tmp_path):
    full = {"pass_1": 80.0, "pass_2": 70.0, "pass_3": 62.0, "pass_4": 55.0}
    make(tmp_path, "m1", {"airline": full, "retail": {"pass_1": 50.0, "pass_2": None, "pass_3": None, "pass_4": None}}, "4 trials")
    make(tmp_path, "A_EXAMPLE_x", {"airline": full})
    (tmp_path / "manifest.json").write_text(json.dumps({"submissions": ["m1"], "voice_submissions": [], "legacy_submissions": []}))
    df = T.load(tmp_path)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.drop_2_pp, r.drop_3_pp, r.drop_4_pp) == (10.0, 18.0, 25.0)
    assert r.category == "text" and r.trials_stated == 4 and r.monotone_non_increasing == 1
    s = T.summarize(df)
    assert s["n_submissions_complete"] == 1 and "no confidence intervals" in s["note"]
