# LLM Agent Leaderboards Often Cannot Separate Their Top Two Entries: code and derived data

Code, derived data and logs behind R. R. Bellibatlu and M. Singh, "LLM Agent Leaderboards Often Cannot Separate Their Top Two Entries".
Canonical citation key: `bellibatlu2026leaderboards` (see `CITATION.cff`). Version 1.1.0. `RESULTS-MAP.md` maps each table and figure of the paper and its supplement to the script and files that produce it.

Repository: https://github.com/rohithreddybc/agent-leaderboard-reliability. Software archive (all versions): https://doi.org/10.5281/zenodo.23174950 (v1.0.0, 2026-10-05; v1.1.0, 2026-10-21). Please cite the paper (see Citation), not the archive DOI.

Licences: code MIT (`LICENSE`); derived data CC BY 4.0 (`LICENSE-data`). Upstream sources keep their own terms; see "Data sources".

## What each folder is

| Path | Contents |
|---|---|
| `analysis/` | Scripts, tests and outputs of the Terminal-Bench 2.0 multi-run re-analysis (`tb2_*.py`, `rank_inversion.py`, `bootstrap_core.py`, `gap_zone_count.py`), the tau2-bench aggregate check (`tau2_bounds.py`), the results report (`make_results_md.py`) and `fetch_data.py`. `robustness_tb2.py` (Terminal-Bench 2.0 missing-reward sensitivity) belongs to the robustness analyses below. `robustness_tb2_r3.py` and `robustness_tb2_r4.py` are the Terminal-Bench 2.0 parts of the round-3 and round-4 robustness analyses. `analysis/results/` holds the derived tables, `results.md` and the `robustness_2026-10-06*`, `robustness_2026-10-19*` and `robustness_2026-10-20*` files. `CHANGES-2026-10-04.md` lists the revisions made after the review round, old and new values. |
| `analysis/leaderboards/` | The 22-leaderboard census: `census.py` (bounds, tiers, release files), `estimate_rho.py` (task-level correlation), `robustness.py` (paired-test validation, calibration, eligibility, multiplicity and run-versus-task analyses), `robustness_r3.py` and `robustness_r4.py` (round-3 and round-4 robustness analyses of the census), data-collection helpers (`fetch_blobs.py`, `fetch_tac.py`, `consolidate_swebench.py`, `inventory_counts.py`, `wayback_check.py`, `make_manifest.py`) and `tests/`. `results/` holds the census outputs; `results/release/` holds the per-entry derived statistics (entry name, score, n, k, source URL, snapshot date, SHA-256 of the source file) and the entry-level exclusions with reasons. |
| `analysis/figures/` | `make_figures.py` (rank intervals, flip probability against gap), `make_tiers.py` (tier strips), `make_schematics.py` (overview, concept and axes schematics), `make_census_figure.py` (census figure) and the shared style module `viz_style.py`. |
| `harness/` | The 20-task pinned run-to-run harness: code (`world.py`, `tasks.py`, `agent.py`, `client.py`, `run.py`), the system prompt and task prompts (`tasks.py`), tests, `results/runs.jsonl` (one record per run), `results/summary.csv`, `results/per_task.csv`, superseded runs and quota events. No API key is stored; the key is read from `GROQ_API_KEY` at run time. |
| `data/` | `SOURCES.md` and `leaderboards-SOURCES.md` (URLs, commits, licences, retrieval dates) and the SHA-256 manifests of the raw files used (`tb2_manifest.csv`, `tau2_manifest.csv`, `leaderboards_manifest.csv`). No raw data. |

## Raw data are not redistributed

No raw leaderboard file and no Terminal-Bench 2.0 trial file is redistributed here. Fifteen of the 22 census sources state no licence or terms, so only derived numbers are released. `analysis/results/tb2_long.csv`, the per-trial table derived from the Terminal-Bench 2.0 trial files, is also left out; `tb2_load.py` regenerates it. The data folder `data/raw/` is empty by design and is ignored by git. MedAgentBench, which the paper discusses in Supplement S10, is not included or redistributed.

To fetch the sources again, use the URLs, commits and hashes in `data/SOURCES.md` and `data/leaderboards-SOURCES.md`, then compare your files with the SHA-256 values in the three manifests:

```bash
# from the repository root
python analysis/fetch_data.py tb2  --out data/raw/tb2     # Terminal-Bench 2.0, about 158 MB; writes data/tb2_manifest.csv
python analysis/fetch_data.py tau2 --out data/raw/tau2    # tau2-bench submissions, 0.2 MB; writes data/tau2_manifest.csv
```

`fetch_data.py` overwrites the shipped manifest in `data/`; keep a copy and compare the two. The other leaderboards (SWE-bench, Terminal-Bench 1.0, TheAgentCompany, AppWorld, BrowserGym, GAIA, MLE-bench, the three Google Sheets, the HAL pages) were retrieved as described in `data/leaderboards-SOURCES.md`, with the helper scripts in `analysis/leaderboards/` where one exists (`fetch_blobs.py`, `fetch_tac.py`, `consolidate_swebench.py`) and by a single request per sheet or page otherwise. They go under `data/raw/leaderboards/<name>/` with the file names listed in `data/leaderboards_manifest.csv`. Live sources may have changed since 2026-10-04; the manifest hashes tell whether a file matches.

## Setup

Python 3.11.7.

```bash
pip install -r requirements.txt            # numpy, pandas, scipy, networkx, PyYAML, requests, pytest, matplotlib, openpyxl
```

## Tests

No network is needed. 206 of the 208 analysis tests need no raw data. The other two (`test_committed_census_subset_and_report` and `test_sectionF_on_committed_census_matches_sectionB_and_round3` in `analysis/leaderboards/tests/test_robustness_r4.py`) read the per-task census snapshots under `data/raw/leaderboards/` (fetch them as described above) and fail with `FileNotFoundError` without them.

```bash
cd analysis && python -m pytest -q         # 208 tests: 69 in analysis/tests, 139 in analysis/leaderboards/tests
cd ../harness && python -m pytest tests -q # 52 tests
```

## Reproduce each table and figure

The numbers in the paper's running text are read from the files named below. Commands run from the repository root. Steps 1 and 2 need the raw data fetched as above; the figure steps need only the shipped outputs.

| Paper item | Command | Output |
|---|---|---|
| Multi-run analysis of Terminal-Bench 2.0 (Methods and Results) | `python analysis/tb2_load.py` then `python analysis/tb2_bootstrap.py` then `python analysis/rank_inversion.py` then `python analysis/rank_inversion.py --exclude MAYA__Claude-4.5-sonnet --suffix _sensitivity_excl_MAYA` then `python analysis/tau2_bounds.py` then `python analysis/gap_zone_count.py` (and with `--sensitivity`) then `python analysis/make_results_md.py` | `analysis/results/*.csv`, `*.json` and `results.md` (seed 20261008, B = 10,000) |
| Tables 4 to 6 (census leaderboards, correlation rho, census results; Tables S3 and S6) | `python analysis/leaderboards/census.py` (also runs `estimate_rho.py`) | `analysis/leaderboards/results/census*.csv`, `rho_*.csv`, `census_summary.md`, `release/` |
| Tier strips (Figure 2) | `python analysis/figures/make_tiers.py --out <dir>` | `fig_tiers.pdf` from `census_entries.csv`, `census_parameters.json` and `census.csv` |
| Census figure (Figure S1) | `python analysis/figures/make_census_figure.py --out <dir>` | `fig_census.pdf` from `census.csv` |
| Overview figure (Figure 1) | `python analysis/figures/make_schematics.py --out <dir>` | `fig_overview.pdf` (and `fig_concept.pdf`, `fig_axes.pdf`) from `census.csv` and `rank_inversion_summary.json` |
| Rank-interval and flip-probability figures (Figures 3 and 4) | `python analysis/figures/make_figures.py --out <dir>` | `fig_rank_intervals.pdf`, `fig_flip_vs_gap.pdf` from `analysis/results/` |
| Table S7 (pinned 20-task setup) | `python harness/summarize.py` then `python harness/summarize_for_paper.py` | `harness/results/summary.csv`, `per_task.csv`; the second script prints every number the section uses |
| New harness runs | `export GROQ_API_KEY=...` then `python harness/run.py --condition default` (and `--condition temp0`) | appends to `harness/results/runs.jsonl`; see `harness/README.md` |

Tables 7 and 8 (sensitivity and approximations of the census) are read from `census_summary.md` and the `robustness_2026-10-06*`, `robustness_2026-10-19*` and `robustness_2026-10-20*` files listed in `RESULTS-MAP.md`. Tables 1 to 3 and 9 and Tables S1, S2 and S8 (related works, systematization, notation, checklist, queries) and the survey-flow figure come from the literature search and the authors' reading, not from code in this repository; the verbatim queries and the remaining checklist items are in the paper's supplement. Without the raw data you can still recompute everything from the shipped derived tables, for example `python analysis/make_results_md.py` and the figure scripts.

## Robustness analyses

Added in response to a simulated peer review (2026-10-06 and 2026-10-07). They write new files only and change no earlier output. Seven analyses, reported in `analysis/results/robustness_2026-10-06.md` (numbers only):

1. Exact paired-test validation of the item-sampling bound (McNemar test, or a paired t-test where an entry averages several runs) on the 7 boards with per-task outcomes.
2. Calibration simulation of the bound: false-separation rate and power over a grid of n, P and rho.
3. Eligibility sensitivity: the census headline recomputed on 26 boards instead of 22.
4. Terminal-Bench 2.0 sensitivity with trials that have no verifier reward treated as missing instead of failed.
5. Multiplicity: Bonferroni and Holm versions of the pairwise verdicts.
6. Exact ties among the top-entry comparisons.
7. Run-versus-task variance share on the 72 Terminal-Bench 2.0 submissions, from the shipped `tb2_bootstrap.csv` and `tb2_within_task_variance.csv`.

`robustness.py` reads the raw census snapshots (fetched as described above), so a full rerun needs them; analysis 4 also needs `analysis/results/tb2_long.csv`, which `tb2_load.py` regenerates. Commands run from the repository root:

```bash
python analysis/tb2_load.py                            # regenerates analysis/results/tb2_long.csv (needs the Terminal-Bench 2.0 raw data)
python analysis/leaderboards/robustness.py             # all seven analyses; writes analysis/results/robustness_2026-10-06*
python analysis/leaderboards/robustness.py --skip-tb2  # without analysis 4
python analysis/robustness_tb2.py                      # analysis 4 alone; writes the _tb2_missing_reward.json file
```

Defaults: 20,000 simulation replicates, B = 10,000 bootstrap replicates and 1,000 splits (`--reps`, `--tb2-n-boot`, `--tb2-n-splits`). The tests (`analysis/leaderboards/tests/test_robustness.py`, `analysis/tests/test_robustness_tb2.py`) need neither network access nor raw data. The released outputs hold per-pair and per-board derived statistics only (the per-trial table `tb2_long.csv` is not released); the entry names and scores in them are those already released in `analysis/leaderboards/results/release/`.

### Rounds 3 and 4 (2026-10-19 and 2026-10-20)

Added after the second and third simulated review rounds. Four scripts, each writing only new files; each reproduces the committed headline numbers first and stops if they differ.

- `analysis/leaderboards/robustness_r3.py` (seed 20261019): 16 sections on the census headline bound, reported in `analysis/results/robustness_2026-10-19.md` and `robustness_2026-10-19_sec01_*` to `_sec16_*`. They cover order confidence, a parametric-bootstrap null, a rho sweep to 0.9, leave-one-family-out and all-pairs rho, the k-adjusted bound, Agresti-Caffo and Wilson standard errors, a cluster-robust bound, a calibration extension, the tau2-bench run-noise share f, the confidence set of possible leaders, agreement with the paired test, a systems-by-tasks variance decomposition, pair-specific rho and the rerun bound at f = 0.369.
- `analysis/robustness_tb2_r3.py` (seed 20261019): Terminal-Bench 2.0 task-set identity, overdispersion of runs, top-20 window correlations, a two-sided flip threshold, run share by k and a generalizability study; reported in `robustness_2026-10-19_tb2.md` and `robustness_2026-10-19_tb2_*`.
- `analysis/leaderboards/robustness_r4.py` (seed 20261020, no random step): the protocol-compatible subset of the census (section B), the bound and minimum-detectable-difference lookup (Table S4, section C), the leader-set check including entry 1 (section D), bound-versus-test agreement (section E) and the full primary analysis of the protocol-compatible subset (section F: ranks 1 versus 2 and 1 versus 5 across rho, tiers, order confidence, leader-set size, bound and paired-test agreement); reported in `robustness_2026-10-20.md` and `robustness_2026-10-20_sec*`. Only the plug-in simulation of section F is random.
- `analysis/robustness_tb2_r4.py` (seed 20261020): whole-run versus within-task resampling for the Terminal-Bench 2.0 submissions whose run directories each hold one trial of every task; reported in `robustness_2026-10-20_tb2.md` and `robustness_2026-10-20_tb2_*`.

```bash
python analysis/leaderboards/robustness_r3.py     # needs the raw census snapshots; defaults --boot-reps 2000 --cal-reps 20000
python analysis/leaderboards/robustness_r4.py     # shipped census outputs and robustness_2026-10-19_sec11_leader_set.csv; section F also needs the raw census snapshots (per-task outcomes)
python analysis/robustness_tb2_r3.py              # needs analysis/results/tb2_long.csv and, unless --skip-raw, data/raw/tb2
python analysis/robustness_tb2_r4.py              # needs analysis/results/tb2_long.csv
```

### Seeds

| Seed | Use |
|---|---|
| 20261008 | Terminal-Bench 2.0 bootstrap (`bootstrap_core.py`, B = 10,000) |
| 20261006 | calibration simulation of the item-sampling bound (`robustness.py`) |
| 20261007 | calibration of the Terminal-Bench 2.0 run-to-run bootstrap (`robustness_tb2_calibration.py`) |
| 20261019 | round-3 robustness (`robustness_r3.py`, `robustness_tb2_r3.py`) |
| 20261020 | round-4 robustness (`robustness_r4.py`, `robustness_tb2_r4.py`) |

The tests of the round-3 and round-4 scripts (`analysis/leaderboards/tests/test_robustness_r3.py`, `test_robustness_r4.py`, `analysis/tests/test_robustness_tb2_r3.py`, `test_robustness_tb2_r4.py`) need no network access. The two r4 tests named under Tests also need the raw census snapshots; the other r3, r4 and Terminal-Bench 2.0 tests need no raw data.

## Data sources

All retrieval dates are 2026-10-04. Full details (URL, revision, licence, local file hash) are in `data/SOURCES.md` and `data/leaderboards-SOURCES.md`.

| Source | URL | Pinned revision | Licence |
|---|---|---|---|
| Terminal-Bench 2.0 leaderboard | https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard | `572b2614be2c0cb2527e14f5b1e4026f1072e6c1` | Apache-2.0 |
| tau2-bench submissions | https://github.com/sierra-research/tau2-bench (`web/leaderboard/public/submissions/`) | `5bfa7e37b36656b37dc6d022156be6563c1007f3` | MIT |
| SWE-bench (five splits) | https://github.com/swe-bench/experiments | `40f164d5b8f1d249bf95a6df8b74b577fd8e519d` | none stated |
| Terminal-Bench 1.0 | https://github.com/laude-institute/terminal-bench-leaderboard | `3c08c2e1291e2e5562c0d89da250530ccd6d5119` | none stated |
| TheAgentCompany | https://github.com/TheAgentCompany/experiments | `f9b1e9411fcb554e76509a0e1bc8fe7f8b0e69d6` | none stated |
| AppWorld | https://github.com/stonybrooknlp/appworld-leaderboard | `c1f56015cf7c3441ff1933f5bfbec879798d7bbe` | none stated |
| BrowserGym leaderboard (WebArena, WorkArena) | https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard | `294ebe1dc36ba83c6916c64c64fb870af3dc8cee` | MIT |
| GAIA results (public) | https://huggingface.co/datasets/gaia-benchmark/results_public | `3be110f52679cf0850f6c29237564855423f9075` | none stated |
| MLE-bench | https://github.com/openai/mle-bench | `507f92e1138bb6e40dac5c6ee7a6758e6424bf97` | MIT |
| OSWorld, BFCL, AgentBench, Aider | see `data/leaderboards-SOURCES.md` | | inventoried and not analysed (or reference only) |
| WebArena, AndroidWorld, AgentBench Google Sheets | see `data/leaderboards-SOURCES.md` | CSV export, no revision id | none stated |
| HAL (nine leaderboards) | https://hal.cs.princeton.edu/ | HTML, no revision id | none stated |

Hashes of every local raw file are in `data/leaderboards_manifest.csv` (150 files), `data/tau2_manifest.csv` and `data/tb2_manifest.csv`.

## Harness models

`openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`, called through the Groq API on the dates recorded per run in `harness/results/runs.jsonl`. Results depend on a hosted service whose models and limits can change.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{bellibatlu2026leaderboards,
  author  = {Bellibatlu, Rohith Reddy and Singh, Manpreet},
  title   = {{LLM} Agent Leaderboards Often Cannot Separate Their Top Two Entries},
  year    = {2026},
  note    = {Manuscript submitted to IEEE Access}
}
```

This entry will be updated with the arXiv DOI and, on acceptance, the journal DOI. Please do not cite the Zenodo archive DOI.
