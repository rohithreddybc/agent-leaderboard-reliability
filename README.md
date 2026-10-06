# Are LLM Agent Leaderboards Reliable? Code and derived data

Code, derived data and logs behind R. R. Bellibatlu and M. Singh, "Are LLM Agent Leaderboards Reliable? A Survey and Multi-Run Analysis of Agent Reliability Evaluation".
Canonical citation key: `bellibatlu2026leaderboards` (see `CITATION.cff`). Version 1.0.0.

Repository: https://github.com/rohithreddybc/agent-leaderboard-reliability. Archive (all versions): https://doi.org/10.5281/zenodo.23174950 (v1.0.0, 2026-10-05).

Licences: code MIT (`LICENSE`); derived data CC BY 4.0 (`LICENSE-data`). Upstream sources keep their own terms; see "Data sources".

## What each folder is

| Path | Contents |
|---|---|
| `analysis/` | Scripts, tests and outputs of the Terminal-Bench 2.0 multi-run re-analysis (`tb2_*.py`, `rank_inversion.py`, `bootstrap_core.py`, `gap_zone_count.py`), the tau2-bench aggregate check (`tau2_bounds.py`), the results report (`make_results_md.py`) and `fetch_data.py`. `analysis/results/` holds the derived tables and `results.md`. `CHANGES-2026-10-04.md` lists the revisions made after the review round, old and new values. |
| `analysis/leaderboards/` | The 22-leaderboard census: `census.py` (bounds, tiers, release files), `estimate_rho.py` (task-level correlation), data-collection helpers (`fetch_blobs.py`, `fetch_tac.py`, `consolidate_swebench.py`, `inventory_counts.py`, `wayback_check.py`, `make_manifest.py`) and `tests/`. `results/` holds the census outputs; `results/release/` holds the per-entry derived statistics (entry name, score, n, k, source URL, snapshot date, SHA-256 of the source file) and the entry-level exclusions with reasons. |
| `analysis/figures/` | `make_figures.py` (rank intervals, flip probability against gap) and `make_census_figure.py` (census figure). |
| `harness/` | The 20-task pinned run-to-run harness: code (`world.py`, `tasks.py`, `agent.py`, `client.py`, `run.py`), the system prompt and task prompts (`tasks.py`), tests, `results/runs.jsonl` (one record per run), `results/summary.csv`, `results/per_task.csv`, superseded runs and quota events. No API key is stored; the key is read from `GROQ_API_KEY` at run time. |
| `data/` | `SOURCES.md` and `leaderboards-SOURCES.md` (URLs, commits, licences, retrieval dates) and the SHA-256 manifests of the raw files used (`tb2_manifest.csv`, `tau2_manifest.csv`, `leaderboards_manifest.csv`). No raw data. |

## Raw data are not redistributed

No raw leaderboard file and no Terminal-Bench 2.0 trial file is redistributed here. Fifteen of the 22 census sources state no licence or terms, so only derived numbers are released. `analysis/results/tb2_long.csv`, the per-trial table derived from the Terminal-Bench 2.0 trial files, is also left out; `tb2_load.py` regenerates it. The data folder `data/raw/` is empty by design and is ignored by git. MedAgentBench, which the paper discusses in Section VIII-B, is not included or redistributed.

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

No network and no raw data are needed.

```bash
cd analysis && python -m pytest -q         # 72 tests
cd ../harness && python -m pytest tests -q # 52 tests
```

## Reproduce each table and figure

The numbers in the paper's running text are read from the files named below. Commands run from the repository root. Steps 1 and 2 need the raw data fetched as above; the figure steps need only the shipped outputs.

| Paper item | Command | Output |
|---|---|---|
| Section VII (multi-run analysis of Terminal-Bench 2.0) | `python analysis/tb2_load.py` then `python analysis/tb2_bootstrap.py` then `python analysis/rank_inversion.py` then `python analysis/rank_inversion.py --exclude MAYA__Claude-4.5-sonnet --suffix _sensitivity_excl_MAYA` then `python analysis/tau2_bounds.py` then `python analysis/gap_zone_count.py` (and with `--sensitivity`) then `python analysis/make_results_md.py` | `analysis/results/*.csv`, `*.json` and `results.md` (seed 20261008, B = 10,000) |
| Tables 4 to 6 (census leaderboards, correlation rho, census results) | `python analysis/leaderboards/census.py` (also runs `estimate_rho.py`) | `analysis/leaderboards/results/census*.csv`, `rho_*.csv`, `census_summary.md`, `release/` |
| Census figure (Figure 2) | `python analysis/figures/make_census_figure.py --out <dir>` | `fig_census.pdf` from `census.csv` |
| Rank-interval and flip-probability figures (Section VII) | `python analysis/figures/make_figures.py --out <dir>` | `fig_rank_intervals.pdf`, `fig_flip_vs_gap.pdf` from `analysis/results/` |
| Table 7 (pinned 20-task setup) | `python harness/summarize.py` then `python harness/summarize_for_paper.py` | `harness/results/summary.csv`, `per_task.csv`; the second script prints every number the section uses |
| New harness runs | `export GROQ_API_KEY=...` then `python harness/run.py --condition default` (and `--condition temp0`) | appends to `harness/results/runs.jsonl`; see `harness/README.md` |

Tables 1 to 3 and 8 (related works, queries, systematization, checklist) and the survey-flow figure come from the literature search and the authors' reading, not from code in this repository; the verbatim queries and the checklist are in the paper's supplement. Without the raw data you can still recompute everything from the shipped derived tables, for example `python analysis/make_results_md.py` and the two figure scripts.

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
  title   = {Are {LLM} Agent Leaderboards Reliable? A Survey and Multi-Run Analysis of Agent Reliability Evaluation},
  year    = {2026},
  note    = {Manuscript submitted to IEEE Access}
}
```

This entry will be updated with the preprint DOI and, on acceptance, the journal DOI.
