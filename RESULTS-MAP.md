# Results map

Which script produces each table and figure of the main paper and of the supplement, and which files it reads or writes. Numbers are those of `paper/main.pdf` (from `main.aux`) and `paper/supplement/supplement.pdf` (tables and figures prefixed S). Paths are relative to the repository root. `LB` is `analysis/leaderboards/results/`, `R` is `analysis/results/`.

Items that no script in this repository produces (literature reading, the framework, the reporting checklist, argued tables) are listed with "none".

## Main paper

| Item | Title | Script | Output files read or written |
|---|---|---|---|
| Fig. 1 | Overview of the paper in four panels | `analysis/figures/make_schematics.py` (`viz_style.py`) | reads `R/robustness_2026-10-20_secF_boards.csv` (variant P1, single-run standard errors: the primary set, 21 boards), `R/rank_inversion_summary.json`; writes `fig_overview.pdf` (panel 1 and the toy chart of panel 3 are illustrative) |
| Table 1 | Works nearest to this paper | none (literature reading) | |
| Table 2 | Framework for reliability measurement | none (framework definition) | |
| Table 3 | Notation | none | |
| Table 4 | The 22 census leaderboards | `analysis/leaderboards/census.py` | `LB/census_summary.md`, `LB/census_entries.csv`, `LB/census.csv`; `data/leaderboards-SOURCES.md`, `data/SOURCES.md` |
| Table 5 | Task-level correlation rho between entries | `analysis/leaderboards/estimate_rho.py` (run by `census.py`) | `LB/rho_by_board.csv`, `LB/rho_adjacent_pairs.csv`, `LB/rho_summary.json`, `LB/census_summary.md` |
| Fig. 2 | Tier strips for six census leaderboards (as-published diagnostic, 22 boards) | `analysis/figures/make_tiers.py` (`viz_style.py`) | reads `LB/census_entries.csv`, `LB/census_parameters.json`, `LB/census.csv`; writes `fig_tiers.pdf` |
| Table 6 | Census results per leaderboard under the headline bound (as-published diagnostic, all 775 entries) | `analysis/leaderboards/census.py` | `LB/census.csv`, `LB/census_summary.md` |
| Table 7 | Sensitivity of the census to the bound (upper rows as-published diagnostic; lower rows the primary set) | `analysis/leaderboards/census.py`; `analysis/leaderboards/robustness_r3.py` (sections 3, 10, 16); `analysis/leaderboards/robustness_r4.py` (section F, primary rows); `analysis/leaderboards/robustness.py` (Bonferroni row) | `LB/census_summary.md`; `R/robustness_2026-10-20_secF_rho_sweep.csv`, `R/robustness_2026-10-20_secF_boards.csv`; `R/robustness_2026-10-19_sec03_rho_sweep_summary.csv`, `R/robustness_2026-10-19_sec10_feasibility.csv`, `R/robustness_2026-10-19_sec16_rerun_summary.csv`; `R/robustness_2026-10-06_multiplicity_boards.csv` |
| Table 8 | Approximations of the census and their measured effect | `analysis/leaderboards/robustness_r3.py` (sections 3 to 8, 15); `analysis/leaderboards/robustness_r4.py` (section B); `analysis/leaderboards/robustness.py` (multiplicity) | `R/robustness_2026-10-19_sec03_rho_sweep_boards.csv`, `_sec04_lofo_boards.csv`, `_sec04_lofo_families.csv`, `_sec05_allpairs_rho_boards.csv`, `_sec06_kadj_boards.csv`, `_sec07_adjusted_se_boards.csv`, `_sec08_cluster_boards.csv`, `_sec08_cluster_pairs.csv`, `_sec15_pair_rho_boards.csv`; `R/robustness_2026-10-20_secB_boards.csv`, `_secB_entry_protocol.csv`; `R/robustness_2026-10-06_multiplicity_boards.csv`; reports `R/robustness_2026-10-19.md`, `R/robustness_2026-10-20.md` |
| Fig. 3 | Mean task success of the 72 Terminal-Bench 2.0 submissions with 95% intervals | `analysis/figures/make_figures.py` (`viz_style.py`) | reads `R/tb2_load_counts.csv`, `R/tb2_bootstrap.csv`, `R/rank_intervals_run.csv`; writes `fig_rank_intervals.pdf`. Inputs come from `tb2_load.py`, `tb2_bootstrap.py`, `rank_inversion.py` |
| Fig. 4 | Bootstrap flip probability against observed gap | `analysis/figures/make_figures.py` (`viz_style.py`) | reads `R/rank_inversion_pairs.csv`, `R/rank_gap_bins.csv`, `R/rank_inversion_summary.json`; writes `fig_flip_vs_gap.pdf`. Inputs come from `rank_inversion.py` |
| Table 9 | Proposed reporting checklist, items 1 to 8 | none (checklist); evidence cells cite `LB/census_summary.md`, `R/results.md`, `R/robustness_2026-10-19.md` and `R/robustness_2026-10-19_tb2.md` | |

The primary inferential set of the paper (the protocol-compatible subset, 653 entries on 21 boards) is read from `R/robustness_2026-10-20.md` section F; see Section F below. In-text Terminal-Bench 2.0 numbers (Methods and Results, multi-run re-analysis) are read from `R/results.md` (`analysis/make_results_md.py`), `R/rank_inversion_summary.json`, `R/split_half.csv`, `R/tb2_load_summary.json` and `R/tau2_summary.json`. The round-3 and round-4 Terminal-Bench 2.0 numbers are read from `R/robustness_2026-10-19_tb2.md` and `R/robustness_2026-10-20_tb2.md` (see the next section).

## Supplement

| Item | Title | Script | Output files read or written |
|---|---|---|---|
| Table S1 | Logged queries | none (search log) | |
| Table S2 | Works closest to this paper, one coding rule | none (literature reading) | |
| Table S3 | Snapshot date of each census leaderboard | `analysis/leaderboards/census.py` | `LB/census_summary.md`; `data/leaderboards-SOURCES.md`, `data/SOURCES.md` |
| Table S4 | Bound and 80%-power minimum detectable difference lookup | `analysis/leaderboards/robustness_r4.py` (section C) | `R/robustness_2026-10-20_secC_lookup.csv`, `R/robustness_2026-10-20.md` |
| Table S5 | Decisions taken after seeing results | none (decision log); effect columns read from the robustness reports | `R/robustness_2026-10-06.md`, `R/robustness_2026-10-19.md`, `R/robustness_2026-10-20.md` (sections B and F), `LB/census.csv` |
| Table S6 | Census results per leaderboard, all kept entries, all three bounds (as-published diagnostic) | `analysis/leaderboards/census.py` | `LB/census.csv`, `LB/census_summary.md` |
| Table S7 | Outcomes on the pinned 20-task setup | `harness/summarize_for_paper.py` (after `harness/summarize.py`) | `harness/results/runs.jsonl`, `harness/results/summary.csv`, `harness/results/per_task.csv` |
| Table S8 | Reporting checklist, items 9 to 14 | none (argued) | |
| Fig. S1 | Share of adjacent score gaps below the bound, per leaderboard (all kept entries, as-published diagnostic) | `analysis/figures/make_census_figure.py` (`viz_style.py`) | reads `LB/census.csv`; writes `fig_census.pdf` |
| Re-analysis supplement (`paper/supplement/supp-reanalysis.tex`), text paragraphs without a table | Task-set identity, missing reward, overdispersion of runs, run share by k, within-task variance, sensitivity to the flagged submission | `analysis/robustness_tb2_r3.py`, `analysis/robustness_tb2_r4.py`, `analysis/robustness_tb2.py` | see the round-3 and round-4 lists below |

## Round-3 and round-4 robustness outputs

Files written by each script, with the section of the report they belong to. All carry derived statistics only.

| Script | Report | Files |
|---|---|---|
| `analysis/leaderboards/robustness_r3.py` (seed 20261019) | `R/robustness_2026-10-19.md` | `R/robustness_2026-10-19_sec01_*` to `_sec16_*` (order confidence, bootstrap null, rho sweep, leave-one-family-out rho, all-pairs rho, k-adjusted bound, adjusted SE, cluster-robust bound, calibration, tau2 run-noise share, leader set, verified top pairs, kappa, variance decomposition, pair-specific rho, bound (b) rerun) |
| `analysis/robustness_tb2_r3.py` (seed 20261019) | `R/robustness_2026-10-19_tb2.md` | `R/robustness_2026-10-19_tb2_*.csv`, `R/robustness_2026-10-19_tb2_summary.json` (task-set identity, overdispersion, windows, two-sided flip threshold, run share by k, G-study) |
| `analysis/leaderboards/robustness_r4.py` (seed 20261020; only the section F plug-in simulation is random) | `R/robustness_2026-10-20.md` | `R/robustness_2026-10-20_secB_*`, `_secC_lookup.csv`, `_secD_leader_set_incl_entry1.csv`, `_secF_boards.csv`, `_secF_rho_sweep.csv`, `_secF_test_pairs.csv` |
| `analysis/robustness_tb2_r4.py` (seed 20261020) | `R/robustness_2026-10-20_tb2.md` | `R/robustness_2026-10-20_tb2_*.csv`, `R/robustness_2026-10-20_tb2_summary.json` (whole-run versus within-task resampling) |

The earlier robustness files (`R/robustness_2026-10-06*`) are written by `analysis/leaderboards/robustness.py`, `analysis/robustness_tb2.py` and `analysis/robustness_tb2_calibration.py`.

## F. Protocol-compatible primary set

The paper's primary inferential set is the protocol-compatible subset of the census: single-run and averaged entries, with selection-based entries, entries of unknown run count and MLE-bench dropped (653 of 775 entries, 21 of 22 boards). The all-entries census is reported as an as-published diagnostic. Both come from `analysis/leaderboards/robustness_r4.py`, which first reproduces the committed census headline and the committed McNemar verdicts of 401 pairs and stops if either differs. Section F needs the raw per-task census snapshots (`data/raw/leaderboards/`).

Variants: P1 keeps the two AppWorld boards (one run per entry assumed, as in the census methods); P2 drops them (19 boards). Standard errors: (i) single-run binomial SE for every entry (the primary choice; the paper's counts are P1 with SE (i)), (ii) k-adjusted SE for averaged entries (f = 0.369).

| Report section | Content | Used for | Files |
|---|---|---|---|
| B (B1 to B3) | Entry classification (single-run, averaged, selection-based, unknown), boards dropped, k-adjusted subset headline | Methods, Table 8 | `R/robustness_2026-10-20_secB_entry_protocol.csv`, `_secB_boards.csv` |
| F0 | Definitions of P1, P2, SE (i) and (ii) | Methods | `R/robustness_2026-10-20.md` |
| F1 | Entry 1 not separable from entry 2 and from entry 5, by rho (headline, 0, 0.25, 0.5, 0.75, 0.9) | abstract, Introduction, Results, Table 7, Fig. 1, conclusion | `R/robustness_2026-10-20_secF_rho_sweep.csv`, `_secF_boards.csv` |
| F2 | Tiers and share of adjacent pairs below the bound (headline rho) | Results, Table 7 | `R/robustness_2026-10-20_secF_boards.csv` |
| F3 | Order confidence c = Phi(gap / SE_D) | Results, Supplement S4 | `R/robustness_2026-10-20_secF_boards.csv`, `_secF_test_pairs.csv` |
| F4 | Flat-prior posterior probability that the observed leader has the highest true score (seed 20261020) | abstract, Introduction, Results, conclusion | `R/robustness_2026-10-20_secF_boards.csv` |
| F5 | Leader-set size including entry 1 | Results, Supplement S4 | `R/robustness_2026-10-20_secF_boards.csv` |
| F6 | Bound versus paired-test agreement on the adjacent pairs of the seven per-task boards (restricted to subset entries) | abstract, Supplement S4 | `R/robustness_2026-10-20_secF_test_pairs.csv` |
| F7 | How rho for k > 1 runs and the empirical-Bayes shrinkage are computed | Methods, Supplement S4 | `R/robustness_2026-10-20.md` |

`R/robustness_2026-10-20_secF_boards.csv` has one row per board, variant and SE choice; `_secF_rho_sweep.csv` one row per variant, SE choice and rho setting; `_secF_test_pairs.csv` one row per adjacent pair of the subset with the test and bound verdicts. All three hold derived statistics only (entry names and scores are those already released in `LB/release/`). Table and figure numbers in this map were re-checked against `paper/main.aux` and `paper/supplement/supplement.aux` of the 2026-10-20 build (main paper 19 pages, supplement 16 pages).
