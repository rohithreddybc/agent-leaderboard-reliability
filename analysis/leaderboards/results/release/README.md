# Derived leaderboard statistics (release)

Derived per-entry statistics of the RQ2 census (`analysis/leaderboards/census.py`). Only these
columns are released: leaderboard, entry name, score (share of the n tasks solved), n_tasks, k_runs,
source_url, snapshot_date and source_file_sha256 (SHA-256 of the file held locally from that
source, as listed in `data/leaderboards_manifest.csv`, `data/tau2_manifest.csv` and
`data/tb2_manifest.csv`). No raw leaderboard file, trajectory, contact detail or free-text field of
a source is redistributed. Sources without a stated licence or terms are listed in
`census_summary.md`; the released values are derived numbers (counts divided by n), and whether
they may be published is the author's decision. `census_exclusions_release.csv` lists entries
excluded by an entry-level check, with the reason.

`rho_by_board_release.csv` and `rho_adjacent_pairs_release.csv` (added 2026-10-04) hold the
estimated task-level correlation rho between adjacent entries, computed from per-task outcomes
that are not released (`analysis/leaderboards/estimate_rho.py`): per board the median, quartiles and
range over adjacent pairs and the rho used in the headline bound, and per adjacent pair the entry
names, scores and rho. They are derived numbers of the same status as the scores.
