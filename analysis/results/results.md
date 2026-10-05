# RQ3 re-analysis results

Numbers only. Every value is produced by a script in `analysis/`; see the CSV and JSON files in this folder.
Seed 20261008; B = 10,000 bootstrap replicates; split-half splits = 1,000.

## Definitions

- Success: verifier reward >= 1.0. A trial with no reward is counted as failure and flagged (`no_reward`).
- Submission mean: mean over tasks of the task success rate (trials / task).
- Leaderboard order: submissions sorted by observed mean, best first. The dataset holds no published score or rank.
- PRIMARY (run-to-run): tasks fixed; trials resampled with replacement within each task, per submission independently. Small-sample correction: resampling N trials from N understates the variance by (N-1)/N, so each task's resampled rate is rescaled about its observed rate by sqrt(N/(N-1)) (`bootstrap_core.py` docstring); `run SD (analytic)` is the closed form sqrt(sum_t phat(1-phat)/(N_t-1))/T, which the corrected bootstrap SD must match.
- SECONDARY (task-sampling): per-task trial means fixed; tasks resampled with replacement, same draw for every submission.
- Flip probability of a pair: share of replicates in which the observed lower-ranked submission scores above the higher-ranked one, plus half the share of exact ties.
- Separable pair: flip probability <= 0.05. Separable rank positions: size of the largest set of submissions in which every pair is separable.
- Gap: observed difference in submission means, in percentage points (pp).
- Analysis set: submissions with at least 5 trials on every task of the task union.

## Terminal-Bench 2.0 data

Dataset revision and licence: `data/SOURCES.md`.

| quantity | value |
|---|---|
| submissions | 75 |
| tasks (union) | 89 |
| run directories | 245 |
| trials | 32,803 |
| task git commits | 1 |
| trials with no verifier reward | 926 |
| (submission, task) cells | 6,675 |
| cells absent | 0 |
| cells with 1-4 trials | 267 |
| submissions with >= 5 trials on every task (analysis set) | 72 |

Trials per (submission, task) cell:

| trials in cell | cells |
|---|---|
| 1 | 267 |
| 5 | 6268 |
| 6 | 51 |
| 10 | 89 |

Excluded from the analysis set: 3 submissions.

| submission | n_trials | n_runs | n_tasks | trials_per_task_min | trials_per_task_max | reason |
|---|---|---|---|---|---|---|
| Mux__Claude-Opus-4.5 | 89 | 1 | 89 | 1 | 1 | 89 tasks with <5 trials |
| Mux__GPT-5.2 | 89 | 1 | 89 | 1 | 1 | 89 tasks with <5 trials |
| OpenCode__Claude-Opus-4.5 | 89 | 1 | 89 | 1 | 1 | 89 tasks with <5 trials |

Trials per task within the analysis set (min / median / max across submissions of each submission's per-task minimum, median, maximum):

| statistic | min | median | max |
|---|---|---|---|
| per-task minimum | 5.0 | 5.0 | 10.0 |
| per-task median | 5.0 | 5.0 | 10.0 |
| per-task maximum | 5.0 | 5.0 | 10.0 |

## Per-submission mean and intervals (analysis set)

Values in percent. `run` = PRIMARY run-to-run; `task` = SECONDARY task-sampling. SD and 95% percentile interval of the resampled mean.

| rank | submission | trials | mean | run SD | run SD (analytic) | run 95% CI | task SD | task 95% CI |
|---|---|---|---|---|---|---|---|---|
| 1 | vix__claude-opus-4-7 | 445 | 89.9 | 1.07 | 1.08 | 87.9 to 91.9 | 2.39 | 84.9 to 94.2 |
| 2 | Wecode__GPT-5.5 | 445 | 88.1 | 0.89 | 0.88 | 86.3 to 89.8 | 2.92 | 81.8 to 93.5 |
| 3 | JJAgent__Multiple | 890 | 87.1 | 0.68 | 0.68 | 85.8 to 88.4 | 2.95 | 81.0 to 92.5 |
| 4 | LemonHarness_GPT-5.3-CodeX | 445 | 84.5 | 1.32 | 1.31 | 82.0 to 87.0 | 2.81 | 78.7 to 89.7 |
| 5 | NexAU-AHE__gpt-5.5 | 445 | 84.5 | 1.09 | 1.09 | 82.2 to 86.5 | 3.18 | 78.0 to 90.3 |
| 6 | logos-latest__claude-opus-4.7 | 445 | 83.6 | 1.32 | 1.32 | 81.1 to 86.1 | 2.96 | 77.3 to 89.0 |
| 7 | Capy__GPT-5.5 | 445 | 83.1 | 1.07 | 1.08 | 80.9 to 85.2 | 3.33 | 76.4 to 89.4 |
| 8 | OB-1_GPT-5.4-GPT-5.3-Codex-Claude-Opus-4.5-Claude-Opus-4.6 | 445 | 82.5 | 0.68 | 0.67 | 81.2 to 83.7 | 3.79 | 74.6 to 89.4 |
| 9 | Polaris__Claude-Opus-4.7-GPT-5.5-Gemini-3.1-Pro | 445 | 82.0 | 1.42 | 1.42 | 79.3 to 84.8 | 2.95 | 76.0 to 87.6 |
| 10 | pilot-real__claude-opus-4-6 | 445 | 82.0 | 0.85 | 0.86 | 80.3 to 83.5 | 3.67 | 74.6 to 89.0 |
| 11 | Forge__GPT-5.4 | 445 | 81.8 | 1.02 | 1.03 | 79.8 to 83.8 | 3.52 | 74.6 to 88.3 |
| 12 | Forge__Opus-4.6 | 445 | 81.8 | 0.88 | 0.87 | 80.0 to 83.6 | 3.68 | 74.4 to 88.5 |
| 13 | CodeBrain-1.5__GPT-5.3-Codex | 445 | 81.3 | 1.19 | 1.18 | 79.1 to 83.6 | 3.39 | 74.4 to 87.6 |
| 14 | Judy__Gemini-3.1-Pro-Preview | 445 | 80.2 | 1.35 | 1.34 | 77.5 to 82.7 | 3.29 | 73.5 to 86.3 |
| 15 | WOZCODE__Claude-Opus-4-7 | 445 | 80.2 | 1.07 | 1.08 | 78.2 to 82.2 | 3.67 | 72.8 to 87.0 |
| 16 | logos-ts__claude-opus-4.7 | 445 | 80.0 | 1.31 | 1.32 | 77.5 to 82.5 | 3.33 | 73.3 to 86.1 |
| 17 | Forge__Gemini-3.1-Pro-Preview | 445 | 78.4 | 0.94 | 0.93 | 76.7 to 80.2 | 3.95 | 70.3 to 85.8 |
| 18 | OpenSage__GPT-5.3-Codex | 445 | 78.4 | 1.13 | 1.12 | 76.2 to 80.7 | 3.75 | 70.8 to 85.4 |
| 19 | LemonCode__GPT-5.3-Codex | 445 | 78.4 | 1.56 | 1.56 | 75.4 to 81.4 | 3.08 | 72.1 to 84.0 |
| 20 | Droid__GPT-5.3-Codex | 445 | 77.3 | 1.09 | 1.10 | 75.3 to 79.6 | 3.88 | 69.4 to 84.7 |
| 21 | Meta-Harness__Claude-Opus-4.6 | 445 | 76.4 | 1.25 | 1.24 | 73.9 to 78.9 | 3.76 | 69.0 to 83.4 |
| 22 | Codelia__GPT-5.3-Codex | 445 | 75.7 | 1.08 | 1.10 | 73.7 to 77.7 | 4.01 | 67.4 to 83.1 |
| 23 | Capy__Claude-Opus-4.6 | 445 | 75.3 | 1.23 | 1.22 | 72.8 to 77.5 | 3.88 | 67.4 to 82.5 |
| 24 | Terminus-KIRA__Gemini-3.1-Pro-Preview | 445 | 74.8 | 1.31 | 1.31 | 72.3 to 77.3 | 3.82 | 67.2 to 82.0 |
| 25 | Mux__GPT-5.3-Codex | 445 | 74.6 | 1.27 | 1.28 | 72.1 to 77.1 | 3.87 | 66.7 to 81.8 |
| 26 | Simple-Codex__GPT-5.3-Codex | 445 | 74.6 | 1.27 | 1.26 | 72.1 to 77.1 | 3.86 | 66.7 to 81.8 |
| 27 | Terminus-KIRA__Claude-Opus-4.6 | 445 | 74.4 | 1.23 | 1.22 | 72.1 to 76.9 | 3.95 | 66.5 to 81.8 |
| 28 | Ante__Gemini-3.1-Pro-Preview | 445 | 73.7 | 1.44 | 1.42 | 70.9 to 76.5 | 3.72 | 66.1 to 80.7 |
| 29 | OB-1_GPT-5.3-Codex-Claude-Opus-4.5-Claude-Opus-4.6 | 445 | 72.4 | 1.18 | 1.18 | 70.1 to 74.6 | 4.10 | 64.0 to 80.0 |
| 30 | Judy__Claude-Opus-4.6 | 445 | 71.9 | 1.35 | 1.37 | 69.4 to 74.4 | 3.92 | 64.0 to 79.3 |
| 31 | MAYA__Claude-4.6-opus | 445 | 71.9 | 1.11 | 1.10 | 69.6 to 74.2 | 4.20 | 63.6 to 79.8 |
| 32 | spoox-o-m__GPT-5.3-Codex | 445 | 71.5 | 1.29 | 1.27 | 68.9 to 74.0 | 4.09 | 62.9 to 79.3 |
| 33 | Junie_CLI__Gemini-3-Flash-Preview-Gemini-3.1-Pro-Preview-Claude-Opus-4.6-GPT-5.3-Codex | 445 | 71.0 | 1.47 | 1.46 | 68.0 to 74.0 | 3.83 | 63.1 to 78.2 |
| 34 | CodeBrain-1__GPT-5.3-Codex | 445 | 70.3 | 1.31 | 1.31 | 67.8 to 72.8 | 4.12 | 62.0 to 78.0 |
| 35 | Droid__Claude-Opus-4.6 | 445 | 69.9 | 1.27 | 1.26 | 67.4 to 72.4 | 4.14 | 61.6 to 77.5 |
| 36 | WozCode__Claude-Opus-4.6 | 445 | 68.1 | 1.23 | 1.23 | 65.8 to 70.6 | 4.33 | 59.3 to 76.2 |
| 37 | Mux__Claude-Opus-4.6 | 445 | 66.5 | 1.29 | 1.29 | 64.0 to 69.0 | 4.30 | 58.2 to 74.6 |
| 38 | logos__claude-opus-4.6 | 445 | 66.3 | 1.31 | 1.32 | 63.8 to 68.8 | 4.28 | 57.5 to 74.4 |
| 39 | clnkr__GPT-5.5 | 445 | 66.1 | 1.26 | 1.27 | 63.6 to 68.6 | 4.35 | 57.3 to 74.2 |
| 40 | Deep-Agents__GPT-5.2-Codex | 445 | 65.8 | 1.49 | 1.48 | 63.1 to 68.9 | 4.05 | 57.8 to 73.5 |
| 41 | Dirac__Gemini-3-Flash-Preview | 445 | 65.2 | 1.27 | 1.26 | 62.7 to 67.7 | 4.38 | 56.4 to 73.5 |
| 42 | OpenSage__Gemini-3-Pro-Preview | 445 | 65.2 | 1.04 | 1.05 | 63.2 to 67.2 | 4.63 | 56.0 to 74.2 |
| 43 | Terminus2__GPT-5.3-Codex | 445 | 64.7 | 1.40 | 1.39 | 62.0 to 67.5 | 4.26 | 56.2 to 72.8 |
| 44 | Ante__Gemini-3-Pro-Preview | 445 | 64.7 | 1.37 | 1.38 | 62.0 to 67.5 | 4.27 | 56.2 to 72.8 |
| 45 | Terminus2__Claude-Opus-4.6 | 445 | 62.9 | 1.36 | 1.36 | 60.2 to 65.4 | 4.37 | 54.2 to 71.2 |
| 46 | CodeBrain-1__Gemini-3-Pro-Preview | 445 | 62.2 | 1.32 | 1.33 | 59.7 to 64.8 | 4.40 | 53.5 to 70.6 |
| 47 | Crux__Claude-Opus-4.6 | 445 | 61.1 | 1.97 | 1.97 | 57.4 to 64.9 | 3.36 | 54.6 to 67.6 |
| 48 | hookele__gpt5.1-codex-mini | 445 | 61.1 | 1.01 | 1.00 | 59.1 to 63.1 | 4.75 | 51.7 to 70.3 |
| 49 | copilot-cli__claude-opus-4.6 | 445 | 60.4 | 1.20 | 1.20 | 58.2 to 62.7 | 4.60 | 51.2 to 69.2 |
| 50 | Gemini_CLI__Gemini-3.1-Pro-Preview | 445 | 59.8 | 1.49 | 1.47 | 56.8 to 62.5 | 4.37 | 51.2 to 68.1 |
| 51 | IndusAGICodingAgent__gpt-5.3-codex | 445 | 58.7 | 1.71 | 1.69 | 55.4 to 61.9 | 4.02 | 50.6 to 66.3 |
| 52 | Simplai-Agent__Claude-Sonnet-4.6 | 445 | 52.1 | 1.40 | 1.39 | 49.4 to 54.9 | 4.51 | 43.4 to 60.9 |
| 53 | Terminus2__GLM-5 | 445 | 51.9 | 1.34 | 1.36 | 49.1 to 54.4 | 4.55 | 42.9 to 60.7 |
| 54 | Gemini_CLI__Gemini-3-Flash-Preview | 445 | 47.4 | 1.55 | 1.55 | 44.4 to 50.4 | 4.30 | 38.9 to 55.7 |
| 55 | just-another-coding-agent__GLM-5 | 445 | 47.4 | 1.63 | 1.60 | 44.1 to 50.7 | 4.23 | 39.1 to 55.7 |
| 56 | MAYA__Claude-4.5-sonnet | 445 | 42.7 | 0.00 | 0.00 | 42.7 to 42.7 | 5.20 | 32.6 to 52.8 |
| 57 | harness-agent__minimax-m2-7-highspeed | 445 | 42.5 | 1.53 | 1.52 | 39.5 to 45.5 | 4.21 | 34.4 to 50.6 |
| 58 | Terminus2__Kimi-k2.5 | 445 | 42.5 | 1.49 | 1.48 | 39.5 to 45.2 | 4.31 | 34.2 to 51.0 |
| 59 | Terminus2__Minimax-m2.5 | 445 | 42.2 | 1.35 | 1.34 | 39.5 to 45.0 | 4.52 | 33.7 to 51.2 |
| 60 | cchuter__minimax-m2.5 | 445 | 42.2 | 1.45 | 1.45 | 39.5 to 45.0 | 4.34 | 33.7 to 50.8 |
| 61 | grok-cli__grok-4.20-0309-reasoning | 496 | 41.5 | 1.46 | 1.44 | 38.6 to 44.3 | 4.20 | 33.2 to 49.8 |
| 62 | 0error-Ledger__Claude-Opus-4.7 | 445 | 39.8 | 2.10 | 2.13 | 35.8 to 43.8 | 2.94 | 33.9 to 45.4 |
| 63 | Terminus2__DeepSeek-V3.2 | 445 | 39.6 | 1.40 | 1.40 | 36.8 to 42.3 | 4.38 | 31.0 to 48.3 |
| 64 | IndusAGICodingAgent__MiniMax-M2.7 | 445 | 34.4 | 1.80 | 1.80 | 30.9 to 37.9 | 3.52 | 27.6 to 41.3 |
| 65 | ClaudeCode__GLM-4.7 | 445 | 33.3 | 1.26 | 1.25 | 30.7 to 35.8 | 4.30 | 24.9 to 41.8 |
| 66 | Terminus2__GLM-4.7 | 445 | 33.0 | 1.37 | 1.38 | 30.3 to 35.8 | 4.17 | 24.9 to 41.3 |
| 67 | dakou__qwen3-coder-480b | 445 | 27.2 | 1.30 | 1.31 | 24.7 to 29.7 | 3.92 | 19.8 to 35.1 |
| 68 | little-coder__qwen3.6-35b-a3b | 445 | 23.8 | 1.35 | 1.33 | 21.1 to 26.3 | 3.65 | 16.9 to 31.2 |
| 69 | spoox-o-m__GPT-5-Nano | 445 | 21.8 | 1.45 | 1.43 | 19.0 to 24.6 | 3.32 | 15.5 to 28.5 |
| 70 | BashAgent__TermiGen-32B | 445 | 19.3 | 1.01 | 1.02 | 17.3 to 21.3 | 3.68 | 12.4 to 26.7 |
| 71 | terminus-2__AfterQuery-GPT-OSS-20B | 445 | 16.9 | 1.25 | 1.25 | 14.3 to 19.4 | 3.07 | 11.0 to 23.1 |
| 72 | little-coder__qwen3.5-9b | 445 | 9.2 | 1.22 | 1.20 | 7.0 to 11.7 | 1.90 | 5.6 to 13.0 |

Run-to-run SD across the analysis set (pp): min 0.00, median 1.30, max 2.10. Task-sampling SD (pp): min 1.90, median 3.93, max 5.20.

Check of the corrected bootstrap against the closed form (submissions with non-zero run variance, 71 of 72): ratio bootstrap SD / analytic SD has median 1.002, min 0.984, max 1.018. Median analytic run SD 1.30 pp.

## Within-task variance (analysis set)

p = task success rate over trials. `p(1-p)` is averaged over tasks; unbiased = p(1-p) n/(n-1). Per-submission rows: `tb2_within_task_variance.csv`.

| quantity (across submissions) | min | median | max |
|---|---|---|---|
| mean_p1mp_plugin | 0.0000 | 0.0602 | 0.1618 |
| mean_p1mp_unbiased | 0.0000 | 0.0753 | 0.2022 |
| share_tasks_mixed | 0.0000 | 0.3034 | 0.7191 |
| share_tasks_all_success | 0.0000 | 0.5056 | 0.7978 |
| share_tasks_all_fail | 0.0225 | 0.1798 | 0.7416 |

## Run-noise share f and the detectability bound (worked numbers)

f = v / (P(1-P)): the share of a single run's outcome variance that is run-to-run noise, with v the unbiased mean within-task variance of a submission and P its mean success. Undefined at P = 0 or 1. The detectability bound is the smallest observed difference between two systems that is significant at 95% (two-sided z = 1.96; 50% power at a true difference equal to the bound), z * sqrt(2 v / (n k)); no power claim beyond that is made.

| quantity (across the analysis set) | value |
|---|---|
| v, median (min to max) | 0.0753 (0.0000 to 0.2022) |
| f, median (IQR; min to max), 72 submissions | 0.369 (0.325 to 0.434; 0.000 to 0.844) |
| f, median (IQR; min to max), without the flagged submission, 71 | 0.369 (0.326 to 0.434; 0.140 to 0.844) |
| corr(v, P) across submissions | -0.334 |

Bound in pp, with v = median v transported absolutely (z = 1.96 two-sided; z = 1.645 is the one-sided 5% rule used by the flip probability) and with v = f x P(1-P) at f = 0.369 for P = 0.2, 0.5, 0.8 (both systems at P):

| case | z = 1.96, v median | z = 1.645, v median | z = 1.96, f median, P = 0.2 | z = 1.96, f median, P = 0.5 | z = 1.96, f median, P = 0.8 |
|---|---|---|---|---|---|
| illustration: n = 100, k = 1 | 7.61 | 6.38 | 6.73 | 8.42 | 6.73 |
| Terminal-Bench 2.0 design: n = 89, k = 5 | 3.61 | 3.03 | 3.19 | 3.99 | 3.19 |
| ReliabilityBench check: n = 20, k = 2 | 12.02 | 10.09 | 10.65 | 13.31 | 10.65 |

For comparison, the bootstrap gap_max_unstable (run-to-run) of the Terminal-Bench 2.0 analysis set is 3.60 pp and gap_min_stable 2.25 pp (Rank inversion table).

## Rank inversion

| quantity | PRIMARY run-to-run | SECONDARY task-sampling |
|---|---|---|
| submissions / tasks / pairs | 72 / 89 / 2556 |  |
| adjacent pairs | 71 | 71 |
| adjacent pairs with flip probability > 5% | 63 (88.7%) | 70 (98.6%) |
| adjacent pairs with flip probability > 20% | 54 (76.1%) | 64 (90.1%) |
| all pairs with flip probability > 5% | 10.0% | 19.7% |
| separable rank positions (exact) | 20 | 13 |
| separable rank positions (greedy from top) | 19 | 12 |
| gap_max_unstable (pp) | 3.60 | 9.66 |
| gap_min_stable (pp) | 2.25 | 3.60 |
| gap_bin_stable (pp) | 3.0 | 6.0 |
| Kendall tau, observed rank vs mean bootstrap rank | 0.9945 | 0.9914 |
| observed rank outside own 95% rank interval | 0 | 0 |
| rank interval width, median (positions) | 9.0 | 14.5 |
| rank interval width, max (positions) | 13.0 | 27.5 |

Files: `rank_inversion_pairs.csv`, `rank_inversion_adjacent.csv`, `rank_intervals_run.csv`, `rank_intervals_task.csv`, `rank_gap_bins.csv`.

## Sensitivity: analysis set without flagged submissions

Flag: every task has identical outcomes across all its trials (zero within-task variance), or trial names repeat within a task. Flagged members of the analysis set: MAYA__Claude-4.5-sonnet (114 repeated trial names, zero within-task variance = 1).

| quantity | PRIMARY run-to-run | SECONDARY task-sampling |
|---|---|---|
| submissions | 71 | 71 |
| adjacent pairs with flip probability > 5% | 62 (88.6%) | 68 (97.1%) |
| adjacent pairs with flip probability > 20% | 53 (75.7%) | 62 (88.6%) |
| separable rank positions (exact) | 19 | 13 |
| gap_max_unstable (pp) | 3.60 | 7.64 |
| gap_min_stable (pp) | 2.25 | 3.60 |
| split-half Spearman-Brown Spearman, mean | 0.9917 |  |

## Split-half reliability of the ranking

1,000 random splits of each task's trials into two disjoint halves; Spearman-Brown m = 2.477. Mean and 2.5%/97.5% quantiles over splits.

| statistic | mean | q2.5 | q97.5 |
|---|---|---|---|
| spearman | 0.9802 | 0.9720 | 0.9869 |
| kendall | 0.8972 | 0.8754 | 0.9185 |
| spearman_sb | 0.9919 | 0.9885 | 0.9947 |
| kendall_sb | 0.9558 | 0.9457 | 0.9654 |

`*_sb`: Spearman-Brown corrected to the full trial count (applied to Kendall tau as a heuristic).

## tau2-bench aggregates

Aggregates only: no per-task or per-trial outcomes, so no trial-level or task-level resampling and no confidence intervals are possible.

Submissions with pass^1 to pass^4 present in at least one domain: 36 (of which with 4 trials stated in the submission file: 16). Submission-domain rows: 74; rows where pass^k increases with k: 0.

Drop pass^1 - pass^4 (pp) by domain:

| domain | submissions | min | median | mean | max |
|---|---|---|---|---|---|
| airline | 15 | 7.0 | 14.5 | 15.7 | 30.5 |
| banking_knowledge | 28 | 4.6 | 15.2 | 14.7 | 23.7 |
| retail | 16 | 12.4 | 24.6 | 24.6 | 32.9 |
| telecom | 15 | 4.4 | 16.5 | 15.9 | 26.5 |

Per-submission drops (pp):

| submission | domain | trials_stated | pass_1 | pass_2 | pass_3 | pass_4 | drop_2_pp | drop_3_pp | drop_4_pp |
|---|---|---|---|---|---|---|---|---|---|
| claude-3-7-sonnet_anthropic_2024-06-20 | retail |  | 72.1 | 66.8 | 63.2 | 59.7 | 5.30 | 8.90 | 12.40 |
| claude-3-7-sonnet_anthropic_2024-06-20 | airline |  | 64.2 | 58.9 | 55.4 | 52.1 | 5.30 | 8.80 | 12.10 |
| claude-3-7-sonnet_anthropic_2024-06-20 | telecom |  | 49.0 | 43.7 | 40.1 | 37.2 | 5.30 | 8.90 | 11.80 |
| claude-fable-5_sierra_2026-08-04 | banking_knowledge |  | 39.69072164948454 | 32.817869415807564 | 30.15463917525773 | 28.865979381443296 | 6.87 | 9.54 | 10.82 |
| claude-opus-4-5_sierra_2026-02-26 | airline | 4 | 84.0 | 77.67 | 73.5 | 70.0 | 6.33 | 10.50 | 14.00 |
| claude-opus-4-5_sierra_2026-02-26 | retail | 4 | 79.61 | 67.4 | 58.77 | 51.75 | 12.21 | 20.84 | 27.86 |
| claude-opus-4-5_sierra_2026-02-26 | telecom | 4 | 92.32 | 86.11 | 81.36 | 78.07 | 6.21 | 10.96 | 14.25 |
| claude-opus-4-5_sierra_2026-02-26 | banking_knowledge | 4 | 24.74 | 17.18 | 13.66 | 11.34 | 7.56 | 11.08 | 13.40 |
| claude-opus-4-6_sierra_2026-05-05 | banking_knowledge | 4 | 27.32 | 18.73 | 14.18 | 11.34 | 8.59 | 13.14 | 15.98 |
| claude-opus-4-7_sierra_2026-05-05 | banking_knowledge |  | 40.20618556701031 | 31.099656357388323 | 27.061855670103093 | 24.742268041237114 | 9.11 | 13.14 | 15.46 |
| claude-opus-4-8_sierra_2026-08-04 | banking_knowledge |  | 39.69072164948454 | 31.271477663230247 | 26.28865979381444 | 22.68041237113402 | 8.42 | 13.40 | 17.01 |
| claude-opus-5_sierra_2026-08-04 | banking_knowledge |  | 48.71134020618557 | 39.175257731958766 | 34.5360824742268 | 31.95876288659793 | 9.54 | 14.18 | 16.75 |
| claude-sonnet-4-5_sierra_2026-02-26 | airline | 4 | 72.0 | 62.0 | 54.5 | 48.0 | 10.00 | 17.50 | 24.00 |
| claude-sonnet-4-5_sierra_2026-02-26 | retail | 4 | 72.37 | 57.46 | 47.37 | 39.47 | 14.91 | 25.00 | 32.90 |
| claude-sonnet-4-5_sierra_2026-02-26 | telecom | 4 | 84.87 | 75.44 | 68.86 | 64.04 | 9.43 | 16.01 | 20.83 |
| claude-sonnet-4-5_sierra_2026-02-26 | banking_knowledge | 4 | 25.26 | 15.81 | 12.63 | 10.31 | 9.45 | 12.63 | 14.95 |
| distyl-buttonagent_distyl_2026-03-25 | banking_knowledge | 4 | 31.19 | 21.47 | 16.19 | 13.4 | 9.72 | 15.00 | 17.79 |
| gemini-2-5-pro_sierra_2026-05-05 | banking_knowledge | 4 | 13.66 | 6.01 | 2.58 | 1.03 | 7.65 | 11.08 | 12.63 |
| gemini-3-1-pro-preview_sierra_2026-05-05 | banking_knowledge | 4 | 26.03 | 15.12 | 11.08 | 9.28 | 10.91 | 14.95 | 16.75 |
| gemini-3-flash_sierra_2026-03-02 | airline | 4 | 82.5 | 76.33 | 72.0 | 68.0 | 6.17 | 10.50 | 14.50 |
| gemini-3-flash_sierra_2026-03-02 | retail | 4 | 76.75 | 65.94 | 57.89 | 51.75 | 10.81 | 18.86 | 25.00 |
| gemini-3-flash_sierra_2026-03-02 | telecom | 4 | 91.23 | 83.48 | 76.54 | 70.18 | 7.75 | 14.69 | 21.05 |
| gemini-3-flash_sierra_2026-03-02 | banking_knowledge | 4 | 27.32 | 15.46 | 10.31 | 7.22 | 11.86 | 17.01 | 20.10 |
| gemini-3-pro_sierra_2026-03-02 | airline | 4 | 80.5 | 74.67 | 70.0 | 66.0 | 5.83 | 10.50 | 14.50 |
| gemini-3-pro_sierra_2026-03-02 | retail | 4 | 75.88 | 63.45 | 54.39 | 47.37 | 12.43 | 21.49 | 28.51 |
| gemini-3-pro_sierra_2026-03-02 | telecom | 4 | 91.01 | 84.5 | 79.17 | 74.56 | 6.51 | 11.84 | 16.45 |
| gemini-3-pro_sierra_2026-03-02 | banking_knowledge | 4 | 18.04 | 9.28 | 5.67 | 4.12 | 8.76 | 12.37 | 13.92 |
| glm-5-2_sierra_2026-08-04 | banking_knowledge |  | 37.11340206185567 | 25.25773195876289 | 18.298969072164947 | 13.402061855670103 | 11.86 | 18.81 | 23.71 |
| glm-5-think_sierra_2026-03-02 | airline | 4 | 82.5 | 76.0 | 72.0 | 70.0 | 6.50 | 10.50 | 12.50 |
| glm-5-think_sierra_2026-03-02 | retail | 4 | 73.68 | 60.38 | 51.1 | 43.86 | 13.30 | 22.58 | 29.82 |
| glm-5-think_sierra_2026-03-02 | telecom | 4 | 86.84 | 76.32 | 68.2 | 62.28 | 10.52 | 18.64 | 24.56 |
| glm-5-think_sierra_2026-03-02 | banking_knowledge | 4 | 9.79 | 7.22 | 5.67 | 3.09 | 2.57 | 4.12 | 6.70 |
| gpt-4-1-mini_openai_2024-06-20 | retail |  | 61.4 | 49.8 | 42.3 | 36.9 | 11.60 | 19.10 | 24.50 |
| gpt-4-1-mini_openai_2024-06-20 | airline |  | 48.7 | 38.9 | 32.1 | 27.8 | 9.80 | 16.60 | 20.90 |
| gpt-4-1-mini_openai_2024-06-20 | telecom |  | 48.9 | 38.7 | 31.2 | 26.1 | 10.20 | 17.70 | 22.80 |
| gpt-4-1_openai_2024-06-20 | retail |  | 74.0 | 64.2 | 58.1 | 52.3 | 9.80 | 15.90 | 21.70 |
| gpt-4-1_openai_2024-06-20 | airline |  | 56.0 | 47.8 | 42.4 | 38.1 | 8.20 | 13.60 | 17.90 |
| gpt-4-1_openai_2024-06-20 | telecom |  | 34.0 | 27.6 | 23.8 | 20.4 | 6.40 | 10.20 | 13.60 |
| gpt-5-2-none_sierra_2026-02-26 | airline | 4 | 52.5 | 35.66666666666667 | 27.0 | 22.0 | 16.83 | 25.50 | 30.50 |
| gpt-5-2-none_sierra_2026-02-26 | retail | 4 | 75.0 | 62.13450292397662 | 53.07017543859649 | 45.614035087719294 | 12.87 | 21.93 | 29.39 |
| gpt-5-2-none_sierra_2026-02-26 | telecom | 4 | 57.23684210526315 | 42.83625730994152 | 35.526315789473685 | 30.701754385964914 | 14.40 | 21.71 | 26.54 |
| gpt-5-2-none_sierra_2026-02-26 | banking_knowledge | 4 | 12.63 | 7.22 | 5.15 | 4.12 | 5.41 | 7.48 | 8.51 |
| gpt-5-2_sierra_2026-02-26 | airline | 4 | 83.0 | 78.33 | 75.0 | 72.0 | 4.67 | 8.00 | 11.00 |
| gpt-5-2_sierra_2026-02-26 | retail | 4 | 81.58 | 69.59 | 59.87 | 51.75 | 11.99 | 21.71 | 29.83 |
| gpt-5-2_sierra_2026-02-26 | telecom | 4 | 89.69 | 82.46 | 76.75 | 71.93 | 7.23 | 12.94 | 17.76 |
| gpt-5-2_sierra_2026-02-26 | banking_knowledge | 4 | 32.22 | 23.88 | 20.62 | 18.56 | 8.34 | 11.60 | 13.66 |
| gpt-5-4_sierra_2026-03-25 | banking_knowledge | 4 | 39.43 | 29.55 | 24.48 | 21.65 | 9.88 | 14.95 | 17.78 |
| gpt-5-5_sierra_2026-05-05 | banking_knowledge |  | 44.58762886597938 | 37.62886597938145 | 33.24742268041237 | 29.89690721649485 | 6.96 | 11.34 | 14.69 |
| gpt-5-6-sol_sierra_2026-08-04 | banking_knowledge |  | 46.90721649484536 | 37.28522336769759 | 31.95876288659793 | 27.835051546391757 | 9.62 | 14.95 | 19.07 |
| gpt-5_sierra_2025-08-09 | retail |  | 81.57894736842105 | 71.78362573099416 | 64.69298245614036 | 58.77192982456141 | 9.80 | 16.89 | 22.81 |
| gpt-5_sierra_2025-08-09 | airline |  | 62.5 | 55.33333333333332 | 51.0 | 48.0 | 7.17 | 11.50 | 14.50 |
| gpt-5_sierra_2025-08-09 | telecom |  | 95.83333333333334 | 91.95906432748538 | 88.37719298245614 | 85.08771929824562 | 3.87 | 7.46 | 10.75 |
| grok-4-1-fast_sierra_2026-05-05 | banking_knowledge | 4 | 13.14 | 7.73 | 5.67 | 5.15 | 5.41 | 7.47 | 7.99 |
| grok-4-2_sierra_2026-05-05 | banking_knowledge | 4 | 18.04 | 12.37 | 10.05 | 8.25 | 5.67 | 7.99 | 9.79 |
| grok-4-5_sierra_2026-08-04 | banking_knowledge |  | 47.93814432989691 | 39.51890034364261 | 35.30927835051546 | 31.95876288659793 | 8.42 | 12.63 | 15.98 |
| grok-4-fast_sierra_2026-05-05 | banking_knowledge | 4 | 15.72 | 8.42 | 5.67 | 4.12 | 7.30 | 10.05 | 11.60 |
| inkling_sierra_2026-08-04 | banking_knowledge |  | 25.0 | 16.666666666666664 | 13.402061855670103 | 11.34020618556701 | 8.33 | 11.60 | 13.66 |
| kimi-k3_sierra_2026-08-04 | banking_knowledge |  | 37.11340206185567 | 25.25773195876289 | 20.103092783505154 | 17.525773195876287 | 11.86 | 17.01 | 19.59 |
| muse-spark-1-1_sierra_2026-08-04 | banking_knowledge |  | 40.4639175257732 | 29.55326460481099 | 23.969072164948454 | 20.61855670103093 | 10.91 | 16.49 | 19.85 |
| o4-mini_openai_2024-06-20 | retail |  | 68.3 | 58.7 | 52.1 | 46.8 | 9.60 | 16.20 | 21.50 |
| o4-mini_openai_2024-06-20 | airline |  | 52.1 | 44.2 | 38.9 | 34.7 | 7.90 | 13.20 | 17.40 |
| o4-mini_openai_2024-06-20 | telecom |  | 50.2 | 42.1 | 36.8 | 32.4 | 8.10 | 13.40 | 17.80 |
| qwen3-8-max_sierra_2026-08-04 | banking_knowledge |  | 55.154639175257735 | 45.18900343642611 | 39.69072164948454 | 35.051546391752574 | 9.97 | 15.46 | 20.10 |
| qwen3-max_qwen_2025-10-30 | retail |  | 75.44 | 66.23 | 60.09 | 55.26 | 9.21 | 15.35 | 20.18 |
| qwen3-max_qwen_2025-10-30 | airline |  | 71.0 | 65.33333333333334 | 62.5 | 60.0 | 5.67 | 8.50 | 11.00 |
| qwen3-max_qwen_2025-10-30 | telecom |  | 95.83333333333334 | 91.95906432748538 | 88.37719298245614 | 85.08771929824562 | 3.87 | 7.46 | 10.75 |
| qwen3-max_qwen_2026-01-23 | retail |  | 79.3859649122807 | 69.44444444444446 | 62.71929824561403 | 57.89473684210527 | 9.94 | 16.67 | 21.49 |
| qwen3-max_qwen_2026-01-23 | airline |  | 69.0 | 65.33333333333333 | 63.5 | 62.0 | 3.67 | 5.50 | 7.00 |
| qwen3-max_qwen_2026-01-23 | telecom |  | 98.24561403508773 | 96.6374269005848 | 95.17543859649122 | 93.859649122807 | 1.61 | 3.07 | 4.39 |
| qwen3.5-397b-a17b-think_sierra_2026-03-02 | airline | 4 | 81.5 | 75.67 | 71.5 | 68.0 | 5.83 | 10.00 | 13.50 |
| qwen3.5-397b-a17b-think_sierra_2026-03-02 | retail | 4 | 84.43 | 74.42 | 66.45 | 59.65 | 10.01 | 17.98 | 24.78 |
| qwen3.5-397b-a17b-think_sierra_2026-03-02 | telecom | 4 | 97.81 | 95.76 | 93.86 | 92.11 | 2.05 | 3.95 | 5.70 |
| qwen3.5-397b-a17b-think_sierra_2026-03-02 | banking_knowledge | 4 | 9.79 | 6.36 | 5.41 | 5.15 | 3.43 | 4.38 | 4.64 |
| raft-30b-a3b_neu_2026-04-29 | retail |  | 82.45614035087719 | 72.95321637426902 | 66.22807017543859 | 61.40350877192983 | 9.50 | 16.23 | 21.05 |
