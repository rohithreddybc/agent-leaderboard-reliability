# sepbound

sepbound checks, from published leaderboard numbers alone, whether two entries can be told apart.
Give it scores, the task count `n` and a task-level correlation `rho`, and it returns the item-sampling bound on a score difference, a separable / below-the-bound verdict, and the order confidence of a pair.
For a whole board it also gives the Bonferroni leader set, the flat-prior probability that the observed leader is truly first, and the greedy tier count.
It also has the rerun bound, for fixed tasks and independent reruns.
The definitions are those of the paper cited below; the package depends on numpy only (scipy is used if installed).

## Install from source

```
cd sepbound
pip install .          # or: pip install -e ".[test]"
```

Python 3.9 or later. This release is local; it has not been uploaded to PyPI.

## Usage

Scores are proportions in the Python API (`sepbound.to_proportions` converts percentages) and bounds are returned as proportions.

```python
import sepbound

# 1. Are 72.4% and 71.8% on 500 tasks separable? (rho = 0.5: positively correlated per-task outcomes)
b = sepbound.item_sampling_bound(0.724, 0.718, 500, rho=0.5)   # 0.0393 = 3.93 pp
sepbound.separable(0.724, 0.718, 500, rho=0.5)                  # False: a 0.6 pp gap is below the bound
sepbound.order_confidence(0.724, 0.718, 500, rho=0.5)           # about 0.62

# 2. A whole board: leader set, leader probability, tiers
names  = ["A", "B", "C", "D", "E"]
scores = [0.50, 0.45, 0.40, 0.35, 0.30]
rep = sepbound.check_leaderboard(names, scores, n=100, rho=0.5)
print(rep["text"])                       # printable report; rep["tiers"], rep["leader_set"], ...
sepbound.leader_set(scores, 100, 0.5)    # [0, 1, 2]: indices not separated from the leader (Bonferroni)

# 3. Fixed tasks, independent reruns: rerun bound with a run-noise share f (here the paper's transported 0.369)
sepbound.rerun_bound(0.5, 0.5, 100, f=0.369, k=1)
```

Command line (scores in percent, or as proportions; percent is assumed if any value exceeds 1):

```
python -m sepbound --n 500 --rho 0.5 72.4 71.8 70.1 69.0 66.2
```

The output gives the gap and bound for #1 vs #2 and #1 vs #5, the Bonferroni and uncorrected leader sets, the leader probability and the tier count.
`n` can be a comma-separated list with one value per entry; `--names a,b,c` labels the entries.

API: `item_sampling_bound`, `separable`, `order_confidence`, `rerun_bound`, `leader_posterior`, `leader_set`, `tiers`, `check_leaderboard`, `to_proportions`, `z_bonferroni`.
A pair is separable when its gap is not below the bound and is not zero; equal scores are always below the bound.
Below the bound is not evidence that two entries do not differ, only that the published scores cannot show that they do.

## Rule of thumb

Item-sampling bound for two entries that both score P = 0.5, in percentage points (computed by this package; the gap must exceed it for the pair to be separable at the 95% level).

| n | rho = 0 | rho = 0.5 |
|---|---|---|
| 50 | 19.6 | 13.9 |
| 100 | 13.9 | 9.8 |
| 200 | 9.8 | 6.9 |
| 500 | 6.2 | 4.4 |
| 1000 | 4.4 | 3.1 |

At rho = 0.5 and equal scores the bound equals the standard error of one score, 1.96 x sqrt(P(1-P)/n). Detecting a true difference with 80% power needs about 1.43 times the bound.

## Assumptions

- Tasks are exchangeable with the published suite, so a score is a mean over tasks drawn from one population.
- Tasks are independent draws. Clustered tasks (templates, repositories) make the binomial standard error too small.
- The standard error of a score is binomial, sqrt(P(1-P)/n), computed at the published score.
- The difference of two scores is normal with the paired standard error SE_D = sqrt(SE_1^2 + SE_2^2 - 2 rho SE_1 SE_2); `rho` is the correlation between the two entries' per-task outcomes and must come from per-task data or an assumption (the paper's pooled median is 0.52).
- The leader probability is a Monte Carlo estimate under a flat prior with the equicorrelated covariance; the leader set covers the best entry with probability of at least 97.5% only under these assumptions with SEs and `rho` treated as known.
- The rerun bound needs a run-noise share `f` that leaderboards do not publish.

## Cite

Bellibatlu, R. R. and Singh, M., "LLM Agent Leaderboards Often Cannot Separate Their Top Two Entries", 2026. DOI: [DOI to be added]

## License

MIT.
