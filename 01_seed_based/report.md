# Project 01 Report: Seed-based Graph Matching with Greedy Percolation

Code and full outputs: `solution.ipynb`. Frozen settings: `config.json`. Per-instance metrics: `results/train_results.csv`, `results/val_results.csv`. Test predictions: `predictions/<instance>/predictions.csv` (18 files).

## Method
Own implementation (numpy/scipy only). `M[u, v]` counts matched neighbour pairs supporting `(u, v)`, initialised from the seeds with one sparse product `A @ P @ B.T`. Greedy loop: match the unmatched pair with the most votes (at least `r`; ties broken by degree similarity), then add a vote to every neighbour pair of the new match. Leftover nodes get a Hungarian assignment (`scipy.optimize.linear_sum_assignment`) keeping pairs with at least `r_final` votes. Optional margin rule and degree normalisation were tried and rejected.

Protocol: configuration grid (r in 1-3, margin 0/1, normalisation none/deg, r_final) scored on Train by mean unseeded F1 at 1% and 5% seeds (random and degree); confirmed once on Val; frozen before Test.

**Frozen config:** `r=1, margin=0, norm=none, r_final=1`. Test predictions use 5% random seeds (assumption; seeds are included in the submitted mapping).

## Validation results (9 Val instances, mean over instances)
| seeds | precision | recall | F1 | coverage | unseeded recall | runtime (s) |
|---|---|---|---|---|---|---|
| random 1% | 0.982 | 0.982 | 0.982 | 1.0 | 0.982 | 0.25 |
| random 5% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 0.25 |
| random 10% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 0.22 |
| degree 1% | 0.993 | 0.993 | 0.993 | 1.0 | 0.993 | 0.24 |
| degree 5% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 0.24 |
| degree 10% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 0.21 |

Baselines at 5% random seeds (unseeded recall): seeds only 0.000, degree-rank 0.016 (Val), percolation 1.000. At 5%, every family and difficulty is at or above 0.998 unseeded recall on Val. Mean unseeded F1 of the chosen config: Train 0.966 (1% and 5% mix), Val 0.994.

## Analysis
- **Seeds:** only 1% seeds (3-10 pairs) cause failures. Percolation may never ignite; worst case is random seeds on preferential-attachment graphs (Train+Val unseeded recall 0.84 vs 0.995 with degree seeds). Degree seeds help at 1%; no difference at 5%+.
- **Family/difficulty:** dense graphs (mean degree around 60) keep enough shared neighbours even at 26% edge replacement, so noise level barely matters once percolation starts. Stochastic-block graphs at 1% seeds are the other fragile case.
- **Tuning:** `r=1` beat `r=2,3`; margin and degree normalisation lowered F1 (my explanation, not tested separately: ties at 1-2 votes early on, and hub votes carry most of the signal). The leftover stage did not change results.
- **Confidence:** accuracy rises with the vote count at match time (about 0.71-0.75 at 1-2 votes, 0.93 at 3, at least 0.99 above 4; Val, 1% seeds). A vote threshold is a usable rejection rule, trading coverage for precision.
- **Runtime:** about 0.02 s (n=300), 0.1 s (n=600), 0.5 s (n=1000).

## Limitations
- Val is near ceiling, so only the 1% regime separates variants; Train is the only tuning set.
- Assumes same node set and edge-perturbation noise only, and fairly dense graphs; sparser or noisier graphs would make votes rarer.
- Wrong early matches propagate; only greedy ordering mitigates this.
- Dense O(n^2) score matrix; fine up to n around 1000.
- Test seed strategy/level assumed (5% random); test accuracy cannot be checked locally.
