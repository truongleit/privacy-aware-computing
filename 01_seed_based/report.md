# Project 01 Report: Seed-based Graph Matching with Seeded Graph Matching (SGM)

Code and full outputs: `solution.ipynb`. Frozen settings: `config.json`. Per-instance metrics: `results/train_results.csv`, `results/val_results.csv` (SGM), `results/comparison_results.csv` (all methods). Figures: `results/figs/`. Test predictions: `predictions/<instance>/predictions.csv` (18 files).

## Method
Own implementation (numpy/scipy only). Seeded graph matching maximises the edge agreement `f(P) = trace(A P B P^T)` over permutations `P` that contain the seed pairs. The unseeded block is relaxed to a doubly-stochastic matrix and optimised with Frank-Wolfe: gradient `A P B` (two sparse-times-dense products), direction from one `scipy.optimize.linear_sum_assignment`, exact line search (the objective is quadratic along the segment), stop when the Frank-Wolfe gap is below 1. The result is projected back to a permutation with one more assignment. Each match's confidence is its number of agreeing neighbour pairs under the final mapping.

Protocol: grid over initialisation (barycentre, seed-vote permutation, 50/50 mix) x maximum iterations (10, 30), scored on Train by mean unseeded F1 at 1% and 5% seeds (random and degree); confirmed once on Val; frozen before Test.

**Frozen config:** `init=bary, max_iter=30`. Test predictions use 5% random seeds (assumption; seeds are included in the submitted mapping).

## Validation results (9 Val instances, mean over instances)
| seeds | precision | recall | F1 | coverage | unseeded recall | iterations | runtime (s) |
|---|---|---|---|---|---|---|---|
| random 1% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 7.3 | 0.39 |
| random 5% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 3.7 | 0.18 |
| random 10% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 3.3 | 0.13 |
| degree 1% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 6.1 | 0.30 |
| degree 5% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 3.3 | 0.14 |
| degree 10% | 1.000 | 1.000 | 1.000 | 1.0 | 1.000 | 3.0 | 0.12 |

The same holds on Train: unseeded recall 1.000 on all 216 Train+Val runs, for every family, difficulty, seed strategy and seed level. Mean unseeded F1 of the chosen config: Train 1.000, Val 1.000.

## Comparison with other methods (Train+Val, mean unseeded recall)
All methods run on the same 216 runs; the alternatives use configurations tuned earlier on Train with the same protocol.

| seeds | SGM | greedy percolation | iterative global | one-shot seed-sim |
|---|---|---|---|---|
| 1% random | 1.000 | 0.913 | 0.848 | 0.044 |
| 1% degree | 1.000 | 0.979 | 0.916 | 0.073 |
| 5% random / degree | 1.000 / 1.000 | 1.000 / 1.000 | 0.999 / 1.000 | 0.365 / 0.566 |
| 10% random / degree | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.681 / 0.879 |

- **Greedy percolation** (match the pair with most matched-neighbour votes, add its votes, repeat; `r=1`, leftover assignment): ties with SGM at 5%+, but at 1% seeds it can fail to ignite; SGM is better on 39% of 1%-seed runs and never worse. Worst case is preferential attachment with 1% random seeds (0.84).
- **Iterative global** (rounds of global assignment, locking in the best 20% of pairs with >= 2 votes): close at 5%+, weakest of the iterative methods at 1% (0.75 on preferential attachment) because early wrong lock-ins feed later rounds.
- **One-shot seed-sim** (similarity of seed-adjacency vectors, one assignment): nodes touching no seed are indistinguishable, so it stays far behind at every level.
- **Baselines** at 5% random seeds: degree rank 0.018, seeds only 0.000.
- **Runtime** at n=1000: SGM 0.54 s, percolation 0.58 s, iterative global 0.21 s, seed-sim 0.03 s.

## Analysis
- **Seeds:** SGM is insensitive to seed level and strategy on these graphs; fewer seeds only cost more iterations (about 7 at 1% vs 3 at 5-10%). The edge structure itself carries the mapping, and seeds only break symmetries.
- **Family/difficulty:** no differences; all at 1.000. The other methods show that preferential attachment with few random seeds is the hardest case for neighbour-vote methods.
- **Optimiser vs objective:** on every Val instance (1% and 5% random seeds) the edge agreement of SGM's mapping equals that of the true mapping, so the optimiser reaches the optimum and the optimum is the truth.
- **Tuning:** barycentre and mix starts with 30 iterations both reach Train F1 1.000; a 10-iteration cap (0.97-0.99) and the pure vote start (0.98) are slightly worse.
- **Confidence:** keeping only matches with agreement score >= 8 keeps 80% of nodes at precision 1.000; since every match is already correct, no rejection threshold is used.

## Limitations
- Train/Val are at ceiling, so they cannot distinguish SGM from a perfect method; harder regimes (sparser, noisier graphs, under 1% seeds) were not tested.
- Non-convex objective and a local search: global optimum found on all 54 graph pairs here, not guaranteed elsewhere.
- Dense O(n^2) memory and a cubic assignment per iteration; fine up to n around 1000.
- Assumes same node set and edge-perturbation noise only.
- Test seed strategy/level assumed (5% random); Test accuracy cannot be checked locally.
- Privacy implication: a few known correspondences plus graph structure are enough to de-anonymize these graphs completely.
