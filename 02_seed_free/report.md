# Project 02 Report: Seed-free Graph Matching with Frank-Wolfe QAP (FAQ)

Code and full outputs: `solution.ipynb`. Frozen settings: `config.json`. Per-instance metrics: `results/train_results.csv`, `results/val_results.csv` (FAQ), `results/comparison_results.csv` (all methods), `results/test_summary.csv`. Figures: `results/figs/`. Test predictions: `predictions/<instance>/predictions.csv` (18 files).

## Data
54 instances (27 Train / 9 Val / 18 Test), families erdos_renyi / stochastic_block / preferential_attachment, n = 300/600/1000. The node set is shared and A is unmodified. B has 0.05% / 0.3% / 0.55% of its edges deleted and replaced (easy / medium / hard), so B is nearly an exact relabelling of A and good methods are at ceiling on the supplied splits. To separate methods I add a **robustness stress test** (my own addition): extra random edge replacement on B (2-30%), with the ground truth unchanged. It is deterministic and built from the same Train/Val graphs.

## Method
Own implementation (numpy/scipy only). FAQ maximises the edge agreement `tr(A P B P^T)` over the doubly-stochastic relaxation with Frank-Wolfe. Each step solves one linear assignment on the gradient `A P B`, takes an exact line search (the objective is quadratic along the segment) and updates; the result is projected to a permutation with a final Hungarian step. With no seeds, the start point carries the information: `P0 = (1 - lam) J/n + lam * Perm`, where `Perm` is the Hungarian matching of structural fingerprints (degree, triangle count, sorted neighbour degrees and sorted neighbour triangle counts; block-wise L1 distance, each block rescaled by its mean magnitude). Confidence of a pair = number of matched neighbour pairs (consistency); pairs below `tau` can be rejected.

**Protocol:** grid over `init_lambda` in {0, 0.5, 0.9} x `iters` in {15, 30}, then `tau` in {0, 1, 2, 3, 5}. Configurations are scored on Train only by mean F1 at extra noise {0, 5%, 10%} (real-data F1 is saturated, so the stressed levels discriminate), with ties broken by runtime. The choice is confirmed once on Val and frozen in `config.json` before Test.

**Frozen config:** `init_lambda=0.5, iters=15, tau=0` (no rejection).

## Results (mean over instances)
**Real data:** precision, recall, F1 and coverage are 1.000 on both Train (27) and Val (9), for every family and difficulty.

**F1 under extra noise on B**
| extra noise | 0 | 2% | 5% | 10% | 15% | 20% | 30% |
|---|---|---|---|---|---|---|---|
| Val | 1.000 | 1.000 | 1.000 | 0.891 | 0.464 | 0.343 | 0.139 |
| Train | 1.000 | 1.000 | 1.000 | 0.892 | 0.494 | 0.344 | 0.204 |

## Analysis
- **Start point:** a uniform start (`init_lambda=0`) fails without seeds (Train F1 0.82 on real data, 0.34 at 5% noise). Any fingerprint blend reaches 1.000 on real data. More iterations and rejection did not help.
- **Family (Train+Val):** preferential attachment stays at 1.000 up to 20% noise (distinctive hubs). Stochastic-block graphs drop after 10% (0.36 at 15%). Erdos-Renyi graphs are hardest (0.68 at 10%, 0.10 at 15%) because nothing beyond degree statistics breaks their symmetries.
- **Confidence (Val, 10% and 15% noise):** pairs with more than 32 consistent votes are always correct. Below that, accuracy does not rise steadily with votes (under heavy noise FAQ can lock onto a consistent but wrong sub-mapping), so a vote threshold is only a coarse rejection rule. `tau=0` was chosen on Train.
- **Runtime:** about 0.01 s (n=300), 0.06 s (n=600), 0.24 s (n=1000).

## Comparison with other methods
All methods run on the same Train+Val instances and noise levels. The alternatives use configurations tuned earlier on Train with the same protocol.

**Val F1 under extra noise**
| method | 0 | 5% | 10% | 15% | 20% | 30% | runtime n=1000 (s) |
|---|---|---|---|---|---|---|---|
| **FAQ (main)** | 1.000 | 1.000 | 0.891 | 0.464 | 0.343 | 0.139 | 0.24 |
| bootstrap percolation | 1.000 | 0.994 | 0.728 | 0.014 | 0.015 | 0.001 | 1.47 |
| FAQ-seeded percolation (ablation) | 1.000 | 1.000 | 0.891 | 0.521 | 0.335 | 0.222 | 0.40 |
| IsoRank | 0.927 | 0.364 | 0.183 | 0.088 | 0.041 | 0.017 | 1.74 |
| fingerprint + Hungarian | 0.923 | 0.342 | 0.188 | 0.106 | 0.064 | 0.026 | 0.10 |

- **Bootstrap percolation** (previous main method: fingerprint mutual-best seeds, greedy percolation, re-seed, `tau=3`) ties FAQ on real data but collapses beyond 5% noise, because fingerprint seeds become wrong and poison the percolation. FAQ is better on 92% of runs at 10% noise and is about 6 times faster.
- **FAQ-seeded percolation** (FAQ pairs with at least 12 votes as seeds, then percolation) matches FAQ up to 10% and is slightly better at 15% and 30% (worse at 20%), mostly on stochastic-block graphs. The gain is small and inconsistent, and its settings were chosen after seeing earlier Val results, so it is reported as an ablation, not the submission.
- **IsoRank and fingerprint + Hungarian** rely on local features. They are already below 0.83 F1 on the real hard instances and fall apart under noise; IsoRank adds almost nothing over its fingerprint prior.

## Test predictions
`predictions/<instance>/predictions.csv`, produced by the frozen FAQ; 18 files, all pass the format checks (unique, in-range node ids, correct columns). Coverage is 1.0 on all 18, and per-instance runtime is at most 0.2 s. Test truth is hidden. A label-free sanity check: matched pairs preserve at least 99.4% of A's edge neighbourhoods. The predictions are identical to those of the previous bootstrap submission on all 18 instances.

## Limitations
- The supplied splits are saturated, so tuning and the method ranking rely on my synthetic stress test. Its noise model (uniform random edge replacement, one deterministic seed per instance, no confidence intervals) is an assumption.
- Non-convex objective and a local search: without seeds, FAQ depends on the fingerprint start, and under heavy noise it settles in wrong optima.
- Assumes the same node set in A and B and edge-only perturbation; nothing handles missing or added nodes.
- Dense n x n matrices and an O(n^3) assignment per iteration limit scale to roughly n = 1000-2000.
- Test is assumed to have the same noise levels as Train/Val (the Test metadata shows this); Test accuracy cannot be checked locally.
