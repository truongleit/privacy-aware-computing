# Project 02 Report: Seed-free Graph Matching

Code and full outputs: `solution.ipynb` (all algorithms are defined in its first cells). Frozen settings: `config.json`. Per-instance metrics: `results/train_results.csv`, `results/val_results.csv`, `results/ablation_val.csv`. Test predictions: `predictions/<instance>/predictions.csv` (18 files). Figures: `results/figs/`.

## Data
54 instances (27 Train / 9 Val / 18 Test), families erdos_renyi / stochastic_block / preferential_attachment, n = 300/600/1000. The node set is shared and A is unmodified; B has 0.05% / 0.3% / 0.55% of its edges deleted and replaced (easy / medium / hard). So B is nearly an exact relabelling of A, and the supplied splits are close to ceiling for good methods. To separate methods I also run a **robustness stress test** (my own addition): extra random edge replacement on B (2-30%), ground truth unchanged, deterministic, built from the same Train/Val graphs.

## Methods (own numpy/scipy implementations)
All start from structural fingerprints per node: degree, triangle count, and the sorted neighbour degrees / neighbour triangle counts. The dissimilarity is the block-wise L1 distance, each block rescaled by its mean magnitude.

1. **Main: bootstrap percolation.** (a) Seeds = mutual-best fingerprint pairs with a margin over the runner-up, top-k by margin. (b) Greedy percolation: repeatedly match the pair with most matched-neighbour votes (at least `r`; ties broken by fingerprint distance), then add a vote to every neighbour pair. (c) Re-seed once from matches with at least `reseed_votes` votes and rerun. (d) Seed-set search: run with several k (1%, 3%, 10% of n) and keep the run preserving the most A-edges (label-free). (e) Reject matches with fewer than `tau` consistent neighbour votes.
2. **Fingerprint + Hungarian** (baseline): linear assignment on the fingerprint dissimilarity.
3. **IsoRank:** `R <- a (A D^-1) R (B D^-1)^T + (1-a) E`, E = fingerprint-similarity prior, Hungarian on R.
4. **FAQ:** Frank-Wolfe on the doubly-stochastic relaxation of `max tr(A P B P^T)`, initialised from a blend of the uniform matrix and the fingerprint matching, exact line search, final Hungarian projection.

The consistency score (matched neighbour pairs of a predicted pair) is method-independent and is the confidence used for rejection.

**Protocol:** configuration grids per method tuned on Train only by mean F1 at extra noise {0, 5%, 10%} (real-data F1 is saturated, so the stressed levels discriminate), ties broken by runtime; confirmed once on Val; frozen in `config.json` before Test.

**Frozen config (main):** `k_fracs=(0.01, 0.03, 0.1), r=2, reseed_votes=4, tau=3`. Others: IsoRank `alpha=0.8, prior_temp=0.3`; FAQ `init_lambda=0.5, iters=15`.

## Validation results (9 Val instances, mean over instances, real data)
| method | precision | recall | F1 | coverage | runtime (s) |
|---|---|---|---|---|---|
| bootstrap percolation (main) | 1.000 | 1.000 | 1.000 | 1.0 | 0.71 |
| FAQ | 1.000 | 1.000 | 1.000 | 1.0 | 0.10 |
| fingerprint + Hungarian | 0.923 | 0.923 | 0.923 | 1.0 | 0.05 |
| IsoRank | 0.927 | 0.927 | 0.927 | 1.0 | 0.47 |

Metrics are computed per instance and averaged, not pooled. Fingerprint and IsoRank lose accuracy only on the hard (0.55% noise) instances: Train+Val mean F1 by difficulty is 1.000 / 0.965 / 0.816 (fingerprint, easy / medium / hard) against 1.0 everywhere for the two leaders.

**Val F1 under extra noise on B (stress test)**
| extra noise | 0 | 2% | 5% | 10% | 15% | 20% | 30% |
|---|---|---|---|---|---|---|---|
| bootstrap (main) | 1.000 | 0.999 | 0.994 | 0.728 | 0.014 | 0.015 | 0.001 |
| FAQ | 1.000 | 1.000 | 1.000 | 0.891 | 0.464 | 0.343 | 0.139 |
| fingerprint | 0.923 | 0.569 | 0.342 | 0.188 | 0.106 | 0.064 | 0.026 |
| IsoRank | 0.927 | 0.590 | 0.364 | 0.183 | 0.088 | 0.041 | 0.017 |

Train shows the same ordering (bootstrap 0.987 at 5% and 0.763 at 10%; FAQ 1.000 and 0.892). Chosen configs were ahead of the other configs mainly at 5-10% noise.

## Analysis
- **Real data:** bootstrap percolation and FAQ are both at ceiling on Train and Val, so the supplied splits cannot rank them. Seed-free, local features alone (fingerprint, IsoRank with a fingerprint prior) are not enough on hard instances. IsoRank adds almost nothing over its prior (0.927 vs 0.923).
- **Robustness:** FAQ degrades most gracefully. The main method is fine to about 5% extra noise and then collapses, because fingerprints stop producing correct seeds (fingerprint F1 is 0.34 at 5%) and wrong seeds poison the percolation. The seed source, not percolation itself, is the weak point: see the ablation.
- **Family:** at 5% extra noise (Train+Val) bootstrap F1 is 0.994 (erdos_renyi), 0.999 (preferential_attachment), 0.974 (stochastic_block); at 10% it is 0.70 / 0.98 / 0.59. Preferential-attachment graphs are easiest (distinctive hubs); stochastic-block graphs are the most fragile for the main method. Erdos-Renyi is hardest for every method above 10% noise.
- **Confidence (Val, main method, extra noise 5%):** accuracy is 0.84 at 3-4 votes, 0.985 at 5-8, 0.994 at 9-16 and 0.999 above 16; pairs with 2 or fewer votes were all wrong. At 10% noise accuracy is 0.58 at 3-4 votes and 0.88 above 16, so the vote threshold trades coverage for precision but cannot repair bad seeds. `tau=3` was chosen on Train and does not reject anything on the real data.
- **Runtime:** main method about 0.06 s (n=300), 0.4 s (n=600), 1.6 s (n=1000); FAQ 0.01 / 0.06 / 0.24 s; the dense n x n score matrices are the cost.
- **Ablation (exploratory, not the submission):** FAQ-seeded percolation (seeds = FAQ pairs with at least 12 consistent votes, same percolation) reaches Val F1 0.521 / 0.335 / 0.222 at 15 / 20 / 30% noise versus 0.464 / 0.343 / 0.139 for FAQ alone, and 0.014 / 0.015 / 0.001 for the fingerprint-seeded main method. So the main method's weakness is largely the seed source. The gain over plain FAQ is small and concentrated in stochastic-block graphs; Erdos-Renyi at 15% is near zero for every method.

## Test predictions
`predictions/<instance>/predictions.csv`, produced by the frozen main method; 18 files, all pass validity checks (unique, in-range node ids). Coverage is 1.0 on all 18, per-instance runtime at most 1.3 s. Test truth is hidden, so accuracy cannot be checked locally. A label-free sanity check: matched pairs preserve at least 99.4% of A's edge neighbourhoods (mean consistency over mean degree), consistent with the near-perfect Val results.

## Limitations
- The supplied splits are saturated, so tuning and the method ranking rely on my synthetic extra-noise stress test; its noise model (uniform random edge replacement) is an assumption.
- Test is assumed to have the same noise levels as Train/Val (metadata of the test files shows this). Under materially higher noise the main method would fail while FAQ would degrade more gently; FAQ is the safer choice for unknown noise even though the plan designated bootstrap percolation as main.
- Assumes the same node set in A and B and edge-only perturbation; nothing handles missing or added nodes.
- Dense n x n matrices limit scale to roughly n = 1000-2000.
- Single deterministic stress-test seed per instance; no confidence intervals. Ablation settings were chosen on Train but the ablation was run after seeing Val results of the main methods, so treat it as exploratory.
