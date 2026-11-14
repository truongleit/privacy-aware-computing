# Project 01: Seed-based Graph Matching

**Goal:** Recover the one-to-one correspondence between anonymized graph A and auxiliary graph B.

## Data

- 27 train, 9 validation, 18 test graph pairs (independent graph instances across splits).
- Structures: random, community, preferential-attachment.
- Sizes: 300 / 600 / 1000 nodes.
- Current teaching version: **v4**.
- Train/validation have `ground_truth.csv`; test truth is hidden.

## Files

Located at `student/seed_based/SPLIT/INSTANCE/`:

- `nodes_a.csv`, `nodes_b.csv`
- `edges_a.csv`, `edges_b.csv`
- `metadata.json`
- Supplied training/validation truth (`ground_truth.csv`)

## Seeds

- Use known seeds as fixed correspondences.
- Compare **random** vs **high-degree** seed selection at **1%, 5%, 10%**.
- Primary comparison uses **5%**.
- Report recall on **unseeded** nodes as well.

## Required Work

- Implement structural features and matching.
- Tune only on Train/Val; freeze the method before Test.
- Analyze graph family, difficulty, matching confidence and runtime.

**Suggested open directions:** neighborhood fingerprints, iterative matching, graph embeddings, assignment objectives, rejection thresholds, robustness.

## Metrics

- Precision, recall, F1, coverage.
- Aggregate across graph instances rather than pooling all nodes.
- Also report unseeded recall. Submitting only known seeds is insufficient.

## Submission

- `predictions.csv` for each Test instance, columns `node_a,node_b`.
- No duplicate or invalid nodes; uncertain nodes may be omitted.
- Include source code, frozen settings, and a report with validation results and limitations.

## Environment

Run from this project folder:

```bash
conda env create -f environment.yml
conda activate privacy_projects
```

## Rules

- Implement your own matching algorithm and validation metrics.
- Test predictions are scored by the instructor using hidden ground truth.
- **Never** search for hidden mappings or teacher generators.
