# Guide: Seed-based Graph Matching with Greedy Percolation (Jupyter notebook)

Working mode: you write the code in a notebook; this guide gives the files, the functions to use, and the reasoning for each step. No full solution is included.

---

## 1. Files you will interact with

Project root: `01_seed_based/`

| Path | What it is | How you use it |
|---|---|---|
| `README.md`, `PROJECT_INSTRUCTIONS.txt` | Task rules | Read only. Rules: own implementation, tune on train/val only, freeze before test, never look for hidden mappings. |
| `environment.yml` | Conda env `privacy_projects` (python 3.11, numpy, scipy, pandas, sklearn, networkx) | `conda env create -f environment.yml`, then select this kernel in the notebook. |
| `student/seed_based/{train,val,test}/<instance>/` | One folder per graph pair. 27 train, 9 val, 18 test. Instance name = `<family>_<difficulty>_<idx>`, e.g. `erdos_renyi_hard_00` | Loop over these with `os.listdir` / `pathlib.Path.glob`. |
| `nodes_a.csv`, `nodes_b.csv` | Column `node_id` (0..n-1) | Gives n. |
| `edges_a.csv`, `edges_b.csv` | Columns `source,target`. Undirected, each edge listed once (a 1000-node, p=0.06 graph has ~29.8k rows) | Build the adjacency matrices. Symmetrize. |
| `ground_truth.csv` | Columns `node_a,node_b`. Only in train/val | Used only for evaluation and tuning. |
| `seeds_{random,degree}_{01,05,10}.csv` | Columns `node_a,node_b`; 1%, 5%, 10% of n known pairs | The starting matches. "random" picks seeds uniformly, "degree" picks high-degree nodes. |
| `metadata.json` | `family`, `difficulty`, `n_a`, `n_b`, edge retention, noise fraction, split | Use `family` and `difficulty` to group results. Do not use noise parameters as features at test time. |

Check first: the test folders have no `ground_truth.csv`; confirm they contain the seed files you need.

Files you create (suggested):

| Path | Purpose |
|---|---|
| `01_seed_based/solution.ipynb` | The notebook with all steps below. |
| `01_seed_based/config.json` | The frozen hyperparameters (r, margin, normalisation choice, ...). Written once, before test. |
| `01_seed_based/results/val_results.csv` | Per-instance metrics table. |
| `01_seed_based/predictions/<instance>/predictions.csv` | Test output, columns `node_a,node_b`. Check the instructor's required submission layout before final export. |
| `01_seed_based/report.md` | Validation results and limitations. |

---

## 2. The idea in one paragraph

B is A with some edges deleted and replaced by random ones, and the node labels are shuffled. If u in A maps to v in B, then most neighbours of u map to neighbours of v. Given a known seed pair (a, b), every neighbour pair (x in N(a), y in N(b)) gets one "vote" that x matches y. A pair that collects at least r votes is a very likely match, because random pairs rarely share r matched neighbours. Matching that pair creates new votes for its neighbours, and the process spreads (percolates). Greedy version: always match the highest-scoring pair next.

---

## 3. Step-by-step

### Step 0: Setup
- Imports: `numpy`, `pandas`, `scipy.sparse`, `time`, `json`, `pathlib`, `matplotlib`.
- Set a `DATA = Path("student/seed_based")` constant.
- Why: all later steps use the same path and the same libraries, so keep them in one cell.

### Step 1: `load_instance`
Signature idea: `load_instance(split, name, seed_kind="random", seed_pct=5)`. Returns `A, B, seeds, truth, meta`.
- Read `nodes_a.csv` for `n`.
- Read edges with `pd.read_csv`. Build `scipy.sparse.coo_matrix((np.ones(m), (src, dst)), shape=(n, n))`, then `A = A + A.T`, convert to `csr_matrix`, and set `A.data[:] = 1` (this removes duplicates if an edge appears in both directions).
- Seeds: `pd.read_csv(f"seeds_{kind}_{pct:02d}.csv")`.
- Truth: `ground_truth.csv` if it exists, else `None`.
- `json.load` for metadata.
- Why: percolation is matrix arithmetic. A symmetric 0/1 sparse matrix is the cleanest form.
- Check: `A.shape == (n, n)`, `(A != A.T).nnz == 0`, `A.diagonal().sum() == 0`, `A.nnz // 2` close to the number of edge rows.

### Step 2: Metrics (write before the matcher)
`evaluate(pred, truth, seeds, n)` where `pred` is a dict `{a: b}`.
- correct = pairs where `truth[a] == b`.
- precision = correct / len(pred); recall = correct / n; F1 = harmonic mean; coverage = len(pred) / n.
- Unseeded versions: drop the seed nodes from `pred` and `truth`, then recompute recall (and precision). This is the number that matters, because seeds are free.
- Aggregate: compute one row per instance, build a DataFrame with `family`, `difficulty`, `n`, then `df.groupby(["family", "difficulty"]).mean(numeric_only=True)`. Never pool all nodes.
- Check: `evaluate(truth_dict, truth, seeds, n)` gives 1.0 everywhere. `evaluate(seeds_dict, ...)` gives unseeded recall 0.
- Why first: every later design choice is judged by these numbers.

### Step 3: Baselines (give you a reference)
- **Seeds only**: unseeded recall 0 (the README says this is insufficient).
- **Degree-rank match**: sort nodes in A and B by degree and pair them by rank. Expect low accuracy; it is your floor.
- Why: shows the percolation gain is real.

### Step 4: The score matrix
- Build the seed indicator matrix `P` (n x n sparse): `P[a, b] = 1` for each matched pair.
- `M = A @ P @ B.T`, so `M[u, v]` = number of matched pairs (x, y) with x a neighbour of u in A and y a neighbour of v in B. Convert to dense with `.toarray()` (n <= 1000, so at most 8 MB at float64).
- Keep `used_a`, `used_b` boolean arrays, initially True for seed nodes.
- Why: this is the vote count from section 2. One sparse product replaces nested loops.
- Check: for one seed pair, `M[x, y] >= 1` for neighbours x of a and y of b.

### Step 5: The greedy percolation loop
```
while True:
    candidates = M with rows/cols of used nodes masked to -inf
    (u, v) = argmax of candidates
    if candidates[u, v] < r: break
    match u -> v; mark used
    M[np.ix_(neighbors_A(u), neighbors_B(v))] += 1
```
- Neighbours: `A.indices[A.indptr[u]:A.indptr[u+1]]` for CSR.
- Cost: n argmax calls over n x n is ~n^3 = 10^9 for n=1000; fine with numpy if you avoid rebuilding the mask each time. Hint: instead of masking a copy, set `M[u, :] = -inf` and `M[:, v] = -inf` when you match (keep a separate counts matrix if you need both).
- r (threshold) is the main hyperparameter. Start with r = 2.
- Why r >= 2: with r = 1 a single noisy edge creates a wrong match, and wrong matches spread more wrong votes.
- Output: `pred` dict including seeds. Evaluate with Step 2.

### Step 6: Make it robust to noise (the tuning part)
Try these one at a time and log each with Step 2 metrics on train:
1. **Threshold r** in {2, 3, 4}. Plot precision vs coverage. Low r: more coverage, more errors.
2. **Margin rule**: accept (u, v) only if its score exceeds the second-best score in its row and in its column by at least `delta`. This is your rejection mechanism: skip uncertain nodes rather than guess.
3. **Degree normalisation**: replace the score with `M[u,v] / sqrt(degA[u] * degB[v])` or a similar form. High-degree nodes collect many random votes, which matters for preferential attachment (hubs).
4. **Tie-breaking**: when scores tie, prefer the pair with the closest degrees (`-abs(degA[u] - degB[v])`).
5. **Processing order**: instead of one pair at a time, accept all pairs that are mutually best (row max and column max) in a round, then update. Faster, usually similar accuracy.

### Step 7 (optional): Finish the leftovers
When percolation stalls with unmatched nodes:
- Build a feature per node = its adjacency vector over the matched nodes (columns re-ordered by the mapping). Compute similarity `S` between A-rows and B-rows (cosine via `sklearn.metrics.pairwise.cosine_similarity` or your own dot product).
- Solve `scipy.optimize.linear_sum_assignment(-S)` on the unmatched nodes.
- Keep only pairs whose similarity exceeds a cutoff.
- Why: percolation cannot cross small gaps, but this global assignment can.

### Step 8: Experiments
- Loop: split in {train, val}, instance, seed kind in {random, degree}, pct in {1, 5, 10}.
- For each, record: precision, recall, F1, coverage, unseeded recall, `time.perf_counter()` runtime, family, difficulty, n.
- Headline table: 5% random vs 5% degree, by family and difficulty.
- Figures: (a) unseeded recall vs seed percentage, per family; (b) precision vs coverage as r/margin vary; (c) confidence (score or margin) vs correctness for matches; (d) runtime vs n.
- Tune on train, confirm on val. Do not re-tune on val results beyond confirming; if val is much worse, note that as overfitting in the report.

### Step 9: Freeze and predict on test
- Write the chosen parameters to `config.json` and do not change them afterwards.
- For each test instance, load graphs and the 5% seed file (use the seed level the instructor specifies; check the test folders), run the frozen method, write `predictions.csv` with header `node_a,node_b`.
- Validation script: no duplicates in either column, all ids in `[0, n)`, seeds present, no pair outside the node sets.
- Never try to infer hidden test mappings.

### Step 10: Report (`report.md`)
Method description, frozen settings, validation tables (by family, difficulty, seed type and %), the figures above, runtime, and limitations.

---

## 4. What to expect (for analysis)
- Percolation has a seed threshold. At 1% seeds (3 to 10 pairs) the process may not take off and recall stays low; at 5 to 10% it should cascade.
- Degree seeds usually ignite the process more easily than random seeds, since hubs have many neighbours to vote for.
- Hard instances have ~26% edge replacement, so true pairs share fewer neighbours; you need a lower r or a better normalisation, at a cost in precision.
- SBM: wrong matches tend to stay inside the correct community. Erdos-Renyi has no structure beyond degrees, so it depends most on seeds.
- Mistakes compound: a wrong early match gives wrong votes. High precision early matters more than coverage.

---

## 5. Suggested order of work
1. Steps 0 to 2, verify with the sanity checks.
2. Step 3 baselines.
3. Steps 4 and 5 on one easy instance, then on all train instances.
4. Step 6 tuning, one change at a time.
5. Step 8 full experiments, then Step 9 and 10.
