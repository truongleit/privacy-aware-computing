# Guide: Seed-based Graph Matching with Seeded Graph Matching (Jupyter notebook)

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
| `01_seed_based/solution.ipynb` | The single notebook with all steps below: main method first, comparison methods at the end. |
| `01_seed_based/config.json` | The frozen SGM hyperparameters (initialisation, max iterations). Written once, before test. |
| `01_seed_based/results/{train,val}_results.csv` | Per-instance metrics of the main method. |
| `01_seed_based/results/comparison_results.csv` | Per-instance metrics of every method (column `method`). |
| `01_seed_based/predictions/<instance>/predictions.csv` | Test output, columns `node_a,node_b`. Check the instructor's required submission layout before final export. |
| `01_seed_based/report.md` | Validation results and limitations. |

---

## 2. The idea in one paragraph

B is A with some edges deleted and replaced by random ones, and the node labels are shuffled. The true mapping is the one under which A's and B's edges agree the most: if u maps to v and x maps to y, then (u, x) is an edge in A about as often as (v, y) is an edge in B. Seeded graph matching (SGM) searches directly for the permutation `P` that maximises the number of agreeing edges, `f(P) = trace(A P B P^T)`, with the seed pairs fixed. Searching over permutations is hard, so the unknown part is relaxed to a doubly-stochastic matrix and optimised with Frank-Wolfe, then rounded back to a permutation. Unlike neighbour-vote methods (percolation), every edge contributes from the first iteration, so very few seeds are enough.

---

## 3. Step-by-step

### Step 0: Setup
- Imports: `numpy`, `pandas`, `scipy.sparse`, `scipy.optimize.linear_sum_assignment`, `joblib`, `time`, `json`, `pathlib`, `matplotlib`.
- Set a `DATA = Path("student/seed_based")` constant.
- Why: all later steps use the same path and the same libraries, so keep them in one cell.

### Step 1: `load_instance`
Signature idea: `load_instance(split, name, seed_kind="random", seed_pct=5)`. Returns `A, B, seeds, truth, meta`.
- Read `nodes_a.csv` for `n`.
- Read edges with `pd.read_csv`. Build `scipy.sparse.coo_matrix((np.ones(m), (src, dst)), shape=(n, n))`, then `A = A + A.T`, convert to `csr_matrix`, drop the diagonal and set `A.data[:] = 1` (removes duplicates).
- Seeds: `pd.read_csv(f"seeds_{kind}_{pct:02d}.csv")`. Truth: `ground_truth.csv` if it exists, else `None`. `json.load` for metadata.
- Why: SGM is matrix arithmetic. A symmetric 0/1 sparse matrix is the cleanest form.

### Step 2: Metrics (write before the matcher)
`evaluate(pred, truth, seeds, n)` where `pred` is a dict `{a: b}`.
- precision = correct / len(pred); recall = correct / n; F1; coverage = len(pred) / n.
- Unseeded versions: drop seed nodes from `pred` and `truth`, recompute. This is the number that matters, because seeds are free.
- Aggregate one row per instance, then average over instances (`groupby`). Never pool all nodes.

### Step 3: The SGM objective and gradient
- Split nodes into seeded (`sa`, `sb`) and unseeded (`ua`, `ub`). Let `S0` be the n x n seed indicator and `X` the m x m unseeded block (m = n - #seeds); the full `P~` is `S0` with `X` written into rows `ua`, columns `ub`.
- Gradient: `G = A @ P~ @ B` (A, B symmetric). Compute it as two sparse-times-dense products: `A @ (B @ P~.T).T`. Only the block `G[ua][:, ub]` matters.
- Why: `f` is quadratic in `P`; its gradient is the "vote matrix" under the current soft mapping, so it generalises the neighbour votes of percolation to all nodes at once.

### Step 4: Frank-Wolfe loop
```
X = uniform m x m (barycentre)                      # or the LAP of the seed-only gradient, or a 50/50 mix
for it in range(max_iter):
    Gb = gradient(embed(X))[ua][:, ub]
    Q  = permutation from linear_sum_assignment(-Gb)  # best vertex of the polytope
    D  = Q - X
    gap = sum(Gb * D);  stop if gap <= tol
    c = 2 * gap;  d = sum(gradient(embed0(D))[ua][:, ub] * D)   # f along the segment: f(X) + c a + d a^2
    alpha = clip(-c / (2 d), 0, 1) if d < 0 else (1 if c + d > 0 else 0)
    X = X + alpha * D
final mapping = linear_sum_assignment(-X)
```
- Why exact line search: the objective is a quadratic along the segment, so the best step has a closed form and no learning rate is needed.
- Check: the objective `sum(G * P~)` must not decrease between iterations.
- Confidence of a match (u, v): `G[u, v]` under the final mapping = number of neighbour pairs that agree.

### Step 5: Tuning on Train, confirm on Val, freeze
- Grid: initialisation in {bary, vote, mix} x `max_iter` in {10, 30}. Score by mean unseeded F1 over Train instances at 1% and 5% seeds (random and degree), so the choice must work in the low-seed regime.
- Confirm once on Val, then write the chosen parameters to `config.json` and do not change them afterwards.
- Why tune the start: the problem is non-convex, so the start point can decide which local optimum Frank-Wolfe reaches.

### Step 6: Experiments with the frozen config
- Loop: split in {train, val}, instance, seed kind in {random, degree}, pct in {1, 5, 10}; run in parallel with `joblib`.
- Record precision, recall, F1, coverage, unseeded recall, iterations, runtime, family, difficulty, n.
- Tables: 5% headline, by family and difficulty, by seed level. Figures: recall vs seed level, iterations, runtime vs n.
- Optimum check on Val: compare `f` of SGM's mapping with `f` of the true mapping. Lower means a local optimum; equal or higher means the objective itself is the limit.
- Confidence: precision and coverage when keeping only matches with agreement score >= tau.

### Step 7: Freeze and predict on test
- Read `config.json`, run SGM on each test instance with the 5% seed file, write `predictions.csv` with header `node_a,node_b`.
- Validation: no duplicates in either column, all ids in `[0, n)`, seeds present.
- Never try to infer hidden test mappings.

### Step 8: Comparison with other methods (end of the notebook)
Run each alternative on the same Train+Val grid, with configurations fixed in advance, and save all per-instance rows to `results/comparison_results.csv`:
- **Greedy percolation**: `M = A @ P_seeds @ B.T` counts matched-neighbour votes; repeatedly match the unmatched pair with the most votes (>= r), then add one vote to every pair in `N_A(u) x N_B(v)`; a final assignment on leftovers.
- **Iterative global**: in rounds, one global assignment on the vote matrix, lock in only the most confident pairs, add them as anchors, repeat.
- **One-shot seed-sim**: describe each node by which seeds it touches; one assignment on that similarity.
- **Baselines**: degree-rank matching and seeds only.
- Show recall by seed level, per-run SGM vs percolation at 1% seeds, runtime vs n.

### Step 9: Report (`report.md`)
Method, frozen settings, validation tables, the comparison, figures, runtime, and limitations.

---

## 4. What to expect (for analysis)
- SGM uses every edge from the start, so it should work even at 1% seeds (3 to 10 pairs), where neighbour-vote methods may never ignite.
- Fewer seeds mostly cost Frank-Wolfe iterations (more from a uniform start), not accuracy, on these graphs.
- Percolation-style methods are hurt most by random seeds on preferential-attachment graphs (few hubs among the seeds); degree seeds help them.
- One-shot methods that only use adjacency to the given seeds cannot separate nodes that touch no seed; expect them far behind.
- Cost: dense n x n matrices and one O(n^3) assignment per iteration; fine for n <= 1000.

---

## 5. Suggested order of work
1. Steps 0 to 2.
2. Steps 3 and 4 on one easy instance (check the objective increases), then on all train instances.
3. Step 5 tuning and freeze.
4. Step 6 experiments, Step 7 test predictions.
5. Step 8 comparison, then Step 9 report.
