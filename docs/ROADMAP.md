# ROADMAP — test / evaluation / simulation

Evaluation plan for the Bayesian‑computing PTGS method vs. the elastic‑net baseline. Every
phase reuses the **same** shared harness (`cv` · `score` · `metrics` · `viz`) and the two
pluggable builders — only the **data source** changes.

## Phase 0 — iid mock (DONE)
`simulate.py` (`simulate_dataset`): GReX ~ iid standard normal, sparse causal genes → trait.
Purpose: wiring/smoke only. Both arms + 5‑fold nested CV + seaborn comparison figures run
end‑to‑end (notebooks 02, 03, 04). *Limitation:* genes are independent — unrealistic; shrinkage
methods look similar here because there's no correlation/collinearity to exploit.

## Phase 1 — simulated TWAS GReX  (DONE; `simulate_twas.py`)
A realistic simulator where GReX has **TWAS structure**, so the elastic‑net‑vs‑Bayesian
comparison is meaningful:
- **cis‑genotypes** with AR(1) LD;
- **sparse cis‑eQTL weights** per gene → `GReX_g = standardize(G_cis · w_g)`, with overlapping
  cis‑windows between adjacent genes inducing **gene–gene correlation** — the key feature the
  Phase‑0 iid mock lacks and where shrinkage priors should help;
- optional **prediction noise** `expr_pred_r2 < 1` (predicted vs. true expression gap);
- **trait** from a sparse causal‑gene set (gaussian/binomial), known ground truth returned.
Deliverable: notebook `01_simulate_smoke.ipynb` (despite its name, it already exercises
`simulate_twas_dataset`, not the Phase‑0 iid mock — the notebook grew into this role rather
than a separate `05_twas_simulation` being added). Ancestry/portability structure remains
out of scope for now.

## Phase 1.5 — structured priors that USE the gene-gene correlation  (DONE)
Phase 1 produced correlated GReX but, until now, every prior in `priors.py` placed **iid**
shrinkage on each gene — none of them actually used the correlation structure. Two new priors
close that gap, sharing the existing `Builder`/`hyper`-dict/`nested_cv` machinery unchanged:
- **`graph_horseshoe`**: the regularized horseshoe's latent draw is correlated via a
  gene-gene correlation matrix (`hyper["corr"]`) instead of iid. `corr` can be the closed-form
  oracle (`simulate_twas.true_grex_correlation`) or a Ledoit-Wolf estimate from training data
  (`structure.estimate_gene_correlation`, or `BayesBuilder(corr_estimate=True)` for
  leakage-safe per-fold estimation in `nested_cv`).
- **`group_horseshoe`**: a shared group-level (e.g. pathway) local scale
  (`hyper["groups"]`), so a known-informative gene group is shrunk together;
  `structure.perturb_group` demonstrates sensitivity to a realistically imperfect group.
Deliverable: notebooks `05_graph_horseshoe.ipynb`, `06_group_horseshoe.ipynb`.

**Empirical result (full-scale notebook runs):** `group_horseshoe` clearly wins with a correct
group (0.324 vs 0.293 `regularized_horseshoe` vs 0.277 elastic net, partial-R²) and its edge
nearly vanishes once the group is corrupted — the expected pattern. `graph_horseshoe` is a
**null result**: oracle correlation scored 0.244, estimated 0.251, plain
`regularized_horseshoe` 0.247, elastic net 0.243 — all within noise, and WAIC agrees. Reshaping
the *continuous* shrinkage geometry with correlation didn't help; encoding *which genes* share
strength (`group_horseshoe`) did. This motivated Phase 1.6.

## Phase 1.6 — spike-and-slab, plain and correlation-aware  (DONE)
Motivated directly by Phase 1.5's finding: correlation should matter more for **discrete
selection** (independent inclusion indicators arbitrarily split credit between correlated
genes) than for continuous shrinkage. NUTS can't sample a true discrete indicator, and a
literal Ising/MRF prior on indicators has an intractable normalizing constant — both new priors
use the standard NUTS-native continuous relaxation instead:
- **`spike_slab`**: George & McCulloch-style continuous two-component mixture per gene (near-
  zero spike vs. diffuse slab), marginalizing the discrete indicator
  (`numpyro.distributions.MixtureSameFamily`). Gives a genuine posterior inclusion probability
  per gene (`priors.spike_slab_pip`, in `ScoreBundle.extras["pip"]`) — no horseshoe variant
  provides this.
- **`graph_spike_slab`**: `spike_slab` whose per-gene inclusion PROBABILITY (not the weight) is
  correlated via `hyper["corr"]`, reusing `graph_horseshoe`'s exact Cholesky-correlated-latent
  mechanism applied to the mixing logit instead of the shrinkage scale — a deliberate
  substitution for a true Ising/MRF prior, avoiding its intractability while still pulling
  correlated genes toward joint inclusion/exclusion.
Verified via a full `nested_cv`/`run_benchmark` integration run (not just eager-mode unit
tests — the earlier `graph_horseshoe`/`group_horseshoe` pass found two real bugs, a `.name`
collision and a JAX-tracing crash, that only appeared under actual NUTS execution). No
dedicated notebook yet — these are backend models awaiting evaluation on real GReX data.

**Explicitly out of scope for now:** cross-ancestry hierarchical shrinkage (postponed by the
user until the current correlation/discovery-focused goals are done). Cross-tissue `Dataset`
support is also deferred, to be picked up after the current objective.

## Phase 2 — real local GReX + phenotype  (entry: `io.load_dataset`)
Run on the **local cohort biobank's** GReX + phenotype (controlled/PHI). Develop locally on
Phase‑0/1 mock, then execute in the governed environment — the `cloud-dev-workflow` (code to
GitHub, run in cloud, results/errors back as text; data never moves). Deliverable: a real‑run
notebook + a firmed‑up `io.load_dataset` for the actual file formats.

## Strategy (2026-09-04) — use both arms, select the winner per regime

No single method dominates (demo: elastic net wins p<n; Bayesian wins p≫n sparse+correlated,
with lower variance). So the goal is **not** to prove one arm superior — it is to **run both and
select the winning arm for the expected data regime**, guided by simulation:
1. **Simulation-guided pre-selection** — simulate the *expected* regime (p, n, sparsity, LD,
   heritability via `simulate_twas_dataset`), see which arm wins there, apply that to real data.
2. **Data-driven selection on real data** — `run_benchmark` both arms through the same nested CV
   and take the winner by the outer-fold metric (unbiased). Future: a small `select_best`/stacking
   helper in `benchmark.py`.
The shared harness + pluggable builders already support this directly.

## Open decisions / inputs needed
- **Phase 2 data:** GReX file format (e.g. PrediXcan/predictdb genes×samples output), phenotype
  format, the trait(s) + covariates, approx dimensions (n_genes, n_samples), and where it runs
  (Terra / UKB RAP / local controlled) — to set up the cloud bridge.
- **Multi-tissue `Dataset` support** (deferred until the current correlation/discovery goals
  are done) and **GReX-estimation-uncertainty propagation** (errors-in-variables — GReX is
  currently treated as fixed/observed by every builder and prior in this package, including
  `spike_slab`/`graph_spike_slab`) are identified next steps, not yet started.
