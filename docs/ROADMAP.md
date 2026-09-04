# ROADMAP — test / evaluation / simulation

Evaluation plan for the Bayesian‑computing PTGS method vs. the elastic‑net baseline. Every
phase reuses the **same** shared harness (`cv` · `score` · `metrics` · `viz`) and the two
pluggable builders — only the **data source** changes.

## Phase 0 — iid mock (DONE)
`simulate.py` (`simulate_dataset`): GReX ~ iid standard normal, sparse causal genes → trait.
Purpose: wiring/smoke only. Both arms + 5‑fold nested CV + seaborn comparison figures run
end‑to‑end (notebooks 01, 04). *Limitation:* genes are independent — unrealistic; shrinkage
methods look similar here because there's no correlation/collinearity to exploit.

## Phase 1 — simulated TWAS GReX  (NEXT; slot: `simulate_twas.py`)
A realistic simulator where GReX has **TWAS structure**, so the elastic‑net‑vs‑Bayesian
comparison is meaningful. Target realism (to settle with the user — see "Open decisions"):
- **cis‑genotypes** with LD (e.g. block / AR(1) correlation), MAF spectrum;
- **sparse cis‑eQTL weights** per gene → `GReX_g = standardize(G_cis · w_g)`;
- **expression heritability** `h²_expr` (predicted vs. true expression gap);
- **gene–gene correlation** (adjacent genes sharing cis‑SNPs, or a latent factor) — the key
  feature iid mock lacks and where shrinkage priors should help;
- optional **ancestry** structure (portability, per the reference paper);
- **trait** from a sparse causal‑gene set (gaussian/binomial), known ground truth returned.
Purpose: quantify when/why the Bayesian shrinkage arm beats elastic net (correlated features,
p≫n), against a known answer. Deliverable: notebook `05_twas_simulation` (planned).

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
- **Phase 1 model:** which realism matters most (LD model; eQTL sparsity; `h²_expr`; gene–gene
  correlation mechanism; ancestry yes/no); fully synthetic generative model vs. seeding from
  real predictdb/GTEx weights + an LD reference panel.
- **Phase 2 data:** GReX file format (e.g. PrediXcan/predictdb genes×samples output), phenotype
  format, the trait(s) + covariates, approx dimensions (n_genes, n_samples), and where it runs
  (Terra / UKB RAP / local controlled) — to set up the cloud bridge.
