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

## Phase 2 — real local GReX + phenotype  (Milestone v0.3.0; entry: `io.load_dataset`)
Run on the **local cohort biobank's** GReX + phenotype (controlled/PHI). Develop locally on
Phase‑0/1 mock, then execute in the governed environment — the `cloud-dev-workflow` (code to
GitHub, run in cloud, results/errors back as text; data never moves).

**Ordered sub-goals (agreed 2026-09-23; work through them in this order, don't skip ahead):**

1. **Real-data I/O.** Finish `io.load_dataset` for real GReX + phenotype/covariate files.
   GReX uncertainty/variation is a **default** consideration for this real-data path — unlike
   simulated data, where it stays out of scope — and may require an architecture change to
   `Dataset`/`Builder` (errors-in-variables), not just a bolt-on. Exact representation (per-
   sample SE vs. posterior draws vs. a reliability score) is undecided on purpose: build a
   generic/flexible slot now, refine once the user supplies real file metadata (headers,
   sample sizes, gene counts, tissue types) and once an actual run surfaces problems. **Claude
   leads this by asking the user one question at a time — logged below so nothing is re-asked
   or forgotten.** Trait/phenotype data **will be provided by the user** (resolved
   2026-09-23) — the promised model comparison (`partial_r2`, WAIC, `nested_cv`/
   `run_benchmark`) is inherently supervised and needs it; this is not deferred.
2. **Validate all 7 priors on simulated data.** `spike_slab`/`graph_spike_slab` were verified
   only via integration tests (Phase 1.6) — no dedicated notebook yet. Build the missing
   simulated-data comparison notebook so every current prior has been benchmarked against
   elastic net before real data enters the picture.
3. **Structured priors from real biological knowledge.** Replace the oracle/synthetic
   correlation and groups with real biological input. Needs discussion — **two separate
   sub-decisions, not one:**
   - `group_horseshoe` wants discrete gene→group membership (e.g. pathway sets from
     MSigDB/KEGG/Reactome).
   - `graph_horseshoe`/`graph_spike_slab` want a gene-gene correlation-like matrix (e.g. a PPI
     network like STRING, or a co-expression atlas like GTEx).
   Database choice, gene-ID conventions, and how each maps into `hyper["groups"]` /
   `hyper["corr"]` are all open.
4. **Cloud-dev-workflow packaging.** A streamlined, repeatable per-model run+evaluation
   script/notebook template, handed to the user to execute themselves in the secured
   environment; errors come back as text for Claude to fix (data never moves).

### Real-data I/O Q&A (sub-goal 1) — answered so far
_Claude asks one question at a time; log each answer here immediately so it's never re-asked._
- **Q1 (2026-09-23):** GReX file structure — file format, orientation (genes×samples vs.
  samples×genes), gene-ID convention, sample-ID convention. **Answered:** wide format, one row
  per sample; first column `GRID` (Vanderbilt BioVU-style de-identified subject ID, single
  column, no separate FID/IID split); remaining columns are genes, header = bare Ensembl gene
  ID with no version suffix (e.g. `ENSG00000002549`). Delimiter/file extension (tab vs.
  comma/space; `.txt`/`.tsv`/`.csv`) not yet confirmed — paste rendering is ambiguous here.
- **Q2 (2026-09-23):** Phenotype/covariate file structure — same `GRID` join key? What columns
  (trait, PCs, age, sex, etc.)? File format. **Answered:** `GRID` is shared with the GReX file;
  covariate columns vary and must be caller-supplied, never hardcoded by name.
  `io.load_dataset`'s existing `covar_cols: list[str]` / `sample_col` params already satisfy
  this — no column names are assumed anywhere in the loader. Extended `load_dataset` with an
  optional `grex_uncertainty_path` that plumbs an uncertainty table straight into
  `Dataset.meta["grex_uncertainty"]` with no shape assumed yet (per sub-goal 1's default-
  uncertainty requirement). Added `tests/test_io.py` using synthetic files shaped like the real
  ones (`GRID` + bare Ensembl IDs, tab-delimited GReX / comma-delimited phenotype, mismatched
  sample sets, the uncertainty path) — all pass, plus the full 35-test suite. Delimiter/file
  extension confirmation still open but no longer blocking: the loader auto-detects it.
- **Sample-mismatch policy (2026-09-23, explicit user instruction):** always use the
  intersection of GReX/phenotype sample sets (no error, no opt-out) — but `load_dataset` must
  inform the caller when a mismatch happens. Implemented as a `UserWarning` naming how many
  samples were GReX-only vs. phenotype-only and how many survived in the intersection. Test
  added confirming the warning fires on mismatch and stays silent when sets match exactly.
  36/36 tests pass.
- **Q3 (2026-09-23):** Is the sample-ID column name shared between GReX and phenotype/trait
  files within one cohort? **Answered: yes** — the ID column is cohort-dependent (`GRID` for
  VUMC, a different convention for AoU, etc.) but consistent across a given cohort's own files.
  No crosswalk/linking table needed. The existing single `sample_col` parameter (caller-
  supplied per cohort, never hardcoded) already covers this correctly — no code change needed.
- **Q4 (2026-09-23):** Is the trait file the same file as the covariate/PC file, or separate?
  **Answered:** same single file, called the "phenotype file" (trait + covariates together,
  as originally assumed) — no separate `trait_path` needed. Trait coding (continuous values vs.
  `1`/`0` case-control) maps directly onto the existing `Dataset.family` (`"gaussian"` /
  `"binomial"`) — no loader change needed there either.
- **Sub-goal 1 status:** `io.load_dataset` design is now validated against every real-data
  question raised so far (file shape, ID column, covariate flexibility, mismatch handling,
  trait/covariate co-location, GReX uncertainty plumbing) with no further code changes pending.
  Remaining: the actual trait/covariate column names for a real run (caller supplies these at
  call time), and the still-undecided GReX uncertainty representation shape (deferred; generic
  `meta["grex_uncertainty"]` slot already in place).

**Reordering (2026-09-23, explicit user instruction):** testing the currently-available models
against real GReX is NOT deferred to after sub-goal 2/3 — it happens as soon as sub-goal 1's
loader is validated, in parallel with (not after) validating the newer priors on simulated
data. Added `notebooks/07_real_data_baseline.ipynb`: mirrors notebooks 02-04 (elastic net +
`regularized_horseshoe`/`horseshoe`/`bayesian_lasso`) but loads via `io.load_dataset` instead
of `simulate_dataset`. Has a clearly-marked config cell (file paths, `sample_col`, `trait`,
`covar_cols`, `family`) that raises if left unedited — meant to be filled in and run in the
secured environment (`cloud-dev-workflow`), with errors reported back as text. No real data
or output should ever be committed — the notebook carries an explicit reminder to clear outputs
first. **Structured-prior notebooks on real data (sub-goal 3) are the step after this one runs
successfully** — not before.

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
- **Phase 2 data specifics** are tracked question-by-question in the "Real-data I/O Q&A" log
  under Phase 2 sub-goal 1 above — don't guess these, ask the user when needed.
- **Multi-tissue `Dataset` support** remains deferred until the current correlation/discovery
  goals are done.
- **GReX-estimation-uncertainty propagation** (errors-in-variables) is **no longer deferred** —
  it's Phase 2 sub-goal 1's default requirement for the real-data path (see above); simulated
  data paths are unaffected.
