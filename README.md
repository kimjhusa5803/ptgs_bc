# PTGS_BC — Bayesian-computing polygenic transcriptomic scores

Develop and benchmark **Bayesian shrinkage-prior** polygenic transcriptomic scores (PTGS)
built on **GReX** (genetically regulated / predicted gene expression), against the
**elastic-net** baseline from the reference method. PTGS scores a trait as a weighted sum of
predicted gene-expression levels rather than SNPs (Liang et al., *Genome Biology* 2022;
`reference/`), which improves cross-ancestry portability and interpretability.

## Method (what the package implements)

- **Input:** GReX (samples × genes) + trait (+ covariates) from the local cohort biobank —
  the `Dataset` contract. Trait family is general (**gaussian** or **binomial**).
- **Two builders, one harness:** the score-construction step is pluggable —
  `ElasticNetBuilder` (baseline) vs. `BayesBuilder` (new). Everything else (folds, scoring,
  metric) is shared, so the comparison is fair.
- **Bayesian arm:** trait ~ GReX under a sparsity prior (regularized horseshoe by default;
  see `priors.py`), inferred with **NumPyro** (NUTS; SVI as the scale escape hatch).
- **Evaluation:** **5-fold nested CV** — outer folds for unbiased performance, each builder
  tunes itself in the inner folds. Metric: partial R² (continuous) / AUC (binary).

## Key decisions

- Package name `ptgs_bc`; **BC = Bayesian Computing** (the approach, not a disease).
- **Bayesian engine = NumPyro** (JAX): fastest NUTS at ~10k genes × nested CV, GPU-capable,
  Python-native; PyMC is the accessible fallback, Stan/brms a correctness oracle. The prior
  family and engine are constructor arguments — the pipeline never changes when they do.
- **Data policy = mock-local:** real controlled GReX/phenotypes never live here; notebooks and
  tests run on `simulate.py` output with the same schema. Data dirs are git-ignored.

## Installation

Requires **Python ≥ 3.10**. Editable install (mandatory — the `src/` layout means
`import ptgs_bc` fails until installed, even from `notebooks/`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, jupyter, matplotlib
```

## Quick start

```bash
python scripts/generate_mock_data.py --family gaussian    # writes data/ (git-ignored)
ptgs-bc smoke                                             # elastic-net benchmark on mock data
ptgs-bc smoke --with-bayes                                # add the (slower) Bayesian arm
pytest                                                    # fast pipeline tests
```

Then work through `notebooks/` in order (`01_simulate_smoke` → `04_comparison`).

## Layout (two-layer: package = engine, notebooks = workbench)

```
src/ptgs_bc/   io · simulate · cv · score · metrics · builders/{elastic_net,bayes} · priors · benchmark · viz · results · cli
notebooks/     01_simulate_smoke · 02_elastic_net · 03_bayesian_ptgs · 04_comparison
scripts/       generate_mock_data.py        tests/  pytest over the package
reference/     the PTRS paper (s13059-021-02591-w.pdf)
```

Rule: reusable methodology lives in the package; notebooks import it, orchestrate, and analyze.

## Evaluation roadmap

See **`docs/ROADMAP.md`**: Phase 0 iid mock (done) → Phase 1 simulated **TWAS GReX**
(`simulate_twas.py`, realistic gene–gene correlation) → Phase 2 real **local GReX + phenotype**
(controlled; `io.load_dataset` + the local→cloud workflow).
