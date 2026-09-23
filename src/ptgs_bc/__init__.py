"""ptgs_bc — Bayesian-computing polygenic transcriptomic scores (PTGS) on GReX.

Method: build a trait score as a weighted sum of genetically regulated expression (GReX),
comparing a **Bayesian shrinkage-prior** builder (the new approach) against an **elastic-net**
baseline (the reference method, Liang et al. 2022), evaluated by **5-fold nested CV** with the
partial-R²/AUC metric. Design: one shared harness (`cv`, `score`, `metrics`) + pluggable
`builders`, so the two arms differ only in weight estimation and the comparison is fair.

Two-layer project (see README): this installable package is the methodology engine; the
numbered `notebooks/` import it to test and run the analysis. Data policy is mock-local —
develop on `simulate.py` output; real controlled GReX runs elsewhere.

Public API is imported from the top level; internal modules can reorganize behind it.
NumPyro/JAX load lazily (only when the Bayesian arm actually runs), so importing this package
stays fast.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .benchmark import per_fold_table, run_benchmark, summary_table
from .builders import BayesBuilder, Builder, ElasticNetBuilder
from .cv import make_folds, nested_cv
from .io import Dataset, load_dataset
from .metrics import evaluate, partial_r2
from .priors import spike_slab_pip
from .results import BenchmarkResult, FoldResult, ScoreBundle
from .score import compute_score, score_dataset
from .simulate import simulate_dataset
from .simulate_twas import simulate_twas_dataset, true_grex_correlation
from .structure import estimate_gene_correlation, perturb_group
from .tuning import bayes_p0_sweep, compare_priors, enet_cv_path
from .viz import (plot_comparison, plot_corr_recovery, plot_enet_path, plot_paired_scatter,
                  plot_param_sweep, plot_performance, plot_prior_comparison)

__all__ = [
    "__version__",
    # data
    "Dataset", "load_dataset", "simulate_dataset", "simulate_twas_dataset",
    # builders
    "Builder", "ElasticNetBuilder", "BayesBuilder",
    # evaluation
    "nested_cv", "make_folds", "run_benchmark", "summary_table", "per_fold_table",
    "compute_score", "score_dataset", "evaluate", "partial_r2",
    # results
    "ScoreBundle", "FoldResult", "BenchmarkResult",
    # gene-gene structure (correlation / pathway groups) for the structured priors
    "true_grex_correlation", "estimate_gene_correlation", "perturb_group",
    # spike-and-slab posterior inclusion probability
    "spike_slab_pip",
    # tuning
    "enet_cv_path", "bayes_p0_sweep", "compare_priors",
    # visualization
    "plot_performance", "plot_comparison", "plot_paired_scatter",
    "plot_enet_path", "plot_param_sweep", "plot_prior_comparison", "plot_corr_recovery",
]
