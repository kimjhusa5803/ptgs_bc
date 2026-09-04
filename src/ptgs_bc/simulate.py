"""simulate.py — synthetic GReX + trait with known ground truth.

The mock-local data policy runs everything (notebooks, tests, the smoke benchmark) on
simulated data that matches the `Dataset` schema. A sparse set of "causal" genes drives the
trait, so we can check that a builder recovers signal and that elastic-net vs. Bayesian are
compared on data with a known answer. Everything is seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Dataset


def simulate_dataset(n_samples: int = 500, n_genes: int = 200, n_causal: int = 20,
                     family: str = "gaussian", n_covars: int = 3,
                     effect_sd: float = 0.4, noise_sd: float = 1.0,
                     seed: int = 0) -> tuple[Dataset, np.ndarray]:
    """Return a `Dataset` and the true gene-weight vector (length n_genes, mostly zero).

    GReX is standard-normal (stand-in for standardized predicted expression). ``n_causal``
    genes get nonzero weights; the linear predictor is the causal signal (+ covariate effects),
    turned into a continuous trait (Gaussian) or case/control labels (binomial via logistic).
    """
    rng = np.random.default_rng(seed)
    n_causal = min(n_causal, n_genes)          # can't have more causal genes than genes
    genes = [f"gene_{i:04d}" for i in range(n_genes)]
    samples = [f"s{i:05d}" for i in range(n_samples)]

    grex = rng.standard_normal((n_samples, n_genes))
    true_w = np.zeros(n_genes)
    causal = rng.choice(n_genes, size=n_causal, replace=False)
    true_w[causal] = rng.normal(0.0, effect_sd, size=n_causal)

    eta = grex @ true_w
    covars = None
    if n_covars:
        C = rng.standard_normal((n_samples, n_covars))
        beta_c = rng.normal(0.0, 0.2, size=n_covars)
        eta = eta + C @ beta_c
        covars = pd.DataFrame(C, index=samples,
                              columns=[f"cov_{j}" for j in range(n_covars)])

    if family == "gaussian":
        y_vals = eta + rng.normal(0.0, noise_sd, size=n_samples)
    elif family == "binomial":
        p = 1.0 / (1.0 + np.exp(-eta))
        y_vals = rng.binomial(1, p).astype(float)
    else:
        raise ValueError(f"family must be 'gaussian' or 'binomial', got {family!r}")

    ds = Dataset(
        grex=pd.DataFrame(grex, index=samples, columns=genes),
        y=pd.Series(y_vals, index=samples, name="trait"),
        covars=covars, family=family,
        meta={"seed": seed, "n_causal": n_causal, "causal_idx": causal.tolist()},
    )
    return ds, true_w
