"""structure.py — gene-grouping/correlation inputs for the structured priors (`priors.py`).

Two things a "does correlation/pathway structure help?" analysis needs, kept separate from
the priors themselves so they're independently testable without touching NumPyro/JAX:

- `estimate_gene_correlation`: a REALISTIC (estimated-from-data) correlation matrix for
  `graph_horseshoe`, for use when no oracle correlation is available (real data, or the
  "how much does the graph_horseshoe prior degrade under estimation noise" question).
- `perturb_group`: corrupts a known group/pathway assignment (drop true members, add false
  ones) to demonstrate `group_horseshoe`'s sensitivity to realistic pathway misannotation.
"""

from __future__ import annotations

import numpy as np


def estimate_gene_correlation(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage correlation estimate of GReX columns (samples x genes).

    Plain `np.corrcoef` is ill-conditioned once `n_genes` approaches or exceeds `n_samples`
    (exactly the PTGS regime). Ledoit-Wolf shrinks the sample covariance toward a scaled
    identity, giving a well-conditioned, positive-definite estimate suitable for the
    `graph_horseshoe` prior's Cholesky factor even when p >> n.
    """
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(X).covariance_
    d = np.sqrt(np.diag(cov))
    d = np.where(d < 1e-12, 1.0, d)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def perturb_group(true_groups: np.ndarray, drop_frac: float = 0.0, add_frac: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Corrupt a binary group membership vector to simulate imperfect pathway annotation.

    `true_groups` is an (n_genes,) 0/1 array (1 = member of the informative group, e.g. the
    true causal-gene set or a candidate pathway). Returns a new 0/1 array with `drop_frac` of
    the true members removed (relabeled 0) and `add_frac` of the non-members added (relabeled
    1) — the two failure modes of a real pathway database (missed genes, spurious inclusions).
    """
    rng = rng or np.random.default_rng()
    g = np.asarray(true_groups).astype(int).copy()
    members = np.flatnonzero(g == 1)
    non_members = np.flatnonzero(g == 0)
    n_drop = round(drop_frac * len(members))
    n_add = round(add_frac * len(non_members))
    if n_drop:
        g[rng.choice(members, size=n_drop, replace=False)] = 0
    if n_add:
        g[rng.choice(non_members, size=n_add, replace=False)] = 1
    return g
