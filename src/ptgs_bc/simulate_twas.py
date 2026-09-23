"""simulate_twas.py — realistic TWAS-structured GReX simulator (Phase 1; see docs/ROADMAP.md).

A self-contained synthetic version of the `twas_sim` generative model (Wang et al. 2023,
Bioinformatics), corroborated by the TWAS power studies He 2022 and Cao 2021: cis-genotypes
with LD → sparse cis-eQTL effects → predicted expression (GReX) → trait from a sparse set of
causal genes. Unlike the iid mock, GReX columns are **correlated** (adjacent genes share
cis-SNP windows, and SNPs are in LD), which is where shrinkage vs. elastic net can diverge.

Model:
  1. one SNP array with AR(1) LD (rho): G[:,j] = rho·G[:,j-1] + sqrt(1-rho²)·eps.
  2. gene g takes a cis-window of `n_snps_per_gene` SNPs; adjacent windows OVERLAP by
     `window_overlap` SNPs -> shared SNPs + LD induce gene-gene correlation.
  3. sparse cis-eQTL weights on a fraction `eqtl_sparsity` of the window; GReX_g =
     standardize(G_window · w_eqtl); optional prediction noise `expr_pred_r2` < 1 mimics an
     imperfect eQTL model (expression cis-heritability / prediction accuracy).
  4. trait: `n_causal_genes` causal genes, Y = Σ α·GReX (+ covars) + noise scaled to
     `trait_pve` (gaussian) or via a logistic link (binomial). Ground-truth α is returned.

Same `Dataset` schema as `simulate.simulate_dataset`, so nothing downstream changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Dataset


def simulate_twas_dataset(
    n_samples: int = 500,
    n_genes: int = 150,
    n_snps_per_gene: int = 20,
    window_overlap: int = 8,        # shared cis-SNPs between adjacent genes -> gene-gene corr
    ld_rho: float = 0.6,            # AR(1) LD between neighboring SNPs
    eqtl_sparsity: float = 0.25,    # fraction of a gene's cis-SNPs that are eQTLs
    expr_pred_r2: float = 1.0,      # GReX prediction accuracy (<1 adds prediction noise)
    n_causal_genes: int = 12,       # genes truly associated with the trait
    trait_pve: float = 0.15,        # variance of the trait explained by the causal GReX (gaussian)
    n_covars: int = 3,
    effect_sd: float = 1.0,
    family: str = "gaussian",
    seed: int = 0,
) -> tuple[Dataset, np.ndarray]:
    """Return a TWAS-structured `Dataset` and the true gene-weight vector (mostly zero)."""
    rng = np.random.default_rng(seed)
    n_causal_genes = min(n_causal_genes, n_genes)
    step = max(n_snps_per_gene - window_overlap, 1)
    n_snps = (n_genes - 1) * step + n_snps_per_gene
    genes = [f"gene_{i:04d}" for i in range(n_genes)]
    samples = [f"s{i:05d}" for i in range(n_samples)]

    # 1. genotypes with AR(1) LD along the SNP array
    G = np.empty((n_samples, n_snps))
    G[:, 0] = rng.standard_normal(n_samples)
    a = np.sqrt(1.0 - ld_rho**2)
    for j in range(1, n_snps):
        G[:, j] = ld_rho * G[:, j - 1] + a * rng.standard_normal(n_samples)

    # 2-3. per-gene predicted expression (GReX) from sparse cis-eQTL weights
    grex = np.empty((n_samples, n_genes))
    n_eqtl = max(1, round(eqtl_sparsity * n_snps_per_gene))
    window_start = np.empty(n_genes, dtype=int)
    eqtl_idx = np.empty((n_genes, n_eqtl), dtype=int)      # local (within-window) SNP index
    eqtl_weight = np.empty((n_genes, n_eqtl), dtype=float)
    for g in range(n_genes):
        s = g * step
        win = G[:, s:s + n_snps_per_gene]
        idx = rng.choice(n_snps_per_gene, size=n_eqtl, replace=False)
        w = rng.standard_normal(n_eqtl)
        window_start[g] = s
        eqtl_idx[g] = idx
        eqtl_weight[g] = w
        gc = win[:, idx] @ w
        gc = (gc - gc.mean()) / (gc.std() + 1e-8)
        if expr_pred_r2 < 1.0:
            gc = np.sqrt(expr_pred_r2) * gc + np.sqrt(1 - expr_pred_r2) * rng.standard_normal(n_samples)
            gc = (gc - gc.mean()) / (gc.std() + 1e-8)
        grex[:, g] = gc

    # 4. trait from a sparse causal-gene set
    true_w = np.zeros(n_genes)
    causal = rng.choice(n_genes, size=n_causal_genes, replace=False)
    true_w[causal] = rng.normal(0.0, effect_sd, size=n_causal_genes)
    genetic = grex @ true_w
    eta = genetic.copy()
    covars = None
    if n_covars:
        C = rng.standard_normal((n_samples, n_covars))
        eta = eta + C @ rng.normal(0.0, 0.2, size=n_covars)
        covars = pd.DataFrame(C, index=samples, columns=[f"cov_{j}" for j in range(n_covars)])

    if family == "gaussian":
        var_g = float(np.var(genetic)) + 1e-12
        noise_sd = np.sqrt(var_g * (1 - trait_pve) / trait_pve)
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
        meta={"seed": seed, "n_causal_genes": n_causal_genes, "causal_idx": causal.tolist(),
              "ld_rho": ld_rho, "window_overlap": window_overlap, "trait_pve": trait_pve,
              "expr_pred_r2": expr_pred_r2, "window_start": window_start,
              "eqtl_idx": eqtl_idx, "eqtl_weight": eqtl_weight},
    )
    return ds, true_w


def true_grex_correlation(meta: dict) -> np.ndarray:
    """Closed-form population correlation of GReX columns, from `simulate_twas_dataset`'s meta.

    No simulation/Monte Carlo involved — this is the exact oracle, useful as an upper bound
    for `graph_horseshoe` and as ground truth for validating `structure.estimate_gene_correlation`.

    Derivation: the AR(1) genotype process has `Cov(G_i, G_j) = ld_rho**|i-j|` exactly (each
    `G[:, j]` is unit-variance by construction). Gene g's raw pre-standardization signal is a
    fixed linear combination `w_g` of its window's SNPs, so
    `Cov(raw_g, raw_h) = w_g^T [ld_rho**|pos_g_k - pos_h_l|]_{k,l} w_h`. Prediction noise
    (`expr_pred_r2 < 1`) is drawn independently per gene, so it dilutes cross-gene correlation
    by exactly `expr_pred_r2` (variance stays 1 after the code's renormalization) while leaving
    the diagonal at 1.
    """
    ld_rho = meta["ld_rho"]
    r2 = meta.get("expr_pred_r2", 1.0)
    window_start = np.asarray(meta["window_start"])
    eqtl_idx = np.asarray(meta["eqtl_idx"])
    eqtl_weight = np.asarray(meta["eqtl_weight"])
    n_genes = len(window_start)

    pos = window_start[:, None] + eqtl_idx                       # (n_genes, n_eqtl) absolute SNP index
    diff = pos[:, None, :, None] - pos[None, :, None, :]         # diff[g,h,k,l] = pos[g,k]-pos[h,l]
    rho_block = ld_rho ** np.abs(diff)                           # (n_genes, n_genes, n_eqtl, n_eqtl)
    cov = np.einsum("gk,ghkl,hl->gh", eqtl_weight, rho_block, eqtl_weight)
    var = np.diag(cov).copy()
    var = np.where(var < 1e-12, 1.0, var)
    corr = cov / np.sqrt(np.outer(var, var))
    corr *= r2
    np.fill_diagonal(corr, 1.0)
    return corr
