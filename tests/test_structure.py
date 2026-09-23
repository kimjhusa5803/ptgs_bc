"""Tests for structure.py (gene-gene correlation / group utilities) and the closed-form
oracle correlation in simulate_twas.true_grex_correlation — fast, no MCMC."""

import numpy as np

from ptgs_bc import simulate_twas_dataset, true_grex_correlation
from ptgs_bc.structure import estimate_gene_correlation, perturb_group


def test_estimate_gene_correlation_valid_corr_matrix():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 150))  # p >> n regime
    corr = estimate_gene_correlation(X)
    assert corr.shape == (150, 150)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-8)
    np.testing.assert_allclose(corr, corr.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() > -1e-8  # positive semi-definite


def test_true_grex_correlation_matches_empirical():
    ds, _ = simulate_twas_dataset(n_samples=20000, n_genes=12, n_snps_per_gene=10,
                                  window_overlap=6, ld_rho=0.7, eqtl_sparsity=0.4,
                                  n_causal_genes=4, seed=3)
    oracle = true_grex_correlation(ds.meta)
    empirical = np.corrcoef(ds.grex.to_numpy(), rowvar=False)
    assert oracle.shape == (12, 12)
    np.testing.assert_allclose(np.diag(oracle), 1.0, atol=1e-8)
    off = ~np.eye(12, dtype=bool)
    np.testing.assert_allclose(oracle[off], empirical[off], atol=0.05)


def test_true_grex_correlation_scales_with_expr_pred_r2():
    # Same window/eQTL structure, only expr_pred_r2 differs -- isolates the scaling behavior
    # without relying on two `simulate_twas_dataset` calls sharing an RNG stream (they don't:
    # the expr_pred_r2<1 branch draws extra noise per gene, shifting later genes' eQTL weights
    # even under the same seed).
    ds, _ = simulate_twas_dataset(n_samples=500, n_genes=10, window_overlap=6, seed=1)
    meta_full = dict(ds.meta, expr_pred_r2=1.0)
    meta_noisy = dict(ds.meta, expr_pred_r2=0.5)
    corr_full = true_grex_correlation(meta_full)
    corr_noisy = true_grex_correlation(meta_noisy)
    off = ~np.eye(10, dtype=bool)
    np.testing.assert_allclose(corr_noisy[off], 0.5 * corr_full[off], atol=1e-8)


def test_perturb_group_drop_and_add_counts():
    rng = np.random.default_rng(0)
    true_groups = np.zeros(100, dtype=int)
    true_groups[:20] = 1
    perturbed = perturb_group(true_groups, drop_frac=0.5, add_frac=0.1, rng=rng)
    assert perturbed.sum() != true_groups.sum() or perturbed.tolist() != true_groups.tolist()
    n_dropped = np.sum((true_groups == 1) & (perturbed == 0))
    n_added = np.sum((true_groups == 0) & (perturbed == 1))
    assert n_dropped == 10          # 50% of 20 true members
    assert n_added == 8             # 10% of 80 non-members


def test_perturb_group_no_op_when_fractions_zero():
    true_groups = np.array([0, 1, 1, 0, 1])
    out = perturb_group(true_groups, drop_frac=0.0, add_frac=0.0)
    np.testing.assert_array_equal(out, true_groups)
