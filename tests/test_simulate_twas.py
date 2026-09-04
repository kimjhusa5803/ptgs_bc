"""Tests for the TWAS-structured simulator: schema + that it induces gene-gene correlation."""

import numpy as np

from ptgs_bc import simulate_dataset, simulate_twas_dataset
from ptgs_bc.io import Dataset


def test_twas_shapes_and_contract():
    ds, w = simulate_twas_dataset(n_samples=120, n_genes=60, n_causal_genes=6, seed=0)
    assert isinstance(ds, Dataset)
    assert ds.n_samples == 120 and ds.n_genes == 60
    assert ds.grex.index.equals(ds.y.index)
    assert w.shape == (60,) and np.count_nonzero(w) == 6


def test_twas_induces_local_correlation():
    tw, _ = simulate_twas_dataset(n_samples=200, n_genes=80, window_overlap=10,
                                  ld_rho=0.7, seed=0)
    iid, _ = simulate_dataset(n_samples=200, n_genes=80, seed=0)
    adj = lambda df: float(np.abs(np.diag(np.corrcoef(df.to_numpy().T), 1)).mean())
    # adjacent genes are correlated in the TWAS sim, ~independent in the iid mock
    assert adj(tw.grex) > 0.1
    assert adj(tw.grex) > adj(iid.grex)


def test_twas_binomial():
    ds, _ = simulate_twas_dataset(n_samples=150, n_genes=50, family="binomial", seed=1)
    assert ds.family == "binomial"
    assert set(np.unique(ds.y.to_numpy())) <= {0.0, 1.0}
