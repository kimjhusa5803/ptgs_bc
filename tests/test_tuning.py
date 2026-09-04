"""Fast tests for the tuning helpers + their plots (matplotlib-guarded; no MCMC here).

The Bayesian sweep / prior comparison run MCMC and are exercised in the notebooks, not the
fast suite.
"""

import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from ptgs_bc import enet_cv_path, plot_enet_path, plot_param_sweep, simulate_dataset


def test_enet_cv_path_and_plot():
    ds, _ = simulate_dataset(n_samples=150, n_genes=40, n_causal=6, seed=0)
    path = enet_cv_path(ds)
    assert {"param", "l1_ratios", "alphas", "chosen"} <= set(path)
    assert "cv_mse" in path  # gaussian
    ax = plot_enet_path(path)
    assert ax is not None


def test_plot_param_sweep():
    df = pd.DataFrame({"p0": [5, 20, 50], "mean": [0.30, 0.41, 0.36],
                       "sd": [0.05, 0.04, 0.06], "metric": ["partial_R2"] * 3})
    ax = plot_param_sweep(df, param="p0")
    assert ax is not None
