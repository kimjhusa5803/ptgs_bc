"""Smoke tests for the comparison figures (skipped if matplotlib isn't installed)."""

import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")  # headless

from ptgs_bc import (ElasticNetBuilder, plot_comparison, plot_paired_scatter,
                     plot_performance, run_benchmark, simulate_dataset)
from ptgs_bc.builders.base import Builder
from ptgs_bc.results import ScoreBundle
import numpy as np


class _ConstBuilder(Builder):
    """A trivial second arm so viz tests don't need the slow Bayesian fit."""
    def __init__(self, scale=1.0): self.name = f"const_{scale}"; self.scale = scale
    def fit(self, train_ds, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.01 * self.scale, size=train_ds.n_genes)
        return ScoreBundle(weights=w, genes=train_ds.genes, family=train_ds.family,
                           builder=self.name, seed=seed)


def _two_arm_results():
    ds, _ = simulate_dataset(n_samples=200, n_genes=40, n_causal=6, seed=0)
    return run_benchmark([ElasticNetBuilder(), _ConstBuilder()], ds, outer_k=5, seed=0)


def test_plot_comparison_returns_axes():
    ax = plot_comparison(_two_arm_results())
    assert ax is not None and len(ax.get_xticklabels()) == 2


def test_plot_paired_scatter_two_arms():
    ax = plot_paired_scatter(_two_arm_results())
    assert ax is not None


def test_plot_performance_figure():
    fig = plot_performance(_two_arm_results())
    assert fig is not None and len(fig.axes) == 2


def test_paired_scatter_requires_two_arms():
    ds, _ = simulate_dataset(n_samples=120, n_genes=20, seed=1)
    one = run_benchmark([ElasticNetBuilder()], ds, outer_k=5, seed=0)
    with pytest.raises(ValueError):
        plot_paired_scatter(one)
