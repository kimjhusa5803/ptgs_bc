"""viz.py — comparison figures (thin plotting helpers; matplotlib is a dev extra).

Kept out of the notebooks so a plot the analysis relies on is defined once and testable.
"""

from __future__ import annotations

from .results import BenchmarkResult


def plot_fold_comparison(results: dict[str, BenchmarkResult], ax=None):
    """Per-fold metric for each arm on the same folds (paired dots + mean line)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for i, (name, r) in enumerate(results.items()):
        ax.scatter([i] * len(r.folds), r.values, alpha=0.6)
        ax.hlines(r.mean, i - 0.2, i + 0.2, color="k")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(list(results), rotation=20)
    metric = next(iter(results.values())).metric_name if results else ""
    ax.set_ylabel(metric)
    ax.set_title("PTGS: elastic-net vs. Bayesian (nested CV)")
    return ax
