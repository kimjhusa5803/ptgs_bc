"""benchmark.py — run the arms through the same nested CV and compare.

The comparison orchestration: given a `Dataset` and a set of builders, run each through the
identical `cv.nested_cv` (same folds, same metric) and collect one `BenchmarkResult` per arm.
This is where "elastic-net vs. Bayesian PTGS" is actually measured.
"""

from __future__ import annotations

import pandas as pd

from .cv import nested_cv
from .io import Dataset
from .results import BenchmarkResult


def run_benchmark(builders: list, ds: Dataset, outer_k: int = 5,
                  seed: int = 0) -> dict[str, BenchmarkResult]:
    """Run every builder through the same seeded nested CV; return {builder_name: result}."""
    return {b.name: nested_cv(b, ds, outer_k=outer_k, seed=seed) for b in builders}


def summary_table(results: dict[str, BenchmarkResult]) -> pd.DataFrame:
    """Tidy per-arm summary: mean ± sd of the metric across the outer folds."""
    rows = [{"builder": name, "metric": r.metric_name, "mean": r.mean,
             "sd": r.sd, "n_folds": len(r.folds)} for name, r in results.items()]
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def per_fold_table(results: dict[str, BenchmarkResult]) -> pd.DataFrame:
    """Long per-fold table (for paired tests / plotting the arms on identical folds)."""
    rows = []
    for name, r in results.items():
        for f in r.folds:
            rows.append({"builder": name, "fold": f.fold, "metric": f.metric_name,
                         "value": f.metric})
    return pd.DataFrame(rows)
