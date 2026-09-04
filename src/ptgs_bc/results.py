"""results.py — structured objects that cross the package↔notebook boundary.

Why dataclasses (not tuples/dicts): notebooks and tests read `bundle.weights`,
`result.metric` by name; internal changes don't silently reshuffle positions. A fitted
score is a *serializable artifact* (``ScoreBundle.save``/``load``) so any analysis can be
reproduced from one file + the recorded seed (two-layer-model-dev conventions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Family = str  # "gaussian" | "binomial" (general trait support; see io.Dataset)


@dataclass
class ScoreBundle:
    """A fitted PTGS: gene weights (+ optional covariate coefficients) and how to apply them.

    The single artifact both builders (elastic-net, Bayesian) produce, so downstream scoring
    and comparison are builder-agnostic. Bayesian fits may also carry posterior summaries in
    ``extras`` (e.g. weight SDs, credible intervals) — absent for point-estimate builders.
    """

    weights: np.ndarray            # (n_genes,) gene-level score weights
    genes: list[str]               # gene ids aligning to weights
    intercept: float = 0.0
    covar_coef: np.ndarray | None = None   # (n_covars,) nuisance coefficients, if adjusted
    covars: list[str] = field(default_factory=list)
    family: Family = "gaussian"
    builder: str = ""              # which builder produced this (provenance)
    seed: int | None = None
    extras: dict = field(default_factory=dict)   # e.g. posterior SDs, chosen hyperparams

    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ScoreBundle":
        import joblib
        return joblib.load(Path(path))


@dataclass
class FoldResult:
    """One outer-fold outcome of nested CV."""

    fold: int
    metric_name: str
    metric: float
    n_train: int
    n_test: int
    chosen: dict = field(default_factory=dict)   # hyperparams selected in the inner folds


@dataclass
class BenchmarkResult:
    """A builder's performance across the outer folds — the unit compared across arms."""

    builder: str
    metric_name: str
    folds: list[FoldResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def values(self) -> np.ndarray:
        return np.array([f.metric for f in self.folds], dtype=float)

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if self.folds else float("nan")

    @property
    def sd(self) -> float:
        return float(self.values.std(ddof=1)) if len(self.folds) > 1 else 0.0

    def __str__(self) -> str:
        return f"{self.builder}: {self.metric_name} = {self.mean:.4f} ± {self.sd:.4f} (n={len(self.folds)})"
