"""builders/base.py — the pluggable score-builder interface + shared fit helpers.

A builder turns a training `Dataset` into a `ScoreBundle`. Both arms (elastic-net baseline,
Bayesian shrinkage) implement the same `.name` / `.fit(train_ds, seed)` surface, so the CV
harness and comparison are builder-agnostic and the two methods differ *only* in how weights
are estimated. Inner-fold tuning (penalty / prior hyperparameters) happens inside ``fit``.

Shared helpers here keep both arms consistent: GReX is standardized for fitting, then weights
are converted back to the **raw** GReX scale so `score.compute_score` is a plain dot product;
for a continuous trait the covariates are optionally regressed out first (evaluation credits
the score only beyond covariates via `metrics.partial_r2`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..io import Dataset
from ..results import ScoreBundle


class Builder(ABC):
    """Interface every score-builder implements."""

    name: str = "builder"

    @abstractmethod
    def fit(self, train_ds: Dataset, seed: int = 0) -> ScoreBundle:
        """Fit gene weights on ``train_ds`` (inner tuning inside) → ScoreBundle."""
        raise NotImplementedError


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-standardize; return (X_std, mean, sd) with sd floored to avoid div-by-zero."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def to_raw_weights(coef_std: np.ndarray, intercept_std: float,
                   mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, float]:
    """Map coefficients fit on standardized X back to raw-GReX scale."""
    w_raw = coef_std / sd
    b_raw = float(intercept_std - np.sum(coef_std * mu / sd))
    return w_raw, b_raw


def residualize_gaussian(y: np.ndarray, covars: np.ndarray | None) -> np.ndarray:
    """Regress a continuous trait on covariates and return residuals (no covars -> centered y)."""
    if covars is None:
        return y - y.mean()
    A = np.column_stack([np.ones(len(y)), covars])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta
