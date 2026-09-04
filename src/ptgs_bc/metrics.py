"""metrics.py — one place for performance metrics, computed identically across both arms.

Following the reference (Liang et al. 2022), the primary continuous metric is the **partial
R²**: the extra variance in the trait explained by the PTGS *after* accounting for covariates
— so the score is credited only for signal beyond age/sex/PCs. For a binary trait we default
to AUC (with partial R² on the liability scale available). Fair comparison depends on both
arms being scored by the exact same function on the same folds.
"""

from __future__ import annotations

import numpy as np


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² of OLS y ~ [1, X]."""
    A = np.column_stack([np.ones(len(y)), X]) if X.ndim and X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def partial_r2(y: np.ndarray, score: np.ndarray, covars: np.ndarray | None = None) -> float:
    """Incremental R² of ``score`` over a covariate-only model (0 if covars is None)."""
    y = np.asarray(y, float); score = np.asarray(score, float).reshape(-1, 1)
    if covars is None:
        return _ols_r2(y, score)
    C = np.asarray(covars, float)
    r2_reduced = _ols_r2(y, C)
    r2_full = _ols_r2(y, np.column_stack([C, score]))
    denom = 1.0 - r2_reduced
    return (r2_full - r2_reduced) / denom if denom > 0 else 0.0


def auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, int), np.asarray(score, float)))


def evaluate(y: np.ndarray, score: np.ndarray, covars: np.ndarray | None = None,
             family: str = "gaussian") -> tuple[str, float]:
    """Return (metric_name, value) — the single metric used to compare arms for this family."""
    if family == "gaussian":
        return "partial_R2", partial_r2(y, score, covars)
    if family == "binomial":
        return "AUC", auc(y, score)
    raise ValueError(f"unknown family {family!r}")
