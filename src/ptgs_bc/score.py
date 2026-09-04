"""score.py — apply a fitted ScoreBundle to GReX to produce the PTGS.

Shared by both arms (elastic-net and Bayesian) so a score is computed identically no matter
who fit it: PTGS = intercept + GReX·weights (+ covariate adjustment when present). Genes are
aligned by name to the bundle, so a target set with a different column order still scores
correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Dataset
from .results import ScoreBundle


def compute_score(bundle: ScoreBundle, grex: pd.DataFrame,
                  covars: pd.DataFrame | None = None) -> np.ndarray:
    """Return the linear PTGS for each sample in ``grex`` (aligned to the bundle's genes)."""
    X = grex.reindex(columns=bundle.genes)
    if X.isna().any().any():
        missing = [g for g in bundle.genes if g not in grex.columns]
        raise ValueError(f"{len(missing)} bundle genes absent from grex (e.g. {missing[:3]}).")
    s = bundle.intercept + X.to_numpy() @ bundle.weights
    if bundle.covar_coef is not None and covars is not None and bundle.covars:
        s = s + covars.reindex(columns=bundle.covars).to_numpy() @ bundle.covar_coef
    return np.asarray(s, dtype=float)


def score_dataset(bundle: ScoreBundle, ds: Dataset) -> np.ndarray:
    return compute_score(bundle, ds.grex, ds.covars)
