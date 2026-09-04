"""cv.py — the nested cross-validation harness (the shared, arm-agnostic evaluation loop).

Nested CV keeps tuning honest: the **outer** folds estimate unbiased test performance; each
builder does its own **inner** tuning on the outer-training data only (penalty for elastic
net; prior hyperparameters for the Bayesian arm), so no test leakage. Both arms run through
this exact loop on the *same* fold assignments (seeded) — that identity is what makes the
elastic-net vs. Bayesian comparison fair. Binary traits use stratified folds.
"""

from __future__ import annotations

import numpy as np

from .io import Dataset
from .metrics import evaluate
from .results import BenchmarkResult, FoldResult
from .score import score_dataset


def make_folds(y, k: int, family: str, seed: int) -> list[np.ndarray]:
    """Return k arrays of *test* positional indices (stratified for binomial)."""
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(y)
    if family == "binomial":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [te for _, te in splitter.split(np.zeros(n), np.asarray(y))]
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [te for _, te in splitter.split(np.zeros(n))]


def nested_cv(builder, ds: Dataset, outer_k: int = 5, seed: int = 0) -> BenchmarkResult:
    """Run one builder through outer-CV; the builder tunes itself in the inner folds.

    ``builder`` implements the `builders.base.Builder` protocol: ``.name`` and
    ``.fit(train_ds, seed) -> ScoreBundle`` (inner tuning happens inside ``fit``).
    """
    test_folds = make_folds(ds.y, outer_k, ds.family, seed)
    all_idx = np.arange(ds.n_samples)
    result = BenchmarkResult(builder=builder.name, metric_name="",
                             meta={"outer_k": outer_k, "seed": seed})
    for i, test_idx in enumerate(test_folds):
        train_idx = np.setdiff1d(all_idx, test_idx)
        train_ds, test_ds = ds.subset(train_idx), ds.subset(test_idx)

        bundle = builder.fit(train_ds, seed=seed + i)
        s_test = score_dataset(bundle, test_ds)
        covars = None if test_ds.covars is None else test_ds.covars.to_numpy()
        name, value = evaluate(test_ds.y.to_numpy(), s_test, covars, ds.family)

        result.metric_name = name
        result.folds.append(FoldResult(
            fold=i, metric_name=name, metric=value,
            n_train=len(train_idx), n_test=len(test_idx),
            chosen=dict(bundle.extras.get("chosen", {})),
        ))
    return result
