"""builders/elastic_net.py — the BASELINE builder (the reference method).

Reproduces the reference approach (Liang et al. 2022): build the PTGS with elastic net, a
sparse regularized regression over the predicted-expression features, tuning the penalty by
inner cross-validation. Gaussian traits use ``ElasticNetCV``; binary traits use
``LogisticRegressionCV`` with an elastic-net penalty. This is the arm the Bayesian shrinkage
method is compared against.
"""

from __future__ import annotations

import numpy as np

from ..io import Dataset
from ..results import ScoreBundle
from .base import Builder, residualize_gaussian, standardize, to_raw_weights


class ElasticNetBuilder(Builder):
    name = "elastic_net"

    def __init__(self, l1_ratios=(0.05, 0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 1.0),
                 inner_k: int = 5, max_iter: int = 5000):
        # l1_ratios widened toward ridge: correlated TWAS genes benefit from elastic net's
        # grouping (low l1_ratio) — pure lasso arbitrarily drops one of a correlated set.
        self.l1_ratios = list(l1_ratios)
        self.inner_k = inner_k
        self.max_iter = max_iter

    def fit(self, train_ds: Dataset, seed: int = 0) -> ScoreBundle:
        X = train_ds.grex.to_numpy(dtype=float)
        Xs, mu, sd = standardize(X)
        C = None if train_ds.covars is None else train_ds.covars.to_numpy(float)

        if train_ds.family == "gaussian":
            from sklearn.linear_model import ElasticNetCV
            y = residualize_gaussian(train_ds.y.to_numpy(float), C)
            m = ElasticNetCV(l1_ratio=self.l1_ratios, cv=self.inner_k,
                             max_iter=self.max_iter, random_state=seed)
            m.fit(Xs, y)
            coef, intercept = m.coef_, float(m.intercept_)
            chosen = {"alpha": float(m.alpha_), "l1_ratio": float(m.l1_ratio_)}
            # capture the inner-CV tuning surface for plotting (alpha x l1_ratio -> mean CV MSE)
            cv_path = {"param": "alpha", "l1_ratios": self.l1_ratios,
                       "alphas": np.asarray(m.alphas_),                 # (n_l1, n_alphas) or (n_alphas,)
                       "cv_mse": np.asarray(m.mse_path_).mean(axis=-1),  # mean over folds
                       "chosen": chosen}
        else:  # binomial
            from sklearn.linear_model import LogisticRegressionCV
            y = train_ds.y.to_numpy(int)
            m = LogisticRegressionCV(
                penalty="elasticnet", solver="saga", l1_ratios=self.l1_ratios,
                Cs=10, cv=self.inner_k, max_iter=self.max_iter, scoring="roc_auc",
                random_state=seed)
            m.fit(Xs, y)
            coef, intercept = m.coef_.ravel(), float(m.intercept_[0])
            chosen = {"C": float(m.C_[0]), "l1_ratio": float(m.l1_ratio_[0])}
            scores = np.asarray(next(iter(m.scores_.values()))).mean(axis=0)  # (n_Cs, n_l1)
            cv_path = {"param": "C", "l1_ratios": self.l1_ratios,
                       "alphas": np.asarray(m.Cs_), "cv_score": scores, "chosen": chosen}

        w_raw, b_raw = to_raw_weights(coef, intercept, mu, sd)
        return ScoreBundle(
            weights=w_raw, genes=train_ds.genes, intercept=b_raw,
            family=train_ds.family, builder=self.name, seed=seed,
            extras={"chosen": chosen, "n_nonzero": int(np.sum(coef != 0)), "cv": cv_path},
        )
