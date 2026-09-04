"""builders/bayes.py — the Bayesian shrinkage-prior PTGS builder (the new method).

Replaces elastic net with a Bayesian regression of the trait on GReX under a sparsity prior
(`priors.py`), inferred numerically with NumPyro. Returns posterior-mean gene weights as the
score, plus posterior SDs in ``extras`` (uncertainty the point-estimate baseline can't give).

Inference: NUTS by default; SVI (variational) is the scalability escape hatch for the full
~10k-gene problem. The prior family and the engine are research decisions — both are
constructor arguments, so the surrounding pipeline (CV, scoring, comparison) never changes.

NOTE: this is a working *initial* default (regularized horseshoe + NUTS), the slot to refine
as the Bayesian-computing approach is developed — not a finalized model.
"""

from __future__ import annotations

import numpy as np

from ..io import Dataset
from ..priors import sample_weights
from ..results import ScoreBundle
from .base import Builder, residualize_gaussian, standardize, to_raw_weights


def _model(X, covars, y, family, prior, hyper):
    """NumPyro model: y ~ f(intercept + X·beta + covars·gamma) with a shrinkage prior on beta."""
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    n_genes = X.shape[1]
    beta = sample_weights(prior, n_genes, **hyper)
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))
    eta = intercept + X @ beta
    if covars is not None:
        gamma = numpyro.sample("gamma", dist.Normal(jnp.zeros(covars.shape[1]), 5.0))
        eta = eta + covars @ gamma
    if family == "gaussian":
        sigma = numpyro.sample("sigma", dist.HalfCauchy(2.0))
        numpyro.sample("y", dist.Normal(eta, sigma), obs=y)
    else:  # binomial
        numpyro.sample("y", dist.Bernoulli(logits=eta), obs=y)


class BayesBuilder(Builder):
    name = "bayes"

    def __init__(self, prior: str = "regularized_horseshoe", inference: str = "nuts",
                 num_warmup: int = 500, num_samples: int = 500, num_chains: int = 1,
                 svi_steps: int = 20000, hyper: dict | None = None):
        self.prior = prior
        self.inference = inference
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.svi_steps = svi_steps
        self.hyper = hyper or {}
        self.name = f"bayes[{prior}]"

    def fit(self, train_ds: Dataset, seed: int = 0) -> ScoreBundle:
        import jax
        import numpyro
        X = train_ds.grex.to_numpy(dtype=float)
        Xs, mu, sd = standardize(X)
        C = None if train_ds.covars is None else train_ds.covars.to_numpy(float)

        if train_ds.family == "gaussian":
            y = residualize_gaussian(train_ds.y.to_numpy(float), C)
            C_model = None                      # covariates already removed from y
        else:
            y = train_ds.y.to_numpy(float)
            C_model = C                          # adjust inside the logistic model

        post = self._infer(Xs, C_model, y, train_ds.family, seed)
        beta_mean = np.asarray(post["beta"].mean(axis=0))
        beta_sd = np.asarray(post["beta"].std(axis=0))
        intercept_std = float(np.asarray(post["intercept"]).mean())

        w_raw, b_raw = to_raw_weights(beta_mean, intercept_std, mu, sd)
        return ScoreBundle(
            weights=w_raw, genes=train_ds.genes, intercept=b_raw,
            family=train_ds.family, builder=self.name, seed=seed,
            extras={"posterior_sd_std_scale": beta_sd, "prior": self.prior,
                    "inference": self.inference, "chosen": {"prior": self.prior}},
        )

    def _infer(self, X, C, y, family, seed):
        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, Predictive
        key = jax.random.PRNGKey(seed)
        Xj = jnp.asarray(X); yj = jnp.asarray(y)
        Cj = None if C is None else jnp.asarray(C)
        args = (Xj, Cj, yj, family, self.prior, self.hyper)

        if self.inference == "nuts":
            mcmc = MCMC(NUTS(_model), num_warmup=self.num_warmup,
                        num_samples=self.num_samples, num_chains=self.num_chains,
                        progress_bar=False)
            mcmc.run(key, *args)
            return mcmc.get_samples()

        # SVI (variational) fallback for large p
        guide = autoguide.AutoNormal(_model)
        svi = SVI(_model, guide, numpyro.optim.Adam(1e-3), Trace_ELBO())
        state = svi.run(key, self.svi_steps, *args, progress_bar=False)
        pred = Predictive(guide, params=state.params, num_samples=self.num_samples)
        return pred(jax.random.PRNGKey(seed + 1), *args)
