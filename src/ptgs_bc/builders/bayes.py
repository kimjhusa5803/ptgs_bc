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
                 target_accept: float = 0.9, p0: float | None = None, sigma: float | None = None,
                 svi_steps: int = 20000, hyper: dict | None = None,
                 corr=None, corr_estimate: bool = False):
        self.prior = prior
        self.inference = inference
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept = target_accept   # raise for horseshoe's funnel geometry (fewer divergences)
        self.p0 = p0                          # prior guess of #associated genes -> global scale tau0
        self.sigma = sigma                    # noise scale for tau0 (gaussian: est. from y; logistic: 2)
        self.svi_steps = svi_steps
        self.hyper = hyper or {}
        self.corr = corr                      # oracle gene-gene correlation (graph_horseshoe); reused as-is every fold
        self.corr_estimate = corr_estimate    # if True and no corr/hyper["corr"] given, estimate per-fold from train X
        tag = f",p0={p0}" if p0 is not None else ""
        if prior in ("graph_horseshoe", "graph_spike_slab"):
            # disambiguate oracle vs. estimated vs. hyper-supplied corr -- otherwise two
            # builders differing only in corr source collide on `.name` and `run_benchmark`'s
            # dict-by-name silently drops one of them.
            tag += ",corr=est" if corr_estimate else (
                ",corr=oracle" if corr is not None else ",corr=hyper")
        self.name = f"bayes[{prior}]" + tag

    def _resolve_hyper(self, X, y, family) -> dict:
        """Derive per-fit hyperparameters that depend on the training data.

        - regularized-horseshoe global scale tau0 from p0 (Piironen-Vehtari).
        - spike_slab/graph_spike_slab's `pi0` from p0 (expected # causal genes / n_genes) --
          same p0 ergonomics as every other prior, just a different derived hyperparameter.
        - graph_horseshoe's `corr`: the oracle matrix if given, else (if `corr_estimate`) a
          Ledoit-Wolf estimate from THIS fold's training GReX only (no CV leakage).
        """
        import numpy as np
        hyper = dict(self.hyper)
        if self.p0 is not None and self.prior in ("spike_slab", "graph_spike_slab"):
            if "pi0" not in hyper:
                hyper["pi0"] = self.p0 / X.shape[1]
        elif self.p0 is not None and "tau0" not in hyper:
            D, n = X.shape[1], X.shape[0]
            sigma = self.sigma if self.sigma is not None else (
                float(np.std(y)) if family == "gaussian" else 2.0)
            hyper["tau0"] = (self.p0 / max(D - self.p0, 1)) * sigma / np.sqrt(n)
        if "corr" not in hyper:
            if self.corr is not None:
                hyper["corr"] = self.corr
            elif self.corr_estimate:
                from ..structure import estimate_gene_correlation
                hyper["corr"] = estimate_gene_correlation(X)
        return hyper

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

        hyper = self._resolve_hyper(Xs, y, train_ds.family)
        post = self._infer(Xs, C_model, y, train_ds.family, seed, hyper)
        beta_mean = np.asarray(post["beta"].mean(axis=0))
        beta_sd = np.asarray(post["beta"].std(axis=0))
        intercept_std = float(np.asarray(post["intercept"]).mean())

        w_raw, b_raw = to_raw_weights(beta_mean, intercept_std, mu, sd)
        extras = {"posterior_sd_std_scale": beta_sd, "prior": self.prior,
                 "inference": self.inference,
                 "chosen": {"prior": self.prior, "p0": self.p0, **hyper}}
        if self.prior in ("spike_slab", "graph_spike_slab"):
            from ..priors import spike_slab_pip
            pi = np.asarray(post["pi"]) if "pi" in post else hyper.get("pi0", 0.1)
            extras["pip"] = spike_slab_pip(
                post["beta"], pi, hyper.get("spike_scale", 0.01), hyper.get("slab_scale", 1.0))
        return ScoreBundle(
            weights=w_raw, genes=train_ds.genes, intercept=b_raw,
            family=train_ds.family, builder=self.name, seed=seed,
            extras=extras,
        )

    def _infer(self, X, C, y, family, seed, hyper):
        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, Predictive
        key = jax.random.PRNGKey(seed)
        Xj = jnp.asarray(X); yj = jnp.asarray(y)
        Cj = None if C is None else jnp.asarray(C)
        args = (Xj, Cj, yj, family, self.prior, hyper)

        if self.inference == "nuts":
            mcmc = MCMC(NUTS(_model, target_accept_prob=self.target_accept),
                        num_warmup=self.num_warmup,
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
