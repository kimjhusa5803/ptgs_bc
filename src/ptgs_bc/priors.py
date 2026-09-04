"""priors.py — shrinkage priors for the Bayesian PTGS builder.

The heart of the method: instead of an elastic-net penalty (a point estimate), place a
sparsity-inducing prior on the gene weights and infer the full posterior. Each function samples
the weight vector inside a NumPyro model, using **non-centered** parameterizations so NUTS
mixes well in high dimensions (~10k genes). The prior family is a research decision (see
`bayes.BayesBuilder(prior=...)`); the registry below makes swapping one for another a
one-word change. NumPyro/JAX are imported lazily so `import ptgs_bc` stays light.

Implemented:
  - "regularized_horseshoe" (Piironen & Vehtari 2017) — default; heavy shrinkage + a slab that
    caps large effects. Robust for p >> n.
  - "horseshoe" — the classic Carvalho–Polson–Scott horseshoe.
  - "bayesian_lasso" — Laplace prior (the Bayesian analog of the elastic-net L1 part).
TODO (as the method is developed): spike-and-slab, group/pathway priors, cross-ancestry
hierarchical shrinkage.
"""

from __future__ import annotations


def sample_weights(prior: str, n_genes: int, **hyper):
    """Sample and return the (n_genes,) gene-weight vector under ``prior``."""
    fn = _REGISTRY.get(prior)
    if fn is None:
        raise ValueError(f"unknown prior {prior!r}; options: {sorted(_REGISTRY)}")
    return fn(n_genes, **hyper)


def _regularized_horseshoe(n_genes, tau0=0.1, slab_scale=2.0, slab_df=4.0, **_):
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_genes)))
    c2 = numpyro.sample("c2", dist.InverseGamma(slab_df / 2.0, slab_df * slab_scale**2 / 2.0))
    lam_t = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(n_genes), 1.0))
    return numpyro.deterministic("beta", z * tau * lam_t)


def _horseshoe(n_genes, tau0=0.1, **_):
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_genes)))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(n_genes), 1.0))
    return numpyro.deterministic("beta", z * tau * lam)


def _bayesian_lasso(n_genes, scale=0.1, **_):
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    b = numpyro.sample("b", dist.HalfCauchy(scale))
    return numpyro.sample("beta", dist.Laplace(jnp.zeros(n_genes), b))


_REGISTRY = {
    "regularized_horseshoe": _regularized_horseshoe,
    "horseshoe": _horseshoe,
    "bayesian_lasso": _bayesian_lasso,
}
