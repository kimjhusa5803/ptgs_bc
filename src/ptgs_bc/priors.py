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
  - "graph_horseshoe" — regularized horseshoe with a gene-gene CORRELATION structure baked
    into the latent geometry (via `hyper["corr"]`), instead of iid genes. See
    `structure.estimate_gene_correlation` / `simulate_twas.true_grex_correlation` for where
    `corr` comes from.
  - "group_horseshoe" — regularized horseshoe with a shared group-level (e.g. pathway) local
    scale (via `hyper["groups"]`), so genes in an informative group are shrunk together.
  - "spike_slab" — George & McCulloch-style continuous spike-and-slab: each gene's weight is a
    two-component Gaussian mixture (near-zero spike vs. diffuse slab), marginalizing the
    discrete inclusion indicator so NUTS only samples continuous parameters. Unlike every
    horseshoe variant, this gives a genuine per-gene posterior inclusion probability (PIP) —
    see `spike_slab_pip`.
  - "graph_spike_slab" — `spike_slab` whose per-gene inclusion PROBABILITY (not the weight
    itself) is correlated across genes via `hyper["corr"]`, using the same Cholesky-correlated
    latent as `graph_horseshoe` applied to the mixing logit instead of the shrinkage scale —
    correlated genes are pulled toward being jointly included or excluded, targeting the
    known failure mode of independent-indicator spike-and-slab under correlated predictors.
TODO (as the method is developed): cross-ancestry hierarchical shrinkage (postponed).
"""

from __future__ import annotations

import numpy as np


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


def _graph_horseshoe(n_genes, corr, tau0=0.1, slab_scale=2.0, slab_df=4.0, **_):
    """Regularized horseshoe whose latent draw is CORRELATED across genes via `corr`.

    Identical tau/lam/c2 global-local-slab structure to `_regularized_horseshoe`; the only
    change is `z ~ MVN(0, corr)` (via a Cholesky factor of `corr`) instead of iid `z ~ N(0,1)`,
    so genes that are correlated (shared cis-eQTLs/LD, or an empirical GReX correlation) pull
    each other's shrinkage in the same direction rather than being shrunk independently.
    `corr` is a fixed (n_genes, n_genes) correlation matrix (not sampled), so its Cholesky
    factor is effectively a compile-time constant under JAX's tracing.
    """
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_genes)))
    c2 = numpyro.sample("c2", dist.InverseGamma(slab_df / 2.0, slab_df * slab_scale**2 / 2.0))
    lam_t = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    L = jnp.linalg.cholesky(jnp.asarray(corr) + 1e-4 * jnp.eye(n_genes))
    z0 = numpyro.sample("z0", dist.Normal(jnp.zeros(n_genes), 1.0))
    z = L @ z0
    return numpyro.deterministic("beta", z * tau * lam_t)


def _group_horseshoe(n_genes, groups, tau0=0.1, **_):
    """Regularized-style horseshoe with a shared GROUP-level local scale (e.g. pathway).

    `groups` is an (n_genes,) int array of group ids. Each gene keeps its own local scale
    `lam_j` (as in the plain horseshoe) but it is additionally multiplied by a scale `tau_g`
    shared across every gene in its group, so a group that collectively carries signal shrinks
    its members less even if individual genes look modest on their own (the classic
    group-horseshoe construction, e.g. Xu et al.).
    """
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    import numpy as np
    # n_groups must be a static Python int, computed BEFORE any jnp conversion: once NUTS
    # JIT-traces this model, `groups` becomes an abstract tracer and `int(jnp.max(...))` on it
    # raises `ConcretizationTypeError` -- np.max on the raw (still-numpy) input avoids that.
    n_groups = int(np.max(np.asarray(groups))) + 1
    groups = jnp.asarray(groups)
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    tau_g = numpyro.sample("tau_g", dist.HalfCauchy(jnp.ones(n_groups)))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_genes)))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(n_genes), 1.0))
    return numpyro.deterministic("beta", z * lam * tau * tau_g[groups])


def _spike_slab(n_genes, pi0=0.1, spike_scale=0.01, slab_scale=1.0, **_):
    """George & McCulloch-style continuous spike-and-slab (discrete indicator marginalized out).

    `beta_j` is a two-component Gaussian mixture: a near-zero "spike" (scale `spike_scale`,
    weight `1-pi0`) and a diffuse "slab" (scale `slab_scale`, weight `pi0`). NUTS samples the
    continuous `beta` directly (no discrete `gamma`), which is the standard NUTS-compatible way
    to fit spike-and-slab. See `spike_slab_pip` to recover a per-gene inclusion probability.
    """
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    probs = jnp.broadcast_to(jnp.array([1.0 - pi0, pi0]), (n_genes, 2))
    scales = jnp.broadcast_to(jnp.array([spike_scale, slab_scale]), (n_genes, 2))
    mixing = dist.Categorical(probs=probs)
    component = dist.Normal(jnp.zeros((n_genes, 2)), scales)
    return numpyro.sample("beta", dist.MixtureSameFamily(mixing, component))


def _graph_spike_slab(n_genes, corr, pi0=0.1, tau_pi=1.0, spike_scale=0.01, slab_scale=1.0, **_):
    """`_spike_slab` whose per-gene inclusion PROBABILITY (not the weight) is correlated.

    Same Cholesky-correlated-latent trick as `_graph_horseshoe`, applied to the mixing logit:
    `logit(pi_j) = logit(pi0) + tau_pi * phi_j`, `phi ~ MVN(0, corr)`. Correlated genes are
    pulled toward the same inclusion probability (jointly in or out), which is where a
    correlation-aware prior should matter more than it did for `graph_horseshoe` -- independent
    per-gene indicators otherwise split credit for correlated signal arbitrarily.
    """
    import numpyro, numpyro.distributions as dist
    import jax.numpy as jnp
    from jax.scipy.special import logit, expit
    L = jnp.linalg.cholesky(jnp.asarray(corr) + 1e-4 * jnp.eye(n_genes))
    phi0 = numpyro.sample("phi0", dist.Normal(jnp.zeros(n_genes), 1.0))
    phi = L @ phi0
    pi = numpyro.deterministic("pi", expit(logit(pi0) + tau_pi * phi))
    probs = jnp.stack([1.0 - pi, pi], axis=-1)
    scales = jnp.broadcast_to(jnp.array([spike_scale, slab_scale]), (n_genes, 2))
    mixing = dist.Categorical(probs=probs)
    component = dist.Normal(jnp.zeros((n_genes, 2)), scales)
    return numpyro.sample("beta", dist.MixtureSameFamily(mixing, component))


def spike_slab_pip(beta_samples, pi, spike_scale, slab_scale):
    """Posterior inclusion probability per gene, from posterior `beta` (and `pi`) draws.

    Per draw and gene: responsibility of the SLAB component under the two-component mixture,
    `r = pi*N(beta;0,slab) / (pi*N(beta;0,slab) + (1-pi)*N(beta;0,spike))`, then averaged over
    draws. `pi` may be a scalar/float (plain `spike_slab`, constant across genes and draws) or
    a `(draws, n_genes)` array (`graph_spike_slab`, `post["pi"]`) -- both broadcast correctly
    against `beta_samples` of shape `(draws, n_genes)`. Pure numpy: no MCMC needed to test it.
    """
    beta_samples = np.asarray(beta_samples)
    pi = np.asarray(pi)

    def _normal_pdf(x, scale):
        return np.exp(-0.5 * (x / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))

    slab_pdf = _normal_pdf(beta_samples, slab_scale)
    spike_pdf = _normal_pdf(beta_samples, spike_scale)
    numer = pi * slab_pdf
    denom = numer + (1.0 - pi) * spike_pdf
    responsibility = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), 0.0)
    return responsibility.mean(axis=0)


_REGISTRY = {
    "regularized_horseshoe": _regularized_horseshoe,
    "horseshoe": _horseshoe,
    "bayesian_lasso": _bayesian_lasso,
    "graph_horseshoe": _graph_horseshoe,
    "group_horseshoe": _group_horseshoe,
    "spike_slab": _spike_slab,
    "graph_spike_slab": _graph_spike_slab,
}
