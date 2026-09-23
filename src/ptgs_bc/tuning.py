"""tuning.py — hyperparameter-tuning helpers + the data behind the optimization plots.

Produces the "how does performance vary across the tuned parameter" information both arms need:

- `enet_cv_path`   : the elastic-net inner-CV surface (metric vs alpha, per l1_ratio).
- `bayes_p0_sweep` : nested-CV performance of the Bayesian arm across the sparsity guess p0
                     (p0 -> global scale tau0), i.e. its main tuning knob.
- `compare_priors` : Bayesian model comparison of the prior families by WAIC (self-contained,
                     from the pointwise log-likelihood) — the principled "which prior" selection.

The plotting counterparts live in `viz` (plot_enet_path / plot_param_sweep / plot_prior_comparison).
NumPyro/JAX/ArviZ import lazily so `import ptgs_bc` stays light.
"""

from __future__ import annotations

from .io import Dataset


def enet_cv_path(ds: Dataset, builder=None, seed: int = 0) -> dict:
    """Fit the elastic-net baseline and return its inner-CV tuning surface (for `plot_enet_path`)."""
    from .builders import ElasticNetBuilder
    b = builder or ElasticNetBuilder()
    return b.fit(ds, seed=seed).extras["cv"]


def bayes_p0_sweep(ds: Dataset, p0_values, outer_k: int = 3, seed: int = 0, **bayes_kw):
    """Nested-CV performance of the Bayesian arm across p0 values -> tidy DataFrame.

    Each p0 sets the horseshoe global scale tau0. Uses a light outer_k/small MCMC by default —
    this is many MCMC fits (len(p0) × outer_k), so keep the grid and sample counts modest.
    """
    import pandas as pd
    from .builders import BayesBuilder
    from .cv import nested_cv
    rows = []
    for p0 in p0_values:
        res = nested_cv(BayesBuilder(p0=p0, **bayes_kw), ds, outer_k=outer_k, seed=seed)
        rows.append({"p0": p0, "mean": res.mean, "sd": res.sd, "metric": res.metric_name})
    return pd.DataFrame(rows)


def compare_priors(ds: Dataset,
                   priors=("regularized_horseshoe", "horseshoe", "bayesian_lasso"),
                   p0: float | None = None, seed: int = 0,
                   num_warmup: int = 400, num_samples: int = 400,
                   hyper_overrides: dict[str, dict] | None = None, **bayes_kw):
    """Fit each prior on the full data and compare by WAIC (from the pointwise log-likelihood).

    Cheaper than nested-CV refits: one MCMC per prior on the training set, then out-of-sample
    predictive accuracy is estimated by WAIC (self-contained; no ArviZ). Returns
    (compare_df ranked best-first by elpd_waic, log_lik dict {prior: (draws × obs)}).

    `hyper_overrides`: optional `{prior_name: {hyper_key: value}}`, merged into that prior's
    resolved hyper before fitting — needed for priors that take a matrix/array hyperparameter
    the shared `**bayes_kw` can't express (e.g. `graph_horseshoe`'s `corr`, `group_horseshoe`'s
    `groups`), since those differ per prior rather than being a single shared knob like `p0`.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pandas as pd
    from numpyro.infer import MCMC, NUTS, log_likelihood
    from .builders.base import residualize_gaussian, standardize
    from .builders.bayes import BayesBuilder, _model

    X = ds.grex.to_numpy(float)
    Xs, _, _ = standardize(X)
    C = None if ds.covars is None else ds.covars.to_numpy(float)
    if ds.family == "gaussian":
        y = residualize_gaussian(ds.y.to_numpy(float), C); C_model = None
    else:
        y = ds.y.to_numpy(float); C_model = C
    Xj = jnp.asarray(Xs); yj = jnp.asarray(y)
    Cj = None if C_model is None else jnp.asarray(C_model)

    lls, rows = {}, []
    for prior in priors:
        b = BayesBuilder(prior=prior, p0=p0, num_warmup=num_warmup,
                         num_samples=num_samples, **bayes_kw)
        hyper = b._resolve_hyper(Xs, y, ds.family)
        if hyper_overrides and prior in hyper_overrides:
            hyper = {**hyper, **hyper_overrides[prior]}
        mcmc = MCMC(NUTS(_model, target_accept_prob=b.target_accept),
                    num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), Xj, Cj, yj, ds.family, prior, hyper)
        ll = np.asarray(log_likelihood(_model, mcmc.get_samples(),
                                       Xj, Cj, yj, ds.family, prior, hyper)["y"])  # (draws, obs)
        m = ll.max(axis=0)
        lppd_n = m + np.log(np.exp(ll - m).mean(axis=0))    # log mean exp over draws, per obs
        p_waic_n = ll.var(axis=0, ddof=1)                    # effective # parameters, per obs
        elpd_n = lppd_n - p_waic_n
        rows.append({"prior": prior, "elpd_waic": float(elpd_n.sum()),
                     "p_waic": float(p_waic_n.sum()),
                     "se": float(np.sqrt(len(elpd_n)) * elpd_n.std(ddof=1))})
        lls[prior] = ll

    compare_df = (pd.DataFrame(rows).sort_values("elpd_waic", ascending=False)
                  .reset_index(drop=True))
    compare_df.insert(0, "rank", range(len(compare_df)))
    return compare_df, lls
