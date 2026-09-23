"""Fast tests for the structured priors (graph_horseshoe, group_horseshoe, spike_slab,
graph_spike_slab) in priors.py.

Deliberately NOT running MCMC here (matches test_tuning.py/test_viz.py's convention: MCMC is
slow and only exercised in the notebooks). Instead, each prior is drawn ONCE from its NumPyro
model under `numpyro.handlers.seed`, which is enough to catch shape errors, NaN/inf sampling
bugs, and registry wiring mistakes without paying for a full NUTS run. NOTE: this eager-mode
draw does NOT catch bugs that only manifest under real NUTS JIT tracing (two such bugs were
found in graph_horseshoe/group_horseshoe only via a full nested_cv integration run) -- treat
these as a first-pass smoke check, not a substitute for that integration test.
"""

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
from numpyro.handlers import seed  # noqa: E402

from ptgs_bc.priors import sample_weights, spike_slab_pip  # noqa: E402


def _draw(prior, n_genes, seed_val=0, **hyper):
    fn = seed(lambda: sample_weights(prior, n_genes, **hyper), rng_seed=seed_val)
    return np.asarray(fn())


def test_graph_horseshoe_identity_corr_matches_shape():
    n = 15
    beta = _draw("graph_horseshoe", n, corr=np.eye(n))
    assert beta.shape == (n,)
    assert np.all(np.isfinite(beta))


def test_graph_horseshoe_correlated_corr_matches_shape():
    n = 15
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    cov = A @ A.T + n * np.eye(n)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    beta = _draw("graph_horseshoe", n, corr=corr)
    assert beta.shape == (n,)
    assert np.all(np.isfinite(beta))


def test_group_horseshoe_shape_and_finite():
    n = 20
    groups = np.array([0] * 10 + [1] * 10)
    beta = _draw("group_horseshoe", n, groups=groups)
    assert beta.shape == (n,)
    assert np.all(np.isfinite(beta))


def test_group_horseshoe_single_group_reduces_to_shared_scale():
    n = 12
    groups = np.zeros(n, dtype=int)  # everything in one group
    beta = _draw("group_horseshoe", n, groups=groups)
    assert beta.shape == (n,)
    assert np.all(np.isfinite(beta))


def test_unknown_prior_raises():
    with pytest.raises(ValueError):
        sample_weights("not_a_real_prior", 5)


def test_spike_slab_shape_and_finite():
    n = 25
    beta = _draw("spike_slab", n, pi0=0.2, spike_scale=0.01, slab_scale=1.0)
    assert beta.shape == (n,)
    assert np.all(np.isfinite(beta))


def test_graph_spike_slab_shape_and_finite():
    n = 15
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    cov = A @ A.T + n * np.eye(n)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    beta = _draw("graph_spike_slab", n, corr=corr, pi0=0.2, tau_pi=1.0)
    assert beta.shape == (n,)
    assert np.all(np.isfinite(beta))


def test_spike_slab_pip_recovers_slab_membership():
    # Hand-built posterior: gene 0's draws sit at the slab scale, gene 1's at the spike scale.
    draws = 500
    beta = np.zeros((draws, 2))
    rng = np.random.default_rng(0)
    beta[:, 0] = rng.normal(0, 1.0, size=draws)     # slab-scale draws
    beta[:, 1] = rng.normal(0, 0.01, size=draws)    # spike-scale draws
    pip = spike_slab_pip(beta, pi=0.2, spike_scale=0.01, slab_scale=1.0)
    assert pip.shape == (2,)
    assert pip[0] > 0.9    # clearly slab
    assert pip[1] < 0.1    # clearly spike


def test_spike_slab_pip_handles_per_gene_pi_array():
    draws = 200
    n = 3
    beta = np.random.default_rng(0).normal(0, 1.0, size=(draws, n))
    pi = np.full((draws, n), 0.5)
    pip = spike_slab_pip(beta, pi=pi, spike_scale=0.01, slab_scale=1.0)
    assert pip.shape == (n,)
    assert np.all((pip >= 0) & (pip <= 1))
