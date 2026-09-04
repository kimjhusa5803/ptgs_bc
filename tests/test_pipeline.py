"""Fast tests over the shared pipeline (simulate -> build -> score -> evaluate) and contracts.

The Bayesian arm (NumPyro) is exercised separately/slowly; these keep the harness honest and
fast so they run on every change.
"""

import numpy as np
import pytest

from ptgs_bc import (ElasticNetBuilder, ScoreBundle, compute_score, evaluate,
                     nested_cv, run_benchmark, simulate_dataset, summary_table)
from ptgs_bc.io import Dataset


def test_simulate_shapes_and_contract():
    ds, w = simulate_dataset(n_samples=120, n_genes=40, n_causal=5, seed=1)
    assert ds.n_samples == 120 and ds.n_genes == 40
    assert ds.grex.index.equals(ds.y.index)
    assert w.shape == (40,) and np.count_nonzero(w) == 5


def test_dataset_alignment_guard():
    ds, _ = simulate_dataset(n_samples=50, n_genes=10, seed=0)
    bad_y = ds.y.reset_index(drop=True)
    with pytest.raises(ValueError):
        Dataset(grex=ds.grex, y=bad_y, family="gaussian")


def test_elastic_net_builds_and_scores():
    ds, _ = simulate_dataset(n_samples=200, n_genes=50, n_causal=8, seed=2)
    bundle = ElasticNetBuilder().fit(ds, seed=0)
    assert isinstance(bundle, ScoreBundle)
    assert bundle.weights.shape == (ds.n_genes,)
    s = compute_score(bundle, ds.grex, ds.covars)
    assert s.shape == (ds.n_samples,)


def test_score_gene_alignment_by_name():
    ds, _ = simulate_dataset(n_samples=100, n_genes=20, seed=3)
    bundle = ElasticNetBuilder().fit(ds, seed=0)
    shuffled = ds.grex[list(ds.grex.columns[::-1])]     # reversed column order
    s_ref = compute_score(bundle, ds.grex)
    s_shuf = compute_score(bundle, shuffled)
    assert np.allclose(s_ref, s_shuf)                    # aligned by name, not position


def test_bundle_roundtrip(tmp_path):
    ds, _ = simulate_dataset(n_samples=80, n_genes=15, seed=4)
    bundle = ElasticNetBuilder().fit(ds, seed=0)
    p = bundle.save(tmp_path / "b.joblib")
    again = ScoreBundle.load(p)
    assert np.allclose(bundle.weights, again.weights)


def test_nested_cv_and_metric_recovers_signal():
    ds, _ = simulate_dataset(n_samples=400, n_genes=60, n_causal=10,
                             effect_sd=0.6, noise_sd=1.0, seed=5)
    res = nested_cv(ElasticNetBuilder(), ds, outer_k=5, seed=0)
    assert len(res.folds) == 5
    assert res.metric_name == "partial_R2"
    assert res.mean > 0.0        # some signal recovered on data with a known answer


def test_partial_r2_and_auc_dispatch():
    y = np.array([0, 1, 0, 1, 1, 0], float)
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    name, val = evaluate(y, s, None, "binomial")
    assert name == "AUC" and 0.5 <= val <= 1.0


def test_run_benchmark_summary():
    ds, _ = simulate_dataset(n_samples=200, n_genes=30, seed=6)
    res = run_benchmark([ElasticNetBuilder()], ds, outer_k=5, seed=0)
    tbl = summary_table(res)
    assert "elastic_net" in set(tbl["builder"])
