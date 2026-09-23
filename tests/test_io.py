"""Tests for io.load_dataset against the real GReX/phenotype file shapes.

GReX header shape confirmed by the user (2026-09-23, ROADMAP.md "Real-data I/O Q&A"):
wide format, one row per sample, first column `GRID` (BioVU-style subject ID), remaining
columns bare Ensembl gene IDs (no version suffix). Phenotype/covariate columns are caller-
supplied, never hardcoded here. These fixtures are synthetic (mock-local data policy) but
match that real shape so the loader is validated against it, not a placeholder convention.
"""

from pathlib import Path

import pytest

from ptgs_bc.io import load_dataset

GRIDS = ["R100001", "R100002", "R100003", "R100004"]
GENES = ["ENSG00000002549", "ENSG00000002822", "ENSG00000002933"]


def _write_grex_tsv(path: Path, grids=GRIDS, genes=GENES, sep="\t"):
    lines = [sep.join(["GRID", *genes])]
    for i, g in enumerate(grids):
        row = [g] + [f"{0.1 * i + 0.01 * j:.4f}" for j in range(len(genes))]
        lines.append(sep.join(row))
    path.write_text("\n".join(lines) + "\n")


def _write_pheno_csv(path: Path, grids=GRIDS, sep=","):
    # Deliberately different delimiter and a covariate set the loader has never seen a
    # name for, to check nothing is hardcoded.
    header = ["GRID", "trait", "pc1", "pc2", "site_batch"]
    lines = [sep.join(header)]
    for i, g in enumerate(grids):
        lines.append(sep.join([g, f"{i * 1.5:.2f}", f"{0.01 * i:.3f}", f"{-0.02 * i:.3f}", str(i % 2)]))
    path.write_text("\n".join(lines) + "\n")


def test_load_dataset_matches_real_grid_ensembl_shape(tmp_path):
    grex_path = tmp_path / "grex.tsv"
    pheno_path = tmp_path / "pheno.csv"
    _write_grex_tsv(grex_path)
    _write_pheno_csv(pheno_path)

    ds = load_dataset(
        grex_path, pheno_path,
        trait="trait", covar_cols=["pc1", "pc2", "site_batch"],
        sample_col="GRID",
    )

    assert ds.n_samples == len(GRIDS)
    assert ds.genes == GENES
    assert list(ds.covars.columns) == ["pc1", "pc2", "site_batch"]
    assert ds.grex.index.tolist() == ds.y.index.tolist() == ds.covars.index.tolist()


def test_load_dataset_handles_mismatched_sample_sets(tmp_path):
    grex_path = tmp_path / "grex.tsv"
    pheno_path = tmp_path / "pheno.csv"
    _write_grex_tsv(grex_path, grids=GRIDS)
    _write_pheno_csv(pheno_path, grids=GRIDS[:3])  # one fewer sample in pheno

    with pytest.warns(UserWarning, match="sample mismatch"):
        ds = load_dataset(grex_path, pheno_path, trait="trait", sample_col="GRID")

    assert ds.n_samples == 3
    assert set(ds.grex.index) == set(GRIDS[:3])


def test_load_dataset_no_warning_when_samples_match(tmp_path, recwarn):
    grex_path = tmp_path / "grex.tsv"
    pheno_path = tmp_path / "pheno.csv"
    _write_grex_tsv(grex_path)
    _write_pheno_csv(pheno_path)

    load_dataset(grex_path, pheno_path, trait="trait", sample_col="GRID")

    assert len(recwarn) == 0


def test_load_dataset_plumbs_optional_grex_uncertainty(tmp_path):
    grex_path = tmp_path / "grex.tsv"
    pheno_path = tmp_path / "pheno.csv"
    unc_path = tmp_path / "grex_se.tsv"
    _write_grex_tsv(grex_path)
    _write_pheno_csv(pheno_path)
    _write_grex_tsv(unc_path)  # shape-compatible stand-in; real format still open

    ds = load_dataset(
        grex_path, pheno_path, trait="trait", sample_col="GRID",
        grex_uncertainty_path=unc_path,
    )

    assert "grex_uncertainty" in ds.meta
    assert ds.meta["grex_uncertainty"].shape[0] == ds.n_samples
