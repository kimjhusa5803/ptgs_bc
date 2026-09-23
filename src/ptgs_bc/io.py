"""io.py — the data contract and loaders for a PTGS analysis.

The whole pipeline speaks one contract: a `Dataset` of GReX (genetically regulated /
predicted expression), a trait, and optional covariates, aligned by sample. Real
GReX/phenotypes are controlled (mock-local data policy): notebooks/tests run on
`simulate.py` output with this same schema, and only the data *source* differs for real runs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Aligned inputs for building/evaluating a PTGS.

    - ``grex``:   (n_samples, n_genes) predicted expression, sample-indexed, gene-columned.
    - ``y``:      (n_samples,) trait, aligned to grex.index.
    - ``covars``: (n_samples, n_covars) nuisance covariates (age/sex/PCs), or None.
    - ``family``: "gaussian" (continuous) | "binomial" (case/control) — trait is general.
    """

    grex: pd.DataFrame
    y: pd.Series
    covars: pd.DataFrame | None = None
    family: str = "gaussian"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.grex.index.equals(self.y.index):
            raise ValueError("grex and y must share the same sample index (order included).")
        if self.covars is not None and not self.grex.index.equals(self.covars.index):
            raise ValueError("covars must share grex's sample index.")
        if self.family not in ("gaussian", "binomial"):
            raise ValueError(f"family must be 'gaussian' or 'binomial', got {self.family!r}")

    @property
    def n_samples(self) -> int: return self.grex.shape[0]
    @property
    def n_genes(self) -> int: return self.grex.shape[1]
    @property
    def genes(self) -> list[str]: return list(self.grex.columns)

    def subset(self, idx) -> "Dataset":
        """Row subset by positional indices (used by the CV harness)."""
        return Dataset(
            grex=self.grex.iloc[idx],
            y=self.y.iloc[idx],
            covars=None if self.covars is None else self.covars.iloc[idx],
            family=self.family, meta=self.meta,
        )


def load_dataset(grex_path: str | Path, pheno_path: str | Path,
                 trait: str, covar_cols: list[str] | None = None,
                 family: str = "gaussian", sample_col: str = "sample_id",
                 grex_uncertainty_path: str | Path | None = None) -> Dataset:
    """Load a `Dataset` from a GReX table + a phenotype table, aligned on ``sample_col``.

    GReX table: ``sample_col`` + one column per gene (gene-ID convention is whatever the
    file's header uses — e.g. bare Ensembl IDs — this loader doesn't assume a scheme).
    Phenotype table: ``sample_col`` + ``trait`` (+ whatever ``covar_cols`` names — never
    hardcoded here, always caller-supplied). Delimiter is auto-detected (tab/comma/space).

    ``grex_uncertainty_path``, if given, is a same-``sample_col``-indexed table loaded as-is
    into ``Dataset.meta["grex_uncertainty"]`` (no shape assumed yet — per-sample SE, posterior
    draws, and a per-gene reliability score are all still open per the roadmap; builders don't
    consume this yet, this just plumbs it through so it isn't lost once available).
    """
    grex = pd.read_csv(grex_path, sep=None, engine="python").set_index(sample_col)
    ph = pd.read_csv(pheno_path, sep=None, engine="python").set_index(sample_col)
    common = grex.index.intersection(ph.index)
    grex_only = grex.index.difference(ph.index)
    ph_only = ph.index.difference(grex.index)
    if len(grex_only) or len(ph_only):
        warnings.warn(
            f"load_dataset: sample mismatch between GReX and phenotype tables — "
            f"{len(grex_only)} sample(s) only in GReX, {len(ph_only)} only in phenotype; "
            f"keeping the {len(common)}-sample intersection.",
            stacklevel=2,
        )
    grex = grex.loc[common]
    y = ph.loc[common, trait]
    covars = ph.loc[common, covar_cols] if covar_cols else None

    meta = {}
    if grex_uncertainty_path is not None:
        unc = pd.read_csv(grex_uncertainty_path, sep=None, engine="python").set_index(sample_col)
        meta["grex_uncertainty"] = unc.loc[unc.index.intersection(common)]

    return Dataset(grex=grex, y=y, covars=covars, family=family, meta=meta)
