"""generate_mock_data.py — write reproducible mock GReX + phenotype tables to data/.

Mock-local policy: real controlled GReX never lives here; this produces same-schema synthetic
files so notebooks/tests run end-to-end offline. Reproducible via --seed. Output is git-ignored.

    python scripts/generate_mock_data.py --n-samples 500 --n-genes 200 --family gaussian
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ptgs_bc.simulate import simulate_dataset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate mock GReX + phenotype tables.")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--n-genes", type=int, default=200)
    p.add_argument("--n-causal", type=int, default=20)
    p.add_argument("--family", choices=["gaussian", "binomial"], default="gaussian")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    ds, true_w = simulate_dataset(n_samples=args.n_samples, n_genes=args.n_genes,
                                  n_causal=args.n_causal, family=args.family, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    grex = ds.grex.copy(); grex.index.name = "sample_id"
    grex.to_csv(args.out / "grex.tsv", sep="\t")

    pheno = ds.y.rename("trait").to_frame()
    if ds.covars is not None:
        pheno = pheno.join(ds.covars)
    pheno.index.name = "sample_id"
    pheno.to_csv(args.out / "phenotype.tsv", sep="\t")
    np.save(args.out / "true_weights.npy", true_w)

    print(f"wrote {args.out}/grex.tsv, phenotype.tsv, true_weights.npy "
          f"(n={ds.n_samples}, genes={ds.n_genes}, family={ds.family})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
