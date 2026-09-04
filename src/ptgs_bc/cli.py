"""cli.py — thin batch entry point (a wrapper over library functions, no logic here).

`ptgs-bc smoke` runs the elastic-net vs. Bayesian comparison on simulated data — a headless
check that the whole pipeline wires together. Real-data runs are driven from notebooks/scripts.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ptgs-bc", description="PTGS_BC utilities")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="run a small elastic-net vs. Bayesian benchmark on mock data")
    s.add_argument("--n-samples", type=int, default=300)
    s.add_argument("--n-genes", type=int, default=100)
    s.add_argument("--family", choices=["gaussian", "binomial"], default="gaussian")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--with-bayes", action="store_true", help="also run the (slow) Bayesian arm")
    args = p.parse_args(argv)

    if args.cmd == "smoke":
        from .benchmark import run_benchmark, summary_table
        from .builders import BayesBuilder, ElasticNetBuilder
        from .simulate import simulate_dataset
        ds, _ = simulate_dataset(n_samples=args.n_samples, n_genes=args.n_genes,
                                 family=args.family, seed=args.seed)
        builders = [ElasticNetBuilder()]
        if args.with_bayes:
            builders.append(BayesBuilder(num_warmup=200, num_samples=200))
        res = run_benchmark(builders, ds, outer_k=5, seed=args.seed)
        print(summary_table(res).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
