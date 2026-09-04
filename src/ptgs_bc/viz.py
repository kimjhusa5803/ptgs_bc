"""viz.py — performance-comparison figures (seaborn; dev extras).

Kept out of the notebooks so a figure the analysis relies on is defined once and testable.
Every arm runs through the SAME nested-CV folds (`cv.nested_cv`), so the comparison is *paired*
per fold, which these plots exploit:

- `plot_comparison`     : per-arm points (seaborn stripplot) + mean ± sd (any #arms).
- `plot_paired_scatter` : arm-vs-arm per fold with a y=x line (2 arms; the paper's Fig-3a view).
- `plot_performance`    : a 1×2 figure combining both (summary only if !=2 arms).

seaborn/matplotlib import lazily (only when a plot is drawn) so `import ptgs_bc` stays light.
All return the axes/figure and never call `show()`, so they work headless and in tests.
"""

from __future__ import annotations

import numpy as np

from .results import BenchmarkResult


def _metric_name(results: dict[str, BenchmarkResult]) -> str:
    return next(iter(results.values())).metric_name if results else ""


def _long_df(results: dict[str, BenchmarkResult]):
    import pandas as pd
    rows = [{"builder": name, "fold": f.fold, "value": f.metric}
            for name, r in results.items() for f in r.folds]
    return pd.DataFrame(rows)


def plot_comparison(results: dict[str, BenchmarkResult], ax=None):
    """Per-arm fold points (stripplot) with a mean ± sd marker overlaid."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    order = list(results)
    df = _long_df(results)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.8 * max(len(order), 2), 4))
    sns.stripplot(data=df, x="builder", y="value", order=order, ax=ax,
                  hue="builder", palette="deep", legend=False,
                  alpha=0.6, size=7, jitter=0.15, zorder=2)
    for i, name in enumerate(order):
        r = results[name]
        ax.errorbar(i, r.mean, yerr=r.sd, fmt="o", color="black", capsize=5,
                    markersize=7, zorder=5)
        ax.annotate(f"{r.mean:.3f}", (i, r.mean), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel(""); ax.set_ylabel(_metric_name(results))
    ax.set_title("PTGS performance (nested CV: mean ± sd, dots = folds)")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20); lbl.set_ha("right")
    ax.margins(x=0.2)
    return ax


def plot_paired_scatter(results: dict[str, BenchmarkResult], ax=None):
    """Per-fold arm-A vs arm-B with a y=x reference (exactly two arms)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    if len(results) != 2:
        raise ValueError("plot_paired_scatter needs exactly two arms.")
    (na, ra), (nb, rb) = list(results.items())
    a = {f.fold: f.metric for f in ra.folds}
    b = {f.fold: f.metric for f in rb.folds}
    folds = sorted(set(a) & set(b))
    xs = np.array([a[f] for f in folds]); ys = np.array([b[f] for f in folds])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    lo = float(min(xs.min(), ys.min())); hi = float(max(xs.max(), ys.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", zorder=1)
    sns.scatterplot(x=xs, y=ys, ax=ax, s=70, zorder=2)
    ax.set_xlabel(f"{na}  ({_metric_name(results)})")
    ax.set_ylabel(f"{nb}  ({_metric_name(results)})")
    wins = int((ys > xs).sum())
    ax.set_title(f"{nb} > {na} in {wins}/{len(folds)} folds")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_performance(results: dict[str, BenchmarkResult]):
    """Combined figure: summary (mean ± sd + folds) and, for two arms, the paired scatter."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    if len(results) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        plot_comparison(results, ax=axes[0])
        plot_paired_scatter(results, ax=axes[1])
    else:
        fig, ax = plt.subplots(figsize=(1.8 * max(len(results), 2), 4))
        plot_comparison(results, ax=ax)
    fig.tight_layout()
    return fig


#  --- hyperparameter-tuning plots (optimization variation across parameters) ---

def plot_enet_path(path: dict, ax=None):
    """Elastic-net inner-CV surface: metric vs the penalty (log scale), one line per l1_ratio.

    `path` comes from `tuning.enet_cv_path` (or a bundle's ``extras['cv']``).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4))
    l1s = path["l1_ratios"]
    alphas = np.atleast_2d(path["alphas"])
    key = "cv_mse" if "cv_mse" in path else "cv_score"
    vals = np.atleast_2d(path[key])
    per_row = vals.shape[0] == len(l1s)     # gaussian: (n_l1, n_alpha); binomial: (n_C, n_l1)
    for i, l1 in enumerate(l1s):
        a = alphas[i] if alphas.shape[0] == len(l1s) else alphas[0]
        v = vals[i] if per_row else vals[:, i]
        ax.plot(np.log10(a), v, marker="", label=f"{l1}")
    ax.set_xlabel(f"log10({path['param']})  (penalty)")
    ax.set_ylabel("inner-CV MSE" if key == "cv_mse" else "inner-CV score")
    ax.set_title(f"Elastic-net tuning — chosen {path['chosen']}")
    ax.legend(title="l1_ratio", fontsize=7, ncol=2)
    return ax


def plot_param_sweep(df, param: str = "p0", ax=None):
    """Performance (mean ± sd band) vs a tuned parameter — e.g. the Bayesian p0 sweep."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    x = df[param].to_numpy(); m = df["mean"].to_numpy(); s = df["sd"].to_numpy()
    ax.plot(x, m, marker="o", zorder=3)
    ax.fill_between(x, m - s, m + s, alpha=0.2, zorder=1)
    best = df.loc[df["mean"].idxmax()]
    ax.axvline(best[param], ls="--", color="gray", zorder=2)
    ylabel = df["metric"].iloc[0] if "metric" in df.columns else "metric"
    ax.set_xlabel(param); ax.set_ylabel(ylabel)
    ax.set_title(f"Bayesian tuning — {ylabel} vs {param} (best {param}={best[param]})")
    return ax


def plot_prior_comparison(compare_df, ax=None):
    """Prior-family WAIC comparison: elpd_waic ± se per prior (higher = better)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    d = compare_df.sort_values("elpd_waic")
    ax.errorbar(d["elpd_waic"], d["prior"], xerr=d["se"], fmt="o", capsize=4)
    ax.set_xlabel("elpd_waic  (higher = better)"); ax.set_ylabel("")
    ax.set_title("Prior comparison (WAIC)")
    return ax


# Back-compat alias (earlier name).
plot_fold_comparison = plot_comparison
