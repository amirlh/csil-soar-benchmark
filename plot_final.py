"""Produce the final per-(env, algo) panel plots showing mean ± std training
curves across K values, with paper-protocol 25th-percentile snapshot tracked.

For each (env, algo): one figure with K curves overlaid, x=episode, y=raw mean
return. Optionally a second figure showing q25 (paper protocol).

Usage:
    .venv/bin/python3 plot_final.py --csv curves_FINAL.csv --outdir plots_FINAL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K_COLORS = {
    1: "#9467bd", 2: "#d62728", 5: "#8c564b",
    10: "#e377c2", 20: "#1f77b4", 50: "#ff7f0e", 100: "#2ca02c",
}
EXPERT = {"CartPole-v1": 500.0, "Acrobot-v1": -61.8}


def ema(arr, a=0.5):
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def plot_one(df, env, algo, metric, outpath, smooth=0.4, cummax=False):
    """Plot per-K mean+/-std curves of `metric` vs episode.

    If cummax=True, applies cummax PER (seed, K) BEFORE the mean-across-seeds
    aggregation. This is the paper-faithful Q25 snapshot view (Watson Fig 4):
    each curve is monotone-increasing, the y-value at iter t = best Q25 seen
    in [0, t] for that seed. Disables EMA smoothing (cummax is already smooth).
    """
    sub = df[(df["env"] == env) & (df["algo"] == algo)]
    if sub.empty:
        print(f"  skip ({env}, {algo}, {metric}) — empty")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    Ks = sorted(sub["K"].unique())
    for K in Ks:
        s = sub[sub["K"] == K].sort_values(["seed", "iter"]).copy()
        if cummax:
            # cummax per (seed) chronologically before averaging across seeds
            s[metric] = s.groupby("seed")[metric].cummax()
        agg = s.groupby("iter").agg(
            ep=("episode", "mean"),
            v=(metric, "mean"),
            std=(metric, "std"),
        ).reset_index().sort_values("iter").fillna(0.0)
        c = K_COLORS.get(int(K), "#444")
        x = agg["ep"].values
        y = agg["v"].values
        s_ = agg["std"].values
        use_smooth = smooth > 0 and not cummax and len(y) > 3
        y_sm = ema(y, a=1 - smooth) if use_smooth else y
        ax.plot(x, y_sm, color=c, lw=2.4, label=f"K = {int(K)}")
        ax.fill_between(x, y - s_, y + s_, color=c, alpha=0.18, lw=0)
        if use_smooth:
            ax.plot(x, y, color=c, lw=0.6, alpha=0.3)

    if env in EXPERT:
        ax.axhline(EXPERT[env], ls="--", color="gray", lw=1.2,
                   label=f"PPO expert ({EXPERT[env]:.0f})")

    if cummax:
        metric_label = "Best Q25 so far (paper Fig 4 snapshot protocol)"
        title_metric = f"{metric} cummax"
    elif metric == "reward":
        metric_label = "Eval return (mean of 20 episodes)"
        title_metric = "mean"
    else:
        metric_label = "Eval return (25th percentile of 20 episodes — paper protocol)"
        title_metric = "q25"
    ax.set_xlabel("Training episodes", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{env} — {algo} ({title_metric})", fontsize=12, pad=8)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="curves_FINAL.csv")
    p.add_argument("--outdir", default="plots_FINAL")
    p.add_argument("--smooth", type=float, default=0.4)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows; envs={sorted(df.env.unique())}, "
          f"algos={sorted(df.algo.unique())}, Ks={sorted(df.K.unique())}")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for env in sorted(df["env"].unique()):
        for algo in sorted(df["algo"].unique()):
            env_tag = env.lower().replace("-", "").replace("v1", "")
            algo_tag = algo.lower().replace("+", "").replace(" ", "")
            plot_one(df, env, algo, "reward",
                     outdir / f"{env_tag}_{algo_tag}_mean.png",
                     smooth=args.smooth)
            plot_one(df, env, algo, "q25",
                     outdir / f"{env_tag}_{algo_tag}_q25paper.png",
                     smooth=args.smooth)
            plot_one(df, env, algo, "q25",
                     outdir / f"{env_tag}_{algo_tag}_q25cummax.png",
                     smooth=0.0, cummax=True)


if __name__ == "__main__":
    main()
