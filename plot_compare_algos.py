"""Cross-algo comparison panel: 1x3 subplots (one per K), each overlaying
CSIL and CSIL+SOAR Q25-cummax curves. Paper-faithful (Watson Fig 4 snapshot).

Usage:
    .venv/bin/python3 plot_compare_algos.py --csv curves_3000eps_snapshot.csv \
        --out plots_3000eps_snapshot/acrobot_compare_q25cummax.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ALGO_STYLES = {
    "CSIL":       dict(color="#1f77b4", linestyle="-",  marker="o"),
    "CSIL+SOAR":  dict(color="#d62728", linestyle="-",  marker="s"),
}
EXPERT = {"CartPole-v1": 500.0, "Acrobot-v1": -61.8}


def plot_panel(df, env, Ks, outpath):
    fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 4.5),
                              sharey=True)
    if len(Ks) == 1:
        axes = [axes]

    for ax, K in zip(axes, Ks):
        for algo, style in ALGO_STYLES.items():
            s = df[(df.env == env) & (df.algo == algo) & (df.K == K)].copy()
            if s.empty:
                continue
            # cummax per seed BEFORE averaging across seeds (correct order)
            s = s.sort_values(["seed", "iter"])
            s["q25_cm"] = s.groupby("seed")["q25"].cummax()
            agg = s.groupby("iter").agg(
                ep=("episode", "mean"),
                v=("q25_cm", "mean"),
                std=("q25_cm", "std"),
            ).reset_index().sort_values("iter").fillna(0.0)
            x = agg["ep"].values
            y = agg["v"].values
            sd = agg["std"].values
            ax.plot(x, y, lw=2.4, label=algo, **style)
            ax.fill_between(x, y - sd, y + sd, color=style["color"], alpha=0.18, lw=0)

        if env in EXPERT:
            ax.axhline(EXPERT[env], ls="--", color="gray", lw=1.0,
                       label=f"PPO expert ({EXPERT[env]:.0f})" if K == Ks[0] else None)

        ax.set_title(f"K = {int(K)}", fontsize=12)
        ax.set_xlabel("Training episodes", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if K == Ks[0]:
            ax.set_ylabel("Best Q25 so far (paper snapshot)", fontsize=11)
            ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    fig.suptitle(f"{env} — CSIL vs CSIL+SOAR (Q25-cummax, paper-faithful)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="curves_3000eps_snapshot.csv")
    p.add_argument("--out", default="plots_3000eps_snapshot/acrobot_compare_q25cummax.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    Ks = sorted(df["K"].unique())
    for env in sorted(df["env"].unique()):
        out = Path(args.out)
        if len(df["env"].unique()) > 1:
            out = out.with_stem(out.stem + f"_{env.lower().replace('-','')}")
        plot_panel(df, env, Ks, out)


if __name__ == "__main__":
    main()
