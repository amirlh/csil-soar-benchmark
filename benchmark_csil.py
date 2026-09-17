"""Multi-seed multi-K benchmark for CSIL and CSIL+SOAR.

Trains an imitation agent for each combination of (seed, K) on a given env.
Records both the BC baseline (post Stage A) and the full algo final reward
(post Stage C) so we can quantify the contribution of the SAC fine-tune.

Algo dispatched by --algo {csil,csil_soar}. CSIL+SOAR adds an L-critic ensemble
with UCB optimistic Q (IL-SOAR paper Algorithm 5, reward-flipped to mean+std).

Outputs:
    benchmark_<algo>_<env>.csv        one row per (seed, K, algo)
    checkpoints/<algo>_<env>_K<K>_seed<seed>.pt

Usage:
    python benchmark_csil.py --env CartPole-v1 --algo csil      --seeds 0 1 2 --Ks 1 5 10 25
    python benchmark_csil.py --env Acrobot-v1  --algo csil_soar --seeds 0 1 2 --Ks 1 2 3 5 10
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from imitation_policies.csil import CSILAgent, default_config
from imitation_policies.csil_soar import CSILSOARAgent, default_config_soar
from imitation_policies.imitation_policy import Transition


def collect_k_trajectories(env, expert, K: int, seed: int = 0) -> list[Transition]:
    transitions: list[Transition] = []
    n_complete = 0
    obs, _ = env.reset(seed=seed)
    while n_complete < K:
        action, _ = expert.predict(obs, deterministic=True)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        transitions.append(Transition(obs, int(action), next_obs, done, reward=reward))
        obs = next_obs
        if done:
            n_complete += 1
            obs, _ = env.reset()
    return transitions


def evaluate(env, agent, n_eps: int = 20, eval_seed_base: int = 10_000) -> tuple[float, float, float]:
    """Returns (mean, std, q25) over n_eps deterministic episodes with fixed seeds.

    Q25 is the 25th percentile — paper-faithful summary statistic (Watson et al.
    2023 Fig 4 caption: "performance is chosen using the highest 25th percentile
    of the episode return during learning").
    """
    rewards = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=eval_seed_base + ep)
        total_r = 0.0
        done = False
        while not done:
            a = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total_r += r
        rewards.append(total_r)
    arr = np.asarray(rewards)
    return float(arr.mean()), float(arr.std()), float(np.percentile(arr, 25))


def rollout(env, agent, n_steps: int, obs=None):
    trs: list[Transition] = []
    if obs is None:
        obs, _ = env.reset()
    n_episodes = 0
    for _ in range(n_steps):
        a = agent.predict(obs, deterministic=False)
        next_obs, reward, term, trunc, _ = env.step(a)
        done = term or trunc
        trs.append(Transition(obs, a, next_obs, done, reward=reward))
        obs = next_obs
        if done:
            n_episodes += 1
            obs, _ = env.reset()
    return trs, obs, n_episodes


def run_one(env_name: str, K: int, seed: int, n_iters_c: int, learner_steps: int,
            algo: str = "csil",
            n_grad_steps_per_iter: int | None = None,
            alpha: float | None = None, beta: float | None = None,
            sigma_clip: float | None = None, L: int | None = None,
            bonus_scale: float | None = None, lr: float | None = None,
            curves_log: list | None = None,
            snapshot: bool = True,
            eval_n_eps: int = 20):
    """One training run for {csil, csil_soar}. Returns (BC_mean, BC_std, BC_q25,
    algo_mean, algo_std, algo_q25, snap_iter).

    Paper-faithful Q25 snapshot (Watson Fig 4 caption): tracks best per-checkpoint
    Q25 across Stage C, saves actor.state_dict() at the peak, restores before
    final evaluation. `snap_iter` is the iter index of the chosen snapshot (0 = BC).

    If curves_log is provided (a list), per-iter (mean, q25) are appended as dicts
    {env, seed, K, algo, iter, episode, reward, q25}. BC checkpoint (iter=0) included.
    """
    env = gym.make(env_name)
    expert = PPO.load(f"ppo_{env_name.lower().replace('-', '')}", env=env, device="cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    torch.manual_seed(seed)
    np.random.seed(seed)

    expert_batch = collect_k_trajectories(env, expert, K, seed=seed)

    if algo == "csil":
        cfg = default_config(env_name)
    elif algo == "csil_soar":
        cfg = default_config_soar(env_name)
        if sigma_clip is not None:
            cfg["sigma_clip"] = sigma_clip
        if L is not None:
            cfg["L"] = L
        if bonus_scale is not None:
            cfg["bonus_scale"] = bonus_scale
    else:
        raise ValueError(f"Unknown algo: {algo}")

    if n_grad_steps_per_iter is not None:
        cfg["n_grad_steps_per_iter"] = n_grad_steps_per_iter
    if alpha is not None:
        cfg["alpha"] = alpha
    if beta is not None:
        cfg["beta"] = beta
    if lr is not None:
        cfg["lr"] = lr
    Path("checkpoints").mkdir(exist_ok=True)
    save_path = f"checkpoints/{algo}_{env_name}_K{K}_seed{seed}.pt"
    AgentCls = CSILAgent if algo == "csil" else CSILSOARAgent
    agent = AgentCls(state_dim, action_dim, save_path=save_path, **cfg)

    agent.update_representation(expert_batch, learner_batch=[])
    r_bc_m, r_bc_s, r_bc_q25 = evaluate(env, agent, n_eps=eval_n_eps)
    algo_label = "CSIL" if algo == "csil" else "CSIL+SOAR"
    if curves_log is not None:
        curves_log.append(dict(env=env_name, seed=seed, K=K, algo=algo_label,
                               iter=0, episode=0, reward=r_bc_m, q25=r_bc_q25))

    agent.update_representation(expert_batch, learner_batch=[])

    # Paper-faithful Q25 snapshot (Watson 2023 Fig 4 caption): track the actor
    # state_dict at the checkpoint with the highest Q25, restore before final eval.
    # snap_iter=0 means BC was the best (Stage C never improved on it).
    best_q25 = r_bc_q25
    best_snapshot = copy.deepcopy(agent.actor.state_dict()) if snapshot else None
    snap_iter = 0

    eval_every = max(1, n_iters_c // 10)

    obs = None
    total_episodes = 0
    for it in range(n_iters_c):
        lb, obs, n_eps_iter = rollout(env, agent, learner_steps, obs)
        total_episodes += n_eps_iter
        agent.update_representation(expert_batch, lb)
        agent.update_policy(lb)

        if (it + 1) % eval_every == 0:
            r_m, _, r_q25 = evaluate(env, agent, n_eps=eval_n_eps)
            if curves_log is not None:
                curves_log.append(dict(env=env_name, seed=seed, K=K, algo=algo_label,
                                       iter=it + 1, episode=total_episodes,
                                       reward=r_m, q25=r_q25))
            if snapshot and r_q25 > best_q25:
                best_q25 = r_q25
                best_snapshot = copy.deepcopy(agent.actor.state_dict())
                snap_iter = it + 1

    if snapshot and best_snapshot is not None:
        agent.actor.load_state_dict(best_snapshot)

    r_csil_m, r_csil_s, r_csil_q25 = evaluate(env, agent, n_eps=eval_n_eps)
    agent.save()
    return r_bc_m, r_bc_s, r_bc_q25, r_csil_m, r_csil_s, r_csil_q25, snap_iter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=["CartPole-v1", "Acrobot-v1"])
    parser.add_argument("--algo", default="csil", choices=["csil", "csil_soar"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument("--n_iters_c", type=int, default=30, help="Stage C pipeline iterations")
    parser.add_argument("--learner_steps", type=int, default=1000, help="env steps per Stage C iter")
    parser.add_argument("--n_grad_steps_per_iter", type=int, default=None,
                        help="override n_grad_steps_per_iter (Q/actor updates per Stage C iter)")
    parser.add_argument("--alpha", type=float, default=None, help="override BC reward temperature")
    parser.add_argument("--beta", type=float, default=None, help="override SAC fine-tune temperature (β<α)")
    parser.add_argument("--sigma_clip", type=float, default=None,
                        help="(csil_soar only) UCB std clipping ceiling")
    parser.add_argument("--L", type=int, default=None,
                        help="(csil_soar only) ensemble size")
    parser.add_argument("--bonus_scale", type=float, default=None,
                        help="(csil_soar only) multiply UCB std bonus by this scale "
                             "(default 1.0 paper-faithful; >1 amplifies optimism for "
                             "greedy discrete actors)")
    parser.add_argument("--lr", type=float, default=None,
                        help="SAC learning rate (paper IL-SOAR Table 2: 1e-3; "
                             "CSIL ref doc: 3e-4)")
    parser.add_argument("--curves_csv", default=None,
                        help="optional path: log per-iter eval rewards to this CSV "
                             "(columns: env, seed, K, algo, iter, episode, reward, q25)")
    parser.add_argument("--no_snapshot", action="store_true",
                        help="disable Q25 snapshot restoration (paper-faithful default ON)")
    parser.add_argument("--eval_n_eps", type=int, default=20,
                        help="episodes per eval; 20 paper-faithful for stable Q25")
    args = parser.parse_args()

    algo_label = "CSIL" if args.algo == "csil" else "CSIL+SOAR"
    print(f"=== Benchmark {algo_label} | env={args.env} | seeds={args.seeds} | Ks={args.Ks} ===")
    print(f"    Stage C: {args.n_iters_c} iters × {args.learner_steps} env_steps = {args.n_iters_c * args.learner_steps:,} total env_steps per run\n")

    rows = []
    curves_log: list = [] if args.curves_csv else None
    total = len(args.seeds) * len(args.Ks)
    done = 0
    for seed in args.seeds:
        for K in args.Ks:
            done += 1
            print(f"[{done}/{total}] seed={seed} K={K} ...")
            (r_bc_m, r_bc_s, r_bc_q25,
             r_algo_m, r_algo_s, r_algo_q25, snap_iter) = run_one(
                args.env, K, seed,
                algo=args.algo,
                n_iters_c=args.n_iters_c,
                learner_steps=args.learner_steps,
                n_grad_steps_per_iter=args.n_grad_steps_per_iter,
                alpha=args.alpha,
                beta=args.beta,
                sigma_clip=args.sigma_clip,
                L=args.L,
                bonus_scale=args.bonus_scale,
                lr=args.lr,
                curves_log=curves_log,
                snapshot=not args.no_snapshot,
                eval_n_eps=args.eval_n_eps,
            )
            rows.append({"env": args.env, "seed": seed, "K": K, "algo": "BC",
                         "reward_mean": r_bc_m, "reward_std": r_bc_s,
                         "reward_q25": r_bc_q25, "snap_iter": 0})
            rows.append({"env": args.env, "seed": seed, "K": K, "algo": algo_label,
                         "reward_mean": r_algo_m, "reward_std": r_algo_s,
                         "reward_q25": r_algo_q25, "snap_iter": snap_iter})
            print(f"        BC:   m={r_bc_m:7.1f} q25={r_bc_q25:7.1f}   |   "
                  f"{algo_label}: m={r_algo_m:7.1f} q25={r_algo_q25:7.1f}   "
                  f"snap@iter={snap_iter}   gap_q25={r_algo_q25 - r_bc_q25:+.1f}")

    df = pd.DataFrame(rows)
    out_path = f"benchmark_{args.algo}_{args.env}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(rows)} rows)")

    if args.curves_csv and curves_log:
        curves_df = pd.DataFrame(curves_log)
        curves_path = args.curves_csv
        if Path(curves_path).exists():
            existing = pd.read_csv(curves_path)
            curves_df = pd.concat([existing, curves_df], ignore_index=True)
        curves_df.to_csv(curves_path, index=False)
        print(f"Saved {curves_path} ({len(curves_log)} new rows)")

    print("\n=== Summary (mean over seeds) ===")
    summary = (
        df.groupby(["env", "K", "algo"])["reward_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["K", "algo"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
