"""Train a PPO expert and save it as ppo_<env>.zip.

Alternative to the binome's `expert_pipeline.py` without the final render-window
step (so it can run in the background without opening a Gym viewer). Output
filename matches `setup_env_and_policies` convention so the resulting checkpoint
is a drop-in replacement.

Usage:
    python train_expert.py --env CartPole-v1                # uses default 200k timesteps
    python train_expert.py --env Acrobot-v1                 # uses default 5M timesteps
    python train_expert.py --env CartPole-v1 --timesteps 500000 --seed 42
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


DEFAULT_TIMESTEPS = {
    "CartPole-v1": 200_000,   # 500/500 reached in ~50k, 200k is comfortable margin
    "Acrobot-v1":  5_000_000, # needed to converge below -65 reward
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Gym env name (e.g. CartPole-v1)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"PPO training timesteps (default per env: {DEFAULT_TIMESTEPS})")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_eval_episodes", type=int, default=20,
                        help="Number of deterministic eval episodes after training")
    args = parser.parse_args()

    timesteps = args.timesteps if args.timesteps is not None else DEFAULT_TIMESTEPS.get(args.env, 1_000_000)
    save_path = f"ppo_{args.env.lower().replace('-', '')}"

    print(f"[expert] env={args.env} timesteps={timesteps:,} seed={args.seed}")
    print(f"[expert] save_path={save_path}.zip")

    env = Monitor(gym.make(args.env))
    model = PPO("MlpPolicy", env, verbose=0, device="cpu", seed=args.seed)
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(save_path)

    print(f"\n[expert] Evaluating over {args.n_eval_episodes} deterministic episodes...")
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes)
    print(f"[expert] DONE: reward {mean_r:.1f} +/- {std_r:.1f}")


if __name__ == "__main__":
    main()
