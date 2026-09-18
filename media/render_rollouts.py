"""Render side-by-side rollouts: PPO expert (left) vs CSIL+SOAR policy (right).

Run from the repository root:
    python media/render_rollouts.py

Writes media/<env>_expert_vs_csil_soar.gif and .mp4 (mp4 needs ffmpeg).
"""
import os
import subprocess
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from imitation_policies.utils import MLP

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    # env, expert zip, actor checkpoint, label, max steps, frame stride, gif fps
    ("CartPole-v1", "ppo_cartpolev1.zip", "checkpoints/csil_soar_CartPole-v1_K1_actor.pt",
     "CSIL+SOAR, 1 demonstration", 500, 4, 12),
    ("Acrobot-v1", "ppo_acrobotv1.zip", "checkpoints/csil_soar_Acrobot-v1_K50_actor.pt",
     "CSIL+SOAR, 50 demonstrations", 300, 1, 15),
]


def load_actor(path, state_dim, action_dim, hidden=(256, 256)):
    actor = MLP(state_dim, action_dim, hidden)
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    return actor


def rollout(env_name, policy, max_steps, seed=10_000):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, ret, done, steps = [env.render()], 0.0, False, 0
    while not done and steps < max_steps:
        a = policy(obs)
        obs, r, term, trunc, _ = env.step(a)
        frames.append(env.render())
        ret += r
        steps += 1
        done = term or trunc
    env.close()
    return frames, ret, steps


def label(img, text, font):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(255, 255, 255))
    d.text((10, 8), text, fill=(20, 20, 20), font=font)
    return img


def main():
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    out_dir = ROOT / "media"
    for env_name, expert_zip, actor_ckpt, right_label, max_steps, stride, fps in CASES:
        env = gym.make(env_name)
        sd, ad = env.observation_space.shape[0], env.action_space.n
        expert = PPO.load(str(ROOT / expert_zip), device="cpu")
        actor = load_actor(ROOT / actor_ckpt, sd, ad)

        def expert_policy(o):
            return int(expert.predict(o, deterministic=True)[0])

        def learner_policy(o):
            with torch.no_grad():
                return int(actor(torch.as_tensor(o, dtype=torch.float32)).argmax())

        f_e, r_e, n_e = rollout(env_name, expert_policy, max_steps)
        f_l, r_l, n_l = rollout(env_name, learner_policy, max_steps)
        print(f"{env_name}: expert return {r_e:.0f} in {n_e} steps, learner return {r_l:.0f} in {n_l} steps")

        n = max(len(f_e), len(f_l))
        f_e += [f_e[-1]] * (n - len(f_e))
        f_l += [f_l[-1]] * (n - len(f_l))
        frames = []
        for k in range(0, n, stride):
            left = label(Image.fromarray(f_e[k]), f"PPO expert   step {min(k, n_e)}", font)
            right = label(Image.fromarray(f_l[k]), f"{right_label}   step {min(k, n_l)}", font)
            canvas = Image.new("RGB", (left.width + right.width + 8, left.height), (255, 255, 255))
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width + 8, 0))
            canvas = canvas.resize((900, int(900 * canvas.height / canvas.width)), Image.LANCZOS)
            frames.append(canvas)
        stem = out_dir / f"{env_name.lower().replace('-v1', '')}_expert_vs_csil_soar"
        frames[0].save(str(stem) + ".gif", save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        tmp = out_dir / "_frames"
        tmp.mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            fr.save(tmp / f"{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", str(stem) + ".mp4"], check=False)
        for p in tmp.glob("*.png"):
            p.unlink()
        tmp.rmdir()
        print("wrote", stem)


if __name__ == "__main__":
    main()
